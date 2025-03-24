import os
import shutil
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
import utils.config
import utils.save
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
from datetime import datetime
from layers.Quantile_loss import *
import numpy as np
import torch
from torch import optim, nn
# scene1
'''
EXP for Flexiable Preidiction
The differences from different EXPs are the way the data is read, the model is tested, and the accuracy is compared
Both other models and D2V models can be trained in EXP_SKIP, so they are not trained separately in order to facilitate comparison of accuracy
'''


class EXP_Flexiable:
    def __init__(self, args):
        assert args.resume_dir == args.output_dir
        if args.output_dir == None or args.output_dir == 'None' or args.output_dir == 'none':
            args.output_dir = None
            utils.config.create_output_dir(args)  # Create a directory for the output
            args.resume_dir = args.output_dir
        else:
            args.output_dir = os.path.join('experiments', args.output_dir)
            args.resume_dir = os.path.join('experiments', args.resume_dir)
        output_path = os.path.join(args.output_dir, args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path
        resume_path = os.path.join(args.resume_dir, args.model_name)
        if not os.path.exists(resume_path):
            raise print('No path was found to read the pre-trained weights')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        # Predicted length of D2V test and baseline training test
        self.pred_len = args.pred_len
        # Predicted length during D2V training
        self.d2v_train_pred_len = args.d2v_train_pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        self.args = args

        self.train_gpu = [1, ]
        self.devices = [0, ]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.e_layers = args.e_layers
        self.d_layers = args.d_layers
        self.patch_len = args.patch_len
        # FIXME stride is 1/3 of the patch_len
        self.stride = args.patch_len//3
        self.n_heads = args.n_heads
        self.info = args.info



        # Calculate the current time and save it for subsequent results
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")



        # -------------------------------------------
        # All data names should be named in a consistent format
        # And the csv should be processed into a uniform format [date,dim1,dim2......]
        # -------------------------------------------

        if self.args.data_name == 'ETTh1':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTh1.csv'

        if self.args.data_name == 'ETTh2':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTh2.csv'

        if self.args.data_name == 'ETTm1':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTm1.csv'

        if self.args.data_name == 'ETTm2':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTm2.csv'

        if self.args.data_name == 'traffic':
            self.date_path = './datasets/traffic/usa_sanfrancisco.csv'
            self.data_path = './datasets/traffic/traffic.csv'

        if self.args.data_name == 'electricity':
            self.date_path = './datasets/electricity/portugal.csv'
            self.data_path = './datasets/electricity/electricity.csv'

        if self.args.data_name == 'exchange':

            self.date_path = None
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'

        if self.args.data_name == 'illness':
            self.date_path = None
            self.data_path = './datasets/illness/national_illness.csv'
        self._get_data()
        self._get_model()

    '''Shuffle the trained, validated set'''

    def _get_data(self):

        # To obtain data, based on different datasets, you need to change the get_data function and the MyDataset function
        train, valid, test, mean, scale, dim = get_data(self.data_path, self.date_path, args=self.args)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        # The new data segmentation method is selected according to the model name, and the new data segmentation method is required for the D2V model
        # The prediction length of the D2V training and validation sets is different from the prediction length of the test
        D2V_trainset = flexible_D2V_Dataset(train, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.d2v_train_pred_len)
        D2V_validset = flexible_D2V_Dataset(valid, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.d2v_train_pred_len)

        # The prediction length of the baseline is the same as that of the D2V test
        baseline_trainset = flexible_baseline_Dataset(train, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.pred_len)
        baseline_validset = flexible_baseline_Dataset(valid, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.pred_len)

        D2V_testset = flexible_baseline_Dataset(test, seq_len=self.seq_len, label_len=self.label_len,
                               pred_len=self.pred_len)


        self.D2V_trainloader = DataLoader(D2V_trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        self.baseline_trainloader = DataLoader(baseline_trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)

        self.D2V_validloader = DataLoader(D2V_validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.baseline_validloader = DataLoader(baseline_validset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.testloader = DataLoader(D2V_testset, batch_size=self.batch_size, shuffle=False, drop_last=True)



        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(D2V_trainset), len(D2V_validset), len(D2V_testset)))

        return
    
    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_loader = self.D2V_trainloader
        self.args.alpha = 0.2
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)  # Get the corresponding values of different seq_len

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def _get_model(self):
        # get model
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.device = self.device
        self.args.output_path = self.output_path

        print('------------Equipment used--------------')
        print(self.device)

        # -------------------------------------------------------------
        # Select a model based on the model name
        # -------------------------------------------------------------
        if 'D2Vformer' in self.model_name:
            if self.args.data_name == 'exchange':
                self.args.mark_index = [1,2]

            if self.args.data_name == 'ETTh1':
                self.args.mark_index = [0,2,4,6,7,11,12,13,14,15,16,18,19,22,24,26]

            if self.args.data_name == 'traffic':
                self.args.mark_index = [0, 1, 2, 6, 7, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26]

        if self.model_name == 'DLinear':
            self.model = DLinear(self.args)

        if self.model_name == 'Autoformer':
            self.model = Autoformer(self.args)

        if self.model_name == 'PatchTST':
            self.model = PatchTST(self.args)

        if self.model_name == 'Fedformer':
            self.model = Fedformer(self.args)

        if self.model_name == 'D2Vformer':
            self.model = D2Vformer_GLAFF(self.args)

        if self.model_name == 'TimeMixer':
            self.model = TimeMixer(self.args)

        if self.model_name == 'FITS':
            self.args.individual = False
            self.model = FITS(self.args)



        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch - 1) // 1))

        # Early stop mechanism
        path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
                                            verbose=self.verbose, path=path, )

        # Loss function
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)

        if self.args.loss == 'normal':
            self.criterion = nn.MSELoss()

        return

    def _process_one_batch_train(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.model_name in ['D2Vformer','DeepTD_patch_koopa_V2']:
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, batch_y[:, -self.pred_len:, :], batch_y_mark[:, -self.pred_len:, :], 'test')
                loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y[:, -self.d2v_train_pred_len:, :], batch_y_mark[:, -self.d2v_train_pred_len:, :], None) # 相当于你的loss训练也不会有gap_len部分的loss
                loss = self.criterion(outputs, batch_y[:, -self.d2v_train_pred_len:, :])

        else:

            if mode == 'test':
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
                    loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
            else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None)
                    loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
            # --------------------------------------------------------
            # Here you need to use slices to remove the label of the prediction part
            # --------------------------------------------------------
        return outputs, loss



    def train(self):
        train_time=[]
        save_manager = utils.save.SaveManager(self.args.output_dir, self.args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(self.args)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("The number of parameters that the model can learn:", self.num_params)
        print('\n', flush=True)

        for e in range(self.epochs):
            begin = datetime.now()
            self.model.train()
            train_loss = []

            if self.model_name in ['DeepTD_patch_koopa_V2','D2Vformer']:
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.D2V_trainloader):
                    # ------------------------------------------------
                    #   [batch_size,seq_len,feture]
                    # ------------------------------------------------

                    self.optimizer.zero_grad()
                    pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                    train_loss.append(loss.item())
                    loss.backward()
                    self.optimizer.step()

                end = datetime.now()
                seconds = (end - begin).total_seconds()
                print(f'Training time:{seconds}')
                train_time.append(seconds)

                self.model.eval()
                valid_loss = []
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.D2V_validloader):

                    pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                    valid_loss.append(loss.item())

                train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
                print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} ".format(e + 1, train_loss,
                                                                                                       valid_loss,
                                                                                                       ))

                self.early_stopping(valid_loss, self.model, e)
                if self.early_stopping.early_stop:
                    break
                self.scheduler.step()
            else:
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.baseline_trainloader):
                    # ------------------------------------------------
                    #   [batch_size,seq_len,feture]
                    # ------------------------------------------------
                    self.optimizer.zero_grad()
                    pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                    train_loss.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                end = datetime.now()
                seconds = (end - begin).total_seconds()
                train_time.append(seconds)

                self.model.eval()
                valid_loss = []
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.baseline_validloader):

                    pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
                    valid_loss.append(loss.item())


                train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
                print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} ".format(e + 1, train_loss,
                                                                                                       valid_loss
                                                                                                       ))

                self.early_stopping(valid_loss, self.model, e)
                if self.early_stopping.early_stop:
                    break
                self.scheduler.step()

            output_path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
            self.load_best_model(path=output_path, args=self.args)

        # TODO load the best model
    def load_best_model(self, path, args=None):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('The path {0} does not exist, and the model parameters are randomly initialized.'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path)

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']

    def test(self):
        if self.args.resume:
            self.load_best_model(path=self.resume_path, args=self.args)
        self.model.eval()
        trues, preds = [], []
        begin = datetime.now()
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            end = datetime.now()
            # Here, both the prediction part and the true value take the final gap_len. Since D2V directly outputs the gap part, the interception here has no effect on it.
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.pred_len:, :])

        mape_error = np.mean(self.mean) * 0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        test_time = (end - begin).total_seconds()

        # Denormalization:
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean
        mape = np.mean(np.abs(dstand_trues - dstand_preds) / (dstand_trues + mape_error))

        print('Felxiable Prediction 1 Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae, mape))

        np.save(os.path.join(self.output_path, self.args.data_name + '_preds'),preds)
        np.save(os.path.join(self.output_path, self.args.data_name + '_trues'),trues)

        # Create a csv file to record the training process
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/flexible_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                            'best_mae', 'mse', 'mape', 'seq_len',
                           'pred_len', 'num_params', 'train_time','test_time','output_path','info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                   'best_mae': mae, 'mse': mse, 'mape': mape, 'seq_len': self.seq_len,
                   'pred_len': self.pred_len,
                  'num_params':self.num_params,
                  'train_time':self.train_time,
                  'test_time':test_time,
                  'output_path':self.output_path,'info': self.info}]
        write_csv_dict(log_path, a_log, 'a+')





