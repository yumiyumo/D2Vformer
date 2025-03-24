import os
import numpy as np
import torch
import shutil
from torch import optim, nn
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


# scene3
'''
EXP for Skip Prediction
The difference between different EXPs lies in the data reading method, model testing, and accuracy comparison
Other models and D2V models can be trained in EXP_SKIP. In order to facilitate accuracy comparison, exp is not separated for training
'''


class EXP_SKIP_V2:
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
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        # Determine the gap length
        self.gap_len = args.gap_len

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
        self.stride = args.stride
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




    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_loader = self.trainloader
        self.args.alpha = 0.2
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)  # Get the values ​​corresponding to different seq_len

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def _get_data(self):

        # To obtain data, based on different datasets, you need to change the get_data function and the MyDataset function
        train, valid, test, mean, scale, dim = get_data(self.data_path, self.date_path, args=self.args)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        # Use a new data segmentation method based on the model name. The D2V model needs to use a new data segmentation method.

        trainset = Skip_Dataset(train, self.gap_len, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.pred_len)
        validset = Skip_Dataset(valid, self.gap_len, seq_len=self.seq_len, label_len=self.label_len,
                                pred_len=self.pred_len)
        testset = Skip_Dataset(test, self.gap_len, seq_len=self.seq_len, label_len=self.label_len,
                               pred_len=self.pred_len)


        self.trainloader = DataLoader(trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)



        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return

    def _get_model(self):
        # get model
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.device = self.device
        self.args.output_path = self.output_path

        print('------------Equipment used---------------')
        print(self.device)

        # -------------------------------------------------------------
        # Select a model based on the model name
        # -------------------------------------------------------------
        self.skip = False

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

        if self.model_name == 'FITS':
            self.args.individual = False
            self.model = FITS(self.args)

        if self.model_name == 'TimeMixer':
            self.model = TimeMixer(self.args)

        if self.model_name == 'D2Vformer':
            self.model = D2Vformer_GLAFF(self.args)
            self.skip = True



        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        self.parameter_num=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("The number of parameters that the model can learn:", self.parameter_num)
        print('\n', flush=True)


        #   早停机制
        path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
                                            verbose=self.verbose, path=path, )

        #   损失函数
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

        # Because batch_y and batch_y_mark cut gap_len more when splitting data, but it is not used during training, so it is intercepted here;

        D2V_batch_y = batch_y[:, :-self.gap_len, :]
        D2V_batch_y_mark = batch_y_mark[:, :-self.gap_len, :]
        D2V_batch_y_mark = D2V_batch_y_mark[:, -self.pred_len:, :]

        if self.model_name in ['D2Vformer','DeepTD_patch_koopa_V2']:
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, D2V_batch_y, D2V_batch_y_mark, 'test')
            else:
                outputs = self.model(batch_x, batch_x_mark, D2V_batch_y, D2V_batch_y_mark, None) # This means that your loss training will not have the gap_len part of the loss
            loss = self.criterion(outputs, D2V_batch_y[:, -self.pred_len:, :])

        elif self.model_name=='FITS':
            pred_len = self.pred_len + self.gap_len
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
                loss = self.criterion(outputs, batch_y[:, -pred_len:, :])
            else:
                outputs,loss_ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None) # This means that your loss training will not have the gap_len part of the loss
                loss = self.criterion(outputs, batch_y[:, -pred_len:, :])
                loss=loss+loss_
        else:
            pred_len = self.pred_len + self.gap_len
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None)
            # --------------------------------------------------------
            # Here we need to use slicing to extract the label of the predicted part
            # --------------------------------------------------------

            loss = self.criterion(outputs, batch_y[:, -pred_len:, :])

        return outputs, loss

    def _process_one_batch_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        # Because skip_prediciton involves data segmentation of real values, in order not to affect the original training, it is divided into training and testing.
        # For D2V, when testing, you only need to enter the mark of the gap part
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        gap_y = batch_y[:, -self.gap_len:, :]
        gap_mark = batch_y_mark[:, -self.gap_len:, :]

        if self.model_name in ['D2Vformer','DeepTD_patch_koopa_V2']:
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, gap_y, gap_mark, 'test')
            else:
                outputs = self.model(batch_x, batch_x_mark, gap_y, gap_mark, None)
        else:
            if mode == 'test':
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None)
                outputs = outputs[:, -self.gap_len:, :]

        # --------------------------------------------------------
        # Here we only compare the prediction accuracy of the jump prediction part, that is, the gap part
        # --------------------------------------------------------
        loss = self.criterion(outputs, gap_y)
        return outputs, loss

    def train(self):
        self.train_time=[]
        save_manager = utils.save.SaveManager(self.args.output_dir, self.args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(self.args)
        for e in range(self.epochs):
            train_time1=datetime.now()
            self.model.train()
            train_loss = []

            # ------------------------------------------------------
            # tqdm is a dynamic progress bar
            # trainloader just adds a batchsize dimension to the input data
            # ------------------------------------------------------
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                # ------------------------------------------------
                #   [batch_size,seq_len,feature_
                # ------------------------------------------------
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_time2 = datetime.now()
            self.train_time.append(train_time2-train_time1)

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):

                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                # This is not the actual test of skip prediction, but the test of normal training, so the _process_one_batch_train function is still used

                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model, e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        output_path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=self.args)
        print(f'Training Time:{np.mean(self.train_time)}', flush=True)

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
        test_time1=datetime.now()
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            if self.skip:
                pred, loss = self._process_one_batch_test(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            else:
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')

            # Here, both the prediction part and the true value take the final gap_len. Since D2V directly outputs the gap part, the interception here has no effect on it.
            preds.extend(pred.detach().cpu().numpy()[:, -self.gap_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.gap_len:, :])
        test_time2= datetime.now()
        self.test_time=test_time2-test_time1
        print(f'Test Time:{self.test_time}', flush=True)
        mape_error = np.mean(self.mean) * 0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        # Denormalization:
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean

        mape = np.mean(np.abs(dstand_trues - dstand_preds) / (dstand_trues + mape_error))

        print('SKIP Prediction Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae, mape))

        # Store the final numerical result
        np.save(os.path.join(self.output_path, self.args.data_name + '_preds'), preds)
        np.save(os.path.join(self.output_path, self.args.data_name + '_trues'), trues)

        # Create a csv file to record the training process
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/skip_pred_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'best_mae', 'mse', 'mape', 'seq_len',
                           'pred_len', 'train_time','test_time','parameters','output_path','info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'best_mae': mae, 'mse': mse, 'mape': mape, 'seq_len': self.seq_len,
                   'pred_len': self.pred_len,
                   'train_time':self.train_time,'test_time':self.test_time,'parameters':self.parameter_num,
                  'output_path':self.output_path,'info': self.info}]
        write_csv_dict(log_path, a_log, 'a+')





