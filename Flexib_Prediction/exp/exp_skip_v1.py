import os
import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from utils import *
import utils.config
import utils.save
from data import *
from model import *
import datetime
from layers.Quantile_loss import *

# scene2 
class EXP_SKIP_V1:
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
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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
            self.date_path = ['./datasets/exchange_rate/china.csv', './datasets/exchange_rate/australia.csv',
                              './datasets/exchange_rate/british.csv',
                              './datasets/exchange_rate/canada.csv', './datasets/exchange_rate/japan.csv',
                              './datasets/exchange_rate/newzealand.csv',
                              './datasets/exchange_rate/singapore.csv', './datasets/exchange_rate/switzerland.csv',
                              './datasets/exchange_rate/usa.csv']
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
        
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        self.parameter_num=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("The number of parameters that the model can learn:", self.parameter_num)
        print('\n', flush=True)


        # Early stop mechanism
        path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
                                            verbose=self.verbose, path=path, )
        self.modelpath = path
        # Loss function
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)

        if self.args.loss == 'normal':
            self.criterion = nn.MSELoss()

        return
            


        # # Special model reading method for multi-gpu training
        # if ngpus_per_node > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.devices)

        # self.model.to(self.device)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        # # Special optimizer and decay method reading for multi-gpu training
        # if ngpus_per_node > 1:
        #     self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
        #     self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)

        # # Ealry stop mechanism
        # self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
        #                                     verbose=self.verbose, path=self.modelpath, )

        # # Loss function
        # if self.args.loss == 'quantile':
        #     self.criterion = QuantileLoss(self.args.quantiles)

        # if self.args.loss == 'normal':
        #     self.criterion = nn.MSELoss()

        # if self.args.resume:
        #     print('Loading a pre-trained model')
        #     checkpoint = torch.load(self.modelpath)  # Read the previously saved weight file (including optimizer and learning rate strategy)
        #     self.model.load_state_dict(checkpoint['model'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # return

    def _process_one_batch_train(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        if self.model_name == 'DeepTD_patch_koopa_V2' or self.model_name == 'D2Vformer':
            batch_y = batch_y[:, self.label_len:].float().to(self.device)
            batch_y_mark = batch_y_mark[:, self.label_len:].float().to(self.device)
            # The batch_y above has already cut off label_len by default
            if mode == 'test':
                # There must be a gap between val and test
                D2V_batch_y = batch_y[:,self.gap_len:, :]
                D2V_batch_y_mark = batch_y_mark[:, self.gap_len:, :]
                outputs = self.model(batch_x, batch_x_mark, D2V_batch_y, D2V_batch_y_mark, 'test')
            else:
                # When training, it is normal training without gap
                D2V_batch_y = batch_y[:, :self.pred_len, :]
                D2V_batch_y_mark = batch_y_mark[:, :self.pred_len, :]
                outputs = self.model(batch_x, batch_x_mark, D2V_batch_y, D2V_batch_y_mark, None)
            loss = self.criterion(outputs, D2V_batch_y)

        # Other models
        else:
            batch_y = batch_y.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            if mode == 'test':
                # loss only calculates the loss of pred_len
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
                outputs=outputs[:,self.gap_len:]
                loss = self.criterion(outputs, batch_y[:,self.label_len+self.gap_len:])
            else:
                # loss needs to calculate the loss of gap_len+pred_len
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None)
                loss = self.criterion(outputs, batch_y[:,self.label_len:])

        return outputs, loss


    def train(self):
        train_time=[]
        # No need to repeat calculations
        # self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Number of learnable parameters of the model:", self.num_params)
        print(flush=True)
        for e in range(self.epochs):
            begin = datetime.datetime.now()
            self.model.train()
            train_loss = []
            # ------------------------------------------------------
            # tqdm is a dynamic progress bar
            # trainloader just adds a batchsize dimension to the input data
            # ------------------------------------------------------
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                # ------------------------------------------------
                #   [batch_size,seq_len,feature]
                # ------------------------------------------------
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            end=datetime.datetime.now()
            seconds=(end-begin).total_seconds()
            train_time.append(seconds)

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))
            print(f'Training Time:{seconds}')
            self.early_stopping(valid_loss, self.model, e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        # Read the previously saved weight file (including optimizer and learning rate strategy)
        # Because it will be sent to the test function for testing, read the optimal model parameters
        
        checkpoint = torch.load(self.modelpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.train_time=train_time

    def test(self):
        self.model.eval()
        trues, preds = [], []
        test_time=[]
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            begin = datetime.datetime.now()
            if self.model_name == 'DeepTD_patch_koopa_V2':
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            else:
                pred, loss = self._process_one_batch_train(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')

            end = datetime.datetime.now()
            seconds=(end-begin).total_seconds()
            test_time.append(seconds)
            preds.extend(pred.detach().cpu().numpy())
            trues.extend(batch_y[:,-self.pred_len:].detach().cpu().numpy())

        mape_error = np.mean(self.mean) * 0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        # Denormalization:
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean

        mape = np.mean(np.abs(dstand_trues - dstand_preds) / (dstand_trues + mape_error))

        print('SKIP Prediction Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae, mape))
        test_time_mean=np.mean(test_time)
        print(f'test time:{test_time_mean}.')

        # Store the final numerical result
        np.save(os.path.join(self.output_path, self.args.data_name + '_preds'), preds)
        np.save(os.path.join(self.output_path, self.args.data_name + '_trues'), trues)

        # Create a csv file to record the training process
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/skip_pred_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse', 'mape', 'seq_len', 'label_len',
                           'pred_len','gap_len', 'd_model', 'd_ff', 'e_layers', 'd_layers',
                           'patch_len', 'stride', 'n_heads', 'T2V_outmodel','num_params','train_time(s/epoch)','test_time(s/batch)', 'info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse, 'mape': mape, 'seq_len': self.seq_len,
                  'label_len': self.label_len, 'pred_len': self.pred_len,'gap_len':self.gap_len,
                  'd_model': self.d_model, 'd_ff': self.d_ff, 'e_layers': self.e_layers, 'd_layers': self.d_layers,
                  'patch_len': self.patch_len, 'stride': self.stride,
                  'n_heads': self.n_heads, "T2V_outmodel": self.args.T2V_outmodel,'num_params':self.parameter_num,
                  'train_time':self.train_time,'test_time':test_time,
                  'info': self.info}]
        write_csv_dict(log_path, a_log, 'a+')





