import os
import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
import datetime
from layers.Quantile_loss import *



class EXP:
    def __init__(self,args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        self.args = args

        self.train_gpu = [1,]
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
        self.is_shuffle=args.is_shuffle

        # Create checkpoints to save training results
        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/'+self.model_name+'/'):
            os.makedirs('./checkpoint/'+self.model_name+'/')


        # Calculate the current time for saving results later
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        self.modelpath = './checkpoint/'+self.model_name+'/'+self.data_name+'_best_model.pkl'
        self.save_path='./checkpoint/'+self.model_name+'/'+self.data_name

        args.save_path=self.save_path # Path for visualization saving

        #-------------------------------------------
        #   All data should follow a unified naming format
        #   CSV files should be processed into the unified format [date, dim1, dim2......]
        #-------------------------------------------

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
            self.date_path = ['./datasets/exchange_rate/china.csv', './datasets/exchange_rate/australia.csv','./datasets/exchange_rate/british.csv',
                              './datasets/exchange_rate/canada.csv','./datasets/exchange_rate/japan.csv','./datasets/exchange_rate/newzealand.csv',
                              './datasets/exchange_rate/singapore.csv','./datasets/exchange_rate/switzerland.csv','./datasets/exchange_rate/usa.csv']
            self.date_path=None
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'
        
        if self.args.data_name == 'illness':
            self.date_path=None
            self.data_path = './datasets/illness/national_illness.csv'
        self._get_data()
        self._get_model()


    '''Shuffle training and validation set instances'''
    def shuffle_data(self, trainloader, validloader, testloader):
        '''Get all instances and shuffle them, keeping the test set unchanged. Only the training and validation sets are shuffled.'''
        '''~_ (underscore) represents shuffled results, while ~ (no underscore) represents unshuffled results.'''
        dataset_ = []
        for j in [trainloader, validloader]:
            for i in j:
                dataset_.append(i)
        np.random.shuffle(dataset_)  # Randomly shuffle the dataset
        # The test set remains unchanged; only training and validation sets are shuffled
        for i in testloader:
            dataset_.append(i)
        # Split the dataset into 60:20:20 for train, validation, and test sets
        total_len = len(dataset_)
        train_data_ = dataset_[:int(0.6 * total_len)]
        vaild_data_ = dataset_[int(0.6 * total_len):int(0.8 * total_len)]
        test_data_ = dataset_[int(0.8 * total_len):]
        trainset_ = MyDataset_shuffle(train_data_)
        vaildset_ = MyDataset_shuffle(vaild_data_)
        testset_ = MyDataset_shuffle(test_data_)

        trainloader_ = DataLoader(trainset_, batch_size=self.batch_size, shuffle=True, drop_last=True)
        validloader_ = DataLoader(vaildset_, batch_size=self.batch_size, shuffle=False, drop_last=True)
        testloader_ = DataLoader(testset_, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return trainloader_, validloader_, testloader_


    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_loader =self.trainloader
        self.args.alpha=0.2
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)  # 得到不同的seq_len对应的值

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def _get_data(self):

        # Get data, based on different datasets. Mainly need to modify get_data function and MyDataset function.
        train, valid, test, mean, scale, dim = get_data(self.data_path, self.date_path, args=self.args)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        if not self.is_shuffle:
            self.trainloader = DataLoader(trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
            self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
            self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if self.is_shuffle:
            trainloader = DataLoader(trainset, batch_size=1, shuffle=True, drop_last=True)
            validloader = DataLoader(validset, batch_size=1, shuffle=False, drop_last=True)
            testloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=True)

            # Get shuffled Dataloader
            self.trainloader, self.validloader, self.testloader = self.shuffle_data(trainloader, validloader,
                                                                                    testloader)

        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return

    def _get_model(self):
        # Get model
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('------------Using device---------------')
        print(self.device)

        # -------------------------------------------------------------
        # Select model based on model name
        # -------------------------------------------------------------

        # Special model loading method for multi-GPU training
        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        # Special optimizer and scheduler loading for multi-GPU training
        if ngpus_per_node > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
            self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)

        # Early stopping mechanism
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience, verbose=self.verbose, path=self.modelpath)

        # Loss function
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)

        if self.args.loss == 'normal':
            self.criterion = nn.MSELoss()

        if self.args.resume:
            print('Loading pre-trained model')
            checkpoint = torch.load(self.modelpath)  # Load previously saved model weights (including optimizer and scheduler)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if mode == 'test':
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, None)
        
        # --------------------------------------------------------
        # Here we need to slice to extract the label part of the prediction
        # --------------------------------------------------------
        loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
        return outputs, loss

    def train(self):

        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            # ------------------------------------------------------
            # tqdm dynamically displays the progress bar
            # trainloader simply adds a batch size dimension to the input data
            # ------------------------------------------------------

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):

                # ------------------------------------------------
                # [batch_size, seq_len, features]
                # ------------------------------------------------
                if self.is_shuffle:
                    batch_x = batch_x.squeeze(1)
                    batch_y = batch_y.squeeze(1)
                    batch_x_mark = batch_x_mark.squeeze(1)
                    batch_y_mark = batch_y_mark.squeeze(1)

                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                if self.is_shuffle:
                    batch_x = batch_x.squeeze(1)
                    batch_y = batch_y.squeeze(1)
                    batch_x_mark = batch_x_mark.squeeze(1)
                    batch_y_mark = batch_y_mark.squeeze(1)
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                if self.is_shuffle:
                    batch_x = batch_x.squeeze(1)
                    batch_y = batch_y.squeeze(1)
                    batch_x_mark = batch_x_mark.squeeze(1)
                    batch_y_mark = batch_y_mark.squeeze(1)
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss, valid_loss, test_loss))

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



    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            if self.is_shuffle:
                batch_x = batch_x.squeeze(1)
                batch_y = batch_y.squeeze(1)
                batch_x_mark = batch_x_mark.squeeze(1)
                batch_y_mark = batch_y_mark.squeeze(1)
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:,  -self.pred_len:, :])

        mape_error = np.mean(self.mean)*0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        # Denormalize：
        dstand_preds = preds*self.scale+self.mean
        dstand_trues = trues*self.scale+self.mean

        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))

        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae,mape))

        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)

        # Create a csv file to record the training process
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse','mape','seq_len','label_len','pred_len','d_model','d_ff','e_layers','d_layers',
                           'patch_len','stride','n_heads','T2V_outmodel','info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse,'mape':mape,'seq_len':self.seq_len,'label_len':self.label_len,'pred_len':self.pred_len,
                  'd_model':self.d_model,'d_ff':self.d_ff,'e_layers':self.e_layers,'d_layers':self.d_layers,'patch_len':self.patch_len,'stride':self.stride,
                  'n_heads':self.n_heads,"T2V_outmodel":self.args.T2V_outmodel,
                  'info':self.info}]
        write_csv_dict(log_path, a_log, 'a+')





