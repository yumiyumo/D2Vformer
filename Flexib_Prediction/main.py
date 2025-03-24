# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('/public3/zya/Skip_Prediction/')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
from exp.exp import EXP
from utils.setseed import set_seed
from exp.exp_flexiable import EXP_Flexiable
from exp.exp_skip_v1 import EXP_SKIP_V1
from exp.exp_skip_v2 import EXP_SKIP_V2
'''Supplementary Flexible Prediction Experiment 1:
D2V
(1) Training: seq96 pred96 without gap
(2) Testing: seq96 pred48,72,96,108 without additional training deployment
Baseline:
(1) Normal training and prediction, redeploy training for different pred
'''
if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v == 'True':
            return True
        if v == 'False':
            return False

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    # Parameters related to the experimental model
    parser.add_argument('--model_name', default='D2Vformer', type=str, help='[DeepTD_LSP,DeepTD_LSP_simplized,DLinear,Autoformer,PatchTST,'
                                                                                        'Fedformer,TimeMixer,FITS'
                                                                                        'D2Vformer]')
    parser.add_argument('--train', default=True, type=str2bool, help='if train')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume from checkpoint')
    parser.add_argument('--save_log', default=True, type=str2bool, help='save log')
    parser.add_argument('--output_dir', default=None,
                        help='If path is None, exp_id add 1 automatically:if train, it wiil be useful')
    parser.add_argument('--resume_dir', default=None,
                        help='if resume is True, it will be useful')

    parser.add_argument('--loss', default='normal', type=str, help='quantile,normal')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    # Call different experiments; some models require special experimental operations
    parser.add_argument('--exp', default='flexiable_prediction', type=str, help='[deep_learning,flexiable_prediction,skip_prediction_v1,skip_prediction_v2]')
    parser.add_argument('--exp_num', type=int, default=-1)
    parser.add_argument('--desc', type=str, default='',
                        help='describe current experiment and save it into your exp{num}')
    # Experiment data-related parameters
    parser.add_argument('--data_name', default='exchange', type=str, help='[data:ETTh1,electricity,exchange,ETTm1,traffic]')
    parser.add_argument('--seq_len', default=96, type=int, help='input sequence len')
    parser.add_argument('--label_len', default=48, type=int, help='transfomer decoder input part')
    parser.add_argument('--pred_len', default=72, type=int,help='prediction len [48,72,96,108]')
    parser.add_argument('--gap_len', default=2, type=int, help='gap_len for skip_prediction')
    # Prediction length during D2V training phase
    parser.add_argument('--d2v_train_pred_len', default=96, type=int,help='D2V prediction len  96')

    parser.add_argument('--d_mark', default=27, type=int, help='date embed dim')
    parser.add_argument('--d_feature', default=8, type=int,
                        help='input data feature dim without date :[Etth1:7 , electricity:321,exchange:8,ETTm1:7,illness:7,traffic:863]')
    parser.add_argument('--c_out', type=int, default=8, help='output size')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # Hyperparameters related to the model
    parser.add_argument('--d_model', default=128, type=int, help='feature dim in model')
    parser.add_argument('--d_ff', default=256, type=int, help='feature dim2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='initial learning rate for discriminator of gan')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epoches', default=200, type=int, help='Train Epoches')
    parser.add_argument('--patience', default=5, type=int, help='Early Stop patience')

    parser.add_argument('--quantiles', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=int,
                        help='ρ parameter of quantile loss')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--factor', type=int, default=5)
    # patch
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='The interval between each patch')

    parser.add_argument('--patch_stride', type=int, default=8)

    parser.add_argument('--fourier_decomp_ratio', type=float, default=0.5, help='Frequency disassembly module parameter settings')
    parser.add_argument('--stronger', type=int, default=2, help='Top-K Fourier bases')

    # D2V
    parser.add_argument('--T2V_outmodel', type=int, default=64, help='T2V_outmodel')

    # fedformer
    parser.add_argument('--version', default='WSAES_LSTM', type=str, help='fourier or wavelet')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')

    parser.add_argument('--weight_decay', default=0.001, type=float)


    # FITS
    parser.add_argument('--cut_freq', type=int, default=0)
    parser.add_argument('--base_T', type=int, default=24)
    parser.add_argument('--H_order', type=int, default=2)
    parser.add_argument('--Real', type=bool, default=False, help='Real FITS.It is always false')
    parser.add_argument('--train_mode', type=int, default=0,
                        help='train on y(mode 0) or on xy(mode 1) or (on y first then on y)(mode 2)')
    parser.add_argument('--reconstruct', type=bool, default=False, help='reconstruct task')

    # TimeMixer
    parser.add_argument('--channel_independence_timemixer', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm_timemixer', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')


    # info
    parser.add_argument('--info', type=str, default='test', help='extra information')

    args = parser.parse_args()
    set_seed(args.seed)
    
    print(f"|{'=' * 101}|")
    # Use __dict__ method to get parameter dictionary and then iterate through it
    for key, value in args.__dict__.items():
        # Convert all data types to str since parameters may not be strings
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")

    if args.exp == 'deep_learning':
        exp = EXP(args)
        if args.train:
            exp.train()
        exp.test()

    # exp for flexiable_prediction_1
    if args.exp == 'flexiable_prediction':
        exp = EXP_Flexiable(args)
        if args.train:
            exp.train()
        exp.test()
    elif args.exp == 'skip_prediction_v1':
        exp = EXP_SKIP_V1(args)
        if args.train:
            exp.train()
        exp.test()
    elif args.exp == 'skip_prediction_v2':
        exp = EXP_SKIP_V2(args)
        if args.train:
            exp.train()
        exp.test()