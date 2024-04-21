from torch.utils.data import Dataset

class MyDataset_shuffle(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # ------------------------------------------------------
        #   通过index来在原始数据中划分seqlen，labellen以及predlen
        #   从index往后seq_len长度
        # ------------------------------------------------------
        instence=self.data[index]

        return instence

    def __len__(self):
        #------------------------------------------------
        #   掐头去尾计算中间的滑动次数
        #------------------------------------------------

        return len(self.data)



'''根据步长扩充训练集-->相当于历史回顾窗口和预测窗口不一定是紧挨着，而是中间有间隔(stride)
注意：(1)label_len不在对应着历史回顾窗口的后labe_len部分;'''
class MyDataset_stride(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96, stride=0):
        self.data = data[0]
        self.stamp = data[1]

        # ---------------------------------------------
        #   label_len是为了Transfomer中的一步式预测使用的
        #   传统的RNN模型不需要考虑label-len
        # ---------------------------------------------
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride

    def __getitem__(self, index):
        e_begin = index
        # ------------------------------------------------------
        #   通过index来在原始数据中划分seqlen，labellen以及predlen
        #   从index往后seq_len长度
        #   中间过度变量s，加入了stride
        # ------------------------------------------------------
        e_end = e_begin + self.seq_len
        s = e_end+self.stride

        d_begin = s - self.label_len
        d_end = s + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #------------------------------------------------
        #   掐头去尾计算中间的滑动次数
        #   stride 表示跳跃步长
        #------------------------------------------------
        # len(self.data) - self.seq_len - self.pred_len + 1
        return len(self.data) - self.seq_len - self.pred_len + 1 - self.stride


class MyDataset_resize_window(Dataset):
    def __init__(self, data, seq_len:int, label_len:int, pred_len:list):
        '''输入序列长度一定，输出序列的长度可以有多个'''
        self.data = data[0]
        self.stamp = data[1]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        '''seq_x and seq_x_mark -->len(·)=1
           seq_y_lst and seq_y_mark_lst-->len(·)!=1'''
        # 这个index是一次+1，因此如果希望的index不是一次+1，要手动乘上对应的值
        e_begin = index
        # ------------------------------------------------------
        #   通过index来在原始数据中划分seqlen，labellen以及predlen
        #   从index往后seq_len长度
        # ------------------------------------------------------
        e_end = e_begin + self.seq_len
        seq_x = self.data[e_begin:e_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_lst=[]
        seq_y_mark_lst=[]
        d_begin = e_end - self.label_len
        for i in range(len(self.pred_len)):
            d_end = e_end + self.pred_len[i]
            seq_y = self.data[d_begin:d_end]
            seq_y_mark = self.stamp[d_begin:d_end]
            seq_y_lst.append(seq_y)
            seq_y_mark_lst.append(seq_y_mark)

        return seq_x, seq_y_lst, seq_x_mark, seq_y_mark_lst

    def __len__(self):
        #------------------------------------------------
        #   掐头去尾计算中间的滑动次数，滑动次数看最大的预测长度
        #------------------------------------------------
        return len(self.data) - self.seq_len - max(self.pred_len) + 1



class MyDataset(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96):
        self.data = data[0]
        self.stamp = data[1]

        # ---------------------------------------------
        #   label_len是为了Transfomer中的一步式预测使用的
        #   传统的RNN模型不需要考虑label-len
        # ---------------------------------------------
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
        # ------------------------------------------------------
        #   通过index来在原始数据中划分seqlen，labellen以及predlen
        #   从index往后seq_len长度
        # ------------------------------------------------------
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.label_len
        d_end = e_end + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #------------------------------------------------
        #   掐头去尾计算中间的滑动次数
        #------------------------------------------------
        # len(self.data) - self.seq_len - self.pred_len + 1
        return len(self.data) - self.seq_len - self.pred_len + 1
