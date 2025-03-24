from torch.utils.data import Dataset

class MyDataset_shuffle(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        instence=self.data[index]

        return instence

    def __len__(self):
        return len(self.data)



class MyDataset_stride(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96, stride=0):
        self.data = data[0]
        self.stamp = data[1]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride

    def __getitem__(self, index):
        e_begin = index
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
        # len(self.data) - self.seq_len - self.pred_len + 1
        return len(self.data) - self.seq_len - self.pred_len + 1 - self.stride


class MyDataset_resize_window(Dataset):
    def __init__(self, data, seq_len:int, label_len:int, pred_len:list):
        self.data = data[0]
        self.stamp = data[1]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
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

        return len(self.data) - self.seq_len - max(self.pred_len) + 1



class MyDataset(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96):
        self.data = data[0]
        self.stamp = data[1]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.label_len
        d_end = e_end + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # len(self.data) - self.seq_len - self.pred_len + 1
        return len(self.data) - self.seq_len - self.pred_len + 1


class Skip_Dataset(Dataset):
    def __init__(self, data, gap_len, seq_len=96, label_len=48, pred_len=96):
        self.data = data[0]
        self.stamp = data[1]

        self.gap_len = gap_len
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.label_len
        d_end = e_end + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end+self.gap_len]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end+self.gap_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # len(self.data) - self.seq_len - self.pred_len + 1
        return len(self.data) - self.seq_len - self.pred_len -self.gap_len + 1