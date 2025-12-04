import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # Index 0 is for training set
        # Index 1 is for validation set
        # Index 2 is for test set
        # Borders for training/validation/testing sets
        # Training border: 0 to 12 months
        # Validation border: 12 months to 16 months
        # Testing border: 16 months to 20 months
        
        # 12 months * 30 days * 24 hours * 4  = 15-minutes intervals
        # 4 * 30 * 24 * 4 = 4 months
        # 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 = 16 months

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            print("cols_data, ", cols_data)
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            print("df_data.shape ", df_data.shape)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
            print("df_stamp shape:", df_stamp.shape)
            print("data_stamp shape:", data_stamp.shape)
        elif self.timeenc == 1:
            print("self.freq:", self.freq)
            print(pd.to_datetime(df_stamp['date'].values))
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            print("data_stamp AAAA shape:", data_stamp.shape)
            print("data_stamp BBBB:", data_stamp)
        
        print("data[border1:border2]", data[border1:border2].shape)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if (self.data_y == self.data_x).all():
            print("Data_y equals Data_x")
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # print( "seq_x.shape", seq_x.shape)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_VVUser(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='VVUser.csv',
                 target='HeadRX', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        print("freq in Dataset_VVUser:", freq)
        self.time_standard_scaler = {}   # per participant
        self.time_minmax_scaler = {}    # per participant
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()
    
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features == 'M' or self.features == 'MS':
            # HeadX, HeadY, HeadZ, HeadRX, HeadRY, HeadRZ
            df_data = df_raw.iloc[:, 2:7] 
        elif self.features == 'S':
            # Only HeadRX = yaw
            print("df_raw columns:", df_raw.columns)
            df_data = df_raw[["HeadRX","ParticipantID"]]
            df_data['HeadRX'] = df_data['HeadRX'].apply(lambda angle_360 : (((angle_360 + 180) % 360) - 180))
            print(df_data['HeadRX'].head(75))
        
        self.windows = []
        for id in df_raw['ParticipantID'].unique():
            participant_data = df_raw[df_raw['ParticipantID'] == id]
            # Only working with univariate for now
            seq = df_data[df_data['ParticipantID'] == id]
            seq.drop(columns=['ParticipantID'], inplace=True)
            seq = seq.values
            timestamp = participant_data['Timer'].values

           
            # Create windows
            N = len(seq)
            # print("Participant ID:", id, "Data points:", N)
            scaler = StandardScaler()
            scaler.fit(timestamp.reshape(-1, 1))
            minMax = MinMaxScaler()
            minMax.fit(timestamp.reshape(-1, 1))
            for i in range(0, N - self.seq_len - self.pred_len):
                if i + self.seq_len < N:
                    x = seq[i:i + self.seq_len]        # input sequence
                    y = seq[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]            # target sequence
                    
                    # print("x shape:", x.shape)
                    # #check TimeFeatureEmbedding
                    # # x_original = x
                    x_sin = np.sin(np.deg2rad(x))
                    x_cos = np.cos(np.deg2rad(x))

                    y_sin = np.sin(np.deg2rad(y))
                    y_cos = np.cos(np.deg2rad(y))

                    x = np.stack([x_sin, x_cos], axis=1).squeeze()
                    y = np.stack([y_sin, y_cos], axis=1).squeeze()

                    # # # print("x shape", x.shape)

                    # # x_back = np.atan2(x[:, 0], x[:, 1])
                    # # y_back = np.atan2(y[:, 0], y[:, 1])

                    # # print(f"x before: {x_original[0]} and after {np.rad2deg(x_back)[0]}")
                    # # print(f"x before: {x_original[23]} and after {np.rad2deg(x_back)[23]}")

                    t = timestamp[i:i+self.seq_len]
                    dt = np.diff(t, prepend=t[0])      
                    x_mark = t.reshape(-1, 1)   
                    # t = scaler.transform(t.reshape(-1, 1))
                    t_min_max = minMax.transform(t.reshape(-1, 1))

                    x_mark = np.stack([t.reshape(-1, 1), t_min_max], axis=1).squeeze()
                    # print("x_mark", x_mark)   

                    t2 = timestamp[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]
                    dt2 = np.diff(t2, prepend=t2[0])
                    # t2 = scaler.transform(t2.reshape(-1, 1))
                    t2_min_max = minMax.transform(t2.reshape(-1, 1))
                    y_mark = np.stack([t2.reshape(-1, 1), t2_min_max], axis=1).squeeze()
                    # total = self.seq_len + self.pred_len
                    # x_mark = (np.arange(self.seq_len) / total).reshape(-1, 1)
                    # y_mark = (np.arange(self.seq_len - self.label_len, self.seq_len + self.pred_len) / total).reshape(-1, 1)
                    # print("x_mark shape:", x_mark.shape)
                    self.windows.append((x, y, x_mark, y_mark, id))





                    self.time_standard_scaler[id] = scaler    # store per participant
                    self.time_minmax_scaler[id] = minMax






                    # x = seq[i:i + self.seq_len]        # input sequence
                    # y = seq[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]            # target sequence
                    
                    # # print("x shape:", x.shape)
                    # # #check TimeFeatureEmbedding
                    # # # x_original = x
                    # x_sin = np.sin(np.deg2rad(x))
                    # x_cos = np.cos(np.deg2rad(x))

                    # y_sin = np.sin(np.deg2rad(y))
                    # y_cos = np.cos(np.deg2rad(y))

                    # x = np.stack([x_sin, x_cos], axis=1).squeeze()
                    # y = np.stack([y_sin, y_cos], axis=1).squeeze()

                    # # # # print("x shape", x.shape)

                    # # # x_back = np.atan2(x[:, 0], x[:, 1])
                    # # # y_back = np.atan2(y[:, 0], y[:, 1])

                    # # # print(f"x before: {x_original[0]} and after {np.rad2deg(x_back)[0]}")
                    # # # print(f"x before: {x_original[23]} and after {np.rad2deg(x_back)[23]}")
                    # t = timestamp[i:i+self.seq_len]
                    # dt = np.diff(t, prepend=t[0])      
                    # x_mark = t.reshape(-1, 1)   

                    # # print("x_mark", x_mark)   

                    # t2 = timestamp[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]
                    # dt2 = np.diff(t2, prepend=t2[0])
                    # y_mark = t2.reshape(-1, 1)
                    # print("x_mark shape:", x_mark.shape)
                    # self.windows.append((x, y, x_mark, y_mark, id))
            # print("Participant ID:", id, "Total windows:", len(self.windows))

        self.number_of_participants = len(df_raw['ParticipantID'].unique())
        
    def __getitem__(self, index):
        seq_info = self.windows[index]
        seq_x = seq_info[0]
        seq_y = seq_info[1]
        seq_x_mark = seq_info[2]
        seq_y_mark = seq_info[3]

        # x_enc shape: torch.Size([32, 96, 1])
        # x_mark_enc shape: torch.Size([32, 96])


        # x_enc shape: torch.Size([32, 96, 1])
        # x_mark_enc shape: torch.Size([32, 96, 4])

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.windows)
    
