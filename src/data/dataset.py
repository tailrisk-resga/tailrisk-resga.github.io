import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os



class Pointwise_Dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.meta = [(f[1], f[2]) for f in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        stock_id, date_str = self.meta[index]
        feature = torch.Tensor(sample['feature'])
        label = torch.Tensor(sample['label'])
        return stock_id, date_str, feature, label

    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        return _split_stock_date_npy(data_dir, start_time, valid_time, test_time, end_time, Pointwise_Dataset)
    

class Temporal_Dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.meta = [(f[1], f[2]) for f in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        stock_id, date_str = self.meta[index]
        feature = torch.Tensor(sample['feature'])
        label = torch.Tensor(sample['label'])
        return stock_id, date_str, feature, label


    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        return _split_stock_date_npy(data_dir, start_time, valid_time, test_time, end_time, Temporal_Dataset)
    
    

    
def _split_stock_date_npy(dataset_dir, start_time, valid_time, test_time, end_time, dataset_cls):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npy')])
    df = pd.DataFrame({'file': files})

    df[['stock_id', 'date_str']] = df['file'].str.replace('.npy', '', regex=False).str.split('_', expand=True)
    df['stock_id'] = df['stock_id'].astype(int)
    df['date_str'] = pd.to_datetime(df['date_str'])
    df['file_path'] = df['file'].apply(lambda x: os.path.join(dataset_dir, x))

    train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
    valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
    test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

    def to_tuple_list(subdf):
        return list(zip(subdf['file_path'], subdf['stock_id'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

    return dataset_cls(to_tuple_list(train_df)), dataset_cls(to_tuple_list(valid_df)), dataset_cls(to_tuple_list(test_df))


class CrossSectional_Dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.date = [f[1] for f in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        date_str = self.date[index]
        feature = torch.Tensor(sample['feature'])  # Shape: (num_stocks, sequence_length, num_features)
        label = torch.Tensor(sample['label'])      # Shape: (num_stocks, 1)
        stock_ids = sample['stock_ids'] # Shape: (num_stocks,)
        return stock_ids, date_str, feature, label


    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        df = pd.DataFrame({'file': files})

        df['date_str'] = df['file'].str.replace('.npy', '')
        df['date_str'] = pd.to_datetime(df['date_str'])

        df['file_path'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

        train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
        valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
        test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

        def to_tuple_list(subdf):
            return list(zip(subdf['file_path'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

        train_info = to_tuple_list(train_df)
        valid_info = to_tuple_list(valid_df)
        test_info = to_tuple_list(test_df)

        return CrossSectional_Dataset(train_info), CrossSectional_Dataset(valid_info), CrossSectional_Dataset(test_info)


class Informer_Dataset(Dataset):
    """
    Dataset class for Informer model, supporting required encoder/decoder input format.
    Each sample returns (stock_id, date_str, x_enc, x_mark_enc, x_dec, x_mark_dec, label).
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.meta = [(f[1], f[2]) for f in file_list] 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        stock_id, date_str = self.meta[index]
        
        x_enc = torch.Tensor(sample['feature'])
        x_mark_enc = torch.Tensor(sample['x_mark_enc'])
        x_dec = torch.Tensor(sample['x_dec'][:2])
        x_mark_dec = torch.Tensor(sample['x_mark_dec'][:2])
        label = torch.Tensor(sample['label'])
    
        return stock_id, date_str, x_enc, x_mark_enc, x_dec, x_mark_dec, label


    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        df = pd.DataFrame({'file': files})

        df[['stock_id', 'date_str']] = df['file'].str.replace('.npy', '').str.split('_', expand=True)
        df['stock_id'] = df['stock_id'].astype(int)
        df['date_str'] = pd.to_datetime(df['date_str'])

        df['file_path'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

        train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
        valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
        test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

        def to_tuple_list(subdf):
            return list(zip(subdf['file_path'], subdf['stock_id'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

        train_info = to_tuple_list(train_df)
        valid_info = to_tuple_list(valid_df)
        test_info = to_tuple_list(test_df)

        return Informer_Dataset(train_info), Informer_Dataset(valid_info), Informer_Dataset(test_info)



class EInformer_Dataset(Dataset):
    """
    Dataset class for Informer model, supporting required encoder/decoder input format.
    Each sample returns (stock_id, date_str, x_enc, x_mark_enc, x_dec, x_mark_dec, label).
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.meta = [(f[1], f[2]) for f in file_list] 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Lazy load only this sample
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        stock_id, date_str = self.meta[index]
        
        x_enc = torch.Tensor(sample['feature'])
        x_mark_enc = torch.Tensor(sample['x_mark_enc'])
        label = torch.Tensor(sample['label'])
        
        return stock_id, date_str, x_enc, x_mark_enc, label

    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        df = pd.DataFrame({'file': files})

        df[['stock_id', 'date_str']] = df['file'].str.replace('.npy', '').str.split('_', expand=True)
        df['stock_id'] = df['stock_id'].astype(int)
        df['date_str'] = pd.to_datetime(df['date_str'])

        df['file_path'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

        train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
        valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
        test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

        def to_tuple_list(subdf):
            return list(zip(subdf['file_path'], subdf['stock_id'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

        train_info = to_tuple_list(train_df)
        valid_info = to_tuple_list(valid_df)
        test_info = to_tuple_list(test_df)

        return EInformer_Dataset(train_info), EInformer_Dataset(valid_info), EInformer_Dataset(test_info)


class DInformer_Dataset(Dataset):
    """
    Dataset class for Informer model, supporting required encoder/decoder input format.
    Each sample returns (stock_id, date_str, x_enc, x_mark_enc, x_dec, x_mark_dec, label).
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.meta = [(f[1], f[2]) for f in file_list] 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Lazy load only this sample
        sample = np.load(self.file_list[index][0], allow_pickle=True).item()
        stock_id, date_str = self.meta[index]
        
        x = torch.Tensor(sample['feature'])
        x_mark = torch.Tensor(sample['x_mark_enc'])
        label = torch.Tensor(sample['label'])
        
        return stock_id, date_str, x, x_mark, label



    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        df = pd.DataFrame({'file': files})

        df[['stock_id', 'date_str']] = df['file'].str.replace('.npy', '').str.split('_', expand=True)
        df['stock_id'] = df['stock_id'].astype(int)
        df['date_str'] = pd.to_datetime(df['date_str'])

        df['file_path'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

        train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
        valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
        test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

        def to_tuple_list(subdf):
            return list(zip(subdf['file_path'], subdf['stock_id'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

        train_info = to_tuple_list(train_df)
        valid_info = to_tuple_list(valid_df)
        test_info = to_tuple_list(test_df)

        return DInformer_Dataset(train_info), DInformer_Dataset(valid_info), DInformer_Dataset(test_info)




class ReSGA_Dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.date = [f[1] for f in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        sample = torch.load(self.file_list[index][0])
        date_str = self.date[index]
        feature = sample['feature'] # Shape: (num_stocks, sequence_length, num_features)
        label = sample['label']      # Shape: (num_stocks, 1)
        stock_ids = sample['stock_ids'] # Shape: (num_stocks,)
        return stock_ids, date_str, feature, label

    @staticmethod
    def split(data_dir, start_time, valid_time, test_time, end_time):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        df = pd.DataFrame({'file': files})

        df['date_str'] = df['file'].str.replace('.pt', '').str.split('_', expand=True)[0]
        df['date_str'] = pd.to_datetime(df['date_str'])

        df['file_path'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

        train_df = df[(df['date_str'] >= start_time) & (df['date_str'] < valid_time)]
        valid_df = df[(df['date_str'] >= valid_time) & (df['date_str'] < test_time)]
        test_df  = df[(df['date_str'] >= test_time) & (df['date_str'] < end_time)]

        def to_tuple_list(subdf):
            return list(zip(subdf['file_path'], subdf['date_str'].dt.strftime('%Y-%m-%d')))

        train_info = to_tuple_list(train_df)
        valid_info = to_tuple_list(valid_df)
        test_info = to_tuple_list(test_df)

        return ReSGA_Dataset(train_info), ReSGA_Dataset(valid_info), ReSGA_Dataset(test_info)
