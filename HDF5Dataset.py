import torch
import h5py
from torch._C import StringType
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, file, data_type):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object[data_type]['data']
        self.labelset = self.file_object[data_type]['label']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if (index >= len(self.dataset)):
          raise IndexError()
        img = self.dataset[index]
        return torch.FloatTensor(img), self.labelset[index]