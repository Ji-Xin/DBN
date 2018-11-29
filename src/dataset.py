import torch
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, xfilename, yfilename):
        'Initialization'
        self.x = torch.load(xfilename)
        self.y = torch.load(yfilename)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)


    def __getitem__(self, index):
        'Generates one sample of data'
        return self.x[index], self.y[index]


    def getLoader(self, batch_size, shuffle=False, num_workers=0):
        return data.DataLoader(self, batch_size, shuffle=shuffle, num_workers=num_workers)