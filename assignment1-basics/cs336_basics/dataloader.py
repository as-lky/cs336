import torch

class LkyDataSet(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, index):
        return self.data[index:index+self.seq_len], self.data[index+1:index+self.seq_len+1]
    