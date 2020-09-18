import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
        an abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, label_i = self.data[item_idx], self.label[item_idx]
        return torch.tensor(data_i).float(), torch.tensor(label_i, dtype=torch.long)
