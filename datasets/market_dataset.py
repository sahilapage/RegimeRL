from torch.utils.data import Dataset

class MarketDataset(Dataset):
    def __init__(self, windows_tensor):
        self.windows = windows_tensor

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        return self.windows[idx]
