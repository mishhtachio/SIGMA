import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Converts a 1D time series into:
    [t1, t2, t3] -> t4
    """

    def __init__(self, series, window_size):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        # number of valid sliding windows
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        """
        Returns:
        X -> (window_size, 1)
        y -> scalar
        """
        X = self.series[idx:idx + self.window_size]
        y = self.series[idx + self.window_size]
        return X.unsqueeze(-1), y
