import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, input_length, X, y, y_seg, mask, file_boundaries, seed, fully_supervised=False):
        self.input_length = input_length

        self.X_long = X  # timestamp, dim
        self.y_long = y[:, np.newaxis]  # timestamp, 1
        self.y_seg_long = y_seg[:, np.newaxis]  # timestamp, 1
        self.mask_long = mask
        self.file_boundaries = file_boundaries
        self.num_class = len(np.unique(y))

        num_ts_per_class = []  # make maximum num_ts_per_class same as minimum num_ts_per_class by multiply the factor
        for i in range(self.num_class):
            num = np.sum(y[mask == 1] == i)
            if num > 0:
                num_ts_per_class.append(num)
            else:
                num_ts_per_class.append(1)

        lr_class = np.min(num_ts_per_class) / np.array(num_ts_per_class)
        self.lr_mask = np.copy(mask)
        if fully_supervised == True:
            pass
        else:
            for i in range(self.num_class):
                self.lr_mask[(mask == 1) & (y == i)] = lr_class[i]

    def __len__(self):
        return len(self.X_long) - self.input_length + 1

    def __getitem__(self, idx):
        windowed_X = self.X_long[idx:idx + self.input_length]
        windowed_y_seg = self.y_seg_long[idx:idx + self.input_length]
        windowed_y = self.y_long[idx:idx + self.input_length]
        windowed_mask = self.lr_mask[idx:idx + self.input_length]

        return torch.from_numpy(windowed_X), torch.from_numpy(windowed_y), \
               torch.from_numpy(windowed_y_seg), torch.from_numpy(windowed_mask)


def window_scoring(X_long, y_long, mask_long, file_boundaries, input_length, slide_size):
    y = y_long.astype(np.float32)
    int_class, counts = np.unique(y_long[mask_long], return_counts=True)
    class_scoring_dict = {}
    for i in range(len(int_class)):
        class_scoring_dict[int_class[i]] = 1 / counts[i]  # low score for high frequency class
    for i in range(len(int_class)):
        y[y_long == i] = class_scoring_dict[int_class[i]]
    y[np.invert(mask_long)] = 0
    window_sample_prob = []
    indice_list = []
    window_size = input_length
    num_iter = (len(X_long) - window_size) // slide_size + 1
    for i in tqdm(range(num_iter), leave=False, desc="window_scoring"):
        label = y[i * slide_size:i * slide_size + window_size]
        mask = mask_long[i * slide_size:i * slide_size + window_size]
        file_boundary = file_boundaries[i * slide_size:i * slide_size + window_size]
        num_label = np.sum(mask)
        if num_label > 0:
            score = np.sum(label[mask]) / num_label
            window_sample_prob.append(score)
            indice_list.append(i * slide_size)  # save window indice where label exist
    window_sample_prob = np.array(window_sample_prob)
    window_sample_prob = window_sample_prob / np.sum(window_sample_prob)
    return window_sample_prob, np.array(indice_list)