import numpy as np
from torch.utils.data import DataLoader
from torch_loader import TimeSeriesDataset
from preprocessing import Preprocessing

data = Preprocessing(data_name="mHealth", boundary_ratio=0.1)
X_long, y_long, y_seg_long, file_boundaries = data.generate_long_time_series()
input_length = 512
seed = 0
mask = np.ones(len(y_long))
ts  = TimeSeriesDataset(input_length, X_long, y_long, y_seg_long, mask, file_boundaries, seed)
tr_loader = DataLoader(ts, 32, shuffle=False)

