import os, glob
import numpy as np
import pandas as pd
import re
import json
from matplotlib import pyplot as plt
from tqdm import tqdm

class Preprocessing():
    def __init__(self, data_name, boundary_ratio):
        self.data_name = data_name
        self.boundary_ratio = boundary_ratio

    def long_time_series_normalization(self, X_long):
        # X_long = (X_long - np.min(X_long, axis=0)) / (np.max(X_long, axis=0) - np.min(X_long, axis=0))
        X_long = (X_long - np.mean(X_long, axis=0)) / np.std(X_long, axis=0)
        return X_long

    def patchwork_random(self, feature_list, label_list, label_seg_list):
        num_file = len(feature_list)
        permuted_file_indices = np.random.permutation(np.arange(num_file))
        length = 0
        X_long = []
        y_long = []
        y_seg_long = []
        file_boundaries = []
        for i in permuted_file_indices:
            length += len(feature_list[i])
            X_long.append(feature_list[i])
            y_long.append(label_list[i])
            y_seg_long.append(label_seg_list[i])
            file_boundaries.append(length)
        return np.concatenate(X_long, axis=0), np.concatenate(y_long, axis=0), np.concatenate(y_seg_long, axis=0), np.array(file_boundaries, dtype=np.int64)

    def patchwork(self, feature_list, label_list, label_seg_list):
        num_file = len(feature_list)
        permuted_file_indices = np.arange(num_file)
        length = 0
        X_long = []
        y_long = []
        y_seg_long = []
        file_boundaries = []
        for i in permuted_file_indices:
            length += len(feature_list[i])
            X_long.append(feature_list[i])
            y_long.append(label_list[i])
            y_seg_long.append(label_seg_list[i])
            file_boundaries.append(length)
        return np.concatenate(X_long, axis=0), np.concatenate(y_long, axis=0), np.concatenate(y_seg_long, axis=0), np.array(file_boundaries, dtype=np.int64)


    def generate_boundary_labels_ratio(self, label_list, mapping_dict):
        boundary_list = []
        segment_len_list = []
        label_seg_list = []

        for video_label in label_list:
            for class_label, class_name in mapping_dict.items():
                video_label[video_label == class_name] = int(class_label) # change class name into class integer

            label_seg_list.append(np.zeros(len(video_label)))
            boundaries = []
            segment_len = []
            length = 0
            for ind, (prev_label, curr_label) in enumerate(zip(video_label, video_label[1:])):
                length += 1
                if prev_label != curr_label:
                    boundaries.append(ind)
                    segment_len.append(length)
                    length = 0
            if length != 0:
                segment_len.append(length)  # put last segment(no boundary at the last of file)
            if len(boundaries) != len(segment_len)-1:
                segment_len.append(1)
            boundary_list.append(boundaries)
            segment_len_list.append(segment_len)

        ratio = self.boundary_ratio # put 10% of min(rhs/lhs segment) as boundary label.
        for i in range(len(boundary_list)):
            for j in range(len(boundary_list[i])):
                lhs_boundary_length = segment_len_list[i][j] * ratio
                rhs_boundary_length = segment_len_list[i][j + 1] * ratio
                boundary_length = np.minimum(lhs_boundary_length, rhs_boundary_length)

                start_ind = int(boundary_list[i][j] - boundary_length) + 1
                end_ind = int(boundary_list[i][j] + boundary_length) + 1
                label_seg_list[i][start_ind:end_ind] = 1
        return label_seg_list

    def generate_boundary_labels(self, label_list, mapping_dict):
        boundary_list = []
        segment_len_list = []
        label_seg_list = []

        for video_label in label_list:
            for class_label, class_name in mapping_dict.items():
                video_label[video_label == class_name] = int(class_label) # change class name into class integer

            label_seg_list.append(np.zeros(len(video_label)))
            boundaries = []
            segment_len = []
            length = 0
            for ind, (prev_label, curr_label) in enumerate(zip(video_label, video_label[1:])):
                length += 1
                if prev_label != curr_label:
                    boundaries.append(ind)
                    segment_len.append(length)
                    length = 0
            if length != 0:
                segment_len.append(length)  # put last segment(no boundary at the last of file)
            if len(boundaries) != len(segment_len)-1:
                segment_len.append(1)
            boundary_list.append(boundaries)
            segment_len_list.append(segment_len)

        for i in range(len(boundary_list)):
            for j in range(len(boundary_list[i])):
                label_seg_list[i][boundary_list[i][j]] = 1
        return label_seg_list


    def generate_long_time_series(self):
        try:
            X_long = np.load(os.path.join("datasets", self.data_name + "_X_long.npy"))
            y_long = np.load(os.path.join("datasets", self.data_name + "_y_long.npy"))
            y_seg_long = np.load(os.path.join("datasets", self.data_name + "_y_seg_long.npy"))
            file_boundaries = np.load(os.path.join("datasets", self.data_name + "_file_boundaries.npy"))
            print(f"{self.data_name} loaded from preprocessed files")
            print(X_long.shape, y_long.shape, y_seg_long.shape)
            return X_long, y_long, y_seg_long, file_boundaries
        except:
            file_boundaries_indice = []
            if self.data_name == "50salads":
                data_path = 'datasets/50salads/features'
                label_path = 'datasets/50salads/groundTruth'
                label_map_file_name = 'datasets/50salads/mapping.txt'
                feature_file_names = sorted(glob.glob(os.path.join(data_path, "*.npy")))
                label_file_names = sorted(glob.glob(os.path.join(label_path, "*.txt")))

                feature_list = [np.load(f).transpose() for f in feature_file_names]
                label_list = [np.array(pd.read_csv(f, sep=" ", index_col=None, header=None)[0].to_numpy()) for f in
                              label_file_names]
                mapping_dict = pd.read_csv(label_map_file_name, sep=" ", index_col=None, header=None)[1].to_dict()

                label_seg_list = self.generate_boundary_labels(label_list, mapping_dict)
                X_long, y_long, y_seg_long, file_boundaries_indice = self.patchwork(feature_list, label_list, label_seg_list)
                y_seg_long = np.array(self.generate_boundary_labels([y_long],{})).flatten()

                X_long = X_long[::2]
                y_long = y_long[::2]
                y_seg_long = y_seg_long[::2]
                file_boundaries_indice = file_boundaries_indice//2

            elif self.data_name == "GTEA":
                data_path = 'datasets/GTEA/features'
                label_path = 'datasets/GTEA/groundTruth'
                label_map_file_name = 'datasets/GTEA/mapping.txt'

                feature_file_names = sorted(glob.glob(os.path.join(data_path, "*.npy")))
                label_file_names = sorted(glob.glob(os.path.join(label_path, "*.txt")))
                mapping_dict = pd.read_csv(label_map_file_name, sep=" ", index_col=None, header=None)[1].to_dict()
                feature_list = [np.load(f).transpose() for f in feature_file_names]
                label_list = [np.array(pd.read_csv(f, sep=" ", index_col=None, header=None)[0].to_numpy()) for f in
                              label_file_names]

                label_seg_list = self.generate_boundary_labels(label_list, mapping_dict)
                X_long, y_long, y_seg_long, file_boundaries_indice = self.patchwork(feature_list, label_list, label_seg_list)
                y_seg_long = np.array(self.generate_boundary_labels([y_long], {})).flatten()


            elif self.data_name == "mHealth":
                data_path = 'datasets/mHealth'
                file_names = sorted(glob.glob(os.path.join(data_path, "*.log")))
                file_names_tr = file_names[:8]
                file_names_tt = file_names[-2:]
                sampling_rate_tr = 50
                sampling_rate_tt = 50
                total_length_tr = 0
                total_length_tt = 0
                file_boundaries_indice_tt = []
                file_boundaries_indice_tr = []
                isfirst_tr = True
                isfirst_tt = True
                for f in tqdm(file_names_tt, leave=False, desc="mHealth stitching"):
                    Xy_tt = np.loadtxt(f)
                    X_long_part_tt = Xy_tt[:,:-1]
                    y_long_part_tt = Xy_tt[:,-1]
                    X_long_part_tt = X_long_part_tt[y_long_part_tt!=0]
                    y_long_part_tt = y_long_part_tt[y_long_part_tt!=0]
                    assert(np.sum(y_long_part_tt==0)==0)
                    total_length_tt += len(X_long_part_tt)
                    if isfirst_tt:
                        X_long_tt = X_long_part_tt
                        y_long_tt = y_long_part_tt
                        isfirst_tt = False
                    else:
                        X_long_tt = np.concatenate([X_long_tt, X_long_part_tt], axis=0)
                        y_long_tt = np.concatenate([y_long_tt, y_long_part_tt], axis=0)
                    file_boundaries_indice_tt.append(total_length_tt)
                
                for f in tqdm(file_names_tr, leave=False, desc="mHealth stitching"):
                    Xy_tr = np.loadtxt(f)
                    X_long_part_tr = Xy_tr[:,:-1]
                    y_long_part_tr = Xy_tr[:,-1]
                    X_long_part_tr = X_long_part_tr[y_long_part_tr!=0]
                    y_long_part_tr = y_long_part_tr[y_long_part_tr!=0]
                    assert(np.sum(y_long_part_tr==0)==0)
                    total_length_tr += len(X_long_part_tr)
                    if isfirst_tr:
                        X_long_tr = X_long_part_tr
                        y_long_tr = y_long_part_tr
                        isfirst_tr = False
                    else:
                        X_long_tr = np.concatenate([X_long_tr, X_long_part_tr], axis=0)
                        y_long_tr = np.concatenate([y_long_tr, y_long_part_tr], axis=0)
                    file_boundaries_indice_tr.append(total_length_tr)
                #print(y_long_tr)
                y_long_tr -= 1
                y_long_tt -= 1
                label_seg_list_tr = self.generate_boundary_labels([y_long_tr], {})
                label_seg_list_tt = self.generate_boundary_labels([y_long_tt], {})
                y_seg_long_tr = label_seg_list_tr[0]
                y_seg_long_tt = label_seg_list_tt[0]


            elif self.data_name == "HAPT":
                data_path = 'datasets/HAPT/RawData'

                acc_files = sorted(glob.glob(os.path.join(data_path, "acc*.txt")))
                gyro_files = sorted(glob.glob(os.path.join(data_path, "gyro*.txt")))
                df_acc = pd.concat((pd.read_csv(f, sep=' ', index_col=None, header=None) for f in acc_files))
                df_gyro = pd.concat((pd.read_csv(f, sep=' ', index_col=None, header=None) for f in gyro_files))

                X = pd.concat([df_acc, df_gyro], axis=1).to_numpy()

                y = np.zeros(len(X))
                file_boundaries_vector = np.zeros(len(X))
                np_label = np.loadtxt(os.path.join(data_path, 'labels.txt'), dtype=np.int32)

                for label_row in np_label:
                    num_exp = label_row[0]
                    if (num_exp - 1 < 10) and (num_exp - 1 > 0):
                        fname = "acc_exp0" + str(num_exp - 1) + "*.txt"
                        f = glob.glob(os.path.join(data_path, fname))[0]
                    elif num_exp == 1:
                        pass
                    else:
                        fname = "acc_exp" + str(num_exp - 1) + "*.txt"
                        f = glob.glob(os.path.join(data_path, fname))[0]
                    if num_exp == 1:
                        offset = 0
                    else:
                        if prev_num_exp != num_exp:
                            offset = len(pd.read_csv(f, sep=' ', index_col=None, header=None)) + offset
                    file_boundaries_vector[offset] = 1
                    start = offset + label_row[3] - 1
                    end = offset + label_row[4]
                    label = label_row[2]

                    prev_num_exp = num_exp

                    y[start:end] = label

                # find transition points
                # make transition label from label into boundary label(1 or 2, as 0 means no label)
                trans_y = np.zeros(len(y))  # zero means unlabeled data
                for ind, cls_label in enumerate(y):
                    if cls_label != 0:
                        trans_y[ind] = 1  # one means labeled but not boundary data

                file_boundaries_indice_prev = np.where(file_boundaries_vector==1)[0]
                start = 0
                new_start = 0
                file_boundaries_indice = []
                for i in range(len(file_boundaries_indice_prev)):
                    prev_length = file_boundaries_indice_prev[i] - start
                    prev_y_file = y[start:file_boundaries_indice_prev[i]+1]
                    start = file_boundaries_indice_prev[i]

                    file_boundaries_indice.append(new_start + prev_length-np.sum(prev_y_file==0))
                    new_start = new_start + prev_length-np.sum(prev_y_file==0)

                X_long = X[y != 0]
                y_long = y[y != 0]
                y_seg_long = trans_y[y != 0]

                y_long = y_long-1
                y_seg_long = y_seg_long-1

                y_seg_long[np.where((y_long == 6) | (y_long == 7) | (y_long == 8) | (y_long == 9) | (y_long == 10) | (y_long == 11))] = 1


                boundary_list = []
                segment_len_list = []
                label_seg_list = []
                for video_label in [y_long]:
                    label_seg_list.append(np.zeros(len(video_label)))
                    boundaries = []
                    segment_len = []
                    length = 0
                    for ind, (prev_label, curr_label) in enumerate(zip(video_label, video_label[1:])):
                        length += 1
                        condition = ((prev_label == 3) & (curr_label == 4)) | ((prev_label == 4) & (curr_label == 3)) | \
                                   ((prev_label == 3) & (curr_label == 5)) | ((prev_label == 5) & (curr_label == 3)) | \
                                   ((prev_label == 4) & (curr_label == 5)) | ((prev_label == 5) & (curr_label == 4))
                        # boundary labels where transition labels do not exist
                        if (not condition) & (prev_label!=curr_label):
                            boundaries.append(ind)
                            segment_len.append(length)
                            length = 0
                    boundary_list.append(boundaries)
                    segment_len_list.append(segment_len)
                segment_len_list[0].append(length) # put last segment length (this is hard coding)

                ratio = self.boundary_ratio  # put 10% of min(rhs/lhs segment) as boundary label.
                for i in range(len(boundary_list)):
                    for j in range(len(boundary_list[i])):
                        lhs_boundary_length = segment_len_list[i][j] * ratio
                        rhs_boundary_length = segment_len_list[i][j + 1] * ratio
                        boundary_length = np.minimum(lhs_boundary_length, rhs_boundary_length)
                        start_ind = int(boundary_list[i][j] - boundary_length)
                        end_ind = int(boundary_list[i][j] + boundary_length)
                        y_seg_long[start_ind:end_ind] = 1

                for (trans_label, (converted1, converted2)) in [(6, (4, 3)), (7, (3, 4)), (8, (3, 5)), (9, (5, 3)),
                                                                (10, (4, 5)), (11, (5, 4))]:
                    ind_list = np.where(y_long == trans_label)[0]
                    prev_j = ind_list[0]
                    duration_list = [[prev_j]]
                    is_first = True
                    for i, (j, k) in enumerate(zip(ind_list, ind_list[1:])):
                        if (j != k - 1) & (is_first):  # finds not continuing index of index list
                            duration_list[0].append(j)
                            prev_j = k
                            is_first = False
                            continue
                        if j != k - 1:
                            duration_list.append([prev_j, j])
                            prev_j = k
                    duration_list.append([prev_j, ind_list[-1]])
                    for start, end in duration_list:
                        y_long[start:start + int(np.rint((end - start)) / 2)] = converted1
                        y_long[start + int(np.rint((end - start)) / 2):end + 1] = converted2


            X_long_tr = self.long_time_series_normalization(X_long_tr)
            X_long_tt = self.long_time_series_normalization(X_long_tt)
            file_boundaries_tr = np.zeros(y_seg_long_tr.shape)
            file_boundaries_tt = np.zeros(y_seg_long_tt.shape)
            if len(file_boundaries_indice_tr) > 0:
                file_boundaries_tr[np.array(file_boundaries_indice_tr)-1]=1
            if len(file_boundaries_indice_tt) > 0:
                file_boundaries_tt[np.array(file_boundaries_indice_tt)-1]=1

            X_long_tr = X_long_tr.astype(np.float32)
            X_long_tt = X_long_tt.astype(np.float32)
            y_long_tr = y_long_tr.astype(np.int32)
            y_long_tt = y_long_tt.astype(np.int32)
            y_seg_long_tr = y_seg_long_tr.astype(np.int32)
            y_seg_long_tt = y_seg_long_tt.astype(np.int32)
            file_boundaries_tr = np.array(file_boundaries_tr).astype(np.int32)
            file_boundaries_tt = np.array(file_boundaries_tt).astype(np.int32)

            np.save(os.path.join("datasets", self.data_name + "_X_long_tr.npy"), X_long_tr)
            np.save(os.path.join("datasets", self.data_name + "_y_long_tr.npy"), y_long_tr)
            np.save(os.path.join("datasets", self.data_name + "_y_seg_long_tr.npy"), y_seg_long_tr)
            np.save(os.path.join("datasets", self.data_name + "_file_boundaries_tr.npy"), file_boundaries_tr)
            print(f"{self.data_name} training file has been preprocessed and saved for further use")

            np.save(os.path.join("datasets", self.data_name + "_X_long_tt.npy"), X_long_tt)
            np.save(os.path.join("datasets", self.data_name + "_y_long_tt.npy"), y_long_tt)
            np.save(os.path.join("datasets", self.data_name + "_y_seg_long_tt.npy"), y_seg_long_tt)
            np.save(os.path.join("datasets", self.data_name + "_file_boundaries_tt.npy"), file_boundaries_tt)
            print(f"{self.data_name} testing file has been preprocessed and saved for further use")
        print(X_long_tr.shape, y_long_tr.shape, y_seg_long_tr.shape)
        print(X_long_tt.shape, y_long_tt.shape, y_seg_long_tt.shape)
        return (X_long_tr.astype(np.float32), y_long_tr.astype(np.int32), y_seg_long_tr.astype(np.int32), np.array(file_boundaries_tr).astype(np.int32)), (X_long_tt.astype(np.float32), y_long_tt.astype(np.int32), y_seg_long_tt.astype(np.int32), np.array(file_boundaries_tt).astype(np.int32))
        #return X_long.astype(np.float32), y_long.astype(np.int32), y_seg_long.astype(np.int32), np.array(file_boundaries).astype(np.int32)




if __name__ == "__main__":
    ratio = 0.1
    for name in ["mHealth"]:
        data = Preprocessing(name, ratio)
        X_long, y_long, y_seg_long, file_boundaries = data.generate_long_time_series()
        print(type(X_long),type(y_long),type(y_seg_long),type(file_boundaries))

        print("########### Data Specification ###########")
        print("Number of timestamp and data dimension:", X_long.shape)
        print("Number of class:", len(np.unique(y_long)))
        print("Class label:", np.unique(y_long))
        for i in range(len(np.unique(y_long))):
            print("Number of timestamp for class " + str(i) + ":", len(np.where(y_long == i)[0]))
        print("Number of boundary class:", len(np.unique(y_seg_long)))
        print("Boundary Class label:", np.unique(y_seg_long))
        for i in range(len(np.unique(y_seg_long))):
            print("Number of timestamp for Boundary Class " + str(i) + ":",
                  len(np.where(y_seg_long == i)[0]))
        print(f"file boundaries {np.where(file_boundaries == 1)[0]}")
        print("\n\n")
