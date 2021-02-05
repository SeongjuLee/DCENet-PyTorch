import os
import torch
import numpy as np
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    def __init__(self, x, x_occu, y, y_occu, mode='train'):
        self.x = x
        self.x_occu = x_occu
        self.y = y
        self.y_occu = y_occu
        self.mode = mode

    def __getitem__(self, index):
        # Shape of x, y: [seq_len, input_len(2)], Shape of x_occu, y_occu: [channel, seq_len, height, width]
        if self.mode == 'train' or self.mode == 'val':
            x = torch.tensor(self.x[index], dtype=torch.float)
            x_occu = torch.tensor(self.x_occu[index], dtype=torch.float).permute(0, 3, 1, 2)
            y = torch.tensor(self.y[index], dtype=torch.float)
            y_occu = torch.tensor(self.y_occu[index], dtype=torch.float).permute(0, 3, 1, 2)

            return x, x_occu, y, y_occu

        elif self.mode == 'test':
            x = torch.tensor(self.x[index], dtype=torch.float)
            x_occu = torch.tensor(self.x_occu[index], dtype=torch.float).permute(0, 3, 1, 2)
            y = torch.tensor(self.y[index], dtype=torch.float)

            return x, x_occu, y

        elif self.mode == 'challenge':
            x = torch.tensor(self.x[index], dtype=torch.float)
            x_occu = torch.tensor(self.x_occu[index], dtype=torch.float).permute(0, 3, 1, 2)

            return x, x_occu
        
        else:
            raise NotImplemented
        
    def __len__(self):
        return len(self.x)


def load_data(config, dataset_list, datatype="train"):
    # Store the data across datasets
    # All the datasets are merged for training
    if datatype == "train" or datatype == "test":
        offsets = np.empty((0, config['obs_seq'] + config['pred_seq'] - 1, 8))
        traj_data = np.empty((0, config['obs_seq'] + config['pred_seq'], 4))
        occupancy = np.empty((0, config['obs_seq'] + config['pred_seq'] - 1, config['enviro_pdim'][0], config['enviro_pdim'][1], 3))

        if dataset_list[0] == "train_merged":
            for i in range(3):
                data = np.load("./processed_data/train/%s.npz" % (dataset_list[0]+str(i)))
                _offsets, _traj_data, _occupancy = data["offsets"], data["traj_data"], data["occupancy"]


                offsets = np.concatenate((offsets, _offsets), axis=0)
                traj_data = np.concatenate((traj_data, _traj_data), axis=0)

                occupancy = np.concatenate((occupancy, _occupancy), axis=0)
            print(dataset_list[0], "contains %.0f trajectories" % len(offsets))
        else:
            for i, dataset in enumerate(dataset_list):
                # Only take the orinal data
                # ToDo, here needs to be test if augumentation will boost the performance
                # if dataset != "train_merged":
                    # ToDo chenge this to make compatible with linus
                data = np.load("./processed_data/train/%s.npz" % (dataset))
                _offsets, _traj_data, _occupancy = data["offsets"], data["traj_data"], data["occupancy"]
                print(dataset, "contains %.0f trajectories" % len(_offsets))
                offsets = np.concatenate((offsets, _offsets), axis=0)
                traj_data = np.concatenate((traj_data, _traj_data), axis=0)
                occupancy = np.concatenate((occupancy, _occupancy), axis=0)

    # NOTE: When load the challenge data, there is no need to merge them
    # The submission requires each challenge data set (in total 20) to be separated
    # Hence, each time only one challenge data set is called
    elif datatype == "challenge":
        offsets = np.empty((0, config['obs_seq'] - 1, 8))
        traj_data = np.empty((0, config['obs_seq'], 4))
        occupancy = np.empty((0, config['obs_seq'] - 1, config['enviro_pdim'][0], config['enviro_pdim'][1], 3))
        for dataset in dataset_list:
            data = np.load("./processed_data/challenge/%s.npz" % (dataset))
            _offsets, _traj_data, _occupancy = data["offsets"], data["traj_data"], data["occupancy"]
            offsets = np.concatenate((offsets, _offsets), axis=0)
            traj_data = np.concatenate((traj_data, _traj_data), axis=0)
            occupancy = np.concatenate((occupancy, _occupancy), axis=0)

    elif datatype == "test":
        assert len(dataset_list) == 1, print("Only one untouched dataset is left fot testing!")
    elif datatype == "challenge":
        assert len(dataset_list) == 1, print("predict one by one")
    if datatype == "train":
        if not os.path.exists("./processed_data/train/train_merged2.npz"):
            # Save the merged training data
            # sigle file storage more than 16G is not supported in linux system, so I tried to store them in 3 files.
            # it's not necessary for windows user to separate data into 3 files
            offsets_list = [offsets[:15000,:],offsets[15000:30000,:],offsets[30000:,:]]
            traj_data_list = [traj_data[:15000,:],traj_data[15000:30000,:],traj_data[30000:,:]]
            occupancy_list = [occupancy[:15000,:],occupancy[15000:30000,:],occupancy[30000:,:]]

            for i in range(len(offsets_list)):
                np.savez("./processed_data/train/train_merged%s.npz"%(i),
                         offsets=offsets_list[i],
                         traj_data=traj_data_list[i],
                         occupancy=occupancy_list[i])

    return offsets, traj_data, occupancy