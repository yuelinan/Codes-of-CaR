import os.path as osp
import pickle as pkl

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, Data


class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        print(root)
        
        self.mode = mode
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])
        
    @property
    def raw_file_names(self):
        # return ['train_posion.npy', 'val.npy', 'test.npy']
        return ['train.npy', 'val.npy', 'test.npy','train_get_crane_tree.npy', 'train_get_crane_ladder.npy', 'train_get_crane_whell.npy', 'train_get_house_tree.npy', 'train_get_house_ladder.npy', 'train_get_house_whell.npy', 'train_get_cycle_tree.npy', 'train_get_cycle_ladder.npy', 'train_get_cycle_whell.npy','train_ban.npy',
        'cycle_ladder.npy','cycle_tree.npy','cycle_whell.npy'
        ]

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt','SPMotif_train_get_crane_tree.pt', 'SPMotif_train_get_crane_ladder.pt', 'SPMotif_train_get_crane_whell.pt', 'SPMotif_train_get_house_tree.pt', 'SPMotif_train_get_house_ladder.pt', 'SPMotif_train_get_house_whell.pt', 'SPMotif_train_get_cycle_tree.pt', 'SPMotif_train_get_cycle_ladder.pt', 'SPMotif_train_get_cycle_whell.pt', 'SPMotif_train_ban.pt','SPMotif_cycle_ladder.pt','SPMotif_cycle_tree.pt','SPMotif_cycle_whell.pt']


    def process(self):
        print('hello')
        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=p,
                        edge_gt_att=torch.LongTensor(ground_truth),
                        name=f'SPMotif-{self.mode}-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        print(self.processed_paths[idx])
        print(len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[idx])