'''
dataset definition
'''
import os
import re
import pandas as pd
import numpy as np
import pickle
import torch
import torch.utils.data as data
import torch_geometric as pyg
from torch_geometric.data import Data

class CMFDataset(data.Dataset):
    def __init__(self,
                 start_level,
                 end_level,
                 split,
                 fold_id,
                 include_edges,
                 get_coords,
                 transform,
                 root_dir='../../Datasets/XXXX/',
                 origin_dir='../../Datasets/XXXX/'
                 ):
        self._root_dir = root_dir
        self.origin_dir = origin_dir
        self._start_level = start_level
        self._end_level = end_level
        self.splitfile = os.path.join(self._root_dir, 'train_test_split', 'fold_{}_{}_file_list.txt'.format(fold_id,split)) # train_valid_test_split
        self._include_edges = include_edges
        self._get_coords = get_coords
        self._transform = transform
        # data dir
        self.datapath = []
        with open(self.splitfile, 'r') as f:
            for line in f:
                path = line.strip()
                self.datapath.append(os.path.join(self._root_dir,path))
        # data file name
        self.datafile = []
        for item in self.datapath:
            pt_fn = os.path.join(item, 'input_data.pt')
            self.datafile.append(pt_fn)
        # # pre load data
        # self.all_items = []
        # for idx in range(len(self.datafile)):
        #     a_sample = self.get_an_item(idx)
        #     self.all_items.append(a_sample)

    def __getitem__(self, index):
        # return self.all_items[index]
        return self.get_an_item(index)
    
    def get_an_item(self, index):
        sample = None
        pt_fn = self.datafile[index]
        saved_tensors = torch.load(pt_fn)
        coords = saved_tensors['vertices'][:self._end_level]
        edges = saved_tensors['edges'][:self._end_level]
        traces = saved_tensors['traces'][:self._end_level-1]    # trace map
        b_pre = saved_tensors['b_pre']
        b_post = saved_tensors['b_post']
        target_1, target_2 = saved_tensors['f_targets']         # coarse & fine targets
        edge_index_2 = saved_tensors['stage_edges']
        faces_2 = saved_tensors['stage_faces']                  # selected faces for unpool
        lap_index_1, lap_index_2 = saved_tensors['laps_coords'] # for calculate laplacian coords
        # landmarks
        pat_n = pt_fn.split('/')[-2].split('_')[0]
        pre_lm_fn = os.path.join(self.origin_dir, pat_n + '/pre_face_landmarks.csv')
        post_lm_fn = os.path.join(self.origin_dir, pat_n + '/post_face_landmarks.csv')
        pre_lm_df = pd.read_csv(pre_lm_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
        pre_lm_xyz = pre_lm_df.values
        post_lm_df = pd.read_csv(post_lm_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
        post_lm_xyz = post_lm_df.values
        # load info.pkl
        info_fn = os.path.join(os.path.dirname(pt_fn), 'info.pkl')
        with open(info_fn, 'rb') as f:
            info = pickle.load(f)
        scale_size = info['scale_size']
        centroid = info['centroid']
        # normalize landmarks
        pre_lm_xyz = (pre_lm_xyz - centroid) / scale_size
        post_lm_xyz = (post_lm_xyz - centroid) / scale_size
        pre_lm_xyz = torch.from_numpy(pre_lm_xyz).float()
        post_lm_xyz = torch.from_numpy(post_lm_xyz).float()
        # construct sample, a graph structure data
        sample = Data(x=coords[0][:, 3:],
                      pos=coords[0][:, :3],                
                      edge_index=edges[0].t().contiguous(   
                      ) if self._include_edges else None,
                      b_pre=b_pre,
                      b_mv=b_post - b_pre,                  # bone movement vectors
                      lm_pre=pre_lm_xyz,                
                      lm_target=post_lm_xyz,
                      target_1=target_1,                    
                      target_2=target_2,                    
                      edge_stage_2=edge_index_2,            
                      add_stage_2=faces_2,                  
                      lap_stage_1=lap_index_1,              
                      lap_stage_2=lap_index_2)
        sample.pt_fn = pt_fn
        # construct data in each layer of MDFE
        nested_meshes = []
        for level in range(1, len(edges)):
            data = Data(edge_index=edges[level].t(
                        ).contiguous() if self._include_edges else None)
            data.trace_index = traces[level-1]
            if self._get_coords:
                data.pos = coords[level][:, :3]
            nested_meshes.append(data)
        # add layer to sample
        sample.num_vertices = []
        for level, nested_mesh in enumerate(nested_meshes):
            setattr(
                sample, f"hierarchy_edge_index_{level+1}", nested_mesh.edge_index)
            setattr(
                sample, f"hierarchy_trace_index_{level+1}", nested_mesh.trace_index)
            sample.num_vertices.append(
                int(sample[f"hierarchy_trace_index_{level+1}"].max() + 1))
            if self._get_coords:
                setattr(sample, f"pos_{level + 1}", nested_mesh.pos)
        # apply transformer
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self.datafile)


class GraphLevelBatch(pyg.data.Batch):
    def __init__(self, batch=None, **kwargs):
        super(GraphLevelBatch, self).__init__(batch, **kwargs)

    @staticmethod
    def from_graph_data_list(data_list, follow_batch=[]):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        batch = pyg.data.Batch()
        for key in keys:
            batch[key] = []
        for key in follow_batch:
            batch['{}_batch'.format(key)] = []
        batch.batch = []
        cumsum = 0
        cumsum_s2 = 0
        hierarchy_cumsum = [0 for _ in range(len(data_list[0].num_vertices))]
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            num_nodes_s2 = data['lap_stage_2'].shape[0]
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if 'hierarchy' in key:
                    level = int(key[-1]) - 1
                    item = item + hierarchy_cumsum[level] if bool(re.search('(index|face)', key)) else item
                else:
                    item = item + cumsum if bool(re.search('(index|face)', key)) else item
                if 'lap_stage' in key:
                    indices = item[:, :-2]
                    invalid_mask = indices < 0
                    item[:, :-1] = item[:, :-1] + cumsum if '_1' in key else item[:, :-1] + cumsum_s2
                    indices[invalid_mask] = -1
                if 'edge_stage' in key:
                    item = item + cumsum_s2
                batch[key].append(item)
            for key in follow_batch:
                size = data[key].size(data.__cat_dim__(key, data[key]))
                item = torch.full((size, ), i, dtype=torch.long)
                batch['{}_batch'.format(key)].append(item)
            cumsum += num_nodes
            cumsum_s2 += num_nodes_s2
            for i in range(len(data.num_vertices)):
                hierarchy_cumsum[i] += data.num_vertices[i]
        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                pass
        batch.batch = torch.cat(batch.batch, dim=-1)
        for key in follow_batch:
            batch['{}_batch'.format(key)] = torch.cat(batch['{}_batch'.format(key)], dim=-1)
        batch['edge_stage_2'] = torch.transpose(batch['edge_stage_2'], 0,1)
        return batch.contiguous()


class GraphLevelDataLoader(data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(GraphLevelDataLoader, self).__init__(
              dataset,
              batch_size,
              shuffle,
              collate_fn=lambda data_list: GraphLevelBatch.from_graph_data_list(
                  data_list, follow_batch),
              **kwargs)
