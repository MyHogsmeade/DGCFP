import os
import open3d as o3d
import numpy as np
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math
import pymeshlab as mlb
import json
import time
from sklearn.neighbors import BallTree
from multiprocessing import Pool
import multiprocessing

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pc_normalize(pc, m=0, centroid=None):
    l = pc.shape[0]
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if m == 0:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = np.matmul(-2 * src, dst.T)
    dist += np.sum(src ** 2, -1).reshape(N, 1)
    dist += np.sum(dst ** 2, -1).reshape(1, M)
    return dist

def reconstruct_crop_rigion(model='DGCFP', 
                            K=3,
                            dataset_dir = '../../Datasets/XXXX',
                            origin_dir = '../../Datasets/XXXX',
                            target_dir = 'predict',
                            test_samples=2,
                            in_patients = None,
                            use_all=False,
                            show_process=False):
    if in_patients is not None:
        patients = in_patients
    if use_all:
        patients = os.listdir(origin_dir)
    for pat in patients:
        if show_process:
            print("[Now Is Processing]",pat)
        pat_dir = os.path.join(origin_dir, pat+'/crop')
        pre_face_fn = os.path.join(pat_dir, 'pre_face.ply')
        landmark_fn = os.path.join(origin_dir, pat+'/pre_face_landmarks.csv')
        landmark_df = pd.read_csv(landmark_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
        landmark_xyz = landmark_df.values
        predict_dirs = []
        for idx in range(test_samples):
            predict_dirs.append(os.path.join(dataset_dir, pat+'_{}'.format(str(idx).zfill(2))))
        for p_dir in predict_dirs:
            pre_face = o3d.io.read_triangle_mesh(pre_face_fn)
            pre_face_v = np.array(pre_face.vertices)
            # load info
            with open(os.path.join(p_dir,'info.pkl'), 'rb') as f:
                info = pickle.load(f)
            m = info['scale_size']
            # normalize origin shape
            centroid = np.mean(pre_face_v, axis=0)
            pre_face_v_n = pc_normalize(pre_face_v, m, centroid=centroid)
            # normalize landmark xyz
            landmark_xyz_n = pc_normalize(landmark_xyz, m, centroid=centroid)
            # load f_pre shape
            pre_face_ds = o3d.io.read_point_cloud(os.path.join(p_dir,'f_pre.ply'))
            pre_face_ds_v = np.array(pre_face_ds.points)
            # load displacement
            with open(os.path.join(p_dir,'%s/dis_%s.pkl'%(target_dir, model)), 'rb') as f:
                displace_s1, displace_s2, coords_s2 = pickle.load(f)
            # interpolation 1
            ball_tree = BallTree(pre_face_ds_v)
            idx = ball_tree.query(pre_face_v_n, k=K)[1]
            pre_face_ds_t = pre_face_ds_v[idx,:]      
            pre_face_t = np.expand_dims(pre_face_v_n, axis=1)  
            pre_face_t = np.tile(pre_face_t,(1,K,1))           
            dists = np.sum((pre_face_t - pre_face_ds_t)**2, axis=-1)   
            dist_recip = 1.0 / (dists + 1e-8)  
            norm = np.sum(dist_recip, axis=-1, keepdims=True)   
            weight = dist_recip / norm  
            interpolated_dis_s1 = np.sum(displace_s1[idx, :] * weight.reshape(weight.shape[0],weight.shape[1],1), axis=1)
            post_face_v_n_s1 = pre_face_v_n + interpolated_dis_s1
            # landmarks
            idx_lm = ball_tree.query(landmark_xyz_n, k=K)[1]   
            pre_face_ds_t = pre_face_ds_v[idx_lm,:]       
            lm_t = np.expand_dims(landmark_xyz_n, axis=1)
            lm_t = np.tile(lm_t,(1,K,1))          
            dists = np.sum((lm_t - pre_face_ds_t)**2, axis=-1) 
            dist_recip = 1.0 / (dists + 1e-8)
            norm = np.sum(dist_recip, axis=-1, keepdims=True)
            weight = dist_recip / norm
            landmark_dis_s1 = np.sum(displace_s1[idx_lm, :] * weight.reshape(weight.shape[0],weight.shape[1],1), axis=1)
            post_landmark_xyz_n_s1 = landmark_xyz_n + landmark_dis_s1
            # interpolation 2
            ball_tree = BallTree(coords_s2)
            idx = ball_tree.query(post_face_v_n_s1, k=K)[1]
            pre_face_ds_t = coords_s2[idx,:] 
            pre_face_t = np.expand_dims(post_face_v_n_s1, axis=1)
            pre_face_t = np.tile(pre_face_t,(1,K,1))   
            dists = np.sum((pre_face_t - pre_face_ds_t)**2, axis=-1)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = np.sum(dist_recip, axis=-1, keepdims=True)
            weight = dist_recip / norm
            interpolated_dis_s2 = np.sum(displace_s2[idx, :] * weight.reshape(weight.shape[0],weight.shape[1],1), axis=1)
            post_face_v_n = post_face_v_n_s1 + interpolated_dis_s2
            # landmarks
            idx_lm = ball_tree.query(post_landmark_xyz_n_s1, k=K)[1]
            pre_face_ds_t = coords_s2[idx_lm,:]
            lm_t = np.expand_dims(post_landmark_xyz_n_s1, axis=1)
            lm_t = np.tile(lm_t,(1,K,1))
            dists = np.sum((lm_t - pre_face_ds_t)**2, axis=-1)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = np.sum(dist_recip, axis=-1, keepdims=True)
            weight = dist_recip / norm
            landmark_dis_s2 = np.sum(displace_s2[idx_lm, :] * weight.reshape(weight.shape[0],weight.shape[1],1), axis=1)
            post_landmark_xyz_n = post_landmark_xyz_n_s1 + landmark_dis_s2
            post_face_v = post_face_v_n * m + centroid
            # construct and save mesh
            post_face = o3d.geometry.TriangleMesh()
            post_face.vertices = o3d.utility.Vector3dVector(post_face_v)
            post_face.triangles = pre_face.triangles
            post_face_fn = os.path.join(p_dir,'%s/post_face_%s.ply'%(target_dir, model))
            o3d.io.write_triangle_mesh(post_face_fn, post_face)
            # save post landmarks
            post_landmark_xyz = post_landmark_xyz_n * m + centroid
            post_landmark_fn = os.path.join(p_dir,'%s/post_landmark_%s.csv'%(target_dir, model))
            post_landmark_df = pd.DataFrame(post_landmark_xyz)
            post_landmark_df.to_csv(post_landmark_fn,index=False)

def mp_reconstruct_crop_rigion(model='DGCFP',
                               K=3,
                               dataset_dir = '../../Datasets/XXXX',
                               origin_dir = '../../Datasets/XXXX',
                               target_dir = 'predict',
                               test_samples=2,
                               in_patients = None,
                               use_all=False,
                               show_process=False):
    if in_patients is not None:
        patients = in_patients
    if use_all:
        patients = os.listdir(origin_dir)
    p = Pool(len(patients))
    for i in range(len(patients)):
        p.apply_async(reconstruct_crop_rigion, args=(model,K,dataset_dir,origin_dir,target_dir,test_samples,
                                                     [patients[i]],False,show_process,))
    p.close()
    p.join()

def get_landmark_error(model='DGCFP',
                       dataset_dir = '../../Datasets/XXXX',
                       origin_dir = '../../Datasets/XXXX',
                       target_dir = 'predict',
                       test_samples=1,
                       regions = None,
                       show_raw_error = False):
    patients = os.listdir(origin_dir)
    all_indices = []
    for i in range(len(regions)):
        all_indices += regions[i]
    all_error = []
    region_error = [[] for i in range(len(regions))]
    for pat in patients:
        landmark_fn = os.path.join(origin_dir, pat+'/post_face_landmarks.csv')
        landmark_df = pd.read_csv(landmark_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
        landmark_xyz = landmark_df.values
        predict_dirs = []
        for idx in range(test_samples):
            predict_dirs.append(os.path.join(dataset_dir, pat+'_{}/{}'.format(str(idx).zfill(2), target_dir)))
        for p_dir in predict_dirs:
            pred_landmark_fn = os.path.join(p_dir, f'post_landmark_{model}.csv')
            pred_landmark_df = pd.read_csv(pred_landmark_fn)
            pred_landmark_xyz = pred_landmark_df.values
            if show_raw_error:
                pred_landmark_fn = os.path.join(origin_dir, pat+'/pre_face_landmarks.csv')
                pred_landmark_df = pd.read_csv(pred_landmark_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
                pred_landmark_xyz = pred_landmark_df.values
            # landmark error
            all_landmark_xyz = landmark_xyz[all_indices, :]
            all_pred_landmark_xyz = pred_landmark_xyz[all_indices, :]
            per_error = np.mean(np.sqrt(np.sum((all_landmark_xyz-all_pred_landmark_xyz)**2, -1)))
            all_error.append(per_error)
            # landmark error regions
            for i in range(len(regions)):
                indices = regions[i]
                region_landmark_xyz = landmark_xyz[indices, :]
                region_pred_landmark_xyz = pred_landmark_xyz[indices, :]
                per_region_error = np.mean(np.sqrt(np.sum((region_landmark_xyz-region_pred_landmark_xyz)**2, -1)))
                region_error[i].append(per_region_error)
    all_landmark_error = np.mean(np.array(all_error))
    print("[All Landmark Error]\n\t", all_landmark_error)
    region_landmark_error = np.mean(np.array(region_error), axis=-1)
    for i in range(len(regions)):
        print(f"[Region {str(i)} Error]\n\t", region_landmark_error[i])
    # save
    results = {'patients': patients,
               'landmark_error': all_error}
    with open('./results/landmark_error.json','w') as f:
        json.dump(results, f)

def get_hausdorff_error(model='DGCFP', 
                        dataset_dir = '../../Datasets/XXXX',
                        origin_dir = '../../Datasets/XXXX',
                        target_dir = 'predict',
                        test_samples=1,
                        in_patients = None,
                        show_raw_error = False,
                        use_all=False,
                        show_process=False):
    patients = in_patients
    hd_mean = []
    hd_rms = []
    hd_max = []
    for pat in list(patients):
        if show_process:
            print("[Now Is Processing]",pat)
        predict_dirs = []
        for idx in range(test_samples):
            predict_dirs.append(os.path.join(dataset_dir, pat+'_{}/{}'.format(str(idx).zfill(2), target_dir)))
        origin_post_face_fn = os.path.join(origin_dir, pat+'/crop/post_face.ply')
        for p_dir in predict_dirs:
            if show_raw_error:
                pred_face_fn = os.path.join(origin_dir, pat+'/crop/pre_face.ply')
            else:
                pred_face_fn = os.path.join(p_dir, 'post_face_%s.ply'%(model))
            ms = mlb.MeshSet()
            ms.load_new_mesh(origin_post_face_fn)
            ms.load_new_mesh(pred_face_fn)
            ms.set_current_mesh(0)
            out_dict = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1)
            hd_mean.append(out_dict['mean'])
            hd_rms.append(out_dict['RMS'])
            hd_max.append(out_dict['max'])
    result = {'patients': patients,
              'mean': np.mean(np.array(hd_mean)),
              'rms': np.mean(np.array(hd_rms)), 
              'max': np.mean(np.array(hd_max)),
              'detail_mean': hd_mean,
              'detail_rms': hd_rms}
    return result

def mp_get_hausdorff_error(model='DGCFP', 
                           dataset_dir = '../../Datasets/XXXX',
                           origin_dir = '../../Datasets/XXXX',
                           target_dir = 'predict',
                           test_samples=1,
                           in_patients = None,
                           show_raw_error = False,
                           use_all=False,
                           show_process=False,
                           save_results=False):
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if in_patients is not None:
        patients = in_patients
    if use_all:
        patients = os.listdir(origin_dir)
    result_list = []
    p = Pool(len(patients))
    for i in range(len(patients)):
        result_list.append(p.apply_async(get_hausdorff_error, args=(model,dataset_dir,origin_dir,target_dir,test_samples,
                                                [patients[i]],show_raw_error,False,
                                                show_process,)))
    p.close()
    p.join()
    result = {'patients': [],
              'mean': [],
              'rms': [], 
              'max': [],
              'detail_mean': [],
              'detail_rms': []}
    for item in result_list:
        a_result = item.get()
        result['patients'] += a_result['patients']
        result['mean'] += [a_result['mean']]
        result['rms'] += [a_result['rms']]
        result['max'] += [a_result['max']]
        result['detail_mean'] += a_result['detail_mean']
        result['detail_rms'] += a_result['detail_rms']
    result['mean'] = np.mean(np.array(result['mean']))
    result['rms'] = np.mean(np.array(result['rms']))
    result['max'] = np.mean(np.array(result['max']))
    if save_results:
        with open('./results/hausdorff_distance.json','w') as f:
            json.dump(result, f)
    return result

if __name__ == "__main__":
    config_file = "./configuration/config_default.json"
    config = json.load(open(config_file))
    model = config["model_name"]
    dataset_dir = config["data_config"]["datadir"]
    origin_dir = config["data_config"]["origin_dir"]
    target_dir = config["inference"]["predict_dir"]
    test_samples = config["data_config"]["test_samples"]
    show_process = True
    regions = []
    get_landmark_error(model=model,dataset_dir=dataset_dir, origin_dir=origin_dir,
                       target_dir=target_dir, test_samples=test_samples,
                       show_raw_error = False, regions=regions)
    mp_get_hausdorff_error(model=model,dataset_dir=dataset_dir, origin_dir=origin_dir,
                        target_dir=target_dir, test_samples=test_samples, 
                        use_all=True, show_raw_error=False,save_results=True)