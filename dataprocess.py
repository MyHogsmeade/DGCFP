'''
preprocess data
'''
import os
import open3d as o3d
import csv
import numpy as np
import point_cloud_utils as pcu
import pickle
import random
import torch
from sklearn.neighbors import BallTree
from typing import List
from collections import defaultdict
import trimesh
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import argparse

class QEMError(Exception):
    pass

def get_max_scale_size(pcs, centroid):
    m_list = []
    for pc in pcs:
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        m_list.append(m)
    return max(m_list)

def pc_normalize(pc, m, centroid):
    pc = pc - centroid
    pc = pc / m
    return pc

def csv2npy(in_file_path, old_vertices, new_vertices):
    old_ball_tree = BallTree(old_vertices[:, :3])
    new_ball_tree = BallTree(new_vertices[:, :3])
    trace = []
    new2old = {}   
    old_nodes_set = set()   
    new_nodes_set = set()   
    num_nodes = 0
    with open(in_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            new_coords = [float(r) for r in row[:3]]
            num_traces = len(row) // 3 - 1
            new_id = new_ball_tree.query(
                np.array(new_coords).reshape(1, -1), k=1)[1].flatten()[0]
            if not new2old.get(new_id):
                new2old[new_id] = []
            else:
                raise QEMError('GRAPH LEVEL GENERATION ERROR')
            new_nodes_set.add(new_id)
            for i in range(num_traces):
                old_coords = [float(r) for r in row[3+3*i:6+3*i]]
                old_id = old_ball_tree.query(
                    np.array(old_coords).reshape(1, -1), k=1)[1].flatten()[0]
                new2old[new_id].append(old_id)
                old_nodes_set.add(old_id)
    all_old_ids = list(
        set([i for i in range(old_vertices.shape[0])]) - old_nodes_set)
    if np.array(old_vertices[all_old_ids]).shape[0] != 0:
        new_ids = new_ball_tree.query(
            np.array(old_vertices[all_old_ids]), k=1)[1].flatten()
        new_nodes_set.update(new_ids.tolist())
        old_nodes_set.update(all_old_ids)
        for id, new_id in enumerate(new_ids):
            if not new2old.get(new_id):
                new2old[new_id] = []
            new2old[new_id].append(all_old_ids[id])
    assert new_vertices.shape[0] == len(new_nodes_set)
    assert old_vertices.shape[0] == len(old_nodes_set)
    reverse_trace = np.empty((len(old_nodes_set)), dtype=np.int32)
    reverse_trace.fill(-1)
    for new_id, old_ids in new2old.items():
        for old_id in old_ids:
            assert reverse_trace[old_id] == -1
            reverse_trace[old_id] = new_id
    return reverse_trace

def randomVC_save(input_mesh, out_mesh, cell_nums):
    in_mesh = o3d.io.read_triangle_mesh(input_mesh)
    aabb = in_mesh.get_axis_aligned_bounding_box()
    box = aabb.get_extent()
    coords = np.array(in_mesh.vertices)
    faces = np.array(in_mesh.triangles)
    in_mesh.compute_vertex_normals()
    k = int(pow(cell_nums, 1/3)) + 1
    voxel_size = box / k
    bins = coords // voxel_size
    unique_bins, reverse_trace_ids = np.unique(
        bins, axis=0, return_inverse=True)
    points_index = []
    for bin_id in range(len(unique_bins)):
        points_in_bin = np.argwhere(reverse_trace_ids == bin_id).flatten()
        cell_point = np.random.choice(points_in_bin)
        points_index.append(cell_point)
    points_index = np.array(points_index)
    new_coords = coords[points_index]
    new_normals = np.array(in_mesh.vertex_normals)[points_index,:]
    faces = reverse_trace_ids[faces]
    # degeneration
    flag = (faces[:,0] == faces[:,1]) | (faces[:,1] == faces[:,2]) | (faces[:,0] == faces[:,2])
    faces = faces[~flag,:]
    # save
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_coords)
    new_mesh.triangles = o3d.utility.Vector3iVector(faces)
    new_mesh.vertex_normals = o3d.utility.Vector3dVector(new_normals)
    new_mesh2 = new_mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(out_mesh, new_mesh2)

def cal_lap_index(mesh_neighbor):
    lap_index = np.zeros([len(mesh_neighbor), 2 + 8]).astype(np.int32)
    for i, j in enumerate(mesh_neighbor):
        lenj = len(j) if len(j)<8 else 8
        lap_index[i][0:lenj] = j[:lenj]
        lap_index[i][lenj:-2] = -1
        lap_index[i][-2] = i
        lap_index[i][-1] = lenj
    return lap_index

def get_face_index(faces, unpool_num):
    face_inskin_idx = np.arange(faces.shape[0])
    faces_ds_idx = np.random.choice(face_inskin_idx,
                                    unpool_num,
                                    replace=False)
    faces_ds_idx = set(faces_ds_idx.tolist())
    faces_remain_idx = np.sort(np.array(list(set(range(faces.shape[0])) - faces_ds_idx)))
    faces_ds_idx = np.sort(np.array(list(faces_ds_idx)))
    return faces_ds_idx, faces_remain_idx

def randomVC_indices(coords, box, cell_nums, sample_num_list, in_rand=False):
    k = int(pow(cell_nums, 1/3)) + 1
    voxel_size = box / k 
    bins = coords // voxel_size
    unique_bins, reverse_trace_ids = np.unique(
        bins, axis=0, return_inverse=True)
    points_index = []
    for bin_id in range(len(unique_bins)):
        points_in_bin = np.argwhere(reverse_trace_ids == bin_id).flatten()
        cell_point = np.random.choice(points_in_bin)
        points_index.append(cell_point)
    points_index = np.array(points_index)
    points_ds_list = []
    if in_rand:
        temp_ds = points_index
        for sample_num in sample_num_list[-1::-1]:
            if temp_ds.shape[0] >= sample_num:
                points_ds = np.random.choice(temp_ds, size=sample_num, replace=False)
            else:
                points_ds = np.random.choice(temp_ds, size=sample_num, replace=True)
            points_ds_list.append(points_ds)
            temp_ds = points_ds
        points_ds_list = points_ds_list[-1::-1]
    else:
        for sample_num in sample_num_list:
            if points_index.shape[0] >= sample_num:
                points_ds = np.random.choice(points_index, size=sample_num, replace=False)
            else:
                points_ds = np.random.choice(points_index, size=sample_num, replace=True)
            points_ds_list.append(points_ds)
    return points_ds_list 

def sample_point_indices(pc, num, method='uniform', pre_bone=None, cell_num=20000):
    l = pc.shape[0]
    idx = range(l)
    if method == 'uniform':
        indices = np.random.choice(idx, size=num, replace=False)
    if method == 'poisson':
        indices = pcu.downsample_point_cloud_poisson_disk(pc, num_samples=num)
        indices = indices[:num]
    if method == 'VC':
        aabb = pre_bone.get_axis_aligned_bounding_box()
        box = aabb.get_extent()
        pre_bone_idx_list = randomVC_indices(pc, box, cell_num+10000, sample_num_list=[num])
        indices = np.array(pre_bone_idx_list[0])
    indices = sorted(indices.tolist())
    return indices

def downsample_post_face(coords, box, cell_nums, sample_num_list):
    k = int(pow(cell_nums, 1/3)) + 1
    voxel_size = box / k 
    bins = coords // voxel_size
    unique_bins, reverse_trace_ids = np.unique(
        bins, axis=0, return_inverse=True)
    points_index = []
    for bin_id in range(len(unique_bins)):
        points_in_bin = np.argwhere(reverse_trace_ids == bin_id).flatten()
        cell_point = np.random.choice(points_in_bin)
        points_index.append(cell_point)
    points_index = np.array(points_index)
    points_ds_list = []
    points_ds = np.random.choice(points_index, size=sample_num_list[0], replace=False)
    points_ds_list.append(points_ds)
    remain_idx = set(points_index.tolist())
    for idx in range(1,len(sample_num_list)):
        sample_num = sample_num_list[idx]
        remain_idx = remain_idx - set(points_ds_list[-1].tolist())
        temp = np.array(list(remain_idx))
        if temp.shape[0] >= sample_num - sample_num_list[idx-1]:
            points_ds = np.random.choice(temp, 
                                         size=sample_num - sample_num_list[idx-1], 
                                         replace=False)
        else:
            points_ds = np.random.choice(temp, 
                                         size=sample_num - sample_num_list[idx-1], 
                                         replace=True)
        new_ds = np.hstack([points_ds_list[-1],points_ds])
        points_ds_list.append(new_ds)
    return points_ds_list 

def down_sample_by_idx(shape, indices):
    vertices = np.array(shape.vertices)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vertices[indices,:])
    return pc

def vertex_clustering(coords: np.ndarray, edges: List[List[int]], voxel_size: float):
    bins = coords // voxel_size
    unique_bins, reverse_trace_ids = np.unique(
        bins, axis=0, return_inverse=True)
    edge_map = defaultdict(set)
    edge_output = []
    traces = []
    for bin_id in range(len(unique_bins)):
        points_in_bin = np.argwhere(reverse_trace_ids == bin_id).flatten()
        traces.append(points_in_bin)
        for point_id in points_in_bin:
            try:
                neigh = edges[point_id]
            except IndexError:
                print('WARNING: single point with no connections')
                continue
            new_edges = reverse_trace_ids[neigh]
            edge_map[bin_id].update(set(new_edges))
    new_edge_list = []
    for key in sorted(edge_map):
        edge_map[key].discard(key)  
        value = list(edge_map[key])
        new_edge_list.append(value)
        for elem in value:
            edge_output.append([key, elem])
    new_coords = np.empty((len(traces), 3), dtype=np.float32)
    for index, trace in enumerate(traces):
        new_coords[index, :] = coords[trace].mean(axis=0)
    return new_coords, reverse_trace_ids, new_edge_list, edge_output

def quadric_error_metric(input_fn, out_fn, ratio, old_vertices):
    os.system(f"tridecimator "
              f"{input_fn} {out_fn} {ratio} -On -C > /dev/null")
    mesh = o3d.io.read_triangle_mesh(out_fn)
    if not mesh.has_vertices():
        raise QEMError('no vertices left')
    coords = np.asarray(mesh.vertices)
    edges_list = edges_from_faces(np.asarray(mesh.triangles))
    edge_out = []
    for key, group in enumerate(edges_list):
        for elem in group:
            edge_out.append([key, elem])
    reverse_trace = csv2npy(out_fn.replace(
        '.ply', '.csv'), old_vertices=old_vertices, new_vertices=coords)
    return coords, edge_out, reverse_trace

def edges_from_faces(faces):
    edges = defaultdict(set)
    for i in range(len(faces)):
        edges[faces[i, 0]].update(faces[i, (1, 2)])
        edges[faces[i, 1]].update(faces[i, (0, 2)])
        edges[faces[i, 2]].update(faces[i, (0, 1)])
    edge_list = []
    for vertex_id in range(len(edges)):
        connected_vertices = edges[vertex_id]
        edge_list.append(list(connected_vertices))
    return edge_list

def construct_multi_level(input_mesh_fn, level_params,
                          simplify_type='VC', use_normal=True):
    mesh_name = input_mesh_fn.split('.ply')[0]
    original_mesh = o3d.io.read_triangle_mesh(input_mesh_fn)
    if use_normal:
        original_vertices = np.column_stack(
                (np.asarray(original_mesh.vertices),
                np.asarray(original_mesh.vertex_normals)))
    else:
        original_vertices = np.asarray(original_mesh.vertices)
    coords = []
    edges_list = []
    edge_output = []
    traces = []
    curr_mesh = original_mesh
    curr_vertices = np.asarray(curr_mesh.vertices)
    edge_list_0 = edges_from_faces(np.asarray(curr_mesh.triangles))
    coords.append(curr_vertices)
    edges_list.append(edge_list_0)
    edge_output_0 = []
    for key, group in enumerate(edge_list_0):
        for elem in group:
            edge_output_0.append([key, elem])
    edge_output.append(np.array(edge_output_0))
    # graph
    input_fn = input_mesh_fn
    for level in range(len(level_params)):
        # VC
        if simplify_type == 'VC':
            coords_l, trace_scatter, edge_list_l, edge_output_l = \
                vertex_clustering(
                    coords[-1], edges_list[-1], float(level_params[level]))
        # QEM
        if simplify_type == 'QEM':
            out_fn = mesh_name + f'_{level}.ply'
            coords_l, edge_output_l, trace_scatter = \
                quadric_error_metric(input_fn, out_fn, int(level_params[level]),
                                        old_vertices=coords[-1])
            input_fn = out_fn
            edge_list_l = None
        coords.append(coords_l)
        traces.append(trace_scatter)
        edges_list.append(edge_list_l)
        edge_output.append(np.array(edge_output_l))
    coords[0] = original_vertices
    pt_data = {}
    vertices = [torch.from_numpy(coords[i]).float()
                for i in range(len(coords))]
    pt_data['vertices'] = vertices
    pt_data['edges'] = [torch.from_numpy(
                        edge_output[i]).long() for i in range(len(edge_output))]
    pt_data['traces'] = [torch.from_numpy(x).long() for x in traces]
    return pt_data

def unpool_face(old_faces, face_remain_idx, face_ds_idx, old_vertices):
    old_faces = np.array(old_faces)
    N = old_vertices.shape[0]
    new_faces = []
    for i, f in enumerate(old_faces[face_ds_idx,:]):
        f = np.sort(f)
        mid_idx = N+i
        new_faces.append([f[0], f[1], mid_idx])
        new_faces.append([f[0], f[2], mid_idx])
        new_faces.append([f[1], f[2], mid_idx])
    new_faces = np.array(new_faces, dtype=np.int32)
    new_faces = np.vstack([old_faces[face_remain_idx,:], new_faces])
    return new_faces

def handle_process(pat, N, sample_times, sample_method, base_dir, target_dir, cell_num, unpool_num,
                   level_params, simplify_type, use_normal):
    print('[Now is processing]',pat)
    # file names
    pat_dir = os.path.join(base_dir, pat+'/crop/')
    pre_bone_fn = os.path.join(pat_dir, 'pre_bone.ply')
    pre_bone_plan_fn = os.path.join(pat_dir, 'pre_bone_plan.ply')
    pre_face_fn = os.path.join(pat_dir, 'pre_face.ply')
    post_face_fn = os.path.join(pat_dir, 'post_face.ply')
    # load all shape
    pre_bone = o3d.io.read_triangle_mesh(pre_bone_fn)
    pre_bone_v = np.array(pre_bone.vertices)
    pre_bone_plan = o3d.io.read_triangle_mesh(pre_bone_plan_fn)
    pre_bone_plan_v = np.array(pre_bone_plan.vertices)
    pre_face = o3d.io.read_triangle_mesh(pre_face_fn)
    pre_face_v = np.array(pre_face.vertices)
    post_face = o3d.io.read_triangle_mesh(post_face_fn)
    post_face_v = np.array(post_face.vertices)
    # normalize
    all_centroid = np.mean(pre_face_v, axis=0)
    m = get_max_scale_size([pre_bone_v, pre_bone_plan_v, pre_face_v, post_face_v], centroid=all_centroid)
    pre_bone_v = pc_normalize(pre_bone_v, m, centroid=all_centroid)
    pre_bone_plan_v = pc_normalize(pre_bone_plan_v, m, centroid=all_centroid)
    post_face_v = pc_normalize(post_face_v, m, centroid=all_centroid)
    pre_bone.vertices = o3d.utility.Vector3dVector(pre_bone_v)
    pre_bone_plan.vertices = o3d.utility.Vector3dVector(pre_bone_plan_v)
    post_face.vertices = o3d.utility.Vector3dVector(post_face_v)
    # down sample
    for i in range(sample_times):
        a_sample_dir = os.path.join(target_dir, pat+'_{}'.format(str(i).zfill(2)))
        if not os.path.exists(a_sample_dir):
            os.makedirs(a_sample_dir)
        input_mesh = pre_face_fn
        out_mesh = os.path.join(a_sample_dir, 'pre_face.ply')
        randomVC_save(input_mesh, out_mesh, cell_num)
        pre_face_ds = o3d.io.read_triangle_mesh(out_mesh)
        pre_face_ds_v = np.array(pre_face_ds.vertices)
        pre_face_ds_v = pc_normalize(pre_face_ds_v, m, centroid=all_centroid)
        pre_face_ds.vertices = o3d.utility.Vector3dVector(pre_face_ds_v)
        ## Stage 1
        # laplacian coordinate
        mesh1 = trimesh.Trimesh(vertices=np.array(pre_face_ds.vertices),
                                faces=np.array(pre_face_ds.triangles),
                                process=False)
        coords1 = np.array(mesh1.vertices, dtype=np.float32)
        lap1 = cal_lap_index(mesh1.vertex_neighbors)
        # sample faces
        faces1 = np.array(mesh1.faces)
        faces1_ds_idx, faces1_remain_idx = get_face_index(faces1, unpool_num)
        ## Stage 2
        faces2 = unpool_face(faces1, faces1_remain_idx, faces1_ds_idx, coords1)
        temp_1_2 = np.mean(coords1[faces1[faces1_ds_idx,:],:], axis=1)
        coords2 = np.vstack([coords1, temp_1_2])
        # laplacian coordinate
        mesh2 = trimesh.Trimesh(vertices=coords2, faces=faces2, process=False) 
        lap2 = cal_lap_index(mesh2.vertex_neighbors)
        # edge graph
        edge_list_2 = mesh2.vertex_neighbors
        edge_output_2 = []
        for key, group in enumerate(edge_list_2):
            for elem in group:
                edge_output_2.append([key, elem])
        # save
        mesh2_fn = os.path.join(a_sample_dir, 'f_pre_stage_2.ply')
        mesh2_ply = o3d.geometry.TriangleMesh()
        mesh2_ply.vertices = o3d.utility.Vector3dVector(coords2)
        mesh2_ply.triangles = o3d.utility.Vector3iVector(faces2)
        o3d.io.write_triangle_mesh(mesh2_fn, mesh2_ply) 
        # bone sampling
        pre_bone_idx = sample_point_indices(pre_bone_v, N, method=sample_method, 
                                            pre_bone=pre_bone, cell_num=cell_num)
        # post_face sampling
        coords = post_face_v
        aabb = post_face.get_axis_aligned_bounding_box()
        box = aabb.get_extent()
        post_face_idx_list = downsample_post_face(coords, box, cell_num+10000+1000, 
                                                  sample_num_list=[pre_face_ds_v.shape[0], coords2.shape[0]])
        # down sample pre bone
        pre_bone_ds = down_sample_by_idx(pre_bone, pre_bone_idx)
        pre_bone_ds_save = os.path.join(a_sample_dir, 'b_pre.ply')
        o3d.io.write_point_cloud(pre_bone_ds_save, pre_bone_ds)
        # pre bone plan same as pre bone
        pre_bone_plan_ds = down_sample_by_idx(pre_bone_plan, pre_bone_idx)
        pre_bone_plan_ds_save = os.path.join(a_sample_dir, 'b_post.ply')
        o3d.io.write_point_cloud(pre_bone_plan_ds_save, pre_bone_plan_ds)
        # down sample pre face
        pre_face_ds_save = os.path.join(a_sample_dir, 'f_pre.ply')
        o3d.io.write_triangle_mesh(pre_face_ds_save, pre_face_ds)
        # down sample post face
        post_face_ds1 = down_sample_by_idx(post_face, post_face_idx_list[0])
        post_face_ds1_save = os.path.join(a_sample_dir, 'f_target_1.ply')
        o3d.io.write_point_cloud(post_face_ds1_save, post_face_ds1)
        post_face_ds2 = down_sample_by_idx(post_face, post_face_idx_list[1])
        post_face_ds2_save = os.path.join(a_sample_dir, 'f_target_2.ply')
        o3d.io.write_point_cloud(post_face_ds2_save, post_face_ds2)
        # generate layers
        pt_data = construct_multi_level(pre_face_ds_save, level_params,
                                        simplify_type=simplify_type, 
                                        use_normal=use_normal)
        # save pt
        pt_data['b_pre'] = torch.from_numpy(
                            np.array(pre_bone_ds.points)).float()
        pt_data['b_post'] = torch.from_numpy(
                            np.array(pre_bone_plan_ds.points)).float()
        pt_data['f_targets'] = [torch.from_numpy(
                                np.array(post_face_ds1.points)).float(),
                                torch.from_numpy(
                                np.array(post_face_ds2.points)).float()]
        pt_data['stage_edges'] = torch.from_numpy(
                                    np.array(edge_output_2)).long()   
        pt_data['stage_faces'] = torch.from_numpy(
                                    faces1[faces1_ds_idx,:]).long()    
        pt_data['laps_coords'] = [torch.from_numpy(lap1).long(),torch.from_numpy(lap2).long()]
        pt_fn = os.path.join(a_sample_dir, 'input_data.pt')
        torch.save(pt_data, pt_fn)       
        # save info
        info = {
            'pre_bone_idx': pre_bone_idx,
            'scale_size': m,
            'centroid': all_centroid
        }
        info_save = os.path.join(a_sample_dir, 'info.pkl')
        with open(info_save, 'wb') as f:
            pickle.dump(info, f)

def mp_generate_dataset(N, sample_times, kfold, test_samples, sample_method, level_params,
                        base_dir, target_dir, split_dir, simplify_type, use_normal, cell_num, 
                        unpool_num):
    N = N        
    sample_times = sample_times
    kfold = kfold
    test_samples = test_samples
    sample_method = sample_method
    patients = os.listdir(base_dir)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    # generate dataset
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    _ = list(pool.map(handle_process, patients, repeat(N), repeat(sample_times), repeat(sample_method),
                      repeat(base_dir), repeat(target_dir), repeat(cell_num), repeat(unpool_num),
                      repeat(level_params), repeat(simplify_type), repeat(use_normal)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base', type=str, default='../../Datasets/XXXX/', help='original data path')
    parser.add_argument('--output_dir', type=str, default='../../Datasets/XXXX/', help='processed data output')
    parser.add_argument('--num_points', type=int, help='num of points')
    parser.add_argument('--sampleTr', type=int, help='train sampling')
    parser.add_argument('--sampleVal', type=int, help='valid sampling')
    parser.add_argument('--folds', type=int, help='cross validation folds')
    parser.add_argument('--sample_type', type=str, help='uniform or VC or possion')
    parser.add_argument('--level_params', metavar='N', type=int, nargs='+', help='mesh simplify parameters')
    parser.add_argument('--mesh_simplify_type', type=str, help='mesh simplify type')
    parser.add_argument('--cell_num', type=int, help='mesh cell num')
    parser.add_argument('--use_normal', default=False, action='store_true', help='whether to use normal')
    parser.add_argument('--upsample_num', type=int, help='num new points')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # data dir
    args = parse_args()
    base_dir = args.data_base
    target_dir = args.output_dir
    split_dir = f'{args.output_dir}train_test_split/'
    # configuration
    N = args.num_points
    sampleTr = args.sampleTr
    sampleVal = args.sampleVal
    kfold = args.folds
    sample_type = args.sample_type
    level_params = args.level_params
    simplify_type = args.mesh_simplify_type
    cell_num = args.cell_num
    use_normal = args.use_normal
    unpool_num = args.upsample_num
    # process
    mp_generate_dataset(N, sampleTr, kfold, sampleVal, sample_type, level_params, 
                        base_dir=base_dir, target_dir=target_dir, split_dir=split_dir,
                        simplify_type=simplify_type, use_normal=use_normal,
                        cell_num=cell_num, unpool_num=unpool_num)