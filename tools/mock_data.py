'''
Generate a mock dataset that matches the file structure expected by
CMFDataset / main.py, so the whole pipeline (including the multi-level
trace maps produced by dataprocess.vertex_clustering) can be exercised
without real CMF patient data.

Usage:
    python tools/mock_data.py --out mock_data/dataset --origin mock_data/origin \
        --npoint 1024 --unpool_num 300 --train_samples 4 --valid_samples 2
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataprocess import (edges_from_faces, vertex_clustering,
                         cal_lap_index, get_face_index, unpool_face)


def build_levels(mesh, voxel_sizes, use_normal=True):
    '''Build multi-level hierarchy (with trace maps) using the repo's
    vertex_clustering, the same way dataprocess.construct_multi_level does.'''
    coords = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    edges_list = edges_from_faces(np.asarray(mesh.faces))
    edge_output = []
    for key, group in enumerate(edges_list):
        for elem in group:
            edge_output.append([key, elem])
    coords_out = [np.column_stack((coords, normals)) if use_normal else coords]
    edges_out = [np.array(edge_output)]
    traces = []
    cur_edges_list = edges_list
    for vs in voxel_sizes:
        coords_l, trace_scatter, new_edges_list, edge_output_l = \
            vertex_clustering(coords_out[-1][:, :3], cur_edges_list, vs)
        coords_out.append(coords_l)
        pairs = [[i, e] for i, lst in enumerate(new_edges_list) for e in lst]
        edges_out.append(np.array(pairs))
        traces.append(torch.from_numpy(trace_scatter).long())
        cur_edges_list = new_edges_list
    return coords_out, edges_out, traces


def knn_edges(coords, k=6):
    '''symmetric kNN edge list (fallback for meshes without faces)'''
    from sklearn.neighbors import BallTree
    tree = BallTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    nbrs = [sorted(set(row) - {i}) for i, row in enumerate(idx)]
    edges = []
    for i, row in enumerate(nbrs):
        for j in row:
            edges.append([i, j])
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='mock_data/dataset')
    parser.add_argument('--origin', default='mock_data/origin')
    parser.add_argument('--npoint', type=int, default=1024)
    parser.add_argument('--unpool_num', type=int, default=300)
    parser.add_argument('--train_samples', type=int, default=4)
    parser.add_argument('--valid_samples', type=int, default=2)
    parser.add_argument('--test_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    # ---- base mesh: unit sphere subdivided ----
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertex_normals
    n0 = len(mesh.vertices)
    voxel_sizes = [0.08, 0.14, 0.22, 0.32]
    coords, edges, traces = build_levels(mesh, voxel_sizes)
    print('[mock] level sizes:', [c.shape[0] for c in coords])

    for split, n_samples in [('train', args.train_samples),
                             ('valid', args.valid_samples),
                             ('test', args.test_samples)]:
        split_dir = os.path.join(args.out, 'train_test_split')
        os.makedirs(split_dir, exist_ok=True)
        lines = []
        # samples grouped contiguously per patient: pat_00, pat_01, ...
        samples_per_pat = 2
        pat_offset = 0
        for split_idx, (s_name, n) in enumerate([('train', args.train_samples),
                                                 ('valid', args.valid_samples),
                                                 ('test', args.test_samples)]):
            if s_name == split:
                pat_offset = sum(x for s, x in [('train', args.train_samples),
                                                ('valid', args.valid_samples),
                                                ('test', args.test_samples)][:split_idx]) \
                             // samples_per_pat
        for i in range(n_samples):
            pat = 'p{:03d}'.format(pat_offset + i // samples_per_pat)
            tag = '{}_{:02d}'.format(pat, i % samples_per_pat)
            lines.append(tag)
            sample_dir = os.path.join(args.out, tag)
            os.makedirs(sample_dir, exist_ok=True)

            # bone point clouds (fixed size == config npoint)
            b_pre = rng.uniform(-1, 1, (args.npoint, 3)).astype(np.float32)
            b_pre /= np.linalg.norm(b_pre, axis=1, keepdims=True)
            b_post = b_pre + 0.02 * rng.randn(args.npoint, 3).astype(np.float32)

            # stage-2 mesh: unpool a random subset of level-0 faces
            faces1 = np.asarray(mesh.faces)
            faces1_ds_idx, faces1_remain_idx = get_face_index(faces1, args.unpool_num)
            faces2 = unpool_face(faces1, faces1_remain_idx, faces1_ds_idx, coords[0][:, :3])
            temp_1_2 = np.mean(coords[0][:, :3][faces1[faces1_ds_idx, :], :], axis=1)
            coords2 = np.vstack([coords[0][:, :3], temp_1_2])
            edge_list_2 = edges_from_faces(faces2)
            edge_output_2 = []
            for key, group in enumerate(edge_list_2):
                for elem in group:
                    edge_output_2.append([key, elem])
            lap1 = cal_lap_index(edges_from_faces(faces1))
            lap2 = cal_lap_index(edge_list_2)

            # targets: post-op face approximated by a slightly moved sphere
            target_1 = coords[0][:, :3] + 0.05 * rng.randn(n0, 3).astype(np.float32)
            target_2 = coords2 + 0.02 * rng.randn(len(coords2), 3).astype(np.float32)

            pt_data = {
                'vertices': [torch.from_numpy(c.astype(np.float32)) for c in coords],
                'edges': [torch.from_numpy(e.astype(np.int64)) for e in edges],
                'traces': traces,
                'b_pre': torch.from_numpy(b_pre),
                'b_post': torch.from_numpy(b_post),
                'f_targets': [torch.from_numpy(target_1.astype(np.float32)),
                              torch.from_numpy(target_2.astype(np.float32))],
                'stage_edges': torch.from_numpy(np.array(edge_output_2)).long(),
                'stage_faces': torch.from_numpy(faces1[faces1_ds_idx, :]).long(),
                'laps_coords': [torch.from_numpy(lap1).long(),
                                torch.from_numpy(lap2).long()],
            }
            torch.save(pt_data, os.path.join(sample_dir, 'input_data.pt'))

            # info.pkl
            import pickle
            info = {'scale_size': float(1.0),
                    'centroid': np.zeros(3, dtype=np.float64)}
            with open(os.path.join(sample_dir, 'info.pkl'), 'wb') as f:
                pickle.dump(info, f)

            # f_pre.ply (point cloud used by reconstruction step)
            pc = trimesh.PointCloud(coords[0][:, :3])
            pc.export(os.path.join(sample_dir, 'f_pre.ply'))

            # origin patient dir: landmarks + crop meshes (for validation metrics)
            pat_dir = os.path.join(args.origin, pat, 'crop')
            os.makedirs(pat_dir, exist_ok=True)
            if not os.path.exists(os.path.join(args.origin, pat, 'pre_face_landmarks.csv')):
                lm = rng.uniform(-1, 1, (10, 3))
                # real landmark files: col0 = landmark id, cols 1-3 = x,y,z
                lm_df = pd.DataFrame({'id': range(len(lm)),
                                      'x': lm[:, 0], 'y': lm[:, 1], 'z': lm[:, 2]})
                lm_df.to_csv(
                    os.path.join(args.origin, pat, 'pre_face_landmarks.csv'), index=False)
                lm_df.iloc[:, 1:] += 0.01
                lm_df.to_csv(
                    os.path.join(args.origin, pat, 'post_face_landmarks.csv'), index=False)
            pre_mesh = trimesh.Trimesh(vertices=coords[0][:, :3],
                                       faces=np.asarray(mesh.faces), process=False)
            pre_mesh.export(os.path.join(pat_dir, 'pre_face.ply'))
            post_mesh = trimesh.Trimesh(vertices=target_1, faces=np.asarray(mesh.faces),
                                        process=False)
            post_mesh.export(os.path.join(pat_dir, 'post_face.ply'))
        with open(os.path.join(split_dir, 'fold_0_{}_file_list.txt'.format(split)), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    print('[mock] dataset written to', args.out, 'and', args.origin)


if __name__ == '__main__':
    main()
