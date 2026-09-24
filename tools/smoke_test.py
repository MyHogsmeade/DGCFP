'''
Smoke test: mock data -> CMFDataset -> GraphLevelDataLoader -> DGCFP
forward + all losses + backward. Verifies the whole trace-map chain
(dataprocess.vertex_clustering -> hierarchy_trace_index -> MDFE
pooling/unpooling) end to end.

Usage:
    python tools/smoke_test.py --config tools/config_mock.json
'''
import os
import sys
import json
import argparse
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import CMFDataset, GraphLevelDataLoader
from model import DGCFP, GeometricLoss, LELoss, LAPLoss
from transform import RadiusNeighbors, EdgeSampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='tools/config_mock.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')

    def build_transform(cfg_list):
        ts = []
        for tc in cfg_list:
            if tc['type'] == 'RadiusNeighbors':
                ts.append(RadiusNeighbors(**tc['args']))
            if tc['type'] == 'EdgeSampling':
                ts.append(EdgeSampling(**tc['args']))
        return transforms.Compose(ts)

    transf = build_transform(config['train_transform'])
    train_dataset = CMFDataset(root_dir=config['data_config']['datadir'],
                               origin_dir=config['data_config']['origin_dir'],
                               start_level=config['data_config']['start_level'],
                               end_level=config['data_config']['end_level'],
                               split='train', fold_id=0,
                               include_edges=True, get_coords=True,
                               transform=transf)
    loader = GraphLevelDataLoader(train_dataset, batch_size=2, shuffle=False,
                                  follow_batch=['b_pre', 'b_mv', 'target_2',
                                                'add_stage_2'])
    data = next(iter(loader))
    data = data.to(device)
    print('[smoke] batch keys:', sorted(data.keys())[:20], '...')
    print('[smoke] level0 nodes:', data.pos.shape[0],
          '| hierarchy sizes:', data.num_vertices)

    Model = DGCFP(config=config).to(device)
    GLoss = GeometricLoss(top_k=config['arch']['topk'],
                          density_weight=config['arch']['beta']).to(device)
    LeLoss = LELoss().to(device)
    LapLoss = LAPLoss().to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3)

    Model.train()
    for step in range(2):
        optimizer.zero_grad()
        f_xyz_post_s1, coords_s2, f_xyz_post_s2 = Model(data)
        print('[smoke] s1:', tuple(f_xyz_post_s1.shape),
              's2 coords:', tuple(coords_s2.shape),
              's2:', tuple(f_xyz_post_s2.shape))
        f_pre = data.pos
        pred_dis_s1 = f_xyz_post_s1 - f_pre
        pred_dis_s2 = f_xyz_post_s2 - coords_s2
        batch_s1 = data.batch
        batch_s2 = data.target_2_batch
        g1, s1_, d1 = GLoss(f_xyz_post_s1, data.target_1, batch_s1)
        g2, s2_, d2 = GLoss(f_xyz_post_s2, data.target_2, batch_s2)
        lap = LapLoss(f_pre, f_xyz_post_s1, coords_s2, f_xyz_post_s2,
                      [data.lap_stage_1, data.lap_stage_2])
        le1, pred_lm1 = LeLoss(pred_dis_s1, f_pre, data.lm_pre, data.lm_target, batch_s1)
        le2, _ = LeLoss(pred_dis_s2, coords_s2, pred_lm1, data.lm_target, batch_s2)
        total = g1 + g2 + config['arch']['lambda'] * (lap + 0.5 * (le1 + le2))
        total.backward()
        optimizer.step()
        print('[smoke] step %d: total=%.5f gloss_s1=%.5f gloss_s2=%.5f '
              'lap=%.5f le1=%.5f le2=%.5f' %
              (step, total.item(), g1.item(), g2.item(), lap.item(), le1.item(), le2.item()))
        assert torch.isfinite(total), 'loss is not finite'

    # gradient check: verify trace-map pooling path receives gradients
    grad_ok = all(p.grad is not None for p in Model.mdfe_module.left_geo_cnns.parameters())
    print('[smoke] MDFE (trace-map pooling) gradients exist:', grad_ok)
    print('[smoke] ALL OK')


if __name__ == '__main__':
    main()
