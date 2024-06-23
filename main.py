'''
main function
'''
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import math
import time
import json
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.optim as optim
from dataset import CMFDataset, GraphLevelDataLoader
from model import DGCFP, GeometricLoss, LELoss, LAPLoss
from torchsummary import summary
from torch import autograd
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from transform import RadiusNeighbors, EdgeSampling
from utils import mp_reconstruct_crop_rigion, mp_get_hausdorff_error, AverageMeter
import pymeshlab as mlb
from multiprocessing import Pool
import wandb

def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def parse_args():
    parser = argparse.ArgumentParser('DGCFP')
    parser.add_argument('--is_train', default=True, action='store_true', help='train model or inference')
    parser.add_argument('--config', type=str, default='./configuration/config_default.json', help='config file path')
    return parser.parse_args()


def save_log(log_data, log_file, item_num, cur_epoch=-1):
    if cur_epoch != -1:
        temp = log_data.copy()
        temp.append(['cur_best_epoch' for i in range(item_num) ])
        temp.append([cur_epoch for i in range(item_num) ])
        temp_pd = pd.DataFrame(columns=['total_loss',
                                        'gloss', 'gloss_s1', 'gloss_s2',
                                        'sloss', 'sloss_s1', 'sloss_s2',
                                        'dloss', 'dloss_s1', 'dloss_s2',
                                        'lm_r_laploss',
                                        'hausdorff distance','landmark error'], data=temp)
        temp_pd.to_csv(log_file,encoding= 'gbk',index=False)
    else:
        train_result_pd = pd.DataFrame(columns=['total_loss',
                                                'gloss', 'gloss_s1', 'gloss_s2',
                                                'sloss', 'sloss_s1', 'sloss_s2',
                                                'dloss', 'dloss_s1', 'dloss_s2',
                                                'lm_r_laploss'], data=log_data)
        train_result_pd.to_csv(log_file,encoding= 'gbk',index=False)


def train_model(config, fold_id, run=None, is_recons=True):
    '''
    Function for training
    '''
    outf = config["data_config"]["outf"]
    if not os.path.exists(os.path.join(outf, 'history')):
        os.makedirs(os.path.join(outf, 'history'))
    seed_torch(seed=config["training_config"]["manualSeed"])
    # data transformer
    transf_list_train = []
    for transform_config in config['train_transform']:
        if transform_config["type"]=="RadiusNeighbors":
            transf_list_train.append(RadiusNeighbors(**transform_config['args']))
        if transform_config["type"]=="EdgeSampling":
            transf_list_train.append(EdgeSampling(**transform_config['args']))
    transf_train = transforms.Compose(transf_list_train)
    # dataset & dataloader
    train_dataset = CMFDataset(root_dir=config["data_config"]["datadir"],
                               origin_dir=config["data_config"]["origin_dir"],
                               start_level=config["data_config"]["start_level"],
                               end_level=config["data_config"]["end_level"],
                               split='train',
                               fold_id=fold_id,
                               transform=transf_train)
    train_dataloader = GraphLevelDataLoader(train_dataset,
                                            batch_size=config["training_config"]["batch_size"],
                                            shuffle=True,
                                            follow_batch=['b_pre', 'b_mv', 'target_2', 'add_stage_2'],
                                            num_workers=4)
    valid_dataset = CMFDataset(root_dir=config["data_config"]["datadir"],
                               origin_dir=config["data_config"]["origin_dir"],
                               start_level=config["data_config"]["start_level"],
                               end_level=config["data_config"]["end_level"],
                               split='valid',
                               fold_id=fold_id,
                               transform=transf_train)
    valid_dataloader = GraphLevelDataLoader(valid_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            follow_batch=['b_pre', 'b_mv', 'target_2', 'add_stage_2'])
    # model
    Model = DGCFP(config=config)
    if config["model"] != '':
        Model.load_state_dict(torch.load(config["model"]))
    if bool(config["usegpu"]):
        Model.cuda()
    # wandb
    run.watch(Model)
    do_log = run is not None
    # Loss
    GLoss = GeometricLoss(top_k=config["arch"]["topk"], density_weight=config["arch"]["beta"])
    LeLoss = LELoss()
    LapLoss = LAPLoss()
    if bool(config["usegpu"]):
        GLoss.cuda()
        LeLoss.cuda()
        LapLoss.cuda()
    # optimizer
    optimizer = optim.Adam(Model.parameters(), lr=config["training_config"]["lr"], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scaler = GradScaler()
    # training
    num_batch = math.ceil(len(train_dataset) / config["training_config"]["batch_size"])
    all_val_result = []
    all_train_result = []
    cur_best = float('inf')
    cur_epoch = 0
    for epoch in range(config["training_config"]["epoch"]):
        train_loss = []
        batch_time = AverageMeter()
        end = time.time()
        Model = Model.train()
        for i, data in enumerate(train_dataloader, 0):
            if bool(config["usegpu"]):
                data = data.cuda()
            # input data
            f_pre = data.pos                 
            f_target_s1 = data.target_1      
            f_target_s2 = data.target_2     
            lm_pre = data.lm_pre             
            lm_post = data.lm_target
            batch_s1 = data.batch            
            batch_s2 = data.target_2_batch
            lap_stage_1 = data.lap_stage_1
            lap_stage_2 = data.lap_stage_2
            # clear gradient
            optimizer.zero_grad()
            # forward
            with autocast():
                f_xyz_post_s1, coords_s2, f_xyz_post_s2 = Model(data)
                pred_dis_s1 = f_xyz_post_s1 - f_pre
                pred_dis_s2 = f_xyz_post_s2 - coords_s2
                # loss
                geometric_loss_s1, shape_loss_s1, density_loss_s1 = GLoss(f_xyz_post_s1, f_target_s1, batch_s1)
                geometric_loss_s2, shape_loss_s2, density_loss_s2 = GLoss(f_xyz_post_s2, f_target_s2, batch_s2)
                lap_loss = LapLoss(f_pre, f_xyz_post_s1, coords_s2, f_xyz_post_s2, [lap_stage_1,lap_stage_2])
                le_loss_s1, pred_lm_s1 = LeLoss(pred_dis_s1, f_pre, lm_pre, lm_post, batch_s1)
                le_loss_s2, _ = LeLoss(pred_dis_s2, coords_s2, pred_lm_s1, lm_post, batch_s2)
                lm_r_lap_loss = lap_loss + 0.5*(le_loss_s1+le_loss_s2)
                total_loss = geometric_loss_s1 + geometric_loss_s2 +\
                             config["arch"]["lambda"]*lm_r_lap_loss
            # backward
            # total_loss.backward()
            # optimizer.step()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # batch log
            train_loss.append([total_loss.item(),
                            0.5*(geometric_loss_s1.item()+geometric_loss_s2.item()), geometric_loss_s1.item(), geometric_loss_s2.item(),
                            0.5*(shape_loss_s1.item()+shape_loss_s2.item()), shape_loss_s1.item(), shape_loss_s2.item(),
                            0.5*(density_loss_s1.item()+density_loss_s2.item()), density_loss_s1.item(), density_loss_s2.item(),
                            lm_r_lap_loss.item()])
            batch_time.update(time.time() - end)
            end = time.time()
            print('[TRAIN]\n\t[epoch %d: %d/%d][Time %f] mean_total_loss: %f\tmean_gloss: %f\tmean_lm_r_lap_loss: %f' 
                  % (epoch+1, i+1, num_batch, batch_time.avg, total_loss.item(), 0.5*(geometric_loss_s1.item()+geometric_loss_s2.item()), lm_r_lap_loss.item()))
        # epoch log
        train_loss = np.mean(np.array(train_loss), axis=0)
        all_train_result.append(train_loss.tolist())
        scheduler.step()
        # valid
        val_loss, mean_hd_error, mean_landmark_error = valid_model(Model, GLoss, LapLoss, LeLoss, valid_dataset, valid_dataloader, config, is_best=False)
        print('[VALID]\n\t[epoch %d/%d] mean_total_loss: %f\tmean_gloss: %f\tmean_laploss: %f' 
            % (epoch+1, config["training_config"]["epoch"], val_loss[0], val_loss[1], val_loss[-1]))
        # wandb record
        if do_log:
            run.log({"[train]total_loss": train_loss[0],
                    "[train]g_loss": train_loss[1],
                    "[train]lm_r_lap_loss": train_loss[-1],

                    "[valid]total_loss": val_loss[0],
                    "[valid]g_loss": val_loss[1],
                    "[valid]lm_r_lap_loss": val_loss[-1],
                    "[valid]hd_error": mean_hd_error,
                    "[valid]le_error": mean_landmark_error,
                    })
        # save best result
        all_val_result.append(val_loss.tolist()+[mean_hd_error, mean_landmark_error])
        item_num = len(val_loss.tolist())+2
        if val_loss[0] < cur_best:
            cur_best = val_loss[0]
            cur_epoch = epoch
            save_log(all_val_result, 
                     os.path.join(outf, 'val_result-fold%d.csv'%(fold_id)), 
                     item_num, cur_epoch=cur_epoch)
            save_log(all_train_result, 
                     os.path.join(outf, 'train_result-fold%d.csv'%(fold_id)), 
                     0, cur_epoch=-1)
            print("[Finding Current Best & Saving ...]")
            torch.save(Model.state_dict(), '%s/cur_best_model-fold%d.pth' % (outf, fold_id))
        # save checkpoint
        if epoch % config["training_config"]["save_interval"] ==0:
            save_log(all_val_result, 
                     os.path.join(outf, 'val_result-fold%d.csv'%(fold_id)), 
                     item_num, cur_epoch=cur_epoch)
            save_log(all_train_result, 
                     os.path.join(outf, 'train_result-fold%d.csv'%(fold_id)), 
                     0, cur_epoch=-1)
            print("[Saving Check Point ...]")
            torch.save(Model.state_dict(), '%s/history/model-fold%d_%d.pth' % (outf, fold_id, epoch))
    # save result
    save_log(all_val_result, 
             os.path.join(outf, 'val_result-fold%d.csv'%(fold_id)), 
             item_num, cur_epoch=cur_epoch)
    save_log(all_train_result, 
             os.path.join(outf, 'train_result-fold%d.csv'%(fold_id)), 
             0, cur_epoch=-1)
    

def valid_model(Model, GLoss, LapLoss, LeLoss, valid_dataset, valid_dataloader, config, is_best=False):
    target_dir = config["inference"]["predict_dir"]
    dataset_dir = config["data_config"]["datadir"]
    origin_dir = config["data_config"]["origin_dir"]
    test_samples = config["data_config"]["test_samples"]
    model = config["model_name"] if is_best else config["model_name"]+'_t'
    val_loss = []
    val_lm = []
    patients = set()
    # valid
    Model = Model.eval()
    with torch.no_grad():
        for j, data in enumerate(valid_dataloader, 0):
            # patient id
            patients.add(valid_dataset.datapath[j].split('/')[-1].split('_')[0])
            if bool(config["usegpu"]):
                data = data.cuda()
            f_pre = data.pos                 
            f_target_s1 = data.target_1      
            f_target_s2 = data.target_2      
            lm_pre = data.lm_pre             
            lm_post = data.lm_target
            batch_s1 = data.batch
            batch_s2 = data.target_2_batch
            lap_stage_1 = data.lap_stage_1
            lap_stage_2 = data.lap_stage_2
            # forward
            f_xyz_post_s1, coords_s2, f_xyz_post_s2 = Model(data)
            pred_dis_s1 = f_xyz_post_s1 - f_pre
            pred_dis_s2 = f_xyz_post_s2 - coords_s2
            # loss
            geometric_loss_s1, shape_loss_s1, density_loss_s1 = GLoss(f_xyz_post_s1, f_target_s1, batch_s1)
            geometric_loss_s2, shape_loss_s2, density_loss_s2 = GLoss(f_xyz_post_s2, f_target_s2, batch_s2)
            le_loss_s1, pred_lm_s1 = LeLoss(pred_dis_s1, f_pre, lm_pre, lm_post, batch_s1)
            le_loss_s2, _ = LeLoss(pred_dis_s2, coords_s2, pred_lm_s1, lm_post, batch_s2)
            lap_loss = LapLoss(f_pre, f_xyz_post_s1, coords_s2, f_xyz_post_s2, [lap_stage_1,lap_stage_2])
            lm_r_lap_loss = lap_loss + 0.5*(le_loss_s1+le_loss_s2)
            total_loss = geometric_loss_s1 + geometric_loss_s2 +\
                         config["arch"]["lambda"]*lm_r_lap_loss
            # batch log
            val_loss.append([total_loss.item(),
                            0.5*(geometric_loss_s1.item()+geometric_loss_s2.item()), geometric_loss_s1.item(), geometric_loss_s2.item(),
                            0.5*(shape_loss_s1.item()+shape_loss_s2.item()), shape_loss_s1.item(), shape_loss_s2.item(),
                            0.5*(density_loss_s1.item()+density_loss_s2.item()), density_loss_s1.item(), density_loss_s2.item(),
                            lm_r_lap_loss.item()
                            ])
            # save prediction
            if bool(config["usegpu"]):
                pred_dis_s1 = pred_dis_s1.cpu().detach().numpy()
                pred_dis_s2 = pred_dis_s2.cpu().detach().numpy()
                coords_s2 = coords_s2.cpu().detach().numpy()
            else:
                pred_dis_s1 = pred_dis_s1.detach().numpy()
                pred_dis_s2 = pred_dis_s2.detach().numpy()
                coords_s2 = coords_s2.detach().numpy()
            displacement = [pred_dis_s1, pred_dis_s2, coords_s2]
            pat_dir = os.path.join(valid_dataset.datapath[j], target_dir)
            if not os.path.exists(pat_dir):
                os.makedirs(pat_dir)
            dis_fn = os.path.join(pat_dir, 'dis_%s.pkl'%(model))
            with open(dis_fn, 'wb') as f:
                pickle.dump(displacement, f)
    # mean
    val_loss = np.mean(np.array(val_loss), axis=0)
    # recover full
    mp_reconstruct_crop_rigion(model=model,
                               dataset_dir=dataset_dir,
                               origin_dir=origin_dir,
                               target_dir=target_dir,
                               test_samples=test_samples,
                               in_patients=list(patients))
    # landmark error
    for pat in list(patients):
        predict_dirs = []
        for idx in range(test_samples):
            predict_dirs.append(os.path.join(dataset_dir, pat+'_{}/{}'.format(str(idx).zfill(2), target_dir)))
        landmark_fn = os.path.join(origin_dir, pat+'/post_face_landmarks.csv')
        landmark_df = pd.read_csv(landmark_fn, header=0, usecols=[1,2,3], dtype=np.float64, names=['x','y','z'])
        landmark_xyz = landmark_df.values
        for p_dir in predict_dirs:
            pred_landmark_fn = os.path.join(p_dir, f'post_landmark_{model}.csv')
            pred_landmark_df = pd.read_csv(pred_landmark_fn)
            pred_landmark_xyz = pred_landmark_df.values
            all_landmark_xyz = landmark_xyz
            all_pred_landmark_xyz = pred_landmark_xyz
            per_error = np.mean(np.sqrt(np.sum((all_landmark_xyz-all_pred_landmark_xyz)**2, -1)))
            val_lm.append(per_error)
    mean_landmark_error = np.mean(np.array(val_lm))
    # multi process hd
    hd_result = mp_get_hausdorff_error(model=model, 
                                       dataset_dir=dataset_dir,
                                       origin_dir=origin_dir,
                                       target_dir=target_dir,
                                       test_samples=test_samples,
                                       in_patients=list(patients))
    mean_hd_error = hd_result['mean']
    return val_loss, mean_hd_error, mean_landmark_error


def test_model(config, fold_id, check=False, target_dir='predict'):
    log_dir = config["data_config"]["outf"]
    seed_torch(seed=config["training_config"]["manualSeed"])
    # transfer
    transf_list_test = []
    for transform_config in config['test_transform']:
        if transform_config["type"]=="RadiusNeighbors":
            transf_list_test.append(RadiusNeighbors(**transform_config['args']))
        if transform_config["type"]=="EdgeSampling":
            transf_list_test.append(EdgeSampling(**transform_config['args']))
    transf_test = transforms.Compose(transf_list_test)
    # dataset
    test_dataset = CMFDataset(root_dir=config["data_config"]["datadir"],
                              origin_dir=config["data_config"]["origin_dir"],
                              start_level=config["data_config"]["start_level"],
                              end_level=config["data_config"]["end_level"],
                              split='test',
                              fold_id=fold_id,
                              transform=transf_test)
    test_dataloader = GraphLevelDataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           follow_batch=['b_pre', 'b_mv', 'target_2', 'add_stage_2'])
    # load model
    model_fn = os.path.join(log_dir, 'cur_best_model-fold%d.pth'%(fold_id))
    Model = DGCFP(config=config)
    Model.load_state_dict(torch.load(model_fn))
    if bool(config["usegpu"]):
        Model.cuda()
    # Loss
    GLoss = GeometricLoss(top_k=config["arch"]["topk"], density_weight=config["arch"]["beta"])
    LeLoss = LELoss()
    LapLoss = LAPLoss()
    if bool(config["usegpu"]):
        GLoss.cuda()
        LeLoss.cuda()
        LapLoss.cuda()
    # check loss
    if check:
        val_loss = []
        Model = Model.eval()
        for j, data in enumerate(test_dataloader, 0):
            if bool(config["usegpu"]):
                data = data.cuda()
            f_pre = data.pos                 
            f_target_s1 = data.target_1      
            f_target_s2 = data.target_2      
            lm_pre = data.lm_pre             
            lm_post = data.lm_target
            batch_s1 = data.batch            
            batch_s2 = data.target_2_batch
            lap_stage_1 = data.lap_stage_1
            lap_stage_2 = data.lap_stage_2
            # forward
            f_xyz_post_s1, coords_s2, f_xyz_post_s2 = Model(data)
            pred_dis_s1 = f_xyz_post_s1 - f_pre
            pred_dis_s2 = f_xyz_post_s2 - coords_s2
            # loss
            geometric_loss_s1, shape_loss_s1, density_loss_s1 = GLoss(f_xyz_post_s1, f_target_s1, batch_s1)         
            geometric_loss_s2, shape_loss_s2, density_loss_s2 = GLoss(f_xyz_post_s2, f_target_s2, batch_s2)
            lap_loss = LapLoss(f_pre, f_xyz_post_s1, coords_s2, f_xyz_post_s2, [lap_stage_1,lap_stage_2])
            le_loss_s1, pred_lm_s1 = LeLoss(pred_dis_s1, f_pre, lm_pre, lm_post, batch_s1)
            le_loss_s2, _ = LeLoss(pred_dis_s2, coords_s2, pred_lm_s1, lm_post, batch_s2)
            lm_r_lap_loss = lap_loss + 0.5*(le_loss_s1+le_loss_s2)
            total_loss = geometric_loss_s1 + geometric_loss_s2 +\
                         config["arch"]["lambda"]*lm_r_lap_loss
            val_loss.append([total_loss.item(),
                             0.5*(geometric_loss_s1.item()+geometric_loss_s2.item()), geometric_loss_s1.item(), geometric_loss_s2.item(),
                             0.5*(shape_loss_s1.item()+shape_loss_s2.item()), shape_loss_s1.item(), shape_loss_s2.item(),
                             0.5*(density_loss_s1.item()+density_loss_s2.item()), density_loss_s1.item(), density_loss_s2.item(),
                             lm_r_lap_loss.item()
                             ])
        val_loss = np.mean(np.array(val_loss), axis=0)
        print('[Checking Loaded Model]\n\tmean_total_loss: %f\tmean_gloss: %f\tmean_lm_r_lap_loss: %f' 
              % (val_loss[0], val_loss[1], val_loss[-1]))
    # save predicted displacements
    Model = Model.eval()
    patients = set()
    for j, data in enumerate(test_dataloader):
        if bool(config["usegpu"]):
                data = data.cuda()
        patients.add(test_dataset.datapath[j].split('/')[-1].split('_')[0])
        f_pre = data.pos                 
        f_target_s1 = data.target_1      
        f_target_s2 = data.target_2      
        lm_pre = data.lm_pre         
        lm_post = data.lm_target
        batch_s1 = data.batch  
        batch_s2 = data.target_2_batch
        # forward
        f_xyz_post_s1, coords_s2, f_xyz_post_s2 = Model(data)
        pred_dis_s1 = f_xyz_post_s1 - f_pre
        pred_dis_s2 = f_xyz_post_s2 - coords_s2
        if bool(config["usegpu"]):
            pred_dis_s1 = pred_dis_s1.cpu().detach().numpy()
            pred_dis_s2 = pred_dis_s2.cpu().detach().numpy()
        else:
            pred_dis_s1 = pred_dis_s1.detach().numpy()
            pred_dis_s2 = pred_dis_s2.detach().numpy()
        displacement = [pred_dis_s1, pred_dis_s2, coords_s2]
        # save
        pat_dir = os.path.join(test_dataset.datapath[j], target_dir)
        if not os.path.exists(pat_dir):
            os.makedirs(pat_dir)
        dis_fn = os.path.join(pat_dir, 'dis_%s.pkl'%(config["model_name"]))
        with open(dis_fn, 'wb') as f:
            pickle.dump(displacement, f)
    # recover
    dataset_dir = config["data_config"]["datadir"]
    origin_dir = config["data_config"]["origin_dir"]
    test_samples = config["data_config"]["test_samples"]
    mp_reconstruct_crop_rigion(model=config["model_name"],
                               dataset_dir=dataset_dir,
                               origin_dir=origin_dir,
                               target_dir=target_dir,
                               test_samples=test_samples,
                               in_patients=list(patients))


def print_args(config):
    data_config = config["data_config"]
    arch = config["arch"]
    t_config = config["training_config"]
    print('[Dataset Parameters]')
    print('\t[dataset dir]',data_config["datadir"])
    print('\t[origin data dir]',data_config["origin_dir"])
    print('\t[log dir]',data_config["outf"])
    print('\t[K folds cross-validation]',data_config["k_fold"])
    print('[Model Structure Parameters]')
    print('[Model Structure Parameters - MDFE]')
    print('\t[layer number of MDFE] from',data_config["start_level"],'to',data_config["end_level"])
    print('\t[num_propagation_steps]',arch["num_propagation_steps"])
    print('\t[geo_filter]',arch["geo_filter"])
    print('\t[euc_filter]',arch["euc_filter"])
    print('\t[pooling_method]',arch["pooling_method"])
    print('\t[aggr]',arch["aggr"])
    print('[Model Structure Parameters - PBE]')
    print('\t[npoint]',arch["npoint"])
    print('\t[nsample]',arch["nsample"])
    print('\t[be_radius]',arch["be_radius"])
    print('\t[be_mlp]', arch["be_mlp"])
    print('\t[bd_mlp]', arch["bd_mlp"])
    print('[Model Structure Parameters - DMT]')
    print('\t[DMT bone movement dimension]', arch["bv_dim"])
    print('\t[self attention dimension]', arch["selfatt_dim"])
    print('\t[use bias]', bool(arch["is_bias"]))
    print('[Model Structure Parameters - Deformer]')
    print('\t[s2_out_dim]', arch["s2_out_dim"])
    print('\t[out_mlp_s1]', arch["out_mlp_s1"])
    print('\t[out_mlp_s2]', arch["out_mlp_s2"])
    print('[Model Structure Parameters - Other]')
    print('\t[range_max]',arch["range_max"])
    print('\t[use_dropout]',bool(arch["is_dropout"]))
    if bool(arch["is_dropout"]):
        print('\t[dropout_rate]',arch["dp_rate"])
    print('\t[use_bn]',bool(arch["is_bn"]))
    print('[Loss Parameters]')
    print('\t[k in density loss]', arch["topk"])
    print('\t[beta]', arch["beta"])
    print('\t[lambda]', arch["lambda"])
    print('\t[use_normal]',bool(arch["use_normal"]))
    print('[Training Parameters]')
    print('\t[batch_size]', t_config["batch_size"])
    print('\t[epoch]', t_config["epoch"])
    print('\t[lr]', t_config["lr"])


def setup_run(config, fold_id):
    wandb_dir = config["data_config"]["outf"]
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    run = wandb.init(
                        project="Postoperative_Facial_Prediction",
                        dir=wandb_dir,
                        config={
                        "dataset": "",
                        "model_name": config["model_name"],
                        "fold_id": fold_id,
                        "outf": config["data_config"]["outf"]
                        }
                    )
    return run


if __name__ == '__main__':
    args = parse_args()
    if args.is_train:
        print('[Training Phase ..]')
    else:
        print('[Inference Phase ..]')
    print(f"[Loading config {args.config}]")
    config = json.load(open(args.config))
    print_args(config)
    if args.is_train:
        for fold_id in range(config["data_config"]["k_fold_start"],config["data_config"]["k_fold"]):
            print('[Processing Fold:',fold_id,'..]')
            run = setup_run(config, fold_id)
            train_model(config, fold_id, run)
            if run is not None:
                run.finish()
    if not args.is_train:
        for fold_id in range(0,1):
            print('[Processing Fold:',fold_id,'..]')
            test_model(config, fold_id, target_dir=config["inference"]["predict_dir"])