'''
model architecture definition
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, functional as F, ReLU, LeakyReLU, BatchNorm1d
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn.conv.edge_conv as edge
from torch.autograd import Variable
from torchsummary import summary
import chamfer
from torch.autograd import Function
from sklearn.neighbors import BallTree
from pytorch3d.ops import knn_points

def check_nan(value):
    cpu_value = value.detach().cpu().numpy()
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    if np.isnan(np.max(cpu_value).item() - np.min(cpu_value).item()) or np.isnan(
                    np.max(cpu_value).item() - np.min(cpu_value).item()):
        color = bcolors.FAIL + '*'
    else:
        color = bcolors.OKGREEN + ' '
    print('%svalue: %.3e ~ %.3e' % (color, np.min(cpu_value).item(), np.max(cpu_value).item()))

def square_distance1(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.cdist(src, dst, p=2)
    dist = dist**2
    return dist

def index_points(points, idx):
    """
    index corresponding points from all points
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    FPS algorithm
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
    
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    find neighbors using ball query
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    sampling and grouping operation in pointnet2
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]             [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) 
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    all input points as a group
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def get_gcn_filter(input_size: int, output_size, activation: nn.Module,
                   inplace: bool = False, aggregation: str = 'mean',
                   module: MessagePassing = edge.EdgeConv):
    """
    returns graph conv module with specified arguments and type
    Input:
        input_size: input size (2 * current vertex feature size!)
        output_size: feature size of new vertex features
        activation: activation function for internal MLP
        inplace: (default: {False})
        aggregation: permutation-invariant feature aggregation of adjacent vertices (default: {'mean'})
        module: graph convolutional module (default: {edge.EdgeConv})
    """
    assert input_size >= 0
    assert output_size >= 0
    inner_module = Seq(
        Lin(input_size, 2 * output_size),
        BatchNorm1d(2 * output_size),
        activation(inplace=inplace),
        Lin(2 * output_size, output_size),
        BatchNorm1d(output_size))
    return module(inner_module, aggr=aggregation)

def pairwise_l2_norm2_batch(x, y):
    '''
    Input:
        x: a point set, matrix (Nx,3)
        y: a point set, matrix (Ny,3)
    Output:
        square_dist: (Nx, Ny)
    '''
    Nx,_ = x.shape
    Ny,_ = y.shape
    xx = torch.unsqueeze(x, dim=2).repeat([1,1,Ny])
    yy = torch.unsqueeze(y, dim=2).repeat([1,1,Nx])
    yy = yy.permute(2,1,0)
    diff = xx - yy
    square_diff = torch.square(diff)
    square_dist = torch.sum(square_diff, dim=1)
    return square_dist


class FeatureEncodingModule(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, is_bn=True):
        super(FeatureEncodingModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.is_bn = is_bn
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        if self.is_bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if self.is_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)
            if self.is_bn:
                bn = self.mlp_bns[i]
                new_points = bn(new_points)
            new_points =  F.relu(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class FeatureDecodingModule(nn.Module):
    def __init__(self, in_channel, mlp, is_bn=True):
        super(FeatureDecodingModule, self).__init__()
        self.is_bn = is_bn
        self.mlp_convs = nn.ModuleList()
        if self.is_bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if self.is_bn:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)
            if self.is_bn:
                bn = self.mlp_bns[i]
                new_points = bn(new_points)
            new_points = F.relu(new_points)
        return new_points

class EdgeConvTransInv(edge.EdgeConv):
    def __init__(self, nn, aggr):
        super(EdgeConvTransInv, self).__init__(nn, aggr)
        self._aggr = aggr

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_j - x_i], dim=1))

    def __repr__(self):
        return '{}(nn={}, aggr={})'.format(self.__class__.__name__, self.nn, self._aggr)

class MDFE(nn.Module):
    def __init__(self, feature_number, num_propagation_steps, geo_filter, euc_filter,
                 activation='ReLU', use_normal=True, pooling_method='mean', 
                 aggr='mean', one_neighborhood=False):
        super(MDFE, self).__init__()
        assert len(geo_filter) == len(euc_filter)
        curr_size = feature_number
        self.use_normal = use_normal
        self._pooling_method = pooling_method
        self._one_neighborhood = one_neighborhood
        if activation == 'ReLU':
            self._activation = ReLU
            self._act = F.relu
        elif activation == 'LeakyReLU':
            self._activation = LeakyReLU
            self._act = F.leaky_relu
        else:
            raise NotImplementedError(f"{activation} is not implemented")
        self.left_geo_cnns = []
        self.left_euc_cnns = []
        self.right_geo_cnns = []
        self.right_euc_cnns = []
        self._graph_levels = len(geo_filter)
        for level in range(len(geo_filter)):
            if level < len(geo_filter) - 1:
                if level == 0:
                    left_geo = [get_gcn_filter(
                        curr_size, geo_filter[level], self._activation, aggregation=aggr, module=EdgeConvTransInv)]
                    left_euc = [get_gcn_filter(
                        curr_size, euc_filter[level], self._activation, aggregation=aggr, module=EdgeConvTransInv)]
                else:
                    left_geo = [get_gcn_filter(
                        2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                    left_euc = [get_gcn_filter(
                        2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]
                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                   self._activation, aggregation=aggr))
                    left_euc.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                   self._activation, aggregation=aggr))
                curr_size = geo_filter[level] + euc_filter[level] + \
                    geo_filter[level+1] + euc_filter[level+1]
                right_geo = [get_gcn_filter(
                    2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                right_euc = [get_gcn_filter(
                    2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]
                for _ in range(num_propagation_steps - 1):
                    right_geo.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                    self._activation, aggregation=aggr))
                    right_euc.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                    self._activation, aggregation=aggr))
                self.right_geo_cnns.append(torch.nn.ModuleList(right_geo))
                self.right_euc_cnns.append(torch.nn.ModuleList(right_euc))
                curr_size = geo_filter[level] + euc_filter[level]
            else:
                left_geo = [get_gcn_filter(
                    2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                left_euc = [get_gcn_filter(
                    2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]
                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(2 * (geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                   self._activation, aggregation=aggr))
                    left_euc.append(get_gcn_filter(2 * (geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                   self._activation, aggregation=aggr))
            self.left_geo_cnns.append(torch.nn.ModuleList(left_geo))
            self.left_euc_cnns.append(torch.nn.ModuleList(left_euc))
        self.left_geo_cnns = torch.nn.ModuleList(self.left_geo_cnns)
        self.left_euc_cnns = torch.nn.ModuleList(self.left_euc_cnns)
        self.right_geo_cnns = torch.nn.ModuleList(self.right_geo_cnns)
        self.right_euc_cnns = torch.nn.ModuleList(self.right_euc_cnns)

    def _residual_steps(self, vertex_features, geo_filters, eucl_filters, geo_edges, eucl_edges, inplace=False, residual_last=True):
        residual_geo = geo_filters[0](vertex_features, geo_edges)
        residual_eucl = eucl_filters[0](vertex_features, eucl_edges)
        vertex_features = self._act(
            torch.cat((residual_geo, residual_eucl), dim=-1), inplace=inplace)
        for step in range(1, len(geo_filters)):
            residual_geo = geo_filters[step](vertex_features, geo_edges)
            residual_eucl = eucl_filters[step](vertex_features, eucl_edges)
            residual_cat = torch.cat((residual_geo, residual_eucl), dim=-1)
            if step < (len(geo_filters)-1) or residual_last:
                vertex_features = vertex_features + residual_cat
            else:
                vertex_features = residual_cat
            vertex_features = self._act(vertex_features, inplace=inplace)
        return vertex_features

    def _simple_residual_steps(self, vertex_features, filters, edges, inplace=False):
        residual = filters[0](vertex_features, edges)
        vertex_features = self._act(residual, inplace=inplace)
        for step in range(1, len(filters)):
            residual = filters[step](vertex_features, edges)
            vertex_features = vertex_features + residual
            vertex_features = self._act(vertex_features, inplace=inplace)
        return vertex_features

    def _pooling(self, vertex_features, edges):
        if self._pooling_method == 'mean':
            return scatter_mean(vertex_features, edges, dim=0)
        if self._pooling_method == 'max':
            return scatter_max(vertex_features, edges, dim=0)[0]
        raise ValueError(f"Unkown pooling type {self._pooling_method}")

    def forward(self, sample):
        if self._one_neighborhood:
            sample[f"euclidean_edge_index"] = sample.edge_index
            for level in range(1, self._graph_levels):
                sample[f"hierarchy_euclidean_edge_index_{level}"] = sample[f"hierarchy_edge_index_{level}"]
        levels = []
        if self.use_normal:
            level1 = torch.cat((sample.pos, sample.x), dim=-1)
        else:
            level1 = sample.pos
        level1 = self._residual_steps(level1, self.left_geo_cnns[0], self.left_euc_cnns[0],
                                      sample.edge_index, sample.euclidean_edge_index)
        levels.append(level1)
        # ENCODER BRANCH
        for level in range(1, self._graph_levels):
            curr_level = self._pooling(levels[-1],
                                       sample[f"hierarchy_trace_index_{level}"])
            curr_level = self._residual_steps(curr_level, self.left_geo_cnns[level], self.left_euc_cnns[level],
                                              sample[f"hierarchy_edge_index_{level}"],
                                              sample[f"hierarchy_euclidean_edge_index_{level}"])
            levels.append(curr_level)
        current = levels[-1]
        # DECODER BRANCH
        for level in range(1, self._graph_levels):
            back = current[sample[f"hierarchy_trace_index_{self._graph_levels - level}"]]
            fused = torch.cat((levels[-(level+1)], back), -1)
            if level == self._graph_levels - 1:
                fused = self._residual_steps(fused, self.right_geo_cnns[-level], self.right_euc_cnns[-level],
                                             sample.edge_index, sample.euclidean_edge_index, residual_last=False)
            else:
                fused = self._residual_steps(fused, self.right_geo_cnns[-level], self.right_euc_cnns[-level],
                                             sample[f"hierarchy_edge_index_{self._graph_levels - level - 1}"],
                                             sample[f"hierarchy_euclidean_edge_index_{self._graph_levels - level - 1}"])
            current = fused
        result = current
        return result

class DMT(nn.Module):
    def __init__(self, f_dim=128, b_dim=128, bv_dim=6, hidden_size=64, is_bias=True):
        super(DMT, self).__init__()
        self.hidden_size = hidden_size
        self.f_dim = f_dim
        self.f_conv = nn.Conv1d(f_dim, hidden_size, 1, bias=is_bias)
        self.b_conv = nn.Conv1d(b_dim, hidden_size, 1, bias=is_bias)
        self.bv_conv = nn.Conv1d(bv_dim, hidden_size, 1, bias=is_bias)
        self.out_conv = nn.Conv1d(hidden_size*2, hidden_size, 1, bias=is_bias)
    
    def forward(self, f_pre_in, f_pre_batch, b_pre_in, bv_in):
        '''
        Inputs:
            f_pre_in: (num_nodes, D)
            f_pre_batch: (num_nodes)
            b_pre_in: (B,D,N)
            bv_in: (B,C,N)
        '''
        B,_,N = b_pre_in.shape
        f_pre_in_f = f_pre_in.transpose(0,1)
        f_pre_in_f = self.f_conv(f_pre_in_f)
        b_pre_feature = self.b_conv(b_pre_in)
        bv_feature = self.bv_conv(bv_in)
        f_pre_features_euc = []     
        f_pre_features_geo = []     
        for bid in range(B):
            f_pre_features_euc.append(f_pre_in_f[ :self.hidden_size//2,f_pre_batch==bid])
            f_pre_features_geo.append(f_pre_in_f[ self.hidden_size//2:,f_pre_batch==bid])
        weight_f_pre_list_euc = []
        weight_f_pre_list_geo = []
        for bid in range(B):
            cur_bv = bv_feature[bid,:,:]
            f_pre_euc = f_pre_features_euc[bid]
            cur_b_pre_euc = b_pre_feature[bid,:self.hidden_size//2,:]
            temp_W_euc = torch.div(torch.matmul(cur_b_pre_euc.transpose(0,1), f_pre_euc), torch.sqrt(torch.tensor(self.hidden_size)))
            W_euc = F.softmax(temp_W_euc, dim=0)
            weight_f_pre_list_euc.append(torch.matmul(cur_bv, W_euc).transpose(0,1).contiguous())

            f_pre_geo = f_pre_features_geo[bid]
            cur_b_pre_geo = b_pre_feature[bid,self.hidden_size//2:,:]
            temp_W_geo = torch.div(torch.matmul(cur_b_pre_geo.transpose(0,1), f_pre_geo), torch.sqrt(torch.tensor(self.hidden_size)))
            W_geo = F.softmax(temp_W_geo, dim=0)
            weight_f_pre_list_geo.append(torch.matmul(cur_bv, W_geo).transpose(0,1).contiguous())
        weight_f_pre_euc = torch.concat(weight_f_pre_list_euc, dim=0)
        weight_f_pre_geo = torch.concat(weight_f_pre_list_geo, dim=0)      
        weight_f_pre = torch.concat([weight_f_pre_euc,weight_f_pre_geo], dim=1).transpose(0,1) 
        weight_f_pre = self.out_conv(weight_f_pre).transpose(0,1)
        return weight_f_pre

class GUnpooling(nn.Module):
    def __init__(self):
        super(GUnpooling, self).__init__()

    def forward(self, coords, point_fe, point_batch, face_ds, face_batch):
        '''
        Inputs:
            coords: coordinates  (batch_size * point_num, 3)
            point_fe: pointwise features (batch_size * point_num, D)
            point_batch: sample id of points in a batch   (batch_size * point_num, )
            face_ds: faces used to add more vertices    (batch_size * face_nums, 3)
            face_batch: sample id of faces in a batch  (batch_size * face_nums, )
        '''
        device = coords.device
        split_coords = []
        split_point_fe = []
        split_face_ds = []
        B = torch.max(point_batch) + 1
        for bid in range(B):
            split_coords.append(coords[point_batch==bid,:])
            split_point_fe.append(point_fe[point_batch==bid,:])
            split_face_ds.append(face_ds[face_batch==bid,:])
        for bid in range(B):
            a_coords = split_coords[bid]
            a_point_fe = split_point_fe[bid]
            a_face_ds = split_face_ds[bid]
            new_coords = torch.mean(a_coords[a_face_ds,:], dim=1)
            new_point_fe = torch.mean(a_point_fe[a_face_ds,:], dim=1)
            split_coords[bid] = torch.cat([a_coords, new_coords])
            split_point_fe[bid] = torch.cat([a_point_fe, new_point_fe])
        out_coords = torch.cat(split_coords)
        out_point_fe = torch.cat(split_point_fe)
        return out_coords, out_point_fe

class DGCFP(nn.Module):
    def __init__(self, config):
        super(DGCFP, self).__init__()
        self.npoint = config["arch"]["npoint"]
        self.range_max = config["arch"]["range_max"]
        self.is_dropout = bool(config["arch"]["is_dropout"])
        self.dp_rate = config["arch"]["dp_rate"]
        self.is_bn = bool(config["arch"]["is_bn"])

        self.feature_number = config["arch"]["feature_number"]
        self.num_propagation_steps = config["arch"]["num_propagation_steps"]
        self.geo_filter = config["arch"]["geo_filter"]
        self.euc_filter = config["arch"]["euc_filter"]
        self.use_normal = bool(config["arch"]["use_normal"])
        self.pooling_method = config["arch"]["pooling_method"]
        self.aggr = config["arch"]["aggr"]

        self.nsample = config["arch"]["nsample"]
        self.be_radius = config["arch"]["be_radius"]
        self.be_mlp = config["arch"]["be_mlp"]
        self.bd_mlp = config["arch"]["bd_mlp"]
        self.bv_dim = config["arch"]["bv_dim"]
        self.selfatt_dim = config["arch"]["selfatt_dim"]
        self.is_bias = bool(config["arch"]["is_bias"])

        self.s2_out_dim = config["arch"]["s2_out_dim"]
        self.out_mlp_s1 = config["arch"]["out_mlp_s1"]
        self.out_mlp_s2 = config["arch"]["out_mlp_s2"]
        # MDFE
        self.mdfe_module = MDFE(self.feature_number, self.num_propagation_steps, self.geo_filter, 
                                self.euc_filter, activation='ReLU', use_normal=self.use_normal, 
                                pooling_method=self.pooling_method, aggr=self.aggr, one_neighborhood=False)
        # PBE
        self.be1 = FeatureEncodingModule(npoint=1024, radius=self.be_radius[0], nsample=self.nsample, 
                                         in_channel=6, mlp=self.be_mlp[0], group_all=False, is_bn=self.is_bn)
        self.be2 = FeatureEncodingModule(npoint=512, radius=self.be_radius[1], nsample=self.nsample, 
                                         in_channel=128+3, mlp=self.be_mlp[1], group_all=False, is_bn=self.is_bn)
        self.be3 = FeatureEncodingModule(npoint=256, radius=self.be_radius[2], nsample=self.nsample, 
                                         in_channel=256+3, mlp=self.be_mlp[2], group_all=False, is_bn=self.is_bn)
        self.be4 = FeatureEncodingModule(npoint=64, radius=self.be_radius[3], nsample=self.nsample, 
                                         in_channel=512+3, mlp=self.be_mlp[3], group_all=False, is_bn=self.is_bn)
        self.bd1 = FeatureDecodingModule(in_channel=1536, mlp=self.bd_mlp[0], is_bn=self.is_bn)
        self.bd2 = FeatureDecodingModule(in_channel=768, mlp=self.bd_mlp[1], is_bn=self.is_bn)
        self.bd3 = FeatureDecodingModule(in_channel=384, mlp=self.bd_mlp[2], is_bn=self.is_bn)
        self.bd4 = FeatureDecodingModule(in_channel=131, mlp=self.bd_mlp[3], is_bn=self.is_bn)
        # DMT module
        self.dmt = DMT(f_dim=self.geo_filter[0]+self.euc_filter[0], b_dim=self.bd_mlp[3][-1], 
                        bv_dim=self.bv_dim, hidden_size=self.selfatt_dim, is_bias=self.is_bias)
        # Unpool Block
        self.unpooling_1_2 = GUnpooling()
        # Stage 2 GCN
        self.s2_in_dim = self.selfatt_dim + 3
        self.s2_gcn = get_gcn_filter(2*self.s2_in_dim, self.s2_out_dim, nn.ReLU, aggregation='mean')
        # displacement prediction module
        self.outblock_s1_list = []
        last_channel = self.selfatt_dim
        for out_channel in self.out_mlp_s1:
            self.outblock_s1_list.append(nn.Conv1d(last_channel, out_channel, 1))
            if self.is_bn:
                self.outblock_s1_list.append(nn.BatchNorm1d(out_channel))
            self.outblock_s1_list.append(nn.ReLU())
            if self.is_dropout:
                self.outblock_s1_list.append(nn.Dropout(self.dp_rate))
            last_channel = out_channel
        self.outblock_s1_list.append(nn.Conv1d(64, 3, 1))
        self.outblock_s1 = nn.Sequential(*self.outblock_s1_list)
        self.outblock_s2_list = []
        last_channel = self.s2_out_dim
        for out_channel in self.out_mlp_s2:
            self.outblock_s2_list.append(nn.Conv1d(last_channel, out_channel, 1))
            if self.is_bn:
                self.outblock_s2_list.append(nn.BatchNorm1d(out_channel))
            self.outblock_s2_list.append(nn.ReLU())
            if self.is_dropout:
                self.outblock_s2_list.append(nn.Dropout(self.dp_rate))
            last_channel = out_channel
        self.outblock_s2_list.append(nn.Conv1d(64, 3, 1))
        self.outblock_s2 = nn.Sequential(*self.outblock_s2_list)

    def forward(self, sample):
        f_xyz_pre = sample.pos
        # get facial features
        facial_feature = self.mdfe_module(sample)
        # get bony features
        b_xyz_pre = sample.b_pre
        b_v = sample.b_mv
        B = torch.max(sample.b_pre_batch) + 1
        N = self.npoint
        C = b_xyz_pre.shape[1]
        b_xyz_pre = b_xyz_pre.reshape((B,N,C)).transpose(1,2)
        b_v = b_v.reshape((B,N,C)).transpose(1,2)
        b_l0_points = b_xyz_pre
        b_l0_xyz = b_xyz_pre
        b_l1_xyz, b_l1_points = self.be1(b_l0_xyz, b_l0_points)
        b_l2_xyz, b_l2_points = self.be2(b_l1_xyz, b_l1_points)
        b_l3_xyz, b_l3_points = self.be3(b_l2_xyz, b_l2_points)
        b_l4_xyz, b_l4_points = self.be4(b_l3_xyz, b_l3_points)
        b_l3_points = self.bd1(b_l3_xyz, b_l4_xyz, b_l3_points, b_l4_points)
        b_l2_points = self.bd2(b_l2_xyz, b_l3_xyz, b_l2_points, b_l3_points)
        b_l1_points = self.bd3(b_l1_xyz, b_l2_xyz, b_l1_points, b_l2_points)
        b_l0_points = self.bd4(b_l0_xyz, b_l1_xyz, b_l0_points, b_l1_points)
        # dmt
        bv_in = torch.cat((b_xyz_pre, b_v), dim=1)
        f_pre_batch = sample.batch
        atten_feature = self.dmt(facial_feature, f_pre_batch, b_l0_points, bv_in)
        total_feature = torch.unsqueeze(atten_feature.transpose(0,1), dim=0)
        # coarse prediction
        temp_displacement = torch.squeeze(
                            self.outblock_s1(total_feature), dim=0)
        displacement_s1 = torch.sigmoid(temp_displacement) * self.range_max * 2 - self.range_max
        displacement_s1 = displacement_s1.transpose(0,1).contiguous()
        f_xyz_post_s1 = f_xyz_pre + displacement_s1
        # Unpooling
        out_fe_s1 = torch.cat([displacement_s1, atten_feature], dim=1)
        point_batch = sample.batch
        face_ds_s2 = sample.add_stage_2
        face_batch_s2 = sample.add_stage_2_batch
        coords_s2, in_fe_s2 = self.unpooling_1_2(f_xyz_post_s1, out_fe_s1, point_batch, 
                                                 face_ds_s2, face_batch_s2)
        # fine GCN
        out_fe_s2 = self.s2_gcn(in_fe_s2, sample.edge_stage_2)
        out_fe_s2 = torch.unsqueeze(out_fe_s2.transpose(0,1), dim=0)
        # fine prediction
        temp_displacement_s2 = torch.squeeze(
                                self.outblock_s2(out_fe_s2), dim=0)
        displacement_s2 = torch.sigmoid(temp_displacement_s2) * self.range_max * 2 - self.range_max
        displacement_s2 = displacement_s2.transpose(0,1).contiguous()
        f_xyz_post_s2 = coords_s2 + displacement_s2
        return f_xyz_post_s1, coords_s2, f_xyz_post_s2


class ChamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        '''
        Inputs:
            xyz1: point coordinates (bs, point_num, 3)
        '''
        # device = xyz1.device

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, _idx1, _idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        
        return gradxyz1, gradxyz2

class ChamferDist(nn.Module):
    def __init__(self):
        super(ChamferDist, self).__init__()

    def forward(self, input1, input2):
        return ChamferFunction.apply(input1, input2)

class GeometricLoss(nn.Module):
    def __init__(self, top_k, density_weight):
        super(GeometricLoss, self).__init__()
        self.top_k = top_k
        self.density_weight = density_weight
        self.chamfer_dist = ChamferDist()

    def forward(self, pred, target, batch):
        '''
        Input:
            pred: (num_nodes, 3)
            target: (num_nodes, 3)
            batch: (num_nodes,)
        '''
        device = pred.device
        split_pre = []
        split_target = []
        B = torch.max(batch) + 1
        for bid in range(B):
            split_pre.append(pred[batch==bid,:])
            split_target.append(target[batch==bid,:])
        shape_loss = 0.0
        density_loss = 0.0
        for bid in range(B):
            cur_pre = torch.unsqueeze(split_pre[bid], dim=0)
            cur_target = torch.unsqueeze(split_target[bid], dim=0)
            dist1, dist2, idx1, idx2 = self.chamfer_dist(cur_pre, cur_target)
            shape_loss += (torch.mean(dist1) + torch.mean(dist2))
            dist_tar_pre = torch.squeeze(knn_points(cur_target,cur_pre,K=self.top_k)[0],0)
            dist_tar_tar = torch.squeeze(knn_points(cur_target,cur_target,K=self.top_k)[0],0)
            density_loss += torch.mean(torch.abs(dist_tar_pre - dist_tar_tar))
        shape_loss = shape_loss / B
        density_loss = density_loss / B
        geometric_loss = shape_loss + density_loss * self.density_weight
        return geometric_loss, shape_loss, density_loss

class LELoss(nn.Module):
    def __init__(self):
        super(LELoss, self).__init__()

    def forward(self, pred_dis, pre_xyz, pre_lm, target_lm, batch):
        '''
        Input:
            pred_dis: (num_nodes, 3)
            pre_xyz: (num_nodes, 3)
            pre_lm: (lm_num*B, 3)
            target_lm: (lm_num*B, 3)
            batch: (num_nodes,)
        '''
        device = pred_dis.device
        split_pred_dis = []
        split_pre_xyz = []     
        split_pre_lm = []  
        split_target_lm = []
        B = torch.max(batch) + 1
        num_lm = pre_lm.shape[0] // B
        for bid in range(B):
            split_pred_dis.append(pred_dis[batch==bid,:])
            split_pre_xyz.append(pre_xyz[batch==bid,:])
            split_pre_lm.append(pre_lm[bid*num_lm:(bid+1)*num_lm,:])
            split_target_lm.append(target_lm[bid*num_lm:(bid+1)*num_lm,:])
        le_loss = 0.0
        all_pred_lm = []
        for bid in range(B):
            cur_pred_dis = split_pred_dis[bid]         
            cur_pre_xyz = split_pre_xyz[bid]
            cur_pre_lm = split_pre_lm[bid]             
            cur_target_lm = split_target_lm[bid]
            dists = pairwise_l2_norm2_batch(cur_pre_lm, cur_pre_xyz)            
            r_tuple = torch.topk(dists, k=3, largest=False)    
            k_dists = 1.0 / (r_tuple[0]+1e-8) 
            k_idx = r_tuple[1]
            weight = k_dists / torch.sum(k_dists, dim=-1, keepdim=True)
            lm_dis = torch.sum(cur_pred_dis[k_idx, :] * torch.reshape(weight, (weight.shape[0],weight.shape[1],1)), dim=1)
            pred_lm = cur_pre_lm + lm_dis
            all_pred_lm.append(pred_lm)
            le_loss += torch.mean(torch.sum(torch.pow(cur_target_lm-pred_lm, 2), dim=-1))
        total_le_loss = le_loss / B
        pred_lm = torch.cat(all_pred_lm, dim=0)
        return total_le_loss, pred_lm

class LAPLoss(nn.Module):
    def __init__(self):
        super(LAPLoss, self).__init__()
    
    def laplace_coord(self, inputs, lap_idx):
        """
        Inputs:
        inputs: nodes Tensor
        lap_idx: laplace index matrix Tensor
        Returens:
        The laplacian coordinates of input with respect to edges as in lap_idx
        """
        device = inputs.device
        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0
        vertices = inputs[:, all_valid_indices]
        # vertices[:, invalid_mask] = 0
        mask = torch.ones(vertices.shape,dtype=torch.float).to(device)
        mask[:,invalid_mask] = 0.0
        vertices = vertices * mask
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :]   # (3,num_nodes)
        return laplace

    def laplace_regularization(self, input1, input2, laplace_idx):
        lap1 = self.laplace_coord(input1, laplace_idx).transpose(0,1)   # (num_nodes,3)
        lap2 = self.laplace_coord(input2, laplace_idx).transpose(0,1)
        laplace_loss = torch.mean(torch.sum(torch.pow(lap1-lap2, 2), dim=-1))
        return laplace_loss

    def forward(self, coarse_input, coarse_pred, fine_input, fine_pred, 
                laplace_idx_list):
        '''
        Input:
            coarse_pred: (num_nodes, 3)
            fine_pred: (num_nodes, 3)
        '''
        lap_loss = 0.
        lap_const = [0.5, 0.5]
        before_coord = [coarse_input.transpose(0,1), fine_input.transpose(0,1)] # (3,num_nodes)
        after_coord = [coarse_pred.transpose(0,1), fine_pred.transpose(0,1)]
        for i in range(2):
            lap = self.laplace_regularization(before_coord[i], after_coord[i], laplace_idx_list[i])
            lap_loss += lap_const[i] * lap
        return lap_loss
