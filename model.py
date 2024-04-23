from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import rearrange
import os
from utils import bilinear_interpolation, trilinear_interpolation

class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))
class DenseGrid(nn.Module):
    def __init__(self, base_lod=4, num_lod=5, feature_size=8, interpolation_type="trilinear"):
        super().__init__()
        self.feat_dim = feature_size  # feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # trilinear

        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]
        print("LODS:", self.LODS)
        self.init_feature_structure()

    def init_feature_structure(self):
        for LOD in self.LODS:
            fts = nn.Parameter(torch.zeros(LOD**3, self.feat_dim), requires_grad=True)
            fts = torch.nn.init.normal_(fts, std=0.01)
            self.codebook.append(fts)

    def forward(self, pts):
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            if self.interpolation_type == "closest":
                x = torch.floor(pts[:, 0] * (res-1)) 
                y = torch.floor(pts[:, 1] * (res-1)) 
                z = torch.floor(pts[:, 2] * (res-1)) 
                features = self.codebook[i][(x + (y * res) + (z * (res**2))).long()]
            elif self.interpolation_type == "trilinear":
                features = trilinear_interpolation(res, self.codebook[i], pts, "NGLOD")
            else:
                raise NotImplementedError
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)

    
class HashGrid(nn.Module):
    def _init_(self, min_grid_res=6, max_grid_res=64, num_LOD=10, band_width=10):
        super()._init_()
        self.feat_dim = 3  # feature dim size
        self.codebook = nn.ParameterList([])
        self.codebook_size = 2**band_width

        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_LOD - 1))
        self.LODS = [int(1 + np.floor(min_grid_res * (b**l))) for l in range(num_LOD)]
        print("LODS:", self.LODS)
        self.init_hash_structure()

    def init_hash_structure(self):
        for LOD in self.LODS:
            fts = nn.Parameter(torch.zeros(min(LOD**2, self.codebook_size), self.feat_dim), requires_grad=True)
            fts = torch.nn.init.normal_(fts, std=0.01)
            self.codebook.append(fts)

    def forward(self, pts):
        _, feat_dim = self.codebook[0].shape
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            features = trilinear_interpolation(res, self.codebook[i], pts, "HASH")
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)
class SingleLODmodel(nn.Module):
    def __init__(self, res=128, feature_size=8, interpolation_type="trilinear"):
        super().__init__()
        self.res = res
        self.interpolation_type = interpolation_type
        self.feature_size = feature_size
        self.init_grid()

    def init_grid(self):
        self.features = nn.Parameter(torch.zeros(self.res**3, self.feature_size), requires_grad=True)
        self.features = torch.nn.init.normal_(self.features, std=0.01)
    
    def forward(self, pts):
        normalized_pts = (pts + torch.ones_like(pts)) / 2  # From -1 - 1 to 0 - 1 range
        features = trilinear_interpolation(self.res, self.features, normalized_pts, "NGLOD")
        return features


class MLP(nn.Module):
    def _init_(self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super()._init_()

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU()))
        for _ in range(num_hidden_layers):
            self.module_list.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()))
        self.module_list.append(nn.Linear(hidden_dim, output_dim, bias=True))

        self.model = torch.nn.Sequential(*self.module_list)
        self.tanh = nn.Tanh()
        self.grid_structure = grid_structure

    def forward(self, coords):
        B, C = coords.shape
        coords = coords.reshape(-1, C)
        eval_flag = 0
        if not torch.is_tensor(coords):
            eval_flag = 1
            coords = torch.from_numpy(coords)

        # 2. Pass the reshaped tensor to the grid structure to get points features
        feat = self.grid_structure(coords).to(torch.float)

        
        # Concat coords with features
        feat = torch.cat((coords.to(torch.float), feat), dim=1)

        # 3. Pass the features to the model to get the output (prediction of color values)
        out = self.model(feat)

        # 4. Reshape the output back to size [B, C]
        out = out.reshape(B, -1)
        out = self.tanh(out)
        if eval_flag:
            out = (out.detach().squeeze().numpy())

        return out
