import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import os
import numpy as np


class PCDataset(Dataset):
    def __init__(self, data_dir, device):
        # self.device = device
        # self.file_path = file_path
        # obj_dataitem = self.file_path
        # pc = trimesh.load(obj_dataitem)
        # self.coords = np.array(pc.vertices)
        # occupancy = np.array(pc.visual.vertex_colors)[:, 0]
        # self.occupancy = (occupancy == 0).astype("float32") * 2 - 1

        self.device = device
        self.file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.obj')]
        self.coords = []
        self.occupancy = []

        for file_path in self.file_paths:
            pc = trimesh.load(file_path)
            self.coords.extend(pc.vertices)
            occupancy = np.array(pc.visual.vertex_colors)[:, 0]
            self.occupancy.extend((occupancy == 0).astype("float32") * 2 - 1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return torch.tensor(self.coords[idx], device = self.device), torch.tensor(self.occupancy[idx], dtype=torch.float)

def collate(batch):
    coords, occupancy = zip(*batch)
    return torch.stack(coords), torch.stack(occupancy)

def get_loader(file_path, device, batch_size):
    # return DataLoader(
    #     PCDataset(file_path, device)
    #     generator = torch.Generator(device = self.device)

    #     data_loader = torch.utils.data.DataLoader(dataset, 
    #     batch_size=batch_size, shuffle=True)
    #     return data_loader
    #     )
    dataset = PCDataset(file_path, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    # Test the data loader
    data_loader = get_loader("./processed/bunny.obj")

