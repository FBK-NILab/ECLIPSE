import os

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from src.utils.utils import load_streamlines


def edge_indexes(n_points):
    """
    Fast edge_index generator using torch (no python loop).
    Returns shape (2, 2*(n_points-1)) long tensor.
    """
    if n_points <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    idx = torch.arange(n_points - 1, dtype=torch.long)
    top = torch.cat([idx, idx + 1])
    bot = torch.cat([idx + 1, idx])

    return torch.stack([top, bot], dim=0).contiguous()


class BilGinData(Dataset):

    def __init__(self, base_path, mode='train', streamline_number=1000, padding=True, max_streamline_length=0, **kwargs):

        self.base_path = base_path
        self.subs = sorted(os.listdir(self.base_path))

        self.mode = mode
        self.streamline_number = streamline_number

        if self.mode == 'test':
            self.streamline_number = None

        self.padding = padding
        self.max_streamline_length = max_streamline_length
        self.plausible_only = kwargs.get('plausible_only', True)
        self.non_plausible_only = kwargs.get('non_plausible_only', False)
        self.return_geometric = kwargs.get('return_geometric', True)
        self.rescale = kwargs.get('rescale', True)

        self.col_min = np.array([-72.31874084472656, -111.03961181640625, -72.49899291992188], dtype=np.float32)
        self.col_max = np.array([74.29684448242188, 74.60054016113281, 84.51304626464844], dtype=np.float32)

        self.scale = self.col_max - self.col_min


        if kwargs.get('use_sublist', True) and kwargs.get('sub_list', None) is not None:
            self.subs = [item for item in self.subs if item in kwargs['sub_list']]

        self._edge_index_cache = {}


    def __len__(self):
        return len(self.subs)


    def set_streamline_number(self, val: int):
        self.streamline_number = val


    def set_plausible_only(self, val: bool):
        self.plausible_only = val


    def set_unplausible_only(self, val: bool):
        self.non_plausible_only = val


    def get_sub_idx(self, sub_name):
        geom = ''
        finta = ''

        if self.return_geometric:
            geom = '_geometric'
        else:
            finta = '_finta'
            if self.padding:
                finta += '_padding'
            else:
                finta += '_resampled'

        try:
            return f'{self.subs.index(sub_name)}{geom}{finta}'
        except:
            raise ValueError(f"Subject {sub_name} not found. Please check if is correct")


    def get_sub_name_header(self, sub_name):

        try:
            idx = self.subs.index(sub_name)
        except:
            raise ValueError(f"Subject {sub_name} not found. Please check if the code is correct")

        sub_path = os.path.join(self.base_path, self.subs[idx])
        trk_file = [item for item in os.listdir(sub_path) if 'trk' in item][0]
        trk_file = os.path.join(sub_path, trk_file)

        return nib.streamlines.load(trk_file, lazy_load=True).header


    def get_edge_index(self, n_points: int):
        """
        Return cached edge_index for given number of points (n_points).
        Cache stores CPU tensors to avoid repeated allocations.
        """

        if n_points in self._edge_index_cache:
            return self._edge_index_cache[n_points]
        e = edge_indexes(n_points)
        self._edge_index_cache[n_points] = e

        return e


    def __getitem__(self, item):

        assert item < len(self.subs), 'Index out of range'

        sub_name = self.subs[item]
        sub_trk_path = os.path.join(self.base_path, self.subs[item])
        trk_file = [item for item in os.listdir(sub_trk_path) if 'trk' in item][0]
        trk_file = os.path.join(sub_trk_path, trk_file)

        np_filename = [item for item in os.listdir(sub_trk_path) if 'npy' in item][0]
        np_filename = os.path.join(sub_trk_path, np_filename)

        lines, lens = load_streamlines(trk_file, idxs=self.streamline_number,
                                       apply_affine=True, return_len=True, container='list',
                                       plausible_only=self.plausible_only, non_plausible_only=self.non_plausible_only,
                                       labels=np_filename, mode=self.mode)

        num_lines = len(lines)
        if not self.return_geometric:
            if self.padding:
                new_lines = list()
                masks = list()
                for idx, line in enumerate(lines):
                    last_point = line[-1]
                    new_line = np.zeros((self.max_streamline_length, 3))
                    mask = np.zeros((self.max_streamline_length, 3))
                    mask[:len(line)] = 1
                    new_line[:len(line)] = line
                    new_line[len(line):] = last_point
                    mask[len(line):] = 0
                    new_lines.append(new_line)
                    masks.append(mask)

                new_lines = np.asarray(new_lines)
                masks = np.asarray(masks)
                if self.rescale:
                    new_lines = (new_lines - self.col_min) / self.scale
                    new_lines = new_lines * 2 - 1

                return {"data": torch.from_numpy(new_lines).to(torch.float32),
                        "mask": torch.from_numpy(masks).to(torch.float32),
                        "sub_name": sub_name,
                        "lengths": lens}

            lines = np.array(lines)
            masks = np.ones((len(lines), self.max_streamline_length, 3))
            if self.rescale:
                lines = (lines - self.col_min) / self.scale
                lines = lines * 2 - 1
            return {"data": torch.from_numpy(lines).to(torch.float32),
                    "mask" : torch.from_numpy(masks).to(torch.float32),
                    "sub_name": sub_name,
                    "lengths": [256] * num_lines}

        else:
            return {"data": self.build_subject_graph_data(lines), "mask": None, "sub_name": sub_name, "lengths": lens}

    def build_subject_graph_data(self, lines):

        if self.rescale:
            lines = [(line - self.col_min) / self.scale for line in lines]
            lines = [line * 2 - 1 for line in lines]

        lines = [torch.as_tensor(line, dtype=torch.float32) for line in lines]
        data_list = []
        # minimize overhead: avoid passing into Data(...) constructor with copies
        for line in lines:
            n_pts = line.shape[0]
            eidx = self.get_edge_index(n_pts)

            d = Data()
            d.x = line
            d.pos = line
            d.edge_index = eidx
            data_list.append(d)

        if self.mode == "test":
            return data_list
        else:
            return Batch.from_data_list(data_list)
