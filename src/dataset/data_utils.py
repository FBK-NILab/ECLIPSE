from typing import Optional

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .BGData import BilGinData
from .HCP import HCPData
from .TractoInferno import TractoInfernoData


class CollateWrapper:

    def __init__(self, mode, num_idxs = 12, **kwargs):
        super().__init__()
        self.mode = mode
        self.num_idxs = num_idxs

    def __call__(self, batch):
        if self.mode == 'test':
            return self.collate_wrapper_test(batch)
        else:
            return self.collate_wrapper(batch)

    @staticmethod
    def collate_wrapper(batch):
        batch = {k: [d.get(k) for d in batch] for k in sorted(set().union(*batch))}
        data = {}
        sub_name = batch.pop("sub_name")
        lengths = batch.pop("lengths")

        for k, v in batch.items():
            if isinstance(v[0], Data):
                data[k] = Batch.from_data_list(v)
            elif torch.is_tensor(v[0]):
                data[k] = torch.stack(v, dim=0)

        return data, sub_name, torch.tensor(lengths)


    def collate_wrapper_test(self, batch):
        batch = {k: [d.get(k) for d in batch] for k in sorted(set().union(*batch))}
        data = {}
        sub_name = batch.pop("sub_name")
        lengths = batch.pop("lengths")

        for k, v in batch.items():
            if isinstance(v[0], list):
                data[k] = list()
                for d in v:
                    nb_streamlines = len(d)
                    if nb_streamlines > self.num_idxs:
                        chunks = [Batch.from_data_list(d[i:i + self.num_idxs]) for i in
                              range(0, len(d), self.num_idxs)]
                    else:
                        chunks = [Batch.from_data_list(d)]
                    data[k].append(chunks)
            elif torch.is_tensor(v[0]):
                    data[k] = list()
                    for d in v:
                        num_streamlines = d.size(0)
                        if num_streamlines > self.num_idxs:
                            chunks = [d[start:start+self.num_idxs] for start in range(0, num_streamlines, self.num_idxs)]
                        else:
                            chunks = [d]
                        data[k].append(chunks)

        return data, sub_name, torch.tensor(lengths)


def get_dataset(cfg: dict):

    valid_names = ['BilGin', 'HCPData', 'TractoInfernoData']
    dataset_name = cfg['name']

    if dataset_name == 'BilGin':
        dataset = BilGinData(
            base_path=cfg['base_path'],
            streamline_number=cfg['streamline_number'],
            padding=cfg['padding'],
            mode=cfg['mode'],  # both train and validation use a subsample of the graph
            sub_list=cfg.get('sub_list', None),
            max_streamline_length=cfg['max_streamline_length'],
            return_geometric=cfg['return_geometric'],
            rescale=cfg.get('rescale', False),
            plausible_only=cfg.get('plausible_only', True),
            non_plausible_only=cfg.get('non_plausible_only', False),
            use_sublist=cfg.get('use_sublist', True)
        )
    elif dataset_name == 'HCPData':
            dataset = HCPData(
                base_path=cfg['base_path'],
                streamline_number=cfg['streamline_number'],
                padding=cfg['padding'],
                mode=cfg['mode'],
                sub_list=cfg.get('sub_list', None),
                max_streamline_length=cfg['max_streamline_length'],
                return_geometric=cfg['return_geometric'],
                rescale=cfg.get('rescale', False),
                plausible_only=cfg.get('plausible_only', True),
                non_plausible_only=cfg.get('non_plausible_only', False),
                use_sublist = cfg.get('use_sublist', False)
            )
    elif dataset_name == 'TractoInfernoData':
        dataset = TractoInfernoData(
            base_path=cfg['base_path'],
            streamline_number=cfg['streamline_number'],
            padding=cfg['padding'],
            mode=cfg['mode'],
            sub_list=cfg.get('sub_list', None),
            max_streamline_length=cfg['max_streamline_length'],
            return_geometric=cfg['return_geometric'],
            rescale=cfg.get('rescale', False),
            pattern=cfg.get('pattern', 'det'),
            use_sublist=cfg.get('use_sublist', False)
        )
    else:
        raise NotImplementedError(f'The dataset {dataset_name} is not implemented yet. Please choose among {valid_names}')

    return dataset


def get_dataloader(dataset: Dataset, shuffle: bool,
                   sampler: Optional[DistributedSampler], cfg: dict, mode:str = 'test', num_idxs: int=1000):

    collate_wrapper = CollateWrapper(mode, num_idxs)
    return DataLoader(dataset, shuffle=shuffle, sampler=sampler, collate_fn=collate_wrapper, **cfg)
