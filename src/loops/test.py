import os

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.dataset import get_dataloader
from src.utils import get_rank, get_world_size, is_master, AverageMeter, Logger
from src.models import get_model
from src.losses import get_loss




class Evaluator:

    def __init__(self, config: dict, test_data: Dataset, curr_seed: int):

        self.config = config
        self.curr_seed = curr_seed
        self.model_config = config["model"]
        self.loss_config = config["loss"]
        self.dataloader_config = config['dataloader']
        self.logger_config = config['logger']
        _ = self.dataloader_config.pop('shuffle_test', False)
        self.num_idxs = self.config["data"].get("streamline_number", 10000)
        self.dataloader_config.pop('shuffle_train', True)
        self.dataloader_config.pop('shuffle_val', False)
        self.dataloader_config['batch_size'] = 1
        self.data = test_data
        self.test_dataloader = get_dataloader(test_data,False, None, self.dataloader_config, mode='test',
                                              num_idxs=self.num_idxs)
        torch.cuda.set_device(get_rank())
        torch.cuda.empty_cache()
        self.model, _ = get_model(self.model_config)
        self.loss, self.loss_name = get_loss(self.loss_config)
        self.loss.to(get_rank())
        self.logger = Logger(self.config['checkpoint_path'], config, "test")
        self.test_dict = dict()


    @staticmethod
    def get_subject_data(subject_id, batch):
        subject_data = {}
        for k, v in batch.items():
            splits = v[subject_id]
            for split_idx, split_value in enumerate(splits):
                subject_data.setdefault(str(split_idx), {})[k] = split_value
        return subject_data


    @torch.no_grad()
    def test(self, test_header: str = None):
        self.load_checkpoint()
        self.model.eval()
        counter = 0
        for batch_idx, (test_batch, sub_name, _) in enumerate(self.test_dataloader):
            for subject_id in range(len(test_batch["data"])):
                subject_data = self.get_subject_data(subject_id, test_batch)
                self.test_dict[counter] = AverageMeter(self.loss_name, get_world_size() >= 1)
                for subject_split_idx, subject_split in subject_data.items():
                    for k, v in subject_split.items():
                        subject_split[k] = v.to(get_rank(), non_blocking=True)
                        if torch.is_tensor(v):
                            subject_split[k] = torch.unsqueeze(subject_split[k], 0)

                    y = subject_split["data"].x if isinstance(subject_split["data"], Data) else subject_split["data"]
                    out = self.model(subject_split["data"])
                    if subject_split.get('mask', None) is not None:
                        item_loss =\
                            self.loss(out * subject_split["mask"], y * subject_split["mask"]) / torch.sum(subject_split["mask"])
                    else:
                        item_loss = self.loss(out, y)
                    self.test_dict[counter].update(item_loss.item(), y.size(0))
                    self.logger.test_logger.debug(f"Processing batch_idx {batch_idx} subject {counter} "
                                                 f"split {subject_split_idx}")
                self.test_dict[counter].all_reduce()
                counter += 1

        aggregated_test_dict = {self.loss_name: AverageMeter(self.loss_name, get_world_size() >= 1)}
        for k, v in self.test_dict.items():
            aggregated_test_dict[self.loss_name].update(v.avg, v.count)
        results = {}
        for k, v in aggregated_test_dict.items():
            v.all_reduce()
            results[k] = v.avg
        if is_master():
            self.logger.log_test_stats(header=test_header,
                                       test_stats=results)


    @torch.no_grad()
    def embedding_distance_and_error(self, geometric):
        self.load_checkpoint()
        self.model.eval()

        lines = list()
        embeddings = list()
        losses = list()
        all_lens = list()
        sub_names = list()
        reconstructions = list()

        for batch_idx, (test_batch, sub_name, lengths) in enumerate(self.test_dataloader):
            for subject_id in range(len(test_batch["data"])):

                sub_lines = list()
                sub_losses = list()
                sub_embeddings = list()
                sub_reconstructions = list()

                all_lens.append(lengths[subject_id])
                sub_names.append(sub_name[subject_id])

                subject_data = self.get_subject_data(subject_id, test_batch)

                for subject_split_idx, subject_split in subject_data.items():
                    for k, v in subject_split.items():
                        subject_split[k] = v.to(get_rank(), non_blocking=True)
                        if torch.is_tensor(v):
                            subject_split[k] = torch.unsqueeze(subject_split[k], 0)

                    out = self.model.embed(subject_split["data"])
                    sub_embeddings.extend(out.cpu())

                    y = subject_split["data"].x if isinstance(subject_split["data"], Data) else subject_split["data"]
                    sub_lines.extend(y.cpu())

                    out = self.model(subject_split["data"])
                    sub_reconstructions.extend(out.cpu())

                    if subject_split.get('mask', None) is not None:
                        item_loss =\
                            self.loss(out * subject_split["mask"], y * subject_split["mask"])
                    else:
                        item_loss = self.loss(out, y)

                    sub_losses.extend(item_loss.cpu())

                if geometric:
                    starts = torch.cat([torch.tensor([0], device='cpu'), torch.cumsum(lengths[subject_id][:-1], 0)])
                    ends = torch.cumsum(lengths[subject_id], 0)
                    sub_lines = [sub_lines[st:en] for st, en in zip(starts, ends)]
                    sub_losses = [sub_losses[st:en] for st, en in zip(starts, ends)]
                    sub_reconstructions = [sub_reconstructions[st:en] for st, en in zip(starts, ends)]
                    lines.append(sub_lines)
                    losses.append(sub_losses)
                    reconstructions.append(sub_reconstructions)
                else:
                    lines.append(torch.vstack(sub_lines))
                    losses.append(torch.vstack(sub_losses))
                    reconstructions.append(torch.vstack(sub_reconstructions))

                embeddings.append(sub_embeddings)

        return lines, embeddings, losses, all_lens, sub_names, reconstructions


    @torch.no_grad()
    def save_embeddings(self, out_path):
        self.load_checkpoint()
        self.model.eval()
        geometric = True

        for batch_idx, (test_batch, sub_names, lengths) in enumerate(self.test_dataloader):
            for subject_id in range(len(test_batch["data"])):
                subject_data = self.get_subject_data(subject_id, test_batch)
                sub_lens = lengths[subject_id]
                sub_lens_cumulative = torch.cat([torch.tensor([0]), torch.cumsum(sub_lens, dim=0)])
                sub_embedding = list()
                sub_lines = list()
                sub_out_path = os.path.join(out_path, sub_names[subject_id])
                os.makedirs(sub_out_path, exist_ok=True)
                for subject_split_idx, subject_split in subject_data.items():
                    for k, v in subject_split.items():
                        subject_split[k] = v.to(get_rank(), non_blocking=True)
                        if torch.is_tensor(v):
                            subject_split[k] = torch.unsqueeze(subject_split[k], 0)

                    out = self.model.embed(subject_split["data"])
                    sub_embedding.extend(out.cpu())
                    if torch.is_tensor(subject_split["data"]):
                        sub_lines.extend(subject_split["data"].cpu())
                        geometric = False
                    else:
                        sub_lines.extend(subject_split["data"].x.cpu())

                header = self.data.get_sub_name_header(sub_names[subject_id])
                sub_embedding = torch.stack(sub_embedding, dim=0)

                if geometric:
                    sub_lines = torch.stack(sub_lines, dim=0)
                    sub_lines = [sub_lines[sub_lens_cumulative[i]:sub_lens_cumulative[i + 1]] for i in range(0, len(sub_lens_cumulative) - 1)]
                else:
                    sub_lines = torch.cat(sub_lines, dim=0)
                    sub_lines = sub_lines.view(-1, sub_lines.shape[-1])
                    sub_lines = [sub_lines[i*256:i*256+sub_lens[i]] for i in range(len(sub_lens))]


                sft = StatefulTractogram(sub_lines, reference=header, space=Space.RASMM)
                sft.data_per_streamline["embeddings"] = sub_embedding
                save_tractogram(sft, os.path.join(sub_out_path, f'{sub_names[subject_id]}_tractogram.trx'))


    def load_checkpoint(self):
        name = self.config['model']['model_name'] + f'_epoch_{self.config.get("which_epoch")}'
        model_path = os.path.join(f"{self.config['checkpoint_path']}/weights/{name}",
                                  f'model.pth.tar')
        map_location = {'cuda:%d' % 0: 'cuda:%d' % get_rank()}
        self.logger.test_logger.info(f"loading model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=map_location)
            self.model.load_state_dict(state_dict['model'], strict=False)
        except FileNotFoundError as e:
            self.logger.test_logger.error(f"failed to load model from {model_path} due to error {e}")
