import os
import math
import time
import warnings

import numpy as np
from yaml import dump

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


from src.utils import get_rank, mkdirs, get_world_size, is_master, AverageMeter, Logger
from src.dataset import get_dataloader
from src.models import get_model, get_optimizer, get_scheduler, get_lr
from src.losses import get_loss

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Trainer(object):
    def __init__(self, config: dict, train_set: Dataset, validation_set: Dataset, test_data: Dataset, curr_seed: int):
        self.config = config
        self.curr_seed = curr_seed
        self.model_config = config["model"]
        self.loss_config = config["loss"]
        self.dataloader_config = config['dataloader']
        self.logger_config = config['logger']
        self.num_idxs = self.config["data"].get("streamline_number", 10000)
        if get_world_size() > 1:
            train_shuffle = False
            val_shuffle = False
            test_shuffle = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, seed=self.curr_seed)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(validation_set, seed=self.curr_seed,
                                                                          shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
            train_shuffle = self.dataloader_config.pop('shuffle_train', True)
            val_shuffle = self.dataloader_config.pop('shuffle_val', False)
            test_shuffle = False

        self.train_dataloader = get_dataloader(train_set, train_shuffle, self.train_sampler, self.dataloader_config,
                                               mode='train')
        batch_size = self.dataloader_config.pop('batch_size')
        self.dataloader_config['batch_size'] =  1
        self.val_dataloader = get_dataloader(validation_set, val_shuffle,
                                             self.val_sampler, self.dataloader_config, mode='val')
        if is_master():
            self.test_dataloader = get_dataloader(test_data, test_shuffle, self.test_sampler,
                                                  self.dataloader_config, mode='test', num_idxs=self.num_idxs)
        self.dataloader_config['batch_size'] = batch_size
        torch.cuda.set_device(get_rank())
        torch.cuda.empty_cache()
        self.model, self.input_shape = get_model(self.model_config)
        if get_world_size() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[get_rank()],
                                                              broadcast_buffers=False,
                                                              find_unused_parameters=False)
        if config.get('print_summary', False):
            self.input_shape = (2, 10, 256, 3) #TODO this works only for FINTA, we need to fix this
            if is_master():
                if self.config["data"]["return_geometric"]:
                    from torch_geometric.nn import summary
                    self.input_data = next(iter(self.train_dataloader))[0]["data"].to(get_rank())
                    print(summary(self.model, self.input_data))
                else:
                    from torchinfo import summary
                    self.input_data = next(iter(self.train_dataloader))[0]["data"].to(get_rank())
                    summary(self.model, input_data=self.input_data)
        self.loss, self.loss_name = get_loss(self.loss_config)
        self.loss.to(get_rank())
        self.optimizer = get_optimizer(self.model, config['optimizer'])
        self.scheduler = get_scheduler(self.optimizer, config['scheduler'])

        if is_master():
            self.logger = Logger(self.config['checkpoint_path'], self.logger_config)

        self.dataset_size = len(self.train_dataloader) * self.dataloader_config["batch_size"]
        self.multiplier = get_world_size()
        self.total_steps = 0
        self.current_epoch = 0
        self.current_step = 0
        self.start_epoch = None
        self.start_epoch_time = None
        self.epoch_iter = None
        self.epoch_loss = 0
        self.print_delta = None
        self.current_lr = None
        self.val_interval = self.config.get('val_freq', 1)
        self.save_interval = self.config.get('save_epoch_freq', 1)
        self.loss_dict = dict()
        self.loss_dict[self.loss_name] = AverageMeter(self.loss_name, get_world_size() >= 1)

        self.eval_dict = dict()
        self.eval_dict[self.loss_name] = AverageMeter(self.loss_name, get_world_size() >= 1)

        self.test_dict = dict()

        if is_master() and not os.path.exists(os.path.join(self.config['checkpoint_path'], "config.yaml")):
            with open(os.path.join(self.config['checkpoint_path'], "config.yaml"), 'w') as yaml_file:
                dump(self.config, yaml_file, default_flow_style=False, Dumper=Dumper)


    def train_epoch(self, epoch):
        """
        Trains one epoch.
        Returns
        -------

        """
        self.start_of_epoch(epoch)
        if is_master():
            self.logger.train_logger.info(f'Epoch {self.current_epoch}/{self.config["epochs"] - 1}')

        for batch_idx, (train_batch, _, _) in enumerate(self.train_dataloader):
            data = self.start_of_iteration(train_batch, batch_idx)
            self.update_model(data)
            self.end_of_iteration()
        self.end_of_epoch()
        self.logger.train_logger.info(f'End of epoch {epoch}: Average MSE Loss: {self.epoch_loss}')

    def start_of_epoch(self, current_epoch):
        """Things to do before an epoch.
        Args:
            current_epoch (int): Current number of epoch.
        """
        if get_world_size() > 1:
            self.train_sampler.set_epoch(current_epoch)
        self.start_epoch_time = time.time()
        self.current_epoch = current_epoch
        self.epoch_loss = 0
        self.current_lr = get_lr(self.optimizer)
        self.loss_dict[self.loss_name].reset()
        self.eval_dict[self.loss_name].reset()

    def start_of_iteration(self, data, current_step):
        """Things to do before an iteration.
        Args:
            data: Data used for the current iteration.
            current_step: Current iteration of an epoch.
        """
        for k, v in data.items():
            data[k] = v.to(get_rank(), non_blocking=True)
        self.current_step = current_step
        self.model.train()
        return data

    def update_model(self, data):
        self.optimizer.zero_grad()
        out = self.model(data["data"])
        y = data["data"].x if isinstance(data["data"], Data) else data["data"]
        if data.get('mask', None) is not None:
            item_loss = self.loss(out * data["mask"], y * data["mask"]) / torch.sum(data["mask"])
        else:
            item_loss = self.loss(out, y)
        item_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_dict[self.loss_name].update(item_loss.item(), y.size(0))

    def end_of_iteration(self):
        self.total_steps += (self.dataloader_config["batch_size"] * self.multiplier)
        self.epoch_iter += (self.dataloader_config["batch_size"] * self.multiplier)
        if self.total_steps % self.logger_config["print_freq"] == self.print_delta:
            errors = dict()
            for k, v in self.loss_dict.items():
                v.all_reduce()
                errors[k] = v.avg
            if is_master():
                self.logger.log_train_stats(epoch=self.current_epoch, i=self.epoch_iter, errors=errors)
                self.logger.plot_train(errors=errors, step=self.total_steps)
                self.logger.plot_lr(lr=self.current_lr, step=self.total_steps)

    def end_of_epoch(self):
        """
        Things to do after an epoch.
        Returns
        -------
        """
        self.loss_dict[self.loss_name].all_reduce()
        self.epoch_loss = self.loss_dict[self.loss_name].avg
        self.epoch_iter = 0
        if is_master():
            self.logger.train_logger.info(f'End of epoch {self.current_epoch}/{self.config["epochs"]}'
                                          f' Time Taken {int(time.time() - self.start_epoch_time)} sec')

    @torch.no_grad()
    def val_epoch(self):

        self.model.eval()
        for batch_idx, (val_batch, _, _) in enumerate(self.val_dataloader):
            for k, v in val_batch.items():
                val_batch[k] = v.to(get_rank(), non_blocking=True)
            out = self.model(val_batch["data"])
            y = val_batch["data"].x if isinstance(val_batch["data"], Data) else val_batch["data"]
            if val_batch.get('mask', None) is not None:
                item_loss = self.loss(out * val_batch["mask"], y * val_batch["mask"]) / torch.sum(val_batch["mask"])
            else:
                item_loss = self.loss(out, y)
            self.eval_dict[self.loss_name].update(item_loss.item(), y.size(0))
        self.eval_dict[self.loss_name].all_reduce()

        results = dict()
        for k, v in self.eval_dict.items():
            v.all_reduce()
            results[k] = v.avg
        if is_master():
            self.logger.log_eval_stats(epoch=self.current_epoch, eval_stats=results)
            self.logger.plot_eval(eval_stats=results, step=self.total_steps)
        return results


    def train(self):

        save_path = self.config['checkpoint_path']
        model_name = self.config['model']['model_name']
        self.start_epoch, self.epoch_iter = self.load_checkpoint()
        self.initialize_deltas(self.start_epoch)

        # Track best by the chosen metric
        best_score = math.inf
        best_state = None

        for epoch in range(self.start_epoch, self.config["epochs"]):
            self.train_epoch(epoch)

            if self.scheduler is not None:
                if self.config['scheduler']['name'].lower() == 'step':
                    if self.current_lr >= float(self.config['scheduler'].get("min_lr", 0)):
                        self.scheduler.step()
                else:
                    self.scheduler.step()

            if epoch % self.val_interval == 0:
                results = self.val_epoch()
                score = results[self.loss_name]
                improved = (score < best_score - 1e-6)
                if improved:
                    best_score = score
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    if is_master():
                        name = model_name + f'_epoch_best'
                        self.save_checkpoint(epoch, name, save_path)

            if self.current_epoch % self.save_interval == 0:
                if is_master():
                    name = model_name + f'_epoch_{epoch}'
                    self.save_checkpoint(epoch, name, save_path)
        if is_master():
            name = model_name + f'_epoch_{self.config["epochs"]}'
            self.save_checkpoint(self.config["epochs"], name, save_path)

        return {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "best_state_model": best_state
        }

    @staticmethod
    def get_subject_data(subject_id, batch):
        subject_data = {}
        for k, v in batch.items():
            splits = v[subject_id]
            for split_idx, split_value in enumerate(splits):
                subject_data.setdefault(str(split_idx), {})[k] = split_value
        return subject_data

    @torch.no_grad()
    def test(self, test_name: str):
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
                        item_loss = \
                            self.loss(out * subject_split["mask"], y * subject_split["mask"]) / torch.sum(
                                subject_split["mask"])
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
            self.logger.log_test_stats(header=f"=== Test ({test_name}) ===", test_stats=results)


    def initialize_deltas(self, start_epoch):
        self.total_steps = (start_epoch - 1) * self.dataset_size
        self.print_delta = self.total_steps % self.logger_config.get("print_freq", 10)

    def save_checkpoint(self, epoch: int, name: str, save_path: str):
        """

        Parameters
        ----------
        epoch: int
        name : str
        save_path: str

        Returns
        -------

        """
        self.logger.train_logger.info(f'saving the model at the end of epoch {epoch}')
        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict()}
        checkpoint_folder = os.path.join(f"{save_path}/weights", f'{name}')
        mkdirs(checkpoint_folder, self.config["y"])
        checkpoint_path = os.path.join(f"{save_path}/weights/{name}", f'model.pth.tar')
        torch.save(checkpoint, checkpoint_path)
        self.logger.train_logger.info("model saved to {}".format(checkpoint_path))
        np.savetxt(os.path.join(f"{checkpoint_folder}", "iter.txt"), (epoch + 1, ), fmt='%d')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                torch.onnx.export(self.model, self.input_data, os.path.join(f"{checkpoint_folder}", 'model.onnx'))
        except Exception as e:
            self.logger.train_logger.info(f'Could not save onnx model due to {e}')

    def load_checkpoint(self):
        if self.config.get("continue_train"):
            name = self.config['model']['model_name'] + f'_epoch_{self.config.get("which_epoch")}'
            model_path = os.path.join(f"{self.config['checkpoint_path']}/weights/{name}",
                                      f'model.pth.tar')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % get_rank()}
            self.logger.train_logger.info(f"loading model from {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=map_location)
                self.model.load_state_dict(state_dict['model'], strict=False)
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.scheduler.load_state_dict(state_dict['scheduler'])
            except Exception as e:
                self.logger.train_logger.error(f"Could not load model from {model_path} due to {e}, start training"
                                              f" from scratch")
                return 1, 0
            try:
                start_epoch = (np.loadtxt(
                    os.path.join(f"{self.config['checkpoint_path']}/weights/{name}",
                                 "iter.txt"),
                    delimiter=',', dtype=int))
            except FileNotFoundError:
                start_epoch = 1
        else:
            start_epoch = 1
        return start_epoch, 0
