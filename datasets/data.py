import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.carla_dataset import *
# from datasets.kitti_dataset import *

dataset_choices = {'carla', 'kitti'}


def get_data_id(args):
    return '{}'.format(args.dataset)


def get_class_weights(freq):
    freq = np.array(freq)
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))
    
    return weights


class DataManager:
    def __init__(self, args):
        self.args = args
        assert self.args.dataset in dataset_choices
        if self.args.dataset == 'carla':
            self.class_frequencies = remap_frequencies_cartesian
            self.comp_weights = get_class_weights(self.class_frequencies).to(torch.float32)
            self.args.num_classes = 11
        elif self.args.dataset == 'kitti':
            raise NotImplementedError("We are going to complete the open-source work related to the SemanticKITTI dataset.")

    def get_train_data(self):
        train_ds = CarlaDataset(directory=self.args.train_data_path, 
                                quantized_directory=self.args.quantized_train_data_path,
                                data_argumentation=self.args.data_argumentation,
                                mode=self.args.mode,
                                prev_stage=self.args.prev_stage,
                                next_stage=self.args.next_stage,
                                prev_data_size=self.args.prev_data_size,
                                next_data_size=self.args.next_data_size,
                                prev_scene_path=self.args.prev_scene_path,
                                infer_data_source=self.args.infer_data_source,
                                mask_ratio=self.args.mask_ratio,
                                mask_prob=self.args.mask_prob,
                                model_type=self.args.model_type,
                                )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.args.distributed else None
        
        dataloader_train = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=train_ds.collate_fn, num_workers=self.args.num_workers)
        
        return dataloader_train, self.args.num_classes, self.comp_weights, train_sampler

    def get_infer_data(self):
        infer_ds = CarlaDataset(directory=self.args.infer_data_path, 
                                quantized_directory=self.args.quantized_infer_data_path,
                                data_argumentation=False,
                                mode=self.args.mode,
                                prev_stage=self.args.prev_stage,
                                next_stage=self.args.next_stage,
                                prev_data_size=self.args.prev_data_size,
                                next_data_size=self.args.next_data_size,
                                prev_scene_path=self.args.prev_scene_path,
                                infer_data_source=self.args.infer_data_source,
                                mask_ratio=self.args.mask_ratio,
                                mask_prob=self.args.mask_prob,
                                model_type=self.args.model_type,
                                )

        dataloader_infer = DataLoader(infer_ds, batch_size=self.args.batch_size, shuffle=False, collate_fn=infer_ds.collate_fn, num_workers=self.args.num_workers)
        
        return dataloader_infer, self.args.num_classes, self.comp_weights
