import torch
import warnings
import os

class InvalidStageError(Exception):
    pass

def set_data_size(stage, stage_name):
    stage_paths = {
        'none': None,
        's_1': (32, 32, 4),
        's_2': (64, 64, 8),
        's_3': (128, 128, 16)
    }
    
    data_size = stage_paths.get(stage)
    if data_size is None and stage != 'none':
        raise InvalidStageError(f"Invalid {stage_name} parameter: {stage}.")
    
    return data_size

def set_log_path(mode, exp_name):
    mode_paths = {
        'train': 'checkpoints',
        'inference': 'generated',
        'infinity_gen': 'infinite_generation'
    }
    
    if mode not in mode_paths:
        raise ValueError(f"Invalid mode: {mode}.")
    
    return os.path.join(mode_paths[mode], exp_name)

def configure_gpu(args):
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable DDP.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node
        
        if args.num_node == 1:
            args.dist_url = "auto"
        else:
            if args.num_node <= 1:
                raise ValueError("num_node must be greater than 1 for distributed training.")

def process_args_conflict(args):
    if args.prev_stage == 'none' and args.next_stage == 's_1':
        args.quantized_infer_data_path = args.infer_data_path
        args.quantized_train_data_path = args.train_data_path
    
    args.log_path = set_log_path(args.mode, args.exp_name)
    
    if args.mode == 'inference':
        args.batch_size = 1
    
    args.prev_data_size = set_data_size(args.prev_stage, 'prev_stage')
    args.next_data_size = set_data_size(args.next_stage, 'next_stage')
    
    configure_gpu(args)