import argparse
import os
import torch
import yaml

from datasets.data import *
from utils.cuda import launch
from utils.multistep import get_optim
from utils.args_process import process_args_conflict
from train import Experiment
from models.conditional_diffusion.con_diffusion import Con_Diffusion
from models.unconditional_diffusion.gen_Diffusion import Diffusion
from models.latent_diffusion.stage1.vqvae import vqvae
from models.latent_diffusion.stage2.gen_diffusion import latent_diffusion

NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, default='', help='Path to the config file')
    parser.add_argument('--exp_name', '-n', type=str, default='default', help='Experiment name')

    args, unknown = parser.parse_known_args()

    config = load_config(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    process_args_conflict(args)
    launch(start, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def start(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    if args.mode == 'train':
        loader, num_classes, comp_weights, train_sampler = DataManager(args).get_train_data()
    elif args.mode in ['inference', 'infinity_gen']:
        loader, num_classes, comp_weights = DataManager(args).get_infer_data()
        train_sampler = None
    else:
        raise ValueError("Invalid mode: {}".format(args.mode))

    args.num_classes = num_classes

    completion_criterion = torch.nn.CrossEntropyLoss(weight=comp_weights)

    if args.model_type == 'uncon':
        model = Diffusion(args, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    elif args.model_type == 'con':
        model = Con_Diffusion(args, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    
    elif args.model_type == 'l_vae':
        model = vqvae(args, completion_criterion).cuda()
        if args.distribution:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    elif args.model_type == 'l_gen':
        Dense = vqvae(args, completion_criterion).cuda()
        dense_check = torch.load(args.vqvae_path)
        model = latent_diffusion(args, Dense, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution:
            Dense = torch.nn.parallel.DistributedDataParallel(Dense, device_ids=[args.gpu], find_unused_parameters=False)
            Dense.module.load_state_dict(dense_check['model'])
            for p in Dense.module.parameters():
                p.requires_grad = False   
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)

    exp = Experiment(args, 
                     model, 
                     optimizer, 
                     scheduler_iter, 
                     scheduler_epoch,
                     loader, 
                     args.log_path, 
                     args.check_every,
                     train_sampler)
    
    exp.run(epochs = args.epochs)

if __name__ == '__main__':
    main()