import torch
import numpy as np
import os
import torch
import torch.nn.functional as F
import warnings

from utils.tables import *
from utils.scene_fusion import save_merged_scenes, infinity_fusion
from utils.generation_mask import generation_mask
from utils.mask_scene import mask_scene
from utils.generation_mask import infinity_mask


class Experiment(object):
    no_log_keys = ['project', 'name','log_tb', 'log_wandb','check_every','device', 'parallelssss', 'pin_memory', 'num_workers']
                   
    def __init__(self, 
                 args, 
                 model, 
                 optimizer, 
                 scheduler_iter, 
                 scheduler_epoch,
                 loader,
                 log_path, 
                 check_every,
                 train_sampler=None,
                 ):

        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model

        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer, self.scheduler_iter, self.scheduler_epoch= optimizer, scheduler_iter, scheduler_epoch
        self.log_path = log_path
        self.check_every = check_every
        self.args = args
        self.current_epoch = 0
        self.train_metrics = {}
        self.loader = loader
        self.train_sampler = train_sampler

        # Store args
        create_folders(args.mode, args.log_path)
        save_args(args, args.log_path)
        self.writer = create_writer(args, args.log_path, self.no_log_keys)
        

    def run(self, epochs):
        if self.args.resume: 
            self.resume()

        if self.args.mode in ['inference', 'infinity_gen']:
            print("Scenes will be generated.")
            self.sample()
            print("Generation process finished.")
        elif self.args.mode == 'train':
            for epoch in range(self.current_epoch, epochs): 
                train_dict = self.train_fn(epoch)
                self.log_metrics(train_dict, self.train_metrics)

                # Checkpoint
                self.current_epoch += 1
                if (epoch+1) % self.check_every == 0:
                    self.checkpoint_save(epoch)
            print("Training process finished.")


    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        if self.args.distribution :
            self.train_sampler.set_epoch(epoch)

        for prev_stage_data, next_stage_data, _ in self.loader:
            self.optimizer.zero_grad()

            if self.args.prev_stage!='none' and self.args.model_type=='con':
                total_loss = 0.0
                num_sub_scenes = len(prev_stage_data[0])
                for block_idx in range(num_sub_scenes):
                    selected_prev_data = [scene[block_idx] for scene in prev_stage_data]
                    selected_next_data = [scene[block_idx] for scene in next_stage_data]

                    if self.args.next_stage=='s_3':
                        selected_masked = [scene[num_sub_scenes + block_idx - 1] for scene in next_stage_data]
                        selected_masked = torch.from_numpy(np.asarray(selected_masked)).long().cuda()
                        selected_masked = selected_masked.unsqueeze(1)
                    prev_data_voxels = torch.from_numpy(np.asarray(selected_prev_data)).long().cuda()
                    next_data_voxels = torch.from_numpy(np.asarray(selected_next_data)).long().cuda()
                    one_hot_labels = F.one_hot(prev_data_voxels, num_classes=self.args.num_classes).permute(0, 4, 1, 2, 3).float()
                    interpolate_labels = F.interpolate(one_hot_labels, size=self.args.next_data_size, mode='trilinear')
                    prev_data_voxels = interpolate_labels.argmax(dim=1).byte().unsqueeze(1)
                    if self.args.next_stage=='s_3':
                        context = torch.cat([prev_data_voxels, selected_masked], dim=1)
                    else:
                        context = prev_data_voxels
                    loss = self.model(next_data_voxels, context)
                    total_loss += loss.item()
                    total_loss.backward() 
                average_loss = total_loss / num_sub_scenes
                loss_to_add = average_loss
            elif self.args.prev_stage=='none' and self.args.next_stage=='s_1' and self.args.model_type=='con':
                next_data_voxels = torch.from_numpy(np.asarray(next_stage_data)).long()
                context = mask_scene(input=next_data_voxels, mask_ratio=self.args.mask_ratio, mask_prob=self.args.mask_prob)
                context = torch.from_numpy(np.asarray(context)).long().unsqueeze(1).cuda()
                loss = self.model(next_data_voxels.cuda(), context)
                loss.backward()
                loss_to_add = loss.detach().cpu().item()
            elif self.args.model_type=='uncon' and self.args.prev_stage=='none':
                prev_data_voxels = torch.from_numpy(np.asarray(prev_stage_data)).long().squeeze(1)
                next_data_voxels = torch.from_numpy(np.asarray(next_stage_data)).long().squeeze(1)
                context = prev_data_voxels.unsqueeze(1).cuda()
                loss = self.model(next_data_voxels.cuda(), context)
                loss.backward()
                loss_to_add = loss.detach().cpu().item()
            else:
                raise ValueError("Please Check the 'prev_stage' and 'next_stage' parameters.")

            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)

            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()

            loss_sum += loss_to_add * len(next_stage_data)
            loss_count += len(next_stage_data)

            print('Training: prev_stage: {}, next_stage: {}, Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(self.args.prev_stage, self.args.next_stage, epoch+1, self.args.epochs, loss_count, len(self.loader.dataset), loss_sum/loss_count), end='\r')

        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'loss': loss_sum/loss_count}


    def sample(self):
        self.model.eval()
        with torch.no_grad():
            dataloader = self.loader
            gen_num_recorder = 0
            for iterate, (prev_stage_data, next_stage_data, _) in enumerate(dataloader):
                if len(prev_stage_data) == self.args.batch_size :
                    gen_num_recorder += 1
                    if self.args.mode != 'infinity_gen':
                        print(f"Working on generation process: {gen_num_recorder} / {self.args.generation_num}")

                    if self.args.prev_stage!='none':
                        num_sub_scenes = len(prev_stage_data[0])
                        if self.args.mode == 'infinity_gen':
                            infinite_scenes = self.args.infinity_size[0] * self.args.infinity_size[1]
                            print(f"Working on INFINITE SCENE generation process: {gen_num_recorder} / {infinite_scenes}")
                        for block_idx in range(num_sub_scenes):
                            selected_prev_data = [scene[block_idx] for scene in prev_stage_data]
                            selected_next_data = [scene[block_idx] for scene in next_stage_data]

                            if self.args.next_stage=='s_3':
                                if self.args.mask_ratio in [0.0625, 0.125, 0.25]:
                                    selected_masked = generation_mask(self.args, block_idx, self.args.mask_ratio)
                                else:
                                    warnings.warn(f"The mask ratio {self.args.mask_ratio} has not been tested for feasibility. Proceeding with caution.")
                                    selected_masked = generation_mask(self.args, block_idx, self.args.mask_ratio)
                                second_context = torch.from_numpy(np.asarray(selected_masked)).long().cuda()
                                _second_context = second_context.unsqueeze(1)

                            prev_data_voxels = torch.from_numpy(np.asarray(selected_prev_data)).long().cuda()
                            next_data_voxels = torch.from_numpy(np.asarray(selected_next_data)).long().cuda()

                            _prev_data_voxels = prev_data_voxels.clone().detach().long()
                            one_hot_labels = F.one_hot(_prev_data_voxels, num_classes=self.args.num_classes).permute(0, 4, 1, 2, 3).float()
                            interpolate_labels = F.interpolate(one_hot_labels, size=self.args.next_data_size, mode='trilinear')
                            _prev_data_voxels = interpolate_labels.argmax(dim=1).byte().unsqueeze(1)

                            if self.args.next_stage=='s_3':
                                context = torch.cat([_prev_data_voxels, _second_context], dim=1)
                            else:
                                context = _prev_data_voxels

                            generated = self.model.sample(context)
                            
                            if self.args.next_stage=='s_3':
                                visualization(args=self.args, 
                                              generated=generated, 
                                              prev_data_voxels=prev_data_voxels, 
                                              next_data_voxels=next_data_voxels, 
                                              iteration = iterate, 
                                              sub_scenes=block_idx, 
                                              second_context=second_context)
                            else:
                                visualization(args=self.args, 
                                              generated=generated, 
                                              prev_data_voxels=prev_data_voxels, 
                                              next_data_voxels=next_data_voxels, 
                                              iteration = iterate, 
                                              sub_scenes=block_idx)
                        
                        if self.args.next_stage == 's_3':
                            print("Working on scene fusion...")
                            save_merged_scenes(log_path=self.args.log_path,
                                               prev_data_size=self.args.prev_data_size,
                                               next_data_size=self.args.next_data_size,
                                               mask_ratio=self.args.mask_ratio,
                                               next_stage=self.args.next_stage,
                                               stage='scene', 
                                               fusion_method=self.args.scene_fusion_method)

                            print("Working on sub-condition Fusion...")
                            save_merged_scenes(log_path=self.args.log_path,
                                               prev_data_size=self.args.prev_data_size,
                                               next_data_size=self.args.next_data_size,
                                               mask_ratio=self.args.mask_ratio,
                                               next_stage=self.args.next_stage,
                                               stage='sub_scene', 
                                               fusion_method=self.args.scene_fusion_method)
                    else:
                        prev_data_voxels = torch.from_numpy(np.asarray(prev_stage_data)).long().squeeze(1).cuda() # (4,1,256,256,32)
                        next_data_voxels = torch.from_numpy(np.asarray(next_stage_data)).long().cuda()            

                        if self.args.model_type == 'l_vae':
                            generated = self.model.sample(next_data_voxels) 
                        else :
                            if self.args.mode=='infinity_gen':
                                for i in range(self.args.infinity_size[0]*self.args.infinity_size[1]):
                                    print(f"Working on INFINITE SCENE generation process: {i + 1} / {self.args.infinity_size[0]*self.args.infinity_size[1]}")
                                    context = infinity_mask(self.args, self.args.infinity_size, i)
                                    context = torch.from_numpy(np.asarray(context)).long().cuda().unsqueeze(0).unsqueeze(1)
                                    generated = self.model.sample(context)
                                    visualization(args=self.args, 
                                                  generated=generated, 
                                                  prev_data_voxels=prev_data_voxels, 
                                                  next_data_voxels=next_data_voxels, 
                                                  iteration = i)
                                print("Working on infinite scene fusion...")
                                infinity_fusion(next_data_size=self.args.next_data_size,
                                                mask_ratio=self.args.infinite_ratio,
                                                log_path=self.args.log_path,
                                                infinity_size=self.args.infinity_size)
                                return 0
                            elif self.args.model_type=='con':
                                context = np.zeros(self.args.next_data_size, dtype=int)
                                context = torch.from_numpy(np.asarray(context)).long().cuda().unsqueeze(0).unsqueeze(1)
                                generated = self.model.sample(context)
                                visualization(args=self.args, 
                                              generated=generated, 
                                              prev_data_voxels=prev_data_voxels, 
                                              next_data_voxels=next_data_voxels, 
                                              iteration = iterate)
                            elif self.args.model_type=='uncon':
                                generated = self.model.sample(prev_data_voxels)
                                visualization(args=self.args, 
                                              generated=generated, 
                                              prev_data_voxels=prev_data_voxels, 
                                              next_data_voxels=next_data_voxels, 
                                              iteration = iterate)
                    if gen_num_recorder >= self.args.generation_num:
                        return 0
                
            if self.args.mode == 'infinity_gen' and self.args.next_stage in ['s_2', 's_3'] and self.args.prev_stage != "none":
                print("Working on infinite scene fusion...")
                infinity_fusion(next_data_size=self.args.next_data_size,
                                mask_ratio=self.args.infinite_ratio,
                                log_path=self.args.log_path,
                                infinity_size=self.args.infinity_size,
                                high_res=True,
                                folder_name='GeneratedFusion')
                    
            return 0

    def resume(self):
        if not os.path.exists(self.args.resume_path):
            raise FileNotFoundError(f"The resume_path '{self.args.resume_path}' does not exist.")
        self.checkpoint_load(self.args.resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]

            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)

    def log_metrics(self, dict, type):
        if len(type)==0:
            for metric_name, metric_value in dict.items():
                type[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in dict.items():
                type[metric_name].append(metric_value)


    def checkpoint_save(self, epoch):        
        checkpoint = {
                      'train_metrics': self.train_metrics,
                      'optimizer': self.optimizer.state_dict(),
                      'model': self.model.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None,
                      }

        epoch_name = 'epoch{}.tar'.format(epoch)
        torch.save(checkpoint, os.path.join(self.log_path, 'models', epoch_name))


    def checkpoint_load(self, resume_path):
        checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

        self.train_metrics = checkpoint['train_metrics']
        print("=> loaded checkpoint '{}'".format(resume_path))