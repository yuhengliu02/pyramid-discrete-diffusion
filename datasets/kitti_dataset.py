import os
import numpy as np
import yaml
from torch.utils.data import Dataset
from utils.spilt_scenes import split_scenes
from utils.mask_scene import mask_scene

SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"]
}

class KITTIDataset(Dataset):
    def __init__(self, 
        directory,
        quantized_directory=None,
        data_argumentation=False,  
        num_frames=4,
        mode='train',
        spilt='train', 
        prev_stage='none',  
        next_stage='s_1', 
        prev_data_size=(32, 32, 4),
        next_data_size=(128, 128, 16),
        prev_data_path=None,
        sliding_ratio=0.5,  
        infer_data_source='dataset',  
        mask_ratio=0.0625,  
        mask_prob=[0.25, 0.25, 0.25, 0.25],  
        ):
        self._directory = directory
        self._num_frames = num_frames
        self.prev_data_size = prev_data_size
        self.next_data_size = next_data_size
        self.prev_data_path = prev_data_path
        self.quantized_directory = quantized_directory
        self.spilt = spilt
        self.data_argumentation = data_argumentation  
        self.mode = mode  
        self.prev_stage = prev_stage  
        self.next_stage = next_stage  
        self.sliding_ratio = sliding_ratio  
        self.infer_data_source = infer_data_source  
        self.mask_ratio = mask_ratio  
        self.mask_prob = mask_prob  
        
        self._grid_size = [256, 256, 16]

        if self.args.kitti_32_to_64 == True:
            self._grid_size = [64, 64, 8]

        if self.debug_mode:
            print("grid_size: ", self._grid_size)
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = [0, -25.6, -2] + [51.2, 25.6, 4.4]
        
        if self.args.kitti_32_to_64 == True:
            self.coor_ranges = [0, -6.4, -2] + [12.8, 6.4, 4.4]

        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)
        self._directory = os.path.join(directory, 'sequences')
        self._num_frames = 4
        self._remap_lut = self.get_remap_lut()

        self._eval_labels = [] 
        self._eval_counts = [] 
        self._frames_list = [] 
        self._completion_list = []
        self._quantized_eval_labels = []

        self._num_frames_scene = []

        self._seqs = SPLIT_SEQUENCES[self.spilt]

        for scene in self._seqs:
            eval_dir = os.path.join(self._directory, scene, 'voxels')
            if self.quantized_directory is not None:
                quantized_dir = os.path.join(self.quantized_directory, scene, 'voxels')
            try:
                self._num_frames_scene.append(len(os.listdir(eval_dir))) 
            except:
                self._num_frames_scene.append(len(os.listdir(quantized_dir)))
            
            try:
                frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(eval_dir)) if filename.endswith('.npy')] 
            except:
                frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(quantized_dir))] 

            self._frames_list.extend(frames_list) 
            self._eval_labels.extend([os.path.join(eval_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])
            self._eval_counts.extend([os.path.join(eval_dir, str(frame).zfill(6) + '.npy') for frame in frames_list])

            self._quantized_eval_labels.extend([os.path.join(quantized_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])

        assert len(self._eval_labels) == np.sum(self._num_frames_scene), f"len(self._eval_labels): {len(self._eval_labels)}, np.sum(self._num_frames_scene): {np.sum(self._num_frames_scene)}"

        if self.second_stage_generation:
            for root_, _, files in os.walk(self.prev_data_path):
                for file in files:
                    if file.endswith('.txt'):
                        self._completion_list.append(os.path.join(root_, file))
            
            self._completion_list.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    def __len__(self):
        if self.mode == 'inference' and self.prev_stage != 'none' and self.infer_data_source == 'generation':
            return len(self._completion_list)
        else:
            return sum(self._num_frames_scene)

    def get_remap_lut(self):
        maxkey = max(LABELS_REMAP.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(LABELS_REMAP.keys())] = list(LABELS_REMAP.values())

        return remap_lut
    
    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        counts_batch = [bi[2] for bi in data]
        return voxel_batch, output_batch, counts_batch

    def __getitem__(self, idx):
        idx_range = self.find_horizon(idx)

        if self.next_stage == 's_1' and self.prev_stage == 'none' and self.infer_data_source == 'dataset':
            next_stage_data = np.load(self._quantized_eval_labels[idx_range[-1]])
            prev_stage_data = next_stage_data.copy()
            quantized_output = np.zeros_like(next_stage_data)
            counts = np.zeros_like(next_stage_data)
        else:
            quantized_output = np.load(self._quantized_eval_labels[idx_range[-1]])
            next_stage_data = np.load(self._eval_labels[idx_range[-1]])
            counts = next_stage_data.copy()
            prev_stage_data = next_stage_data.copy()

        if self.data_argumentation:
            if np.random.randint(2):
                next_stage_data = np.flip(next_stage_data, axis=0)
                counts = np.flip(counts, axis=0)
                quantized_output = np.flip(quantized_output, axis=0)
            
            if np.random.randint(2):
                next_stage_data = np.flip(next_stage_data, axis=1)
                counts = np.flip(counts, axis=1)
                quantized_output = np.flip(quantized_output, axis=1)
            
            rotation_type = np.random.randint(4)
            if rotation_type == 1:
                next_stage_data = np.rot90(next_stage_data, 1, axes=(0, 1))
                counts = np.rot90(counts, 1, axes=(0, 1))
                quantized_output = np.rot90(quantized_output, 1, axes=(0, 1))
            elif rotation_type == 2:
                next_stage_data = np.rot90(next_stage_data, -1, axes=(0, 1))
                counts = np.rot90(counts, -1, axes=(0, 1))
                quantized_output = np.rot90(quantized_output, -1, axes=(0, 1))
            elif rotation_type == 3:
                next_stage_data = np.rot90(next_stage_data, 2, axes=(0, 1))
                counts = np.rot90(counts, 2, axes=(0, 1))
                quantized_output = np.rot90(quantized_output, 2, axes=(0, 1)) 
        
        if self.mode == 'inference' and self.prev_stage != 'none' and self.infer_data_source == 'generation':
            _output = np.zeros(self.prev_data_size)
            with open(self._completion_list[idx_range[-1]], 'r') as file:
                for line in file:
                    label, x, y, z = map(float, line.strip().split())
                    label = int(label)
                    x, y, z = int(x), int(y), int(z)
                    _output[x, y, z] = label
            _output = _output.astype(np.uint8)

        if self.mode == 'inference' and self.next_stage == 's_3' and self.prev_stage!='none' and self.infer_data_source == 'dataset':
            sub_scenes_high, sub_scenes_low, sub_scenes_high_counts = split_scenes(high_resolution=next_stage_data,
                                                                                   low_resolution=quantized_output,
                                                                                   high_resolution_counts=counts,
                                                                                   target_resolution_high=self.next_data_size,
                                                                                   sliding_split=True,
                                                                                   sliding_ratio=1 - self.mask_ratio)
            next_stage_data = sub_scenes_high
            prev_stage_data = sub_scenes_low
            counts = sub_scenes_high_counts
        
        if self.mode == 'train' and self.next_stage == 's_3' and self.prev_stage != 'none':
            sub_scenes_high, sub_scenes_low, sub_scenes_high_counts = split_scenes(high_resolution=next_stage_data, 
                                                                                   low_resolution=quantized_output, 
                                                                                   high_resolution_counts=counts,
                                                                                   target_resolution_high=self.next_data_size,
                                                                                   sliding_split=False)
            next_stage_data = sub_scenes_high
            prev_stage_data = sub_scenes_low
            counts = sub_scenes_high_counts
        
        if self.mode == 'inference' and self.next_stage == 's_3' and self.prev_stage != 'none' and self.infer_data_source == 'generation':
            sub_scenes_high, sub_scenes_low, sub_scenes_high_counts = split_scenes(high_resolution=next_stage_data, 
                                                                                   low_resolution=_output, 
                                                                                   high_resolution_counts=counts,
                                                                                   target_resolution_high=self.next_data_size,
                                                                                   sliding_split=True,
                                                                                   sliding_ratio=1 - self.mask_ratio)
            next_stage_data = sub_scenes_high
            prev_stage_data = sub_scenes_low
            counts = sub_scenes_high_counts
        

        if self.mode == 'train' and self.next_stage in ['s_3', 's_1']:
            if self.next_stage == 's_3' and self.prev_stage != 'none':
                masked_scene = mask_scene(input=next_stage_data, mask_ratio=self.mask_ratio, mask_prob=self.mask_prob)
                next_stage_data = np.concatenate((next_stage_data, masked_scene))
            elif self.next_stage == 's_1' and self.prev_stage == 'none':
                masked_scene = mask_scene(input=next_stage_data, mask_ratio=self.mask_ratio, mask_prob=self.mask_prob)
                next_stage_data = np.concatenate((next_stage_data, masked_scene))
        

        return prev_stage_data, next_stage_data, counts
        
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range