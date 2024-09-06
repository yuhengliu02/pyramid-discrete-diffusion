import os
from collections import defaultdict
import random

def merge_voxel_files_vote(files, offsets):
    voxel_votes = defaultdict(lambda: defaultdict(int))  # Structure: {coordinate: {label: count}}
    merged_voxels = []

    for file_path, offset in zip(files, offsets):
        adjusted_voxels = adjust_coordinates(file_path, offset[0], offset[1])
        for voxel in adjusted_voxels:
            label, x, y, z = voxel
            coord = (x, y, z)
            voxel_votes[coord][label] += 1

    for coord, label_votes in voxel_votes.items():
        max_votes = max(label_votes.values())
        chosen_labels = [label for label, count in label_votes.items() if count == max_votes]
        chosen_label = random.choice(chosen_labels)
        merged_voxels.append((chosen_label, *coord))

    return merged_voxels

def adjust_coordinates(file_path, x_offset, y_offset):
    adjusted_voxels = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.split()
            label, x, y, z = float(values[0]), float(values[1]), float(values[2]), float(values[3])
            x += x_offset
            y += y_offset
            adjusted_voxels.append((label, x, y, z))
    return adjusted_voxels

def merge_voxel_files(files, offsets, fusion_method):
    if fusion_method == 'vote':
        return merge_voxel_files_vote(files, offsets)
    
    merged_voxels = []
    added_coordinates = set()

    for file_path, offset in zip(files, offsets):
        adjusted_voxels = adjust_coordinates(file_path, offset[0], offset[1])
        for voxel in adjusted_voxels:
            _, x, y, z = voxel
            coord = (x, y, z)

            if fusion_method == 'discard' and coord in added_coordinates:
                continue
            elif fusion_method == 'force':
                pass
            
            merged_voxels.append(voxel)
            added_coordinates.add(coord)

    return merged_voxels

def get_scene_files(directory_path, stage='scene'):
    max_scene_number = 0
    for file_name in os.listdir(directory_path):
        parts = file_name.split('_')
        if len(parts) < 3:
            continue
        try:
            scene_number = int(parts[1])
            if scene_number > max_scene_number:
                max_scene_number = scene_number
        except ValueError:
            continue
    
    scene_files = defaultdict(list)

    for file_name in os.listdir(directory_path):
        if stage == 'scene':
            _start = ('result_', 'gt_')
        elif stage == 'sub_scene':
            _start = 'prev_'
        if file_name.startswith(_start) and file_name.endswith('.txt'):
            parts = file_name.split('_')
            if len(parts) < 3:
                continue
            try:
                scene_number = int(parts[1])
                if scene_number == max_scene_number:
                    scene_files[scene_number].append(os.path.join(directory_path, file_name))
            except ValueError:
                continue
    
    for scene_number, files in scene_files.items():
        scene_files[scene_number] = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return scene_files


def save_merged_scenes(log_path, 
                       prev_data_size, 
                       next_data_size, 
                       mask_ratio, 
                       next_stage, 
                       mask_allow_exceed=False, 
                       stage='scene', 
                       fusion_method='discard' ):
    if stage == 'scene':
        input_folder = os.path.join(log_path, 'Generated')
        output_folder = os.path.join(log_path, 'GeneratedFusion')

        gt_input_folder = os.path.join(log_path, 'GroundTruth')
        gt_output_folder = os.path.join(log_path, 'GroundTruthFusion')
    elif stage == 'sub_scene':
        input_folder = os.path.join(log_path, 'PrevSceneContext')
        output_folder = os.path.join(log_path, 'PrevSceneContextFusion')

    scene_files = get_scene_files(input_folder, stage)

    if stage == 'scene':
        gt_scene_files = get_scene_files(gt_input_folder, stage)
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if stage == 'scene':
        if not os.path.exists(gt_output_folder):
            os.makedirs(gt_output_folder)
    
    for scene_number, files in scene_files.items():
        if all(f is not None for f in files):
            if stage == 'scene':
                offsets = compute_offsets_scene(next_data_size=next_data_size,
                                                mask_ratio=mask_ratio,
                                                next_stage=next_stage,
                                                mask_allow_exceed=mask_allow_exceed)
            elif stage == 'sub_scene':
                offsets = compute_offsets_sub_scene(next_data_size=next_data_size,
                                                    prev_data_size=prev_data_size,
                                                    mask_ratio=mask_ratio,
                                                    next_stage=next_stage,
                                                    mask_allow_exceed=mask_allow_exceed)
            merged_voxels = merge_voxel_files(files, offsets, fusion_method)
            output_file_path = os.path.join(output_folder, f'merged_{scene_number}.txt')
            with open(output_file_path, 'w') as f:
                for voxel in merged_voxels:
                    f.write(f"{voxel[0]:.18e} {voxel[1]:.18e} {voxel[2]:.18e} {voxel[3]:.18e}\n")
    
    if stage == 'scene':
        for scene_number, files in gt_scene_files.items():
            if all(f is not None for f in files):
                if stage == 'scene':
                    offsets = compute_offsets_scene(next_data_size=next_data_size,
                                mask_ratio=mask_ratio,
                                next_stage=next_stage,
                                mask_allow_exceed=mask_allow_exceed)
                merged_voxels = merge_voxel_files(files, offsets, fusion_method)
                output_file_path = os.path.join(gt_output_folder, f'merged_{scene_number}.txt')
                with open(output_file_path, 'w') as f:
                    for voxel in merged_voxels:
                        f.write(f"{voxel[0]:.18e} {voxel[1]:.18e} {voxel[2]:.18e} {voxel[3]:.18e}\n")


def compute_offsets_scene(next_data_size, mask_ratio, next_stage, mask_allow_exceed):
    final_scene_size = (256, 256, 16)
    
    step = (int(next_data_size[0] * (1 - mask_ratio)), int(next_data_size[1] * (1 - mask_ratio))) if next_stage=='s_3' else next_data_size[:2]
    
    offsets = []
    for y in range(0, final_scene_size[1], step[1]):
        for x in range(0, final_scene_size[0], step[0]):
            if not mask_allow_exceed and (x + next_data_size[0] > final_scene_size[0] or y + next_data_size[1] > final_scene_size[1]):
                continue
            offsets.append((x, y))
            
    offsets.sort(key=lambda coord: (-coord[1], coord[0]))
    return offsets

def compute_offsets_sub_scene(next_data_size, prev_data_size, mask_ratio, next_stage, mask_allow_exceed):
    ratio = next_data_size[0] / 256
    sub_scene_size = (int(prev_data_size[0] * ratio), int(prev_data_size[1] * ratio))
    
    step = (int(sub_scene_size[0] * (1 - mask_ratio)), int(sub_scene_size[1] * (1 - mask_ratio))) if next_stage=='s_3' else sub_scene_size
    
    offsets = []
    for y in range(0, prev_data_size[1], step[1]):
        for x in range(0, prev_data_size[0], step[0]):
            if not mask_allow_exceed and (x + sub_scene_size[0] > prev_data_size[0] or y + sub_scene_size[1] > prev_data_size[1]):
                continue
            offsets.append((x, y))
    
    offsets.sort(key=lambda coord: (-coord[1], coord[0]))
    return offsets

# --------------------------------------------------------------------------------

def infinity_fusion(next_data_size, mask_ratio, log_path, infinity_size, fusion_method='discard', high_res=False, folder_name=None):
    folder_path = os.path.join(log_path, 'Generated')
    output_folder = os.path.join(log_path, 'InfiniteScene')
    
    if high_res == False:
        INIT_SIZE = next_data_size
    else:
        INIT_SIZE = (256, 256, 16)
    MASK_WIDTH = int(INIT_SIZE[0] * (1 - mask_ratio))
    MASK_HEIGHT = int(INIT_SIZE[1] * (1 -mask_ratio))

    if high_res == False:
        files = [os.path.join(folder_path, f"result_{i * infinity_size[0] + j}.txt") 
                for i in range(infinity_size[1]) for j in range(infinity_size[0])]
    else:
        file_path = os.path.join(log_path, folder_name)
        files = [os.path.join(file_path, f"merged_{i * infinity_size[0] + j}.txt") 
                for i in range(infinity_size[1]) for j in range(infinity_size[0])]
        parent_folder = os.path.dirname(file_path)
        output_folder = os.path.join(parent_folder, 'InfiniteScene')
    
    offsets = [(j * MASK_WIDTH, i * MASK_HEIGHT) 
               for i in range(infinity_size[1]) for j in range(infinity_size[0])]
    
    merged_voxels = merge_voxel_files(files, offsets, fusion_method)
    
    if high_res == False:
        output_file_path = os.path.join(output_folder, f'infinity_scene_0.txt')
    else:
        output_file_path = os.path.join(output_folder, f'high_res_infinity_scene_0.txt')
    with open(output_file_path, 'w') as f:
        for voxel in merged_voxels:
            f.write(f"{voxel[0]:.18e} {voxel[1]:.18e} {voxel[2]:.18e} {voxel[3]:.18e}\n")
