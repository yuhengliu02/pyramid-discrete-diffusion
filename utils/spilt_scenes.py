from numba import jit
import numpy as np

@jit(nopython=True)
def pad_scene(scene, pad_shape):
    padded_scene = np.zeros(pad_shape, dtype=scene.dtype)
    padded_scene[:scene.shape[0], :scene.shape[1], :scene.shape[2]] = scene
    return padded_scene

@jit(nopython=True)
def split(scene, target_resolution, sliding_split, sliding_ratio, scale_factor=(1, 1, 1), allow_exceed=False):
    scaled_target_resolution = (
        int(target_resolution[0] * scale_factor[0]),
        int(target_resolution[1] * scale_factor[1]),
        int(target_resolution[2] * scale_factor[2])
    )
    
    if sliding_split:
        step_size = (
            int(scaled_target_resolution[0] * sliding_ratio),
            int(scaled_target_resolution[1] * sliding_ratio),
            scene.shape[2]
        )
    else:
        step_size = scaled_target_resolution

    if allow_exceed:
        splits_along_x = (scene.shape[0] + step_size[0] - 1) // step_size[0]
        splits_along_y = (scene.shape[1] + step_size[1] - 1) // step_size[1]
    else:
        splits_along_x = 1 + (scene.shape[0] - scaled_target_resolution[0]) // step_size[0]
        splits_along_y = 1 + (scene.shape[1] - scaled_target_resolution[1]) // step_size[1]
    
    num_sub_scenes = splits_along_x * splits_along_y
    sub_scenes = np.empty((num_sub_scenes, scaled_target_resolution[0], scaled_target_resolution[1], scaled_target_resolution[2]), dtype=scene.dtype)
    
    sub_scene_idx = 0
    y = scene.shape[1] - scaled_target_resolution[1]  # Start from the top
    while y >= 0 and (allow_exceed or sub_scene_idx < num_sub_scenes):
        x = 0
        while x < scene.shape[0] and (allow_exceed or sub_scene_idx < num_sub_scenes):
            if not allow_exceed and (scene.shape[0] - x < scaled_target_resolution[0] or y + scaled_target_resolution[1] > scene.shape[1]):
                break
            
            temp_scene = np.zeros(scaled_target_resolution, dtype=scene.dtype)
            
            extract_x = min(scaled_target_resolution[0], scene.shape[0] - x)
            extract_y = min(scaled_target_resolution[1], y + scaled_target_resolution[1])
            
            temp_scene[:extract_x, :extract_y, :] = scene[x:x+extract_x, y:y+extract_y, :]
            
            sub_scenes[sub_scene_idx] = temp_scene
            sub_scene_idx += 1
            
            x += step_size[0]
        y -= step_size[1]
    
    return sub_scenes[:sub_scene_idx]

def split_scenes(high_resolution, low_resolution, high_resolution_counts, target_resolution_high, sliding_split, sliding_ratio=0.0, allow_exceed=False):    
    scale_factor = (
        low_resolution.shape[0] / high_resolution.shape[0], 
        low_resolution.shape[1] / high_resolution.shape[1], 
        low_resolution.shape[2] / high_resolution.shape[2]
    )
    
    sub_scenes_high = split(high_resolution, target_resolution_high, sliding_split, sliding_ratio, allow_exceed=allow_exceed)
    sub_scenes_low = split(low_resolution, target_resolution_high, sliding_split, sliding_ratio, scale_factor, allow_exceed=allow_exceed)
    sub_scenes_counts = split(high_resolution_counts, target_resolution_high, sliding_split, sliding_ratio, allow_exceed=allow_exceed)
    
    return sub_scenes_high, sub_scenes_low, sub_scenes_counts
