import numpy as np
from scipy.ndimage import zoom
import yaml
from numba import njit
import os
from tqdm import tqdm
import dask.array as da
import argparse

@njit
def get_max_occurrence_label(block):
    counts = np.bincount(block)
    nonzero_counts = counts[1:]
    if nonzero_counts.size > 0:
        max_label = np.argmax(nonzero_counts) + 1
        return max_label
    else:
        return 0

@njit
def resample(labels, quantize_size):
    orignial_size = labels.shape
    reshaped = labels.reshape((quantize_size[0], 
                                orignial_size[0]//quantize_size[0], 
                                quantize_size[1], 
                                orignial_size[1]//quantize_size[1], 
                                quantize_size[2], 
                                orignial_size[2]//quantize_size[2]))
    result = np.zeros(quantize_size, dtype=labels.dtype)
    for i in range(quantize_size[0]):
        for j in range(quantize_size[1]):
            for k in range(quantize_size[2]):
                block = reshaped[i, :, j, :, k, :]
                result[i, j, k] = get_max_occurrence_label(block.flatten())

    return result

def resample_first(labels, scale_factors):
    scale_factors = scale_factors / np.asarray(labels.shape)
    labels_resampled = zoom(labels, scale_factors, order=0, mode='nearest', prefilter=False)
    for index, _ in np.ndenumerate(labels_resampled):
        original_index = tuple(slice(int(i / scale), int((i + 1) / scale)) for i, scale in zip(index, scale_factors))
        original_block = labels[original_index]
        labels_resampled[index] = next((x for x in original_block.flat if x), 0)
    return labels_resampled

def resample_value_max(labels, scale_factors):
    shape = np.array(labels.shape)
    reduction_factor = tuple(old_dim // new_dim for old_dim, new_dim in zip(shape, scale_factors))

    reduction_dict = {i: factor for i, factor in enumerate(reduction_factor)}

    labels_dask = da.from_array(labels)

    labels_resampled = da.coarsen(np.max, labels_dask, reduction_dict)

    return labels_resampled.compute()

class Quantize:
    def __init__(self, 
                 config_path,
                 data_base_path,
                 save_base_path,
                 quantize_size=(1., 1., 1.),
                 method='max',
                 ):
        self.config_path = config_path
        self.data_base_path = data_base_path
        self.save_base_path = save_base_path
        self.quantize_size = quantize_size

        self.labels_map = None

        config_file = yaml.safe_load(open(self.config_path, 'r'))
        self.scenes = [os.path.join(data_base_path, subset, scene, 'cartesian') 
                       for subset in ['Train', 'Val', 'Test'] 
                       for scene in os.listdir(os.path.join(data_base_path, subset))]
        self.eval_fine_list = []

        for scene in self.scenes:
            eval_fine_dir = os.path.join(scene, 'evaluation_fine')
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(eval_fine_dir)) if filename.endswith('.label')]
            self.eval_fine_list.extend([os.path.join(eval_fine_dir, str(frame).zfill(6)+'.label') for frame in frames_list])

        self.__process_config(config_file)
        self.process_and_save_labels(method)

    def __process_config(self, config_file):
        _labels_map = config_file['learning_map']
        self.labels_map = np.asarray(list(_labels_map.values()))

    def __get_eval_fine(self, label_path):
        eval_fine = np.fromfile(label_path, dtype=np.uint8)
        return eval_fine

    def process_and_save_labels(self, method):
        for label_path in tqdm(self.eval_fine_list, desc="Processing labels"):
            eval_label = self.__get_eval_fine(label_path)
            eval_label = self.labels_map[eval_label]
            
            eval_label = eval_label.reshape(256, 256, 16)

            if self.quantize_size == (256, 256, 16):
                resampled_label = eval_label
            else:
                if method == 'max':
                    resampled_label = resample(eval_label, self.quantize_size)
                elif method == 'first':
                    resampled_label = resample_first(eval_label, self.quantize_size)
                elif method == 'value_max':
                    resampled_label = resample_value_max(eval_label, self.quantize_size)
            
            relative_path = os.path.relpath(label_path, self.data_base_path)
            scene_dir = os.path.dirname(relative_path)
            quantized_dir = os.path.join(self.save_base_path, f'CarlaSC_quantized_{self.quantize_size[0]}_{self.quantize_size[1]}_{self.quantize_size[2]}/Cartesian', scene_dir, 'quantized')
            
            if not os.path.exists(quantized_dir):
                os.makedirs(quantized_dir)
            
            save_path = os.path.join(quantized_dir, os.path.basename(label_path).replace('.label', '.npy'))
            np.save(save_path, resampled_label)

def main():
    parser = argparse.ArgumentParser(description="Quantize and resample labels.")
    parser.add_argument('--quantize_size', type=int, nargs=3, required=True,
                        help="Set the quantize size (format: x y z)")
    parser.add_argument('--data_base_path', type=str, default='../../data/Cartesian',
                        help="Set the base path for data (default: '../../data/Cartesian')")
    parser.add_argument('--save_base_path', type=str, default='../../data',
                        help="Set the base path for saving output (default: '../../data')")
    parser.add_argument('--config_path', type=str, default='carla.yaml',
                        help="Set the config file path (default: 'carla.yaml')")
    parser.add_argument('--method', type=str, default='max', choices=['max', 'first', 'value_max'],
                        help="Set the resampling method (default: 'max')")

    args = parser.parse_args()

    Quantize(config_path=args.config_path, 
             data_base_path=args.data_base_path,
             save_base_path=args.save_base_path,
             quantize_size=tuple(args.quantize_size),
             method=args.method
             )
    
if __name__ == '__main__':
    main()
