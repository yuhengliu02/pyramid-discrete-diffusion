import numpy as np
import os

def read_txt_to_array(file_path, next_data_size):
    array = np.zeros(next_data_size)
    with open(file_path, 'r') as f:        
        for line in f:
            label, x, y, z = line.strip().split()
            label, x, y, z = int(float(label)), int(float(x)), int(float(y)), int(float(z))
            array[x, y, z] = label
    return array


def generation_mask(args, idx, mask_ratio):
    folder_path = os.path.join(args.log_path, 'Generated')
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt") and f.split("_")[2]])
    all_files.sort(key=lambda x: int(x.split("_")[1]))

    if idx == 0:
        return [np.zeros(args.next_data_size) for _ in range(args.batch_size)]

    elif idx == 1:
        arrays = []
        files_with_zero = [f for f in all_files if f.split("_")[2] == "0.txt"]
        selected_files = files_with_zero[-args.batch_size:]
        for f in selected_files:
            array = read_txt_to_array(os.path.join(folder_path, f), args.next_data_size)
            x_dim = array.shape[1]
            mask_width = int(mask_ratio * x_dim)
            new_array = np.zeros(array.shape)
            new_array[:mask_width, :, :] = array[-mask_width:, :, :]
            arrays.append(new_array)

        return arrays
    
    elif idx == 2:
        arrays = []
        files_with_zero = [f for f in all_files if f.split("_")[2] == "0.txt"]
        selected_files = files_with_zero[-args.batch_size:]
        for f in selected_files:
            array = read_txt_to_array(os.path.join(folder_path, f), args.next_data_size)
            y_dim = array.shape[0]
            mask_height = max(1, int(mask_ratio * y_dim))
            new_array = np.zeros(array.shape)
            new_array[:, -mask_height:, :] = array[:, :mask_height, :]
            arrays.append(new_array)

        return arrays

    elif idx == 3:
        arrays = []
        files_with_one = [f for f in all_files if f.split("_")[2] == "1.txt"]
        files_with_two = [f for f in all_files if f.split("_")[2] == "2.txt"]
        
        selected_files_1 = files_with_one[-args.batch_size:]
        selected_files_2 = files_with_two[-args.batch_size:]
        
        for f1, f2 in zip(selected_files_1, selected_files_2):
            array_1 = read_txt_to_array(os.path.join(folder_path, f1), args.next_data_size)
            array_2 = read_txt_to_array(os.path.join(folder_path, f2), args.next_data_size)
            
            combined_array = np.zeros(args.next_data_size)
            
            y_dim, x_dim = array_1.shape[0], array_1.shape[1]
            mask_height = max(1, int(mask_ratio * y_dim))
            mask_width = int(mask_ratio * x_dim)
            
            combined_array[:, -mask_height:, :] = array_1[:, :mask_height, :]
            combined_array[:mask_width, :, :] = array_2[-mask_width:, :, :]
            
            overlap = combined_array[:mask_width, -mask_height:, :].copy()
            for i in range(overlap.shape[0]):
                for j in range(overlap.shape[1]):
                    for k in range(overlap.shape[2]):
                        if overlap[i, j, k] != array_2[i, j + x_dim - mask_width, k] and overlap[i, j, k] != 0:
                            overlap[i, j, k] = np.random.choice([overlap[i, j, k], array_2[i, j + x_dim - mask_width, k]])
            
            combined_array[:mask_width, -mask_height:, :] = overlap
            arrays.append(combined_array)
        
        return arrays


def infinity_mask(args, infinity_size, idx=0):
    folder_path = os.path.join(args.log_path, 'Generated')
    if idx == 0:
        return np.zeros(args.next_data_size, dtype=int)
    
    new_array = np.zeros(args.next_data_size, dtype=int)

    y_dim, x_dim = args.next_data_size[0], args.next_data_size[1]
    mask_height = max(1, int(args.mask_ratio * y_dim))
    mask_width = int(args.mask_ratio * x_dim)
    
    i, j = divmod(idx, infinity_size[0])

    if j > 0:
        prev_idx = i * infinity_size[0] + j - 1
        array = read_txt_to_array(os.path.join(folder_path, f"result_{prev_idx}.txt"), args.next_data_size)
        new_array[:mask_width, :, :] = array[-mask_width:, :, :]
    if i > 0:
        prev_idx = (i - 1) * infinity_size[0] + j
        array_1 = read_txt_to_array(os.path.join(folder_path, f"result_{prev_idx}.txt"), args.next_data_size)
        new_array[:, -mask_height:, :] = array_1[:, :mask_height, :]
        if j > 0:
            prev_idx = i * infinity_size[0] + j - 1
            array_2 = read_txt_to_array(os.path.join(folder_path, f"result_{prev_idx}.txt"), args.next_data_size)
            new_array[:mask_width, :, :] = array_2[-mask_width:, :, :]
    
    return new_array
