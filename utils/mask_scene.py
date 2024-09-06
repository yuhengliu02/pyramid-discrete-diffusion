import numpy as np

def mask_scene(input, mask_ratio, mask_prob):
    x, y, _ = input[0].shape
    masked_scenes = []
    
    for sub_scene in input:
        random_number = np.random.choice(4, 1, p=mask_prob)[0]
        
        masked_sub_scene = np.copy(sub_scene)

        if random_number == 0:  
            y_limit = int(y * mask_ratio)
            masked_sub_scene[:, :-y_limit, :] = 0
        elif random_number == 1:  
            x_limit = int(x * mask_ratio)
            masked_sub_scene[x_limit:, :, :] = 0
        elif random_number == 2:  
            y_limit = int(y * mask_ratio)
            x_limit = int(x * mask_ratio)
            masked_sub_scene[x_limit:, :-y_limit, :] = 0
        elif random_number == 3:  
            masked_sub_scene[:, :, :] = 0
        
        masked_scenes.append(masked_sub_scene)
    
    return masked_scenes
