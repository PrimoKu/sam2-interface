import numpy as np

def create_mask(sam_predictor, frame_idx, obj_id, coords, labels):
    return sam_predictor.generate_mask(frame_idx, obj_id, coords, labels)

def update_prompts(prompts, object_id, x, y, click_type_val):
    if object_id not in prompts:
        prompts[object_id] = (np.array([[x, y]]), np.array([click_type_val]))
    else:
        coords, labels = prompts[object_id]
        new_coords = np.append(coords, [[x, y]], axis=0)
        new_labels = np.append(labels, [click_type_val])
        prompts[object_id] = (new_coords, new_labels)
    return prompts

def propagate_masks(sam_predictor):
    return sam_predictor.propagate_masks()