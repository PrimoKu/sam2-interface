import numpy as np

def create_mask(sam_predictor, frame_idx, obj_id, coords, labels):
    out_mask_logits = sam_predictor.generate_mask_with_points(frame_idx, obj_id, coords, labels)
    
    if len(out_mask_logits) > 0:
        return (out_mask_logits[obj_id] > 0.0).cpu().numpy()
    else:
        return np.zeros((sam_predictor.inference_state['height'], sam_predictor.inference_state['width']), dtype=bool)

def update_click_prompts(prompts, object_id, x, y, click_type_val):
    if object_id not in prompts:
        prompts[object_id] = (np.array([[x, y]]), np.array([click_type_val]))
    else:
        coords, labels = prompts[object_id]
        new_coords = np.append(coords, [[x, y]], axis=0)
        new_labels = np.append(labels, [click_type_val])
        prompts[object_id] = (new_coords, new_labels)
    return prompts

def update_box_prompts(prompts, object_id, x_min, y_min, x_max, y_max):
    if object_id not in prompts:
        prompts[object_id] = (np.array([[x_min, y_min, x_max, y_max]]))
    else:
        box_coords = prompts[object_id]
        new_coords = np.append(box_coords, [[x_min, y_min, x_max, y_max]], axis=0)
        prompts[object_id] = new_coords
    return prompts

def propagate_masks(sam_predictor):
    return sam_predictor.propagate_masks()