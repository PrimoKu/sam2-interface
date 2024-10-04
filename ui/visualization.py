import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout

def show_mask(mask, ax, obj_id=None, random_color=False):
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask_with_contours_and_bbox(mask, ax, obj_id=None, random_color=False):
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask_binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        contour = contour.reshape(-1, 2)
        ax.plot(contour[:, 0], contour[:, 1], color="yellow", linewidth=2)
    
    x, y, w, h = cv2.boundingRect(mask_binary)
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    return [x, y, x+w, y+h]