import json
import numpy as np
import cv2
import os

class COCOExporter:
    def __init__(self, output_file, use_existing=False):
        self.output_file = output_file
        self.coco_data = self.load_existing_data() if use_existing else {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = max([ann['id'] for ann in self.coco_data['annotations']], default=0) + 1

    def load_existing_data(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "images": [],
                "annotations": [],
                "categories": []
            }

    def initialize_categories(self, categories):
        existing_categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        for category in categories:
            if category['id'] in existing_categories:
                existing_categories[category['id']]['name'] = category['name']
            else:
                existing_categories[category['id']] = category
        
        self.coco_data['categories'] = list(existing_categories.values())

    def add_image(self, frame_number, file_name, width, height):
        for image in self.coco_data['images']:
            if image['id'] == frame_number:
                return frame_number
        
        image_info = {
            "id": frame_number,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        self.coco_data['images'].append(image_info)
        return frame_number

    def add_annotation(self, image_id, category_id, mask):
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == image_id and ann['category_id'] == category_id:
                contours, bbox = self.get_contours_and_bbox(mask)
                segmentation = self.contours_to_segmentation(contours)
                area = float(mask.sum())
                ann.update({
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                })
                return

        contours, bbox = self.get_contours_and_bbox(mask)
        segmentation = self.contours_to_segmentation(contours)
        area = float(mask.sum())

        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }
        self.coco_data['annotations'].append(annotation)
        self.annotation_id += 1

    @staticmethod
    def get_contours_and_bbox(mask):
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask_binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(mask_binary)
        return contours, [float(x), float(y), float(w), float(h)]

    @staticmethod
    def contours_to_segmentation(contours):
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                segmentation.append(contour)
        return segmentation

    def save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.coco_data, f)

    def update_file(self):
        self.save()