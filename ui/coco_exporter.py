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
        # self.annotation_id = max([ann['id'] for ann in self.coco_data['annotations']], default=0)

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
        existing_categories = {cat['name']: cat for cat in self.coco_data['categories']}
        
        updated_categories = []
        for category in categories:
            if category['name'] in existing_categories:
                updated_cat = existing_categories[category['name']]
                updated_cat['id'] = category['id']
                updated_categories.append(updated_cat)
            else:
                updated_categories.append(category)
        
        self.coco_data['categories'] = sorted(updated_categories, key=lambda x: x['id'])


    def add_image(self, frame_number, file_name, width, height):
        coco_image_id = frame_number + 1
        
        for image in self.coco_data['images']:
            if image['id'] == coco_image_id:
                image.update({
                    "file_name": file_name,
                    "width": width,
                    "height": height
                })
                return coco_image_id
        
        image_info = {
            "id": coco_image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        self.coco_data['images'].append(image_info)
        return coco_image_id

    def add_annotation(self, image_id, category_id, mask):
        
        if not np.any(mask):
            return
    
        contours, bbox = self.get_contours_and_bbox(mask)
        segmentation = self.contours_to_segmentation(contours)
        area = float(mask.sum())

        existing_annotation = next((ann for ann in self.coco_data['annotations'] 
                                    if ann['image_id'] == image_id and ann['category_id'] == category_id), None)
        
        if existing_annotation:
            existing_annotation.update({
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
            })
        else:
            annotation = {
                "id": None,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            self.coco_data['annotations'].append(annotation)

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

    def regenerate_annotation_ids(self):
        self.coco_data['annotations'].sort(key=lambda x: (x['image_id'], x['category_id']))

        for idx, annotation in enumerate(self.coco_data['annotations'], start=1):
            annotation['id'] = idx

    def save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.coco_data, f)

    def update_file(self):
        self.regenerate_annotation_ids()
        self.save()