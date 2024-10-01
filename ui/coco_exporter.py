import json
import numpy as np
import cv2

class COCOExporter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = 0

    def initialize_categories(self, categories):
        self.coco_data["categories"] = [{"id": i, "name": name} for i, name in enumerate(categories)]

    def add_image(self, frame_number, file_name, width, height):
        image_info = {
            "id": frame_number,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        self.coco_data["images"].append(image_info)
        return frame_number

    def add_annotation(self, image_id, category_id, mask):
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
        self.coco_data["annotations"].append(annotation)
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