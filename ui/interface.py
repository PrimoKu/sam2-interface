import os
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QMessageBox, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sam2_predictor import SAM2Predictor
from visualization import show_mask, show_points, show_mask_with_contours_and_bbox
from coco_exporter import COCOExporter
from video_loader import load_video_frames, load_frame, navigate_frame
from mask_operations import create_mask, update_prompts, propagate_masks
from ui_utils import create_button, create_vertical_layout, create_horizontal_layout, get_object_color

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(16, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout = create_vertical_layout(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.figure.tight_layout()

    def clear(self):
        self.ax.clear()
        self.canvas.draw()

    def show_image(self, image):
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.canvas.draw()

class SAM2Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_attributes()
        self.initUI()

    def init_attributes(self):
        self.sam2_predictor = SAM2Predictor()
        self.video_dir = None
        self.frame_names = []
        self.current_frame_idx = 0
        self.current_image = None
        self.clicks = []
        self.click_type = "positive"
        self.object_ids = []
        self.prompts = {}
        self.video_segments = {}
        self.coco_exporter = None
        self.masks_propagated = False
        self.first_mask_created = False

    def initUI(self):
        self.setWindowTitle('SAM2 Interface')
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        main_layout.addWidget(self.create_left_panel())
        main_layout.addWidget(self.create_center_panel())
        main_layout.addWidget(self.create_right_panel())
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.disable_all_buttons()
        self.load_btn.setEnabled(True)

    def create_left_panel(self):
        self.load_btn = create_button('Load Video', self.load_video_or_frames)
        self.add_obj_btn = create_button('New Object', self.add_new_object)
        self.propagate_btn = create_button('Propagate Masks', self.propagate_masks)
        self.click_type_btn = create_button(f'Click Type: {self.click_type.capitalize()}', self.toggle_click_type)
        self.export_btn = create_button('Start COCO Export', self.initialize_coco_export)
        
        left_layout = create_vertical_layout(
            self.load_btn, self.add_obj_btn, self.propagate_btn,
            self.click_type_btn, self.export_btn
        )
        
        left_layout.addStretch(1)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        return left_widget

    def create_center_panel(self):
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.setFixedSize(1600, 900)
        self.mpl_widget.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.prev_btn = create_button('Prev Frame', lambda: self.navigate_frame('left'))
        self.next_btn = create_button('Next Frame', lambda: self.navigate_frame('right'))
        nav_layout = create_horizontal_layout(self.prev_btn, self.next_btn)
        
        self.frame_info_label = QLabel()
        
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.mpl_widget)
        center_layout.addLayout(nav_layout)
        center_layout.addWidget(self.frame_info_label)
        
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        
        return center_widget

    def create_right_panel(self):
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Category', 'Color'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedWidth(200)

        scroll = QScrollArea()
        scroll.setWidget(self.table)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)

        right_layout = QVBoxLayout()
        right_layout.addWidget(scroll)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        return right_widget

    def disable_all_buttons(self):
        self.add_obj_btn.setEnabled(False)
        self.propagate_btn.setEnabled(False)
        self.click_type_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def enable_buttons_after_video_load(self):
        self.add_obj_btn.setEnabled(True)
        self.click_type_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

    def load_video_or_frames(self):
        self.video_dir = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if self.video_dir:
            self.frame_names = load_video_frames(self.video_dir)
            if self.frame_names:
                self.current_frame_idx = 0
                self.current_image = load_frame(self.video_dir, self.frame_names[self.current_frame_idx])
                self.update_display(self.current_image)
                self.sam2_predictor.initialize_predictor(self.video_dir)
                print(f"Loaded video frames from {self.video_dir}")
                self.masks_propagated = False
                self.first_mask_created = False
                self.enable_buttons_after_video_load()
            else:
                print("No frames found in the folder.")
        else:
            print("No folder selected.")

    def update_display(self, image):
        if image is not None:
            self.mpl_widget.clear()
            self.mpl_widget.show_image(image)
            self.frame_info_label.setText(f'Current Frame: {self.current_frame_idx + 1} / {len(self.frame_names)}')

    def navigate_frame(self, direction):
        new_idx = navigate_frame(self.current_frame_idx, direction, len(self.frame_names))
        if new_idx != self.current_frame_idx:
            if direction == "right" and self.coco_exporter:
                self.export_current_frame_to_coco()
            self.current_frame_idx = new_idx
            self.current_image = load_frame(self.video_dir, self.frame_names[self.current_frame_idx])
            self.update_display(self.current_image)
            self.clicks = []
            self.prompts = {}
            if self.current_frame_idx in self.video_segments:
                self.display_propagated_masks()

    def on_click(self, event):
        if self.current_image is None or event.inaxes != self.mpl_widget.ax:
            return
        
        if len(self.object_ids) == 0:
            print("No objects added. Please add an object first.")
            return
        
        x, y = event.xdata, event.ydata
        click_type_val = 1 if self.click_type == "positive" else 0
        object_id = self.object_ids[-1]
        
        self.clicks.append({"coord": (x, y), "type": click_type_val, "id": object_id})
        self.prompts = update_prompts(self.prompts, object_id, x, y, click_type_val)

        self.update_mask()
        
        if not self.first_mask_created:
            self.first_mask_created = True
            self.propagate_btn.setEnabled(True)

    def update_mask(self):
        if self.current_image is not None:
            self.mpl_widget.clear()
            self.mpl_widget.show_image(self.current_image)
            
            masks = self.use_sam2_with_clicks(self.current_frame_idx, self.clicks)
            
            for i, mask in enumerate(masks):
                show_mask(mask, self.mpl_widget.ax, i)
                show_mask_with_contours_and_bbox(mask, self.mpl_widget.ax, i)

            for obj_id, (coords, labels) in self.prompts.items():
                show_points(coords, labels, self.mpl_widget.ax)

            self.mpl_widget.canvas.draw()

    def use_sam2_with_clicks(self, frame_idx, clicks):
        masks = []
        for obj_id in range(len(self.object_ids)):
            obj_clicks = [click for click in clicks if click["id"] == obj_id]
            obj_coords = np.array([click["coord"] for click in obj_clicks], dtype=np.float32)
            obj_labels = np.array([click["type"] for click in obj_clicks], dtype=np.int32)
            
            if len(obj_coords) > 0:
                mask = create_mask(self.sam2_predictor, frame_idx, obj_id, obj_coords, obj_labels)
                masks.append(mask)
        return masks

    def propagate_masks(self):
        if not self.video_dir:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
            return

        if not self.object_ids:
            QMessageBox.warning(self, "Warning", "Please add at least one object before propagating masks.")
            return

        self.video_segments = propagate_masks(self.sam2_predictor)
        print("Propagation completed across all frames.")
        self.masks_propagated = True
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Propagation Complete", "Mask propagation is complete. You can now start COCO export.")

    def display_propagated_masks(self):
        self.mpl_widget.clear()
        self.mpl_widget.show_image(self.current_image)
        for out_obj_id, out_mask in self.video_segments[self.current_frame_idx].items():
            show_mask(out_mask, self.mpl_widget.ax, out_obj_id)
            show_mask_with_contours_and_bbox(out_mask, self.mpl_widget.ax, out_obj_id)
        self.mpl_widget.canvas.draw()

    def add_new_object(self):
        new_obj_id = len(self.object_ids)
        self.object_ids.append(new_obj_id)
        print(f"Added new object with ID {new_obj_id}")
        self.update_table()

    def update_table(self):
        self.table.setRowCount(len(self.object_ids))
        for i, obj_id in enumerate(self.object_ids):
            self.table.setItem(i, 0, QTableWidgetItem(f"Object {obj_id}"))
            color_item = QTableWidgetItem()
            color = get_object_color(obj_id)
            color_item.setBackground(QBrush(color))
            self.table.setItem(i, 1, color_item)

    def toggle_click_type(self):
        self.click_type = "positive" if self.click_type == "negative" else "negative"
        self.click_type_btn.setText(f'Click Type: {self.click_type.capitalize()}')
        print(f"Click type switched to {self.click_type}")

    def initialize_coco_export(self):
        if not self.video_dir:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
            return

        if not self.masks_propagated:
            QMessageBox.warning(self, "Warning", "Please propagate masks before starting COCO export.")
            return

        output_file, _ = QFileDialog.getSaveFileName(self, "Save COCO JSON", "", "JSON Files (*.json)")
        if output_file:
            self.coco_exporter = COCOExporter(output_file)
            categories = [f"Object {i}" for i in range(len(self.object_ids))]
            self.coco_exporter.initialize_categories(categories)
            QMessageBox.information(self, "COCO Export", "COCO export initialized. Data will be written as you navigate through frames.")

    def export_current_frame_to_coco(self):
        if self.coco_exporter is None:
            return

        image_id = self.coco_exporter.add_image(
            frame_number=self.current_frame_idx,
            file_name=self.frame_names[self.current_frame_idx],
            width=self.current_image.shape[1],
            height=self.current_image.shape[0]
        )

        if self.current_frame_idx in self.video_segments:
            for obj_id, mask in self.video_segments[self.current_frame_idx].items():
                self.coco_exporter.add_annotation(image_id, obj_id, mask)

        self.coco_exporter.update_file()
        print(f"Exported COCO data for frame {self.current_frame_idx + 1}")

def run_interface():
    app = QApplication(sys.argv)
    ex = SAM2Interface()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_interface()