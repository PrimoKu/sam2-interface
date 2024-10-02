import os
import cv2
import sys
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QMessageBox, QVBoxLayout, QHBoxLayout,
                             QProgressDialog)
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt, QTimer
from sam2_predictor import SAM2Predictor
from visualization import show_mask, show_points, show_mask_with_contours_and_bbox
from coco_exporter import COCOExporter
from ui_utils import (create_button, create_vertical_layout, create_horizontal_layout,
                      get_object_color, CenteredCheckBox, AlignDelegate, MatplotlibWidget)
from object_manager import ObjectManager

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

class SAM2UI:
    def __init__(self, interface):
        self.interface = interface
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.left_panel = self.create_left_panel()
        self.center_panel = self.create_center_panel()
        self.right_panel = self.create_right_panel()
        
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.center_panel)
        self.main_layout.addWidget(self.right_panel)
        
        self.main_widget.setLayout(self.main_layout)

    def create_left_panel(self):
        self.load_btn = create_button('Load Video', self.interface.load_video_or_frames)
        self.add_obj_btn = create_button('New Object', self.interface.prepare_new_object)
        self.propagate_btn = create_button('Propagate Masks', self.interface.propagate_masks)
        self.export_btn = create_button('Start COCO Export', self.interface.initialize_coco_export)
        self.reset_btn = create_button('Reset Tracking', self.interface.reset_inference_state)
        
        left_layout = create_vertical_layout(
            self.load_btn, self.add_obj_btn, self.propagate_btn, self.export_btn, self.reset_btn
        )
        
        left_layout.addStretch(1)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        return left_widget

    def create_center_panel(self):
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.setFixedSize(1600, 900)
        self.mpl_widget.canvas.mpl_connect('button_press_event', self.interface.on_click)
        
        self.prev_btn = create_button('Prev Frame', lambda: self.interface.navigate_frame('left'))
        self.next_btn = create_button('Next Frame', lambda: self.interface.navigate_frame('right'))
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
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Re-segment', 'Category', 'Color'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedWidth(250)
        self.table.itemChanged.connect(self.interface.on_category_name_change)

        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

        align_delegate = AlignDelegate(self.table)
        self.table.setItemDelegateForColumn(1, align_delegate)
        self.table.setItemDelegateForColumn(2, align_delegate)

        scroll = QScrollArea()
        scroll.setWidget(self.table)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(270) 

        right_layout = QVBoxLayout()
        right_layout.addWidget(scroll)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        return right_widget

    def disable_all_buttons(self):
        self.add_obj_btn.setEnabled(False)
        self.propagate_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def enable_buttons_after_video_load(self):
        self.add_obj_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

    def update_table(self):
        self.table.setRowCount(len(self.interface.object_manager.get_all_objects()))
        for i, (obj_id, obj_data) in enumerate(self.interface.object_manager.get_all_objects().items()):
            checkbox_widget = CenteredCheckBox()
            checkbox_widget.checkbox.stateChanged.connect(lambda state, x=obj_id: self.interface.on_resegment_checked(state, x))
            self.table.setCellWidget(i, 0, checkbox_widget)
            
            name_item = QTableWidgetItem(obj_data['category_name'])
            name_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, name_item)
            
            color_item = QTableWidgetItem()
            color_item.setBackground(QBrush(obj_data['color']))
            self.table.setItem(i, 2, color_item)

        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, 30)
            
class SAM2Interface:
    def __init__(self):
        self.ui = SAM2UI(self)
        self.sam2_predictor = SAM2Predictor()
        self.object_manager = ObjectManager()
        self.coco_exporter = None
        self.default_load_dir = os.path.abspath("../data/")
        self.default_export_dir = os.path.abspath("../output/")
        self.input_folder_name = None
        self.coco_export_file = None
        self.video_dir = None
        self.frame_names = []
        self.current_frame_idx = 0
        self.current_image = None
        self.prompts = {}
        self.video_segments = {}
        self.masks_propagated = False
        self.first_mask_created = False
        self.current_object_id = None
        self.masks = {}
        self.object_bboxes = {}

    def run(self):
        self.window = QMainWindow()
        self.window.setCentralWidget(self.ui.main_widget)
        self.window.show()
        self.ui.disable_all_buttons()
        self.ui.load_btn.setEnabled(True)

    # Video and Frame Management
    # --------------------------
    def load_video_or_frames(self):
        if self.video_dir:
            reply = QMessageBox.question(self.window, 'Confirm Reload', 
                                        'Are you sure you want to load a new video? This will reset all current work.',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.video_dir = QFileDialog.getExistingDirectory(self.window, "Select Video Directory", self.default_load_dir)
        if self.video_dir:
            self.frame_names = [
                f for f in os.listdir(self.video_dir) if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'))
            ]
            self.frame_names.sort(key=lambda f: int(os.path.splitext(f)[0]))

            if self.frame_names:
                self.current_frame_idx = 0
                self.current_image = cv2.imread(os.path.join(self.video_dir, self.frame_names[self.current_frame_idx]))
                self.update_display(self.current_image)

                progress = QProgressDialog("Initializing SAM2 Predictor...", None, 0, 0, self.window)
                progress.setWindowModality(Qt.WindowModal)
                progress.setWindowTitle("Loading")
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()

                def update_progress(status):
                    progress.setLabelText(status)
                    QApplication.processEvents()

                try:
                    self.sam2_predictor.initialize_predictor(self.video_dir, progress_callback=update_progress)
                except Exception as e:
                    QMessageBox.critical(self.window, "Error", f"Failed to initialize SAM2 Predictor: {str(e)}")
                    progress.close()
                    return

                progress.close()

                print(f"Loaded video frames from {self.video_dir}")
                self.masks_propagated = False
                self.first_mask_created = False
                self.ui.enable_buttons_after_video_load()
                self.input_folder_name = os.path.basename(self.video_dir)
            else:
                QMessageBox.warning(self.window, "Warning", "No frames found in the selected folder.")
        else:
            print("No folder selected.")

    def navigate_frame(self, direction):
        new_idx = self.current_frame_idx

        if direction == "left" and self.current_frame_idx > 0:
            new_idx = self.current_frame_idx - 1
        elif direction == "right" and self.current_frame_idx < len(self.frame_names) - 1:
            new_idx = self.current_frame_idx + 1

        if new_idx != self.current_frame_idx:
            if direction == "right" and self.coco_exporter:
                self.export_current_frame_to_coco()
            self.current_frame_idx = new_idx
            self.current_image = cv2.imread(os.path.join(self.video_dir, self.frame_names[self.current_frame_idx]))
            self.update_display(self.current_image)
            self.prompts = {}
            if self.current_frame_idx in self.video_segments:
                self.display_propagated_masks()
            else:
                self.masks.clear()
                self.redraw_all_masks()
    
    # Display Update
    # --------------
    def update_display(self, image):
        if image is not None:
            self.ui.mpl_widget.clear()
            self.ui.mpl_widget.show_image(image)
            self.ui.frame_info_label.setText(f'Current Frame: {self.current_frame_idx + 1} / {len(self.frame_names)}')

    # Mask Creation and Management
    # ----------------------------
    def on_click(self, event):
        if self.current_image is None or event.inaxes != self.ui.mpl_widget.ax:
            return
        
        if self.current_object_id is None:
            QMessageBox.warning(self.window, "Warning", "Please select an object to edit or add a new object.")
            return
        
        x, y = event.xdata, event.ydata
        
        if event.button == 1:
            click_type_val = 1
        elif event.button == 3:
            click_type_val = 0
        else:
            return
        
        self.update_click_prompts(self.current_object_id, x, y, click_type_val)
        self.update_mask()
        
        if not self.first_mask_created:
            self.first_mask_created = True
            self.ui.propagate_btn.setEnabled(True)

    def update_mask(self):
        if self.current_image is not None:
            self.ui.mpl_widget.clear()
            self.ui.mpl_widget.show_image(self.current_image)

            if self.current_object_id is not None:
                coords, labels = self.prompts.get(self.current_object_id, (None, None))
                if coords is not None and len(coords) > 0:
                    mask = self.create_mask(self.current_frame_idx, self.current_object_id, coords, labels)
                    self.masks[self.current_object_id] = mask
            
            for obj_id, mask in self.masks.items():
                show_mask(mask, self.ui.mpl_widget.ax, obj_id)
                show_mask_with_contours_and_bbox(mask, self.ui.mpl_widget.ax, obj_id)
                self.object_manager.update_last_valid_mask(obj_id, mask)

            if self.current_object_id in self.prompts:
                coords, labels = self.prompts[self.current_object_id]
                show_points(coords, labels, self.ui.mpl_widget.ax)

            self.ui.mpl_widget.canvas.draw()

    def create_mask(self, frame_idx, obj_id, coords, labels):
        out_mask_logits = self.sam2_predictor.generate_mask_with_points(frame_idx, obj_id, coords, labels)
        
        if len(out_mask_logits) > 0:
            return (out_mask_logits[obj_id] > 0.0).cpu().numpy()
        else:
            return np.zeros((self.sam2_predictor.inference_state['height'], self.sam2_predictor.inference_state['width']), dtype=bool)

    def update_click_prompts(self, object_id, x, y, click_type_val):
        if object_id not in self.prompts:
            self.prompts[object_id] = (np.array([[x, y]]), np.array([click_type_val]))
        else:
            coords, labels = self.prompts[object_id]
            new_coords = np.append(coords, [[x, y]], axis=0)
            new_labels = np.append(labels, [click_type_val])
            self.prompts[object_id] = (new_coords, new_labels)
    
    def redraw_all_masks(self):
        self.ui.mpl_widget.clear()
        self.ui.mpl_widget.show_image(self.current_image)
        
        for obj_id, mask in self.masks.items():
            show_mask(mask, self.ui.mpl_widget.ax, obj_id)
            show_mask_with_contours_and_bbox(mask, self.ui.mpl_widget.ax, obj_id)
        
        self.ui.mpl_widget.canvas.draw()

    # Mask Propagation
    # ----------------
    def propagate_masks(self):
        if not self.video_dir:
            QMessageBox.warning(self.window, "Warning", "Please load a video first.")
            return

        if not self.object_manager.get_all_objects():
            QMessageBox.warning(self.window, "Warning", "Please add at least one object before propagating masks.")
            return

        progress = QProgressDialog("Propagating masks...", None, 0, 0, self.window)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Processing")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        def update_progress(frame_idx):
            progress.setLabelText(f"Propagating masks... (Frame {frame_idx + 1})")
            QApplication.processEvents()

        try:
            self.video_segments = self.sam2_predictor.propagate_masks(progress_callback=update_progress)
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"An error occurred during mask propagation: {str(e)}")
            progress.close()
            return

        progress.close()

        print("Propagation completed across all frames.")
        self.masks_propagated = True
        self.ui.export_btn.setEnabled(True)
        self.ui.reset_btn.setEnabled(True)
        self.ui.add_obj_btn.setEnabled(False)
        QMessageBox.information(self.window, "Propagation Complete", "Mask propagation is complete. You can now start COCO export.")

    def display_propagated_masks(self):
        self.ui.mpl_widget.clear()
        self.ui.mpl_widget.show_image(self.current_image)
        
        if self.current_frame_idx in self.video_segments:
            for out_obj_id, out_mask in self.video_segments[self.current_frame_idx].items():
                self.masks[out_obj_id] = out_mask
                show_mask(out_mask, self.ui.mpl_widget.ax, out_obj_id)
                bbox = show_mask_with_contours_and_bbox(out_mask, self.ui.mpl_widget.ax, out_obj_id)
                if self.current_frame_idx not in self.object_bboxes:
                    self.object_bboxes[self.current_frame_idx] = {}
                self.object_bboxes[self.current_frame_idx][out_obj_id] = bbox
        
        self.ui.mpl_widget.canvas.draw()

    # Object Management
    # -----------------
    def prepare_new_object(self):
        new_obj_id = max(self.object_manager.get_all_objects().keys(), default=-1) + 1
        category_name = f"Object {new_obj_id}"
        color = get_object_color(new_obj_id)
        self.object_manager.add_object(new_obj_id, category_name, color)
        self.ui.update_table()
        self.current_object_id = new_obj_id
        print(f"Prepared new object with ID {new_obj_id}")

    def on_category_name_change(self, item):
        if item.column() == 1:  # Category name column
            new_name = item.text()
            obj_id = list(self.object_manager.get_all_objects().keys())[item.row()]
            self.object_manager.update_category_name(obj_id, new_name)
            print(f"Category {obj_id} renamed to: {new_name}")

    def on_resegment_checked(self, state, obj_id):
        if state == Qt.Checked:
            for row in range(self.ui.table.rowCount()):
                checkbox_widget = self.ui.table.cellWidget(row, 0)
                if checkbox_widget.isChecked() and list(self.object_manager.get_all_objects().keys())[row] != obj_id:
                    checkbox_widget.setChecked(False)
            
            self.current_object_id = obj_id
            QMessageBox.information(self.window, "Object Selected", f"You can now edit Object {obj_id}")
        else:
            self.current_object_id = None

    # COCO Export
    # -----------
    def initialize_coco_export(self):
        if not self.video_dir:
            QMessageBox.warning(self.window, "Warning", "Please load a video first.")
            return

        if not self.masks_propagated:
            QMessageBox.warning(self.window, "Warning", "Please propagate masks before starting COCO export.")
            return

        if self.input_folder_name is None:
            QMessageBox.warning(self.window, "Warning", "Input folder name not recorded. Please reload the video.")
            return

        if self.coco_export_file == None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.coco_export_file = os.path.join(
                self.default_export_dir,
                f"coco_export_{self.input_folder_name}_{timestamp}.json"
            )

        use_existing = False
        if os.path.exists(self.coco_export_file):
            reply = QMessageBox.question(self.window, 'File Exists', 
                                         'An export file already exists. Do you want to update it?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                use_existing = True
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.coco_export_file = os.path.join(
                    self.default_export_dir,
                    f"coco_export_{self.input_folder_name}_{timestamp}.json"
                )

        self.coco_exporter = COCOExporter(self.coco_export_file, use_existing)
        categories = [{"id": obj_id, "name": obj_data['category_name']} 
                      for obj_id, obj_data in self.object_manager.get_all_objects().items()]
        self.coco_exporter.initialize_categories(categories)
        QMessageBox.information(self.window, "COCO Export", f"COCO export initialized.\nData will be {'updated' if use_existing else 'written'} to {self.coco_export_file}")

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

    # State Management
    # ----------------
    def reset_inference_state(self):
        current_masks = self.masks.copy()

        self.sam2_predictor.reset_state()
        self.video_segments = {}
        self.masks_propagated = False
        
        self.reinitialize_masks(current_masks)
        
        self.update_display(self.current_image)
        self.ui.update_table()

        self.redraw_all_masks()
        
        self.ui.export_btn.setEnabled(False)
        self.ui.reset_btn.setEnabled(False)
        self.ui.add_obj_btn.setEnabled(True)

        QMessageBox.information(self.window, "Reset Complete", "Inference state has been reset.\nExisting masks are preserved. You can now edit objects or add new ones.")

    def reinitialize_masks(self, current_masks):
        for obj_id, mask in current_masks.items():
            bbox = self.object_bboxes.get(self.current_frame_idx, {}).get(obj_id)
            if bbox is not None:
                new_mask = self.sam2_predictor.generate_mask_with_box(self.current_frame_idx, obj_id, bbox)
                self.masks[obj_id] = new_mask
            else:
                self.masks[obj_id] = mask.cpu().numpy() if torch.is_tensor(mask) else mask

def run_interface():
    app = QApplication(sys.argv)
    interface = SAM2Interface()
    interface.run()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_interface()