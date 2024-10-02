from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLayout,
                             QCheckBox, QStyledItemDelegate)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

def create_button(text, callback):
    button = QPushButton(text)
    button.clicked.connect(callback)
    return button

def create_vertical_layout(*items):
    layout = QVBoxLayout()
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        elif isinstance(item, QLayout):
            layout.addLayout(item)
    return layout

def create_horizontal_layout(*items):
    layout = QHBoxLayout()
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        elif isinstance(item, QLayout):
            layout.addLayout(item)
    return layout

def get_object_color(obj_id):
    return QColor(*[int(c * 255) for c in plt.cm.tab10(obj_id % 10)[:3]])

class CenteredCheckBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def isChecked(self):
        return self.checkbox.isChecked()

    def setChecked(self, state):
        self.checkbox.setChecked(state)

class AlignDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)

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