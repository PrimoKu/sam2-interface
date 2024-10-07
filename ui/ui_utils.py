from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLayout,
                             QCheckBox, QStyledItemDelegate)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal
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
    return QColor(*[int(c * 255) for c in plt.cm.tab20(obj_id % 20)[:3]])

class CenteredCheckBox(QWidget):
    stateChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.checkbox.stateChanged.connect(self.stateChanged.emit)

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
        self.ax.set_position([0, 0, 1, 1])  # Make axes occupy the entire figure
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.setMinimumSize(1600, 900)  # Set fixed size for the widget

    def clear(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw()

    def show_image(self, image):
        self.clear()
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.set_xlim(0, image.shape[1])
        self.ax.set_ylim(image.shape[0], 0)  # Invert y-axis for correct image orientation
        self.canvas.draw()

    def resizeEvent(self, event):
        width = max(event.size().width(), 1600)
        height = max(int(width * 9 / 16), 900)
        self.setMinimumSize(width, height)
        super().resizeEvent(event)