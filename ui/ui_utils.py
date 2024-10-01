from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLayout
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt

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