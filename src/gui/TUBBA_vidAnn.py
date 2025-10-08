import os
import cv2
import numpy as np
import csv
import pandas as pd
from matplotlib import cm
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,QMessageBox, QInputDialog,
                             QPushButton, QSlider, QFileDialog, QListWidget,QMenu, QAction,
                             QGroupBox, QComboBox,QSizePolicy, QShortcut, QCheckBox, QMainWindow)
from PyQt5.QtGui import QPixmap, QImage, QTransform, QPainter, QColor, QPen, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
import random
import h5py
import traceback
import seaborn as sns
import importlib.util
import av
import platform


class ZoomableVideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.pixmap_original = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = None

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap
        self.update_view()

    def update_view(self):
        """Redraw the image with current scale and offset."""
        if self.pixmap_original is None:
            return

        canvas = QPixmap(self.size())
        canvas.fill(Qt.black)

        scaled = self.pixmap_original.scaled(
            self.pixmap_original.size() * self.scale,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        painter = QPainter(canvas)
        painter.drawPixmap(self.offset, scaled)
        painter.end()

        super().setPixmap(canvas)

    def wheelEvent(self, event):
        if self.pixmap_original is None:
            return

        mouse_pos = event.pos()

        # Coordinates relative to scaled image
        old_img_x = (mouse_pos.x() - self.offset.x()) / self.scale
        old_img_y = (mouse_pos.y() - self.offset.y()) / self.scale

        # Zoom
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1 / 1.1
        new_scale = self.scale * zoom_factor
        new_scale = max(0.5, min(new_scale, 10.0))  # Restrict between 0.5x and 10x

        # Update offset to keep point under mouse fixed
        if new_scale != self.scale:
            self.scale = new_scale
            new_img_x = old_img_x * self.scale
            new_img_y = old_img_y * self.scale
            self.offset = QPoint(
                int(mouse_pos.x() - new_img_x),
                int(mouse_pos.y() - new_img_y)
            )
            self.clamp_offset()

        self.update_view()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos and self.scale > 0.5:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.clamp_offset()
            self.update_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event):
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.update_view()

    def clamp_offset(self):
        """Ensure that we cannot drag the image outside the widget area."""
        if self.pixmap_original is None:
            return

        scaled_size = self.pixmap_original.size() * self.scale

        if scaled_size.width() <= self.width():
            # Center horizontally
            self.offset.setX((self.width() - scaled_size.width()) // 2)
        else:
            # Clamp x offset
            min_x = self.width() - scaled_size.width()
            max_x = 0
            self.offset.setX(min(max(self.offset.x(), min_x), max_x))

        if scaled_size.height() <= self.height():
            # Center vertically
            self.offset.setY((self.height() - scaled_size.height()) // 2)
        else:
            # Clamp y offset
            min_y = self.height() - scaled_size.height()
            max_y = 0
            self.offset.setY(min(max(self.offset.y(), min_y), max_y))

class BehaviorPanel(QWidget):
    def __init__(self, behaviors, parent=None):
        super().__init__(parent)
        self.initUI(behaviors)

    def initUI(self, behaviors):
        layout = QVBoxLayout()
        self.buttons = []
        self.behavior_colors = {}

        button_style = """
                                        QPushButton {
                                            background-color: black;
                                            color: white;
                                            border: 1px solid gray;
                                            border-style: outset;
                                            border-radius: 4px;
                                            padding: 4px;
                                            font: bold 12px;
                                        }
                                        QPushButton:pressed {
                                            border-style: inset;
                                        }
                                    """

        palette = sns.color_palette('hls', n_colors=len(behaviors))
        for i, behavior in enumerate(behaviors):
            rgb = palette[i]
            self.behavior_colors[behavior] = tuple(int(c * 255) for c in rgb)

            row_layout = QHBoxLayout()

            for label in [behavior, f"NOT {behavior}"]:
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setStyleSheet(button_style)
                btn.clicked.connect(self.handle_button_click)
                btn.behavior_label = label
                row_layout.addWidget(btn)
                self.buttons.append(btn)

            layout.addLayout(row_layout)

        layout.addStretch()
        self.setLayout(layout)

    def handle_button_click(self):

        button_style = """
                                        QPushButton {
                                            background-color: black;
                                            color: white;
                                            border: 1px solid gray;
                                            border-style: outset;
                                            border-radius: 4px;
                                            padding: 4px;
                                            font: bold 12px;
                                        }
                                        QPushButton:pressed {
                                            border-style: inset;
                                        }
                                    """

        sender = self.sender()
        for btn in self.buttons:
            if btn is not sender:
                btn.setChecked(False)
                btn.setStyleSheet(button_style)

        behavior_name = sender.behavior_label.replace("NOT ", "")
        if sender.isChecked():
            if sender.behavior_label.startswith("NOT"):
                sender.setStyleSheet("background-color: #808080; color: black; border: 1px solid gray; border-style: outset;"
                                    "border-radius: 4px; padding: 4px; font: bold 12px;")
            else:
                rgb = self.behavior_colors[behavior_name]
                sender.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: black; border: 1px solid gray; border-style: outset;"
                                    "border-radius: 4px; padding: 4px; font: bold 12px;")
        else:
            sender.setStyleSheet(button_style)

    def get_selected_behavior(self):
        for btn in self.buttons:
            if btn.isChecked():
                return btn.behavior_label
        return None

class TimelineCanvas(QWidget):
    def __init__(self, annotator, parent=None):
        super().__init__(parent)
        self.annotator = annotator

        self.setMinimumHeight(150)
        self.setMouseTracking(True)

        self.is_scrubbing = False
        self.last_scrub_pos = None
        self.scrub_fractional_frames = 0
        self.last_scrub_dir = 0

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_context_menu)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        video_info = self.annotator.project['videos'][self.annotator.current_video_idx]
        total_frames = video_info['nFrames']
        annotations = video_info.get('annotations', {})
        behaviors = self.annotator.project['behaviors']
        fps = self.annotator.fps

        width = self.width()
        height = self.height()
        row_height = height / max(len(behaviors), 1)

        center_frame = self.annotator.current_frame_idx
        frames_around = int(fps * 10)  # show ±10 seconds
        left_edge = max(0, center_frame - frames_around)
        right_edge = min(total_frames, center_frame + frames_around)
        frames_shown = right_edge - left_edge

        ### Draw time grid lines every 1 second
        painter.setPen(QPen(QColor(200, 200, 200, 100), 1))  # Light gray, semi-transparent
        start_sec = int(left_edge / fps)
        end_sec = int(right_edge / fps)

        for sec in range(start_sec, end_sec + 1):
            frame_num = sec * fps
            x_pos = int((frame_num - left_edge) / frames_shown * width)
            painter.drawLine(x_pos, 0, x_pos, height)

        ### Draw behavior bars or inference confidences
        for idx, behavior in enumerate(behaviors):
            behavior_y_lo = idx * row_height
            behavior_y_hi = (idx + 1) * row_height

            color_rgb = self.annotator.behavior_panel.behavior_colors.get(behavior, (255, 255, 255))

            if self.annotator.display_inference and self.annotator.current_inference and self.annotator.conf_array is not None:
                conf_slice = self.annotator.conf_array[idx, left_edge:right_edge]
                h, w = 1, len(conf_slice)
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for ch in range(3):
                    rgb[:, :, ch] = int(color_rgb[ch]) * conf_slice[None, :]
                qimage = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                painter.drawPixmap(QRect(0, int(behavior_y_lo), width, int(row_height)), pixmap)

            else:
                # Draw manual annotations as intervals
                behavior_data = annotations.get(behavior, [])
                for interval in behavior_data:
                    startF, endF, val = interval
                    if endF == 0 or endF < startF:
                        endF = center_frame

                    if endF < left_edge or startF > right_edge:
                        continue

                    # Clip interval inside visible region
                    startF = max(startF, left_edge)
                    endF = min(endF, right_edge)

                    x_start = int((startF - left_edge) / frames_shown * width)
                    x_end = int((endF - left_edge) / frames_shown * width)

                    interval_color = color_rgb if val == 1 else (128, 128, 128)  # Gray if NOT behavior
                    painter.fillRect(x_start, int(behavior_y_lo), x_end - x_start, int(row_height), QColor(*interval_color))

        ### Add behavior names to the canvas
        painter.setPen(Qt.white)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        for idx, behavior in enumerate(behaviors):
            behavior_y_lo = idx * row_height
            painter.drawText(5, int(behavior_y_lo + row_height / 2 + 4), behavior)

        ### Draw vertical line for current frame
        if self.is_scrubbing:
            painter.setPen(QPen(QColor(255, 50, 50, 255), 2))  # Brighter red while scrubbing
        else:
            painter.setPen(QPen(QColor(255, 100, 100, 180), 2))  # Normal soft red

        x_cursor = int((center_frame - left_edge) / frames_shown * width)
        painter.drawLine(x_cursor, 0, x_cursor, height)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x_click = event.x()

            video_info = self.annotator.project['videos'][self.annotator.current_video_idx]
            total_frames = video_info['nFrames']
            fps = self.annotator.fps

            width = self.width()
            center_frame = self.annotator.current_frame_idx
            frames_around = int(fps * 10)
            left_edge = max(0, center_frame - frames_around)
            right_edge = min(total_frames, center_frame + frames_around)
            frames_shown = right_edge - left_edge

            x_cursor = int((center_frame - left_edge) / frames_shown * width)

            # Allow small range around red line
            tolerance = 10  # pixels
            if abs(x_click - x_cursor) < tolerance:
                # Start scrubbing
                self.is_scrubbing = True
                self.last_scrub_pos = x_click
                self.scrub_fractional_frames = 0
                self.last_scrub_dir = 0
                self.setCursor(Qt.ClosedHandCursor)
            else:
                # Jump normally if clicking elsewhere
                clicked_frame = int(left_edge + (x_click / width) * frames_shown)
                clicked_frame = max(0, min(total_frames - 1, clicked_frame))

                self.annotator.current_frame_idx = clicked_frame
                self.annotator.show_frame(clicked_frame)

    def mouseMoveEvent(self, event):
        x = event.x()

        video_info = self.annotator.project['videos'][self.annotator.current_video_idx]
        total_frames = video_info['nFrames']
        fps = self.annotator.fps

        width = self.width()
        center_frame = self.annotator.current_frame_idx
        frames_around = int(fps * 10)
        left_edge = max(0, center_frame - frames_around)
        right_edge = min(total_frames, center_frame + frames_around)
        frames_shown = right_edge - left_edge

        x_cursor = int((center_frame - left_edge) / frames_shown * width)

        # Adjust cursor depending on proximity
        tolerance = 10  # pixels
        if self.is_scrubbing:
            self.setCursor(Qt.ClosedHandCursor)
        elif abs(x - x_cursor) < tolerance:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)

        # Handle scrubbing if active
        if self.is_scrubbing:
            deltaX = (x - self.last_scrub_pos) * 0.5
            if deltaX < 1:
                deltaX = deltaX / 2

            if np.sign(deltaX) != self.last_scrub_dir:
                self.scrub_fractional_frames = 0

            currFrame = self.annotator.current_frame_idx
            newFrame = currFrame + deltaX + self.scrub_fractional_frames
            newFrame = max(0, min(total_frames - 1, newFrame))

            if np.sign(deltaX) > 0:
                self.scrub_fractional_frames = newFrame % 1
            else:
                self.scrub_fractional_frames = -1 + (newFrame % 1)

            self.annotator.current_frame_idx = int(round(newFrame))
            self.annotator.show_frame(self.annotator.current_frame_idx)

            self.last_scrub_pos = x
            self.last_scrub_dir = np.sign(deltaX)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_scrubbing = False
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def open_context_menu(self, position):
        x_click = position.x()
        y_click = position.y()

        # Get dimensions and frames displayed
        video_info = self.annotator.project['videos'][self.annotator.current_video_idx]
        total_frames = video_info['nFrames']
        fps = self.annotator.fps
        width = self.width()
        height = self.height()
        behaviors = self.annotator.project['behaviors']
        row_height = height / len(behaviors)

        center_frame = self.annotator.current_frame_idx
        frames_around = int(fps * 10)
        left_edge = max(0, center_frame - frames_around)
        right_edge = min(total_frames, center_frame + frames_around)
        frames_shown = right_edge - left_edge

        clicked_frame = int(left_edge + (x_click / width) * frames_shown)
        behavior_idx = int(y_click / row_height)

        if behavior_idx < 0 or behavior_idx >= len(behaviors):
            return  # Clicked outside behavior rows

        clicked_behavior = behaviors[behavior_idx]

        annotations = video_info.get('annotations', {}).get(clicked_behavior, [])

        # Find if we clicked inside an annotation segment
        clicked_interval = None
        for interval in annotations:
            startF, endF, val = interval
            if endF == 0 or endF < startF:
                endF = center_frame  # Open interval handling

            if startF <= clicked_frame <= endF:
                clicked_interval = interval
                break

        if clicked_interval is None:
            return  # Not clicked on any annotation

        # Context menu setup
        context_menu = QMenu(self)

        delete_action = QAction("🗑️ Delete Annotation", self)
        context_menu.addAction(delete_action)

        action = context_menu.exec_(self.mapToGlobal(position))

        if action == delete_action:
            self.delete_annotation(clicked_behavior, clicked_interval)

    def delete_annotation(self, behavior, interval):
        annotations = self.annotator.project['videos'][self.annotator.current_video_idx]['annotations']

        if behavior in annotations and interval in annotations[behavior]:
            annotations[behavior].remove(interval)
            self.update()
            print(f"🗑️ Deleted annotation: {behavior}, interval {interval}")
        else:
            QMessageBox.warning(self, "Deletion Error",
                                "Failed to delete annotation. It may already have been removed.")

class VideoAnnotator(QMainWindow):
    def __init__(self, project):
        super().__init__()

        self.project = project  # Project dictionary
        self.current_video_idx = 0  # Which video are we looking at?
        self.current_frame_idx = 0
        self.annotation_mode = False  # True when annotating (after pressing 's')
        self.annotation_start_frame = None
        self.project_path = project['project_path']
        self.currentModelType = 'full'  # Default model type
        self.availModels = ['full', 'lite']  # Available model types

        self.display_inference = False
        self.current_inference = None

        # Detect OS and set video backend
        self.os_type = platform.system()
        self.use_pyav = self.os_type in ['Windows', 'Linux']

        # Set modifier key based on platform
        if self.os_type == 'Darwin':  # macOS
            self.modifier_key = Qt.MetaModifier
            self.modifier_name = "Cmd"
        else:  # Windows, Linux
            self.modifier_key = Qt.ControlModifier
            self.modifier_name = "Ctrl"

        if self.use_pyav:
            print(f"🎥 Using PyAV backend for {self.os_type}")
        else:
            print(f"🎥 Using OpenCV backend for {self.os_type}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # --- Shortcuts ---
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_project_feedback)

        save_as_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        save_as_shortcut.activated.connect(self.save_project_as)

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_last_annotation)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("TUBBA Video Annotator")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet("background-color: black; color: white;")

        main_layout = QHBoxLayout()

        button_style = """
                        QPushButton {
                            background-color: black;
                            color: white;
                            border: 1px solid gray;
                            border-style: outset;
                            border-radius: 4px;
                            padding: 4px;
                            font: bold 12px;
                        }
                        QPushButton:pressed {
                            border-style: inset;
                        }
                    """

        # Start with left side of the gui
        left_panel = QVBoxLayout()

        # Create behavior panel for annotation controls
        behavior_group = QGroupBox("Behaviors")
        behavior_group.setStyleSheet("QGroupBox {border: 1px solid lightgray; margin-top: 2ex;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center;}")
        self.behavior_layout = QVBoxLayout()

        self.behavior_panel = BehaviorPanel(self.project['behaviors'])
        self.behavior_layout.addWidget(self.behavior_panel)

        self.start_button = QPushButton("Start Annotation (s)")
        self.start_button.clicked.connect(self.start_annotation)
        self.start_button.setStyleSheet(button_style)
        self.behavior_layout.addWidget(self.start_button)

        self.end_button = QPushButton("End Annotation (e)")
        self.end_button.clicked.connect(self.end_annotation)
        self.end_button.setStyleSheet(button_style)
        self.behavior_layout.addWidget(self.end_button)

        behavior_group.setLayout(self.behavior_layout)
        behavior_group.setMaximumWidth(250)
        behavior_group.setMinimumHeight(500)
        behavior_group.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        left_panel.addWidget(behavior_group)

        # Move to create a controls panel under the behavior panel
        video_controls_group = QGroupBox("Controls")
        video_controls_group.setStyleSheet("QGroupBox {border: 1px solid lightgray; margin-top: 2ex;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center;}")
        controls_layout = QVBoxLayout()

        video_select_label = QLabel("Select Video:")
        video_select_label.setStyleSheet("color: white; font-weight: bold;")
        self.video_selector = QComboBox()
        self.video_selector.addItems([vid['name'] for vid in self.project['videos']])
        self.video_selector.currentIndexChanged.connect(self.switch_video)

        controls_layout.addWidget(video_select_label)
        controls_layout.addWidget(self.video_selector)

        model_select_label = QLabel("Select Model:")
        model_select_label.setStyleSheet("color: white; font-weight: bold;")
        self.model_selector = QComboBox()
        self.model_selector.addItems([model for model in self.availModels])
        self.model_selector.currentIndexChanged.connect(self.set_model)

        controls_layout.addWidget(model_select_label)
        controls_layout.addWidget(self.model_selector)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play)
        self.play_pause_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.play_pause_button)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        self.save_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.save_button)

        # Add train and infer buttons
        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train_TUBBAmodel)
        self.train_button.setStyleSheet("QPushButton {background-color: #17FFD2; color: black; border: 1px solid gray; border-style: outset;"
                                        "border-radius: 4px; padding: 4px; font: bold 12px;} QPushButton:pressed {border-style: inset}")
        controls_layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Run Inference on Video")
        self.predict_button.clicked.connect(self.run_model_inference)
        self.predict_button.setStyleSheet(
            "QPushButton {background-color: #157FFF; color: black; border: 1px solid gray; border-style: outset;"
            "border-radius: 4px; padding: 4px; font: bold 12px;} QPushButton:pressed {border-style: inset}")
        controls_layout.addWidget(self.predict_button)

        self.batch_predict_button = QPushButton("Run Inference on Project")
        self.batch_predict_button.clicked.connect(self.batch_run_model_inference)
        self.batch_predict_button.setStyleSheet(
            "QPushButton {background-color: #9BFF1A; color: black; border: 1px solid gray; border-style: outset;"
            "border-radius: 4px; padding: 4px; font: bold 12px;} QPushButton:pressed {border-style: inset}")
        controls_layout.addWidget(self.batch_predict_button)

        # Inference display toggle
        self.inference_toggle = QCheckBox("Show Inferred Behaviors")
        self.inference_toggle.setChecked(self.display_inference)
        self.inference_toggle.setStyleSheet("color: white; font-weight: bold;")
        self.inference_toggle.stateChanged.connect(self.handle_inference_toggle)
        controls_layout.addWidget(self.inference_toggle)

        controls_layout.addStretch()
        video_controls_group.setLayout(controls_layout)
        video_controls_group.setMaximumWidth(250)
        left_panel.addWidget(video_controls_group)

        # Now move to the right side of the gui, which will contain the video and timeline
        right_panel = QVBoxLayout()

        video_group = QGroupBox("Video")
        video_group.setStyleSheet("QGroupBox {border: 1px solid lightgray; margin-top: 2ex;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center;}")
        video_layout = QVBoxLayout()

        self.video_label = ZoomableVideoLabel()
        video_layout.addWidget(self.video_label)

        slider_style = """
                        QSlider::groove:horizontal {
                            border: 1px solid gray;
                            height: 10px;
                        }
                        QSlider::handle:horizontal {
                            background: #fff;
                            width: 10px;
                            margin: -1px 1px;
                            border-radius: 5px;
                            border: 1px solid #5555ff;
                        }
                        QSlider::handle:horizontal:hover {
                            background: #000;
                        }
                        QSlider::add-page:horizontal {
                            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #B1B1B1, stop: 1 #c4c4c4);
                        }
                        QSlider::sub-page:horizontal {
                            background: #5555ff;
                        }
                    """

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        self.slider.setStyleSheet(slider_style)
        video_layout.addWidget(self.slider)

        video_group.setLayout(video_layout)
        video_group.setMinimumHeight(550)
        video_group.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        right_panel.addWidget(video_group)

        timeline_group = QGroupBox("Ethogram")
        timeline_group.setStyleSheet("QGroupBox {border: 1px solid lightgray; margin-top: 2ex;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center;}")
        timeline_layout = QVBoxLayout()

        self.timeline = TimelineCanvas(self)
        timeline_layout.addWidget(self.timeline)

        timeline_group.setLayout(timeline_layout)
        timeline_group.setFixedHeight(200)
        right_panel.addWidget(timeline_group)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 4)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Add menubar section here ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Create actions and connect them
        save_as_act = QAction('Save Project As...', self)
        add_behavior_act = QAction('Add Behavior...', self)
        remove_behavior_act = QAction('Remove Behavior...', self)
        add_video_act = QAction('Add Video...', self)
        remove_video_act = QAction('Remove Video...', self)
        import_ann_act = QAction('Import Video Annotations', self)
        export_ann_act = QAction('Export Video Annotations to CSV', self)
        export_ann_batch_act = QAction('Export ALL Video Annotations', self)
        shortcutHelp_act = QAction('Shortcuts...', self)

        save_as_act.triggered.connect(self.save_project_as)
        add_behavior_act.triggered.connect(self.add_behavior)
        remove_behavior_act.triggered.connect(self.remove_behavior)
        add_video_act.triggered.connect(self.add_video)
        remove_video_act.triggered.connect(self.remove_video)
        import_ann_act.triggered.connect(self.import_annotations)
        export_ann_act.triggered.connect(self.export_annotations)
        export_ann_batch_act.triggered.connect(self.batch_export_annotations)
        shortcutHelp_act.triggered.connect(self.show_shortcuts)

        file_menu.addAction(save_as_act)
        file_menu.addSeparator()
        file_menu.addAction(add_behavior_act)
        file_menu.addAction(remove_behavior_act)
        file_menu.addSeparator()
        file_menu.addAction(add_video_act)
        file_menu.addAction(remove_video_act)
        file_menu.addSeparator()
        file_menu.addAction(import_ann_act)
        file_menu.addAction(export_ann_act)
        file_menu.addAction(export_ann_batch_act)
        file_menu.addSeparator()
        file_menu.addAction(shortcutHelp_act)

        self.load_video(self.project['videos'][self.current_video_idx]['folder'])

    def load_video(self, folder):
        mp4s = [f for f in os.listdir(folder) if f.endswith('.mp4')]
        if len(mp4s) == 0:
            print(f"⚠️ No video found in {folder}")
            return

        video_path = os.path.join(folder, mp4s[0])

        if self.use_pyav:
            # PyAV backend (Windows/Linux)
            self.container = av.open(video_path)
            self.video_stream = self.container.streams.video[0]

            self.total_frames = self.video_stream.frames
            self.fps = float(self.video_stream.average_rate)

            # Initialize frame iterator for smooth playback
            self.frame_iterator = None
            self.last_decoded_frame = None
        else:
            # OpenCV backend (macOS)
            self.cap = cv2.VideoCapture(video_path)

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.slider.setMaximum(self.total_frames - 1)
        print(f"🔊 Video info: {self.total_frames} frames at {self.fps} fps")

        self.show_frame(0)
        self.current_frame_idx = 0

    def show_frame(self, idx):
        if self.use_pyav:
            self._show_frame_pyav(idx)
        else:
            self._show_frame_opencv(idx)

    def _show_frame_pyav(self, idx):
        """PyAV implementation for Windows/Linux"""
        if not hasattr(self, 'container') or self.container is None:
            return

        try:
            # If jumping to a different frame (not sequential), reset iterator
            if (self.frame_iterator is None or
                    abs(idx - self.current_frame_idx) > 1):
                # Seek to nearest keyframe before target
                timestamp = int(idx / self.fps * av.time_base)
                self.container.seek(timestamp, stream=self.video_stream, backward=True)
                self.frame_iterator = self.container.decode(video=0)

            # Decode frames until we reach the target
            for frame in self.frame_iterator:
                if frame.pts * self.video_stream.time_base * self.fps >= idx:
                    # Convert to numpy array
                    img = frame.to_ndarray(format='rgb24')

                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(qt_img)

                    self.video_label.setPixmap(pixmap)
                    self.slider.blockSignals(True)
                    self.slider.setValue(idx)
                    self.slider.blockSignals(False)
                    self.timeline.update()

                    self.last_decoded_frame = idx
                    break

        except Exception as e:
            print(f"⚠️ Error showing frame {idx}: {e}")
            # Reset iterator on error
            self.frame_iterator = None

    def _show_frame_opencv(self, idx):
        """OpenCV implementation for macOS"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.video_label.setPixmap(pixmap)
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.timeline.update()

    def switch_video(self, idx):
        # Close old video resources
        if self.use_pyav:
            if hasattr(self, 'container') and self.container is not None:
                self.container.close()
                self.frame_iterator = None
        else:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

        self.current_video_idx = idx
        self.load_video(self.project['videos'][idx]['folder'])

        # Try to load inference from file if path exists
        video = self.project['videos'][idx]
        if 'inferred_path' in video:
            import pickle
            try:
                abs_path = os.path.join(os.path.dirname(self.project_path), video['inferred_path'])
                with open(abs_path, 'rb') as f:
                    self.current_inference = pickle.load(f)
                self.display_inference = True
                print(f"🧠 Loaded inference for {video['name']}")
                self.update_confidence_pixmap()
            except Exception as e:
                print(f"⚠️ Failed to load inference for {video['name']}: {e}")
                self.current_inference = None
                self.display_inference = False
        else:
            self.current_inference = None
            self.display_inference = False

        self.inference_toggle.setChecked(self.display_inference)

    def set_model(self, idx):
        self.currentModelType = self.availModels[idx]

    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.show_frame(self.current_frame_idx)
        else:
            self.timer.stop()

    def previous_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.show_frame(self.current_frame_idx)
        else:
            self.timer.stop()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            interval = int(1000 / self.fps)
            self.timer.start(interval)
            self.play_pause_button.setText("Pause")

    def start_annotation(self):
        selected_behavior = self.behavior_panel.get_selected_behavior()
        if not selected_behavior:
            print("⚠️ No behavior selected!")
            return

        if self.annotation_start_frame is not None:
            QMessageBox.warning(self, "Warning", "You must end the current annotation before starting a new one!")
            return

        is_not_behavior = selected_behavior.startswith("NOT")
        behavior_name = selected_behavior.replace("NOT ", "")
        val = -1 if is_not_behavior else 1  # <-- you forgot to define val!

        video = self.project['videos'][self.current_video_idx]
        if 'annotations' not in video:
            video['annotations'] = {}
        if behavior_name not in video['annotations']:
            video['annotations'][behavior_name] = []

        # Append a new open interval [start_frame, 0, val]
        video['annotations'][behavior_name].append([self.current_frame_idx, 0, val])

        # Store active annotation metadata
        self.annotation_start_frame = self.current_frame_idx
        self.annotation_behavior = behavior_name
        self.annotation_val = val

        print(f"📝 Start {selected_behavior} at frame {self.current_frame_idx}")

        self.timeline.update()

    def end_annotation(self):
        if self.annotation_start_frame is None:
            print("⚠️ No annotation started!")
            return

        behavior_name = self.annotation_behavior
        video = self.project['videos'][self.current_video_idx]
        intervals = video['annotations'].get(behavior_name, [])

        if not intervals:
            print("⚠️ No open interval to end!")
            return

        # Find the last open interval and close it
        last_interval = intervals[-1]
        if last_interval[1] == 0:
            startF = last_interval[0]
            endF = self.current_frame_idx

            # Ensure proper order
            if startF > endF:
                startF, endF = endF, startF

            last_interval[0] = startF
            last_interval[1] = endF
        else:
            print("⚠️ Last interval already closed!")

        self.annotation_start_frame = None
        self.annotation_behavior = None
        self.annotation_val = None

        print(f"📝 End {behavior_name} at frame {self.current_frame_idx}")

        self.timeline.update()

    def undo_last_annotation(self):
        selected_behavior = self.behavior_panel.get_selected_behavior()
        if not selected_behavior:
            print("⚠️ No behavior selected!")
            return

        behavior_name = selected_behavior.replace("NOT ", "")

        video = self.project['videos'][self.current_video_idx]
        if 'annotations' not in video or behavior_name not in video['annotations']:
            print("⚠️ No annotations for this behavior.")
            return

        if len(video['annotations'][behavior_name]) == 0:
            print("⚠️ No bouts to undo.")
            return

        # Remove the last bout
        zapped = video['annotations'][behavior_name].pop()
        print(f"↩️ Removed last bout: {zapped}")

        # Refresh ethogram
        self.timeline.update()

    def slider_moved(self):
        self.current_frame_idx = self.slider.value()
        self.show_frame(self.current_frame_idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.start_annotation()
        elif event.key() == Qt.Key_E:
            self.end_annotation()
        elif event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:
            self.previous_frame()
        elif event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_S and (event.modifiers() & self.modifier_key):
            self.save_project()
        elif event.key() == Qt.Key_Z and (event.modifiers() & self.modifier_key):
            self.undo_last_annotation()
        else:
            super().keyPressEvent(event)

    def show_shortcuts(self):
        shortcuts = (
            "<b>Available Keybind Shortcuts:</b><br><br>"
            "<b>S</b>: Start annotation<br>"
            "<b>E</b>: End annotation<br>"
            "<b>Space</b>: Play/Pause video<br>"
            "<b>Left Arrow</b>: Previous frame<br>"
            "<b>Right Arrow</b>: Next frame<br>"
            f"<b>{self.modifier_name} + S</b>: Save project<br>"
            f"<b>{self.modifier_name} + Z</b>: Undo last annotation"
        )

        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def save_project(self):
        import json
        from PyQt5.QtWidgets import QFileDialog

        if not hasattr(self, 'project_path') or self.project_path is None:
            # If no original path, ask the user
            print("💬 No project path set. Asking user where to save...")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project As...",
                os.path.expanduser("~/TUBBA_project.json"),
                "JSON Files (*.json)"
            )
            if not path:
                print("⚠️ Save cancelled.")
                return
            self.project_path = path  # Store new path for future saves!

        print(f"💾 Saving project to {self.project_path}...")

        try:
            with open(self.project_path, 'w') as f:
                json.dump(self.project, f, indent=4)
            print("✅ Project saved successfully!")
        except Exception as e:
            print(f"❌ Error saving project: {e}")

    def save_project_feedback(self):
        self.save_project()

        # Blink the save button
        orig_style = self.save_button.styleSheet()
        self.save_button.setStyleSheet("background-color: green; color: black; border: 1px solid white; padding: 4px;")
        QTimer.singleShot(300, lambda: self.save_button.setStyleSheet(orig_style))

    def save_project_as(self):
        from PyQt5.QtWidgets import QFileDialog
        import json

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As...",
            os.path.expanduser("~/TUBBA_project.json"),
            "JSON Files (*.json)"
        )
        if not path:
            print("⚠️ Save As cancelled.")
            return

        self.project_path = path
        print(f"💾 New project path: {self.project_path}")

        # After setting new path, save
        try:
            with open(self.project_path, 'w') as f:
                json.dump(self.project, f, indent=4)
            print("✅ Project saved successfully!")
        except Exception as e:
            print(f"❌ Error saving project: {e}")

    def add_behavior(self):
        from PyQt5.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(self, "Add Behavior", "Enter new behavior name:")
        if ok and text:
            behavior = text.strip().capitalize()
            if behavior not in self.project['behaviors']:
                self.project['behaviors'].append(behavior)
                print(f"➕ Added behavior: {behavior}")

                # Remove old panel
                self.behavior_layout.removeWidget(self.behavior_panel)
                self.behavior_panel.deleteLater()

                # Add new updated panel
                self.behavior_panel = BehaviorPanel(self.project['behaviors'])
                self.behavior_layout.insertWidget(0, self.behavior_panel)

    def add_video(self):
        from PyQt5.QtWidgets import QFileDialog
        import os
        import glob

        folder = QFileDialog.getExistingDirectory(self, "Select Folder for New Video")
        if not folder:
            print("⚠️ No folder selected.")
            return None

        print(f"📂 Selected folder: {folder}")

        # Try to preprocess it
        try:
            # Check feature extraction algorithm used in project creation
            selected_script = self.project['featureExtractionAlgorithm']
            if not selected_script:
                return None

            # Build the full path to the script
            script_dir = os.path.join(os.path.dirname(__file__), '..', 'features')
            script_path = os.path.join(script_dir, selected_script)

            # Import the module dynamically
            spec = importlib.util.spec_from_file_location("feature_module", script_path)
            feature_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(feature_module)

            # assume same sampling rate as previous videos
            spatialSR = self.project['videos'][self.current_video_idx]['samplingRate']
            print("Starting preprocessing...")
            vidInfo = feature_module.tracksToFeatures(folder, spatialSR)  # Assume no downsampling for new
            print("✅ Preprocessing successful.")

            # Determine the video name
            video_files = glob.glob(os.path.join(folder, '*.mp4'))
            if video_files:
                video_name = os.path.basename(video_files[0])
            else:
                video_name = "Unknown.mp4"

            # Add video entry
            video_entry = {
                'name': vidInfo.get('name', video_name),
                'folder': vidInfo.get('dir', folder),
                'nFrames': vidInfo.get('nframes', None),
                'frameRate': vidInfo.get('frameRate', self.project['videos'][self.current_video_idx]['frameRate']),
                'samplingRate': vidInfo.get('samplingRate', self.project['videos'][self.current_video_idx]['samplingRate']),
                'featureFile': vidInfo.get('featureFile', 'perframe_feats.h5'),
                'annotations': {},
            }
            self.project['videos'].append(video_entry)

            # Update video selector
            self.video_selector.addItem(video_entry['name'])
            new_index = self.video_selector.count() - 1
            self.video_selector.setCurrentIndex(new_index)

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            filename, line, func, text = tb[-1]  # Get last traceback entry (where the error happened)
            print(f"❌ Error in function '{func}', line {line}: {e}")
            traceback.print_exc()

    def import_annotations(self):

        video = self.project['videos'][self.current_video_idx]
        video_dir = os.path.splitext(video['folder'])[0]

        # Open file dialog for CSV selection
        csv_file, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation CSV", video_dir, "CSV Files (*.csv)")

        if not csv_file:
            print("No file selected.")
            return

        # Load annotations from CSV
        annotations_df = pd.read_csv(csv_file)

        if len(annotations_df) != video['nFrames']:
            print(f"Mismatch in number of frames! CSV has {len(annotations_df)} rows; video has {video['nFrames']} frames.")
            return

        existing_behaviors = set(self.project['behaviors'])
        csv_behaviors = annotations_df.columns

        # Add new behaviors if they don't exist
        for behavior in csv_behaviors:
            newName = behavior.strip().capitalize()
            if newName not in existing_behaviors:
                self.project['behaviors'].append(newName)
                print(f"Added new behavior found in annotations: {newName}")

            behavior_col = annotations_df[behavior].fillna(0).astype(int)

            annotations = []
            current_val = 0
            start_frame = 0

            for frame_idx, val in enumerate(behavior_col):
                if val != current_val:
                    if current_val != 0:
                        annotations.append([start_frame, frame_idx - 1, current_val])
                    start_frame = frame_idx
                    current_val = val

            # Handle behavior continuing to the last frame
            if current_val != 0:
                annotations.append([start_frame, len(behavior_col) - 1, current_val])

            # Store annotations
            video['annotations'][newName] = annotations

        # Remove and replace old panel
        self.behavior_layout.removeWidget(self.behavior_panel)
        self.behavior_panel.deleteLater()
        self.behavior_panel = BehaviorPanel(self.project['behaviors'])
        self.behavior_layout.insertWidget(0, self.behavior_panel)

        print("Annotations successfully imported.")

    def export_annotations(self):

        video = self.project['videos'][self.current_video_idx]
        video_name = os.path.splitext(video['name'])[0]
        output_path = os.path.join(video['folder'], f"{video_name}_annotations.csv")

        nFrames = video['nFrames']
        behaviors = self.project['behaviors']

        # Initialize [nFrames x nBehaviors] matrix with zeros
        frame_annotations = np.zeros((nFrames, len(behaviors)), dtype=int)

        for b_idx, behavior in enumerate(behaviors):
            spans = video['annotations'].get(behavior, [])
            for start, end, value in spans:
                frame_annotations[start:end + 1, b_idx] = value

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame'] + behaviors)
            for i in range(nFrames):
                writer.writerow([i] + list(frame_annotations[i]))

        print(f"✅ Per-frame annotations exported to {output_path}")

    def remove_video(self):
        video = self.project['videos'][self.current_video_idx]
        reply = QMessageBox.question(
            self, 'Remove Video',
            f"Are you sure you want to remove '{video['name']}' from the project? All annotations will be lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.project['videos'].pop(self.current_video_idx)
            self.video_selector.removeItem(self.current_video_idx)
            self.current_video_idx = max(0, self.current_video_idx - 1)
            self.load_video(self.project['videos'][self.current_video_idx]['folder'])
            print("Video successfully removed from the project.")

    def remove_behavior(self):
        behaviors = self.project['behaviors']

        behavior, ok = QInputDialog.getItem(
            self, 'Remove Behavior',
            'Select behavior to remove:', behaviors, 0, False)

        if ok and behavior:
            reply = QMessageBox.question(
                self, 'Confirm Remove Behavior',
                f"Are you sure you want to remove behavior '{behavior}'? All associated annotations will be lost.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.project['behaviors'].remove(behavior)
                for video in self.project['videos']:
                    if behavior in video['annotations']:
                        video['annotations'].pop(behavior)

                # Refresh behavior panel
                self.behavior_layout.removeWidget(self.behavior_panel)
                self.behavior_panel.deleteLater()
                self.behavior_panel = BehaviorPanel(self.project['behaviors'])
                self.behavior_layout.insertWidget(0, self.behavior_panel)

                print(f"Behavior '{behavior}' successfully removed.")

    def batch_export_annotations(self):

        behaviors = self.project['behaviors']

        for video in self.project['videos']:
            video_name = os.path.splitext(video['name'])[0]
            output_path = os.path.join(video['folder'], f"{video_name}_annotations.csv")
            nFrames = video['nFrames']

            frame_annotations = np.zeros((nFrames, len(behaviors)), dtype=int)

            for b_idx, behavior in enumerate(behaviors):
                spans = video['annotations'].get(behavior, [])
                for start, end, value in spans:
                    frame_annotations[start:end + 1, b_idx] = value

            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame'] + behaviors)
                for i in range(nFrames):
                    writer.writerow([i] + list(frame_annotations[i]))

            print(f"📁 Exported: {output_path}")

    def train_TUBBAmodel(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import TUBBA_train

        # Save first to ensure that all annotations are included
        print(f"Training model...")
        self.save_project()

        if self.currentModelType == 'full':
            model_path = TUBBA_train.train_TUBBAmodel(self.project_path)
        else:
            model_path = TUBBA_train.train_TUBBAmodel_lite(self.project_path)

        self.project['models'] = model_path

        # Save to retain model path
        self.save_project()

        return None

    def run_model_inference(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from TUBBA_utils import save_inference_to_disk
        from models import TUBBA_infer

        print(f"Inferring behaviors...")
        self.save_project()

        video = self.project['videos'][self.current_video_idx]
        if self.currentModelType == 'full':
            results = TUBBA_infer.TUBBA_modelInference(self.project_path, video['name'], video['folder'])
        else:
            results = TUBBA_infer.TUBBA_modelInference_lite(self.project_path, video['name'], video['folder'])

        # save for now
        inferred_path = save_inference_to_disk(video, results)
        self.project['videos'][self.current_video_idx]['inferred_path'] = os.path.relpath(inferred_path)

        # allow viewing of current results
        self.display_inference = True
        self.current_inference = results
        self.update_confidence_pixmap()
        self.inference_toggle.setChecked(self.display_inference)
        self.timeline.update()

        return None

    def batch_run_model_inference(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from TUBBA_utils import save_inference_to_disk
        from models import TUBBA_infer

        print(f"Starting batch inference...")
        self.save_project()

        for video in self.project['videos']:
            print(f"Inferring behaviors on video {video['name']}...")
            if self.currentModelType == 'full':
                results = TUBBA_infer.TUBBA_modelInference(self.project_path, video['name'], video['folder'])
            else:
                results = TUBBA_infer.TUBBA_modelInference_lite(self.project_path, video['name'], video['folder'])

            # save for now
            inferred_path = save_inference_to_disk(video, results)
            video['inferred_path'] = os.path.relpath(inferred_path)

        self.save_project()

        # allow viewing of current video
        self.display_inference = True
        self.current_inference = results
        self.update_confidence_pixmap()
        self.inference_toggle.setChecked(self.display_inference)
        self.timeline.update()

        return None

    def handle_inference_toggle(self, state):
        checked = (state == 2)  # Qt.Checked

        video = self.project['videos'][self.current_video_idx]

        if checked:
            if 'inferred_path' not in video:
                QMessageBox.warning(
                    self,
                    "No Inference Available",
                    "You need to run inference on this video before toggling it on."
                )
                self.inference_toggle.setChecked(False)
                return

        self.display_inference = checked
        self.timeline.update()

    def update_confidence_pixmap(self):
        if not self.current_inference:
            self.confidence_pixmap = None
            return

        behaviors = self.project['behaviors']
        confidences = [self.current_inference['confidence'].get(b, []) for b in behaviors]

        if not confidences or len(confidences[0]) == 0:
            self.confidence_pixmap = None
            return

        self.conf_array = np.array(confidences)  # shape: [behaviors x frames]
