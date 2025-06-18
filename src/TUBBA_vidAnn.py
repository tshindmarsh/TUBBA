import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,QMessageBox,
                             QPushButton, QSlider, QFileDialog, QListWidget,
                             QGroupBox, QComboBox,QSizePolicy, QShortcut)
from PyQt5.QtGui import QPixmap, QImage, QTransform, QPainter, QColor, QPen, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QPoint
import random
import h5py

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

        for behavior in behaviors:
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            color = (r, g, b)
            self.behavior_colors[behavior] = color

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

        ### Draw behavior bars
        for idx, behavior in enumerate(behaviors):
            behavior_y_lo = idx * row_height
            behavior_y_hi = (idx + 1) * row_height

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

                # Color
                color = self.annotator.behavior_panel.behavior_colors.get(behavior, (255,255,255))
                if val == -1:
                    color = (128, 128, 128)  # Gray for NOT behavior

                qcolor = QColor(*color)
                painter.fillRect(x_start, int(behavior_y_lo), x_end - x_start, int(row_height), qcolor)

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

class VideoAnnotator(QWidget):
    def __init__(self, project):
        super().__init__()

        self.project = project  # Project dictionary
        self.current_video_idx = 0  # Which video are we looking at?
        self.current_frame_idx = 0
        self.annotation_mode = False  # True when annotating (after pressing 's')
        self.annotation_start_frame = None

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

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play)
        self.play_pause_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.play_pause_button)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        self.save_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.save_button)

        self.add_behavior_button = QPushButton("Add Behavior")
        self.add_behavior_button.clicked.connect(self.add_behavior)
        self.add_behavior_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.add_behavior_button)

        self.add_video_button = QPushButton("Add Video")
        self.add_video_button.clicked.connect(self.add_video)
        self.add_video_button.setStyleSheet(button_style)
        controls_layout.addWidget(self.add_video_button)

        # Add train and infer buttons
        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train_TUBBAmodel)
        self.train_button.setStyleSheet("QPushButton {background-color: #17FFD2; color: black; border: 1px solid gray; border-style: outset;"
                                        "border-radius: 4px; padding: 4px; font: bold 12px;} QPushButton:pressed {border-style: inset}")
        controls_layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Infer behaviors")
        self.predict_button.clicked.connect(self.TUBBA_modelInference)
        self.predict_button.setStyleSheet(
            "QPushButton {background-color: #157FFF; color: black; border: 1px solid gray; border-style: outset;"
            "border-radius: 4px; padding: 4px; font: bold 12px;} QPushButton:pressed {border-style: inset}")
        controls_layout.addWidget(self.predict_button)

        controls_layout.addStretch()
        video_controls_group.setLayout(controls_layout)
        video_controls_group.setMaximumWidth(250)
        left_panel.addWidget(video_controls_group)

        left_panel.addStretch()

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
                            border_color: #000;
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

        self.setLayout(main_layout)

        self.load_video(self.project['videos'][self.current_video_idx]['folder'])

    def load_video(self, folder):
        mp4s = [f for f in os.listdir(folder) if f.endswith('.mp4')]
        if len(mp4s) == 0:
            print(f"⚠️ No video found in {folder}")
            return

        video_path = os.path.join(folder, mp4s[0])
        self.cap = cv2.VideoCapture(video_path)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider.setMaximum(self.total_frames - 1)

        self.show_frame(0)

    def switch_video(self, idx):
        self.current_video_idx = idx
        self.load_video(self.project['videos'][idx]['folder'])

    def show_frame(self, idx):
        if not self.cap.isOpened():
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
        elif event.key() == Qt.Key_S and (event.modifiers() & Qt.MetaModifier):
            self.save_project()
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.MetaModifier):
            self.undo_last_annotation()
        else:
            super().keyPressEvent(event)

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
            return

        print(f"📂 Selected folder: {folder}")

        # Try to preprocess it
        try:
            from getTUBBAFeats import dlcToFeatures

            vidInfo = dlcToFeatures(folder, spatialSR=0.5)  # Assume no downsampling for new
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
            print(f"❌ Error adding video: {e}")

    def train_TUBBAmodel(self):
        from TUBBA_train import train_TUBBAmodel

        # Save first to ensure that all annotations are included
        print(f"Training model...")
        self.save_project()

        model_path = train_TUBBAmodel(self.project_path)
        self.project['models'] = model_path

        # Save to retain model path
        self.save_project()

        return None

    def TUBBA_modelInference(self):
        from TUBBAutils import save_inference_to_disk
        from TUBBA_infer import TUBBA_modelInference

        print(f"Inferring behaviors...")
        self.save_project()

        video = self.project['videos'][self.current_video_idx]
        results = TUBBA_modelInference(self.project_path, video['name'], video['folder'])

        # Here, we should save the current state of the project, and then pass the directory to the training script
        return None

