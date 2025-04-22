import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QListWidget,
    QInputDialog,QFileDialog,QListView, QTreeView, QCheckBox, QProgressDialog, QDialog,
    QProgressBar)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

## Custom Widgets
class DeletableListWidget(QListWidget):
    def __init__(self, parent=None, backing_list=None):
        super().__init__(parent)
        self.backing_list = backing_list

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            selected_rows = sorted([self.row(item) for item in self.selectedItems()], reverse=True)
            for row in selected_rows:
                item = self.item(row)
                if self.backing_list is not None:
                    print(f"Removing {item.text()} from backing list.")
                    self.backing_list[:] = [x for x in self.backing_list if x != item.text()]
                self.takeItem(row)
            self.clearSelection()
        else:
            super().keyPressEvent(event)

class ProcessingWindow(QDialog):
    def __init__(self, total_folders, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Folders")
        self.setFixedSize(400, 100)
        self.setStyleSheet("background-color: white; color: black;")

        layout = QVBoxLayout()

        self.label = QLabel(f"Starting preprocessing...")
        self.label.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total_folders)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def update_progress(self, current_idx, total_folders):
        self.label.setText(f"Processing folder {current_idx} of {total_folders}...")
        self.progress_bar.setValue(current_idx)
        QApplication.processEvents()

    def finish(self):
        self.label.setText("✅ Preprocessing Complete!")
        self.progress_bar.setValue(self.progress_bar.maximum())
        QApplication.processEvents()
        self.close()

## Main Functions
class TUBBALauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TUBBA Launcher")
        self.setFixedSize(600, 400)
        self.setStyleSheet("background-color: black;")

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Header image
        pixmap = QPixmap("../xtra/TUBBAHeader.png")
        scaled_pixmap = pixmap.scaled(600, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(scaled_pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)

        # Buttons layout
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
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

        # New Project Button
        new_proj_btn = QPushButton("New Project")
        new_proj_btn.setFixedSize(100, 45)
        new_proj_btn.setStyleSheet(button_style)
        new_proj_btn.clicked.connect(self.new_project)
        button_layout.addWidget(new_proj_btn)

        # Load Project Button
        load_proj_btn = QPushButton("Load Project")
        load_proj_btn.setFixedSize(100, 45)
        load_proj_btn.setStyleSheet(button_style)
        load_proj_btn.clicked.connect(self.load_project)
        button_layout.addWidget(load_proj_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def new_project(self):
        print("New project initiated")
        self.new_project_window = NewProjectWindow()
        self.new_project_window.show()

    def load_project(self):
        from TUBBAvidAnn import VideoAnnotator

        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "JSON Files (*.json)",
            options=options
        )
        if path:
            try:
                with open(path, 'r') as f:
                    project = json.load(f)

                # Very important:
                project['project_path'] = path

                print("✅ Project loaded. Launching Annotator...")
                self.video_annotator_window = VideoAnnotator(project)
                self.video_annotator_window.show()
                self.close()
            except Exception as e:
                print(f"❌ Failed to load project: {e}")

class NewProjectWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.project = None
        self.setWindowTitle("Define New Project")
        self.setMinimumSize(900, 400)
        self.setStyleSheet("background-color: black;")

        self.behaviors = []
        self.folders = []
        self.processed_videos = []

        self.downsampledData = False
        self.downsampling_factor = 1.0
        self.project_path = ''

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        listWin_style = """background-color: black;
                            color: white;
                            border: 1px solid gray;
                            border-style: outset;
                            border-radius: 4px;
                            padding: 4px;
                            font: bold 14px;
                        """

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

        # Behaviors List
        self.behaviors_list = DeletableListWidget(backing_list=self.behaviors)
        self.behaviors_list.setStyleSheet(listWin_style)
        top_layout.addWidget(self.behaviors_list)

        # Folders List
        self.folders_list = DeletableListWidget(backing_list=self.folders)
        self.folders_list.setStyleSheet(listWin_style)
        top_layout.addWidget(self.folders_list)

        # Processed Videos List
        self.processed_list = QListWidget()
        self.processed_list.setStyleSheet(listWin_style)
        top_layout.addWidget(self.processed_list)

        layout.addLayout(top_layout)

        # Buttons layout
        button_layout = QHBoxLayout()

        add_behavior_btn = QPushButton("Add Behavior")
        add_behavior_btn.setStyleSheet(button_style)
        add_behavior_btn.clicked.connect(self.add_behavior)
        button_layout.addWidget(add_behavior_btn)

        add_data_btn = QPushButton("Add Data")
        add_data_btn.setStyleSheet(button_style)
        add_data_btn.clicked.connect(self.add_data)
        button_layout.addWidget(add_data_btn)

        preprocess_btn = QPushButton("Pre-process Data")
        preprocess_btn.setStyleSheet(button_style)
        preprocess_btn.clicked.connect(self.preprocess_data)
        button_layout.addWidget(preprocess_btn)

        layout.addLayout(button_layout)

        # Bottom layer (checkbox and Next button)
        bottom_layout = QHBoxLayout()

        self.downsample_checkbox = QCheckBox("Downsampled Videos?")
        self.downsample_checkbox.setStyleSheet("color: white;")
        self.downsample_checkbox.stateChanged.connect(self.handle_downsample_checkbox)
        bottom_layout.addWidget(self.downsample_checkbox, alignment=Qt.AlignCenter)

        bottom_layout.addStretch()

        next_btn = QPushButton("Next")
        next_btn.setStyleSheet(button_style)
        next_btn.setMinimumSize(100, 30)
        next_btn.clicked.connect(self.next_step)
        bottom_layout.addWidget(next_btn, alignment=Qt.AlignRight)

        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def add_behavior(self):
        dialog = QInputDialog(self)
        dialog.setStyleSheet("background-color: black; color: white;")
        dialog.setWindowTitle('Add Behavior')
        dialog.setLabelText('Enter behavior name(s) (comma separated):')
        ok = dialog.exec_()
        text = dialog.textValue()
        if ok and text:
            behaviors = [b.strip().capitalize() for b in text.split(',') if b.strip()]

            for behavior in behaviors:
                if not any(self.behaviors_list.item(i).text() == behavior for i in range(self.behaviors_list.count())):
                    self.behaviors.append(behavior)  # Update the internal list
                    self.behaviors_list.addItem(behavior)  # Add visually to the widget

    def add_data(self):
        old_style = self.styleSheet()  # Save your black style
        self.setStyleSheet("")

        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Data Folders")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        # Enable multiple selection
        dialog.findChild(QListView, 'listView').setSelectionMode(QListView.MultiSelection)
        dialog.findChild(QTreeView).setSelectionMode(QTreeView.MultiSelection)

        if dialog.exec_():
            selected_folders = dialog.selectedFiles()

            if selected_folders:
                # If more than one folder, assume first is parent and skip
                if len(selected_folders) > 1:
                    selected_folders = selected_folders[1:]
                print(self.folders)
                for folder in selected_folders:
                    if folder not in self.folders:
                        self.folders.append(folder)
                        self.folders_list.addItem(folder)

        # Restore your window's style
        self.setStyleSheet(old_style)

    def handle_downsample_checkbox(self, state):
        if state == Qt.Checked:
            dialog = QInputDialog(self)
            dialog.setStyleSheet("background-color: black; color: white;")
            dialog.setWindowTitle('Downsampling')
            dialog.setLabelText('Enter downsampling factor (e.g., 0.5x):')
            ok = dialog.exec_()
            text = dialog.textValue()
            if ok and text:
                try:
                    factor = float(text)
                    if 0 < factor < 1:
                        self.downsampling_factor = factor
                        self.downsampledData = True
                    else:
                        self.downsample_checkbox.setChecked(False)
                except ValueError:
                    self.downsample_checkbox.setChecked(False)
        else:
            self.downsampledData = False
            self.downsampling_factor = 1.0

    def preprocess_data(self):
        from getTUBBAFeats import dlcToFeatures

        if not self.folders:
            print("Error: No folders to process!")
            return

        if self.downsampledData:
            spatialSR = self.downsampling_factor
        else:
            spatialSR = 1.0

        processed_list = []
        video_statuses = []

        # Create our simple progress window
        progress_window = ProcessingWindow(len(self.folders), self)
        progress_window.show()

        # Create our simple progress window
        progress_window = ProcessingWindow(len(self.folders), self)
        progress_window.show()

        for i, folder in enumerate(self.folders):
            progress_window.update_progress(i + 1, len(self.folders))

            try:
                vidInfo = dlcToFeatures(folder, spatialSR)
                folder_name = os.path.basename(folder)
                if vidInfo['status'] > 0:
                    processed_list.append(f"{folder_name}  ✓")
                else:
                    processed_list.append(f"{folder_name}  ✗")
                video_statuses.append(vidInfo)
            except Exception as e:
                print(f"❌ Error processing {folder}: {e}")

        self.processing_window.finish()

        # Update your GUI list
        self.processed_list.clear()
        self.processed_list.addItems(processed_list)

        self.processed_videos = processed_list
        self.video_statuses = video_statuses

    def next_step(self):
        from TUBBAvidAnn import VideoAnnotator

        # --- Check that we have at least 1 behavior and 1 video
        if len(self.behaviors) == 0:
            print("⚠️ No behaviors added!")
            return
        if len(self.folders) == 0:
            print("⚠️ No video folders added!")
            return

        # --- Build project dictionary
        project = {
            'behaviors': self.behaviors,
            'models': {},  # No models yet
            'videos': [],
        }

        for idx, folder in enumerate(self.folders):
            folder_name = os.path.basename(folder)

            # Try to load vidInfo if exists (later we can load it if needed)
            feature_file = 'perframe_feats.h5'
            feature_path = os.path.join(folder, feature_file)

            # Fallback defaults if no pre-processing was done
            nFrames = None
            frameRate = None
            samplingRate = getattr(self, 'spatialSR', 1.0)

            # You had already processed them earlier (processed_videos)
            if idx < len(self.video_statuses):
                vidInfo = self.video_statuses[idx]
                nFrames = vidInfo.get('nframes', None)
                frameRate = vidInfo.get('frameRate', 50)
                samplingRate = vidInfo.get('samplingRate', 1.0)

                # Build video entry
                video_entry = {
                    'name': vidInfo.get('name', None),
                    'folder': folder,
                    'nFrames': nFrames,
                    'frameRate': frameRate,
                    'samplingRate': samplingRate,
                    'featureFile': feature_file,
                    'annotations': {},  # Start empty
                }
                project['videos'].append(video_entry)

        self.project = project
        
        # --- Ask user for save location
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project File",
            os.path.expanduser("~/TUBBA_project.json"),  # Default filename suggestion
            "JSON Files (*.json)"
        )

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(project, f, indent=4)

            print(f"✅ Project saved to {save_path}")
            self.project_path = self.project_path

        print("🚀 Launching Video Annotator...")
        self.video_annotator_window = VideoAnnotator(self.project)
        self.video_annotator_window.show()
        self.close()

        self.close()  # Close the NewProjectWindow

## Entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TUBBALauncher()
    window.show()
    sys.exit(app.exec_())
