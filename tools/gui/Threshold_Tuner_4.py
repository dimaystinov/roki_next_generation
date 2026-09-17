from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from libcamera import controls

from Threshold_Tuner_Layout import Ui_MainWindow

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("Threshold_Tuner_config.json")
DEFAULT_THRESHOLDS = REPO_ROOT / "Init_params" / "Real" / "Real_Thresholds.json"

sys.path.insert(0, str(REPO_ROOT))

from Soccer.Vision.camera import Camera
from Soccer.Vision.reload import Image


def load_json(path, fallback):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return fallback


class Stream(QtCore.QObject):
    new_text = pyqtSignal(str)

    def write(self, text):
        self.new_text.emit(str(text))

    def flush(self):
        pass


class CameraThread(QThread):
    frame_ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent

    def run(self):
        self.window.camera_loop(self)


class ThresholdTunerWindow(QtWidgets.QMainWindow):
    DETECTION_KEYS = {
        1: None,
        2: "orange ball",
        3: "blue posts",
        4: "yellow posts",
        5: "white posts",
    }

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label_Color_Frame.setText("")
        self.ui.label_Binary_Frame.setText("")

        sys.stdout = Stream(new_text=self.on_console_text)
        self.ui.textBrowser_Console.ensureCursorVisible()

        self.config = load_json(
            CONFIG_PATH,
            {
                "defaultFile": "Init_params/Real/Real_Thresholds.json",
                "demo": {"th": [0, 100, -127, 127, -127, 127], "pixel": 200, "area": 200},
            },
        )
        config_default_file = self.config.get("defaultFile", "Init_params/Real/Real_Thresholds.json")
        self.filename = str((REPO_ROOT / config_default_file).resolve()) if not Path(config_default_file).is_absolute() else config_default_file
        self.thresholds = {
            "demo": self.config.get("demo", {"th": [0, 100, -127, 127, -127, 127], "pixel": 200, "area": 200}),
            "exposure": 10000,
            "gain": 1.7,
        }
        self.threshold_file_is_loaded = False
        self.current_device = "demo"
        self.blob_detection = 0
        self.camera = None
        self.camera_thread = None
        self.stop_event = threading.Event()
        self.slider_event = threading.Event()
        self.thresholds_are_changing = False
        self.blobs_are_changing = False
        self.new_timer = 0.0
        self.bitmap_color = QPixmap()
        self.bitmap_binary = QPixmap()
        self.blob_menu_items_list = [
            self.ui.actionBlobs_detection_OFF,
            self.ui.actionBlobs_detection_ON,
            self.ui.actionOrange_Ball_on_Green_Field,
            self.ui.actionBlue_Post_on_Green_Field,
            self.ui.actionYellow_Post_on_Green_Field,
            self.ui.actionWhite_Post_on_Green_Field,
        ]

        self.ui.comboBox.addItem("demo")
        self.ui.lineEdit_Pixel_TH.setInputMask("99999")
        self.ui.lineEdit_Area_TH.setInputMask("99999")

        self.load_threshold_file(self.filename, initial=True)
        self.display_values()
        self.signal_connection()

    def signal_connection(self):
        self.ui.pushButton_Quit.clicked.connect(self.on_quit)
        self.ui.pushButton_Reset_LAB.clicked.connect(self.on_reset_lab)
        self.ui.pushButton_SaveExit.clicked.connect(self.on_save_and_exit)
        self.ui.pushButton_Load_File.clicked.connect(self.on_load_file)
        self.ui.pushButton_Start_Camera.clicked.connect(self.on_start_camera)

        self.ui.actionQuit.triggered.connect(self.on_quit)
        self.ui.actionLoad_from_File.triggered.connect(self.on_load_file)
        self.ui.actionSave.triggered.connect(self.on_save)
        self.ui.actionSave_as.triggered.connect(self.on_save_as)
        self.ui.actionAbout.triggered.connect(self.on_about)
        self.ui.actionQuick_Start.triggered.connect(self.on_quick_start)

        self.ui.lineEdit_Pixel_TH.returnPressed.connect(self.on_pixel_changed)
        self.ui.lineEdit_Area_TH.returnPressed.connect(self.on_area_changed)
        self.ui.lineEdit_Exposure.returnPressed.connect(self.on_exposure_changed)
        self.ui.lineEdit_Gain.returnPressed.connect(self.on_gain_changed)
        self.ui.comboBox.currentIndexChanged.connect(self.on_device_selector)

        self.ui.actionAutoExposure_OFF.triggered.connect(self.on_auto_exposure_off)
        self.ui.actionAutoExposure_ON.triggered.connect(self.on_auto_exposure_on)

        sliders = [
            self.ui.horizontalSlider_Lmin,
            self.ui.horizontalSlider_Lmax,
            self.ui.horizontalSlider_A_min,
            self.ui.horizontalSlider_A_max,
            self.ui.horizontalSlider_B_min,
            self.ui.horizontalSlider_B_max,
        ]
        for slider in sliders:
            slider.sliderMoved.connect(self.on_slider_move)

        for action in self.blob_menu_items_list:
            action.triggered.connect(self.on_blobs_change)

    def display_values(self):
        device = self.thresholds[self.current_device]
        self.ui.lineEdit_Pixel_TH.setText(str(device["pixel"]))
        self.ui.lineEdit_Area_TH.setText(str(device["area"]))
        self.ui.lineEdit_Exposure.setText(str(self.thresholds["exposure"]))
        self.ui.lineEdit_Gain.setText(str(self.thresholds["gain"]))
        self.ui.horizontalSlider_Lmin.setValue(device["th"][0])
        self.ui.horizontalSlider_Lmax.setValue(device["th"][1])
        self.ui.horizontalSlider_A_min.setValue(device["th"][2])
        self.ui.horizontalSlider_A_max.setValue(device["th"][3])
        self.ui.horizontalSlider_B_min.setValue(device["th"][4])
        self.ui.horizontalSlider_B_max.setValue(device["th"][5])

    def update_combo_items(self):
        self.ui.comboBox.blockSignals(True)
        self.ui.comboBox.clear()
        keys = [key for key in self.thresholds.keys() if key not in {"exposure", "gain"}]
        self.ui.comboBox.addItems(keys)
        if self.current_device not in keys:
            self.current_device = "demo"
        self.ui.comboBox.setCurrentText(self.current_device)
        self.ui.comboBox.blockSignals(False)

    def load_threshold_file(self, filename, initial=False):
        data = load_json(filename, None)
        if not data:
            if initial:
                print("Threshold file is not available:", filename)
            return False

        required = [
            "orange ball",
            "blue posts",
            "yellow posts",
            "white posts",
            "green field",
            "white marking",
            "line_follow_1",
            "line_follow_2",
            "line_follow_3",
            "line_follow_4",
        ]
        self.filename = filename
        self.config["defaultFile"] = filename
        self.thresholds["demo"] = self.config.get("demo", self.thresholds["demo"])
        self.thresholds["exposure"] = data.get("exposure", self.thresholds["exposure"])
        self.thresholds["gain"] = data.get("gain", self.thresholds["gain"])
        for key in required:
            if key in data:
                self.thresholds[key] = data[key]
        self.threshold_file_is_loaded = all(key in self.thresholds for key in required)
        if self.threshold_file_is_loaded:
            self.ui.actionOrange_Ball_on_Green_Field.setEnabled(True)
            self.ui.actionBlue_Post_on_Green_Field.setEnabled(True)
            self.ui.actionYellow_Post_on_Green_Field.setEnabled(True)
            self.ui.actionWhite_Post_on_Green_Field.setEnabled(True)
        self.update_combo_items()
        self.display_values()
        print("threshold_file_is_loaded =", self.threshold_file_is_loaded)
        return True

    def persist_config(self):
        self.config = {
            "defaultFile": str(Path(self.filename).resolve().relative_to(REPO_ROOT)),
            "demo": self.thresholds["demo"],
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f)

    def on_console_text(self, text):
        cursor = self.ui.textBrowser_Console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.ui.textBrowser_Console.setTextCursor(cursor)
        self.ui.textBrowser_Console.ensureCursorVisible()

    def on_about(self):
        msg = QMessageBox(self)
        msg.setText(
            "Threshold Tuner v4\n"
            "Local camera tuner for robot vision thresholds.\n"
            "(C) 2022-2026\n"
            "Cleaned for local-only workflow."
        )
        msg.exec()

    def on_quick_start(self):
        print(
            "1. Load a thresholds file.\n"
            "2. Start the local camera.\n"
            "3. Choose a device in the combo box and tune LAB ranges.\n"
            "4. Save back to the same file or use Save as.\n"
            "5. Demo thresholds and the last file path are stored in the local v4 config."
        )

    def on_pixel_changed(self):
        self.thresholds[self.current_device]["pixel"] = int(self.ui.lineEdit_Pixel_TH.text() or "0")
        self.ui.lineEdit_Pixel_TH.setText(str(self.thresholds[self.current_device]["pixel"]))
        self.blobs_are_changing = True
        self.slider_event.set()

    def on_area_changed(self):
        self.thresholds[self.current_device]["area"] = int(self.ui.lineEdit_Area_TH.text() or "0")
        self.ui.lineEdit_Area_TH.setText(str(self.thresholds[self.current_device]["area"]))
        self.blobs_are_changing = True
        self.slider_event.set()

    def on_exposure_changed(self):
        new_exposure = min(int(self.ui.lineEdit_Exposure.text() or "0"), 16500)
        self.thresholds["exposure"] = new_exposure
        self.ui.lineEdit_Exposure.setText(str(new_exposure))
        if self.camera:
            self.camera.set_controls({"ExposureTime": new_exposure})
        self.slider_event.set()

    def on_gain_changed(self):
        new_gain = float(self.ui.lineEdit_Gain.text() or "0")
        self.thresholds["gain"] = new_gain
        self.ui.lineEdit_Gain.setText(str(new_gain))
        if self.camera:
            self.camera.set_controls({"AnalogueGain": new_gain})
        self.slider_event.set()

    def on_auto_exposure_off(self):
        if not self.camera:
            print("Camera is not started")
            return
        self.camera.set_controls({"AeEnable": False})
        self.ui.actionAutoExposure_ON.setChecked(False)

    def on_auto_exposure_on(self):
        if not self.camera:
            print("Camera is not started")
            return
        self.camera.set_controls({"AeEnable": True})
        self.camera.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
        time.sleep(1)
        self.thresholds["exposure"] = self.camera.capture_metadata()["ExposureTime"]
        self.thresholds["gain"] = self.camera.capture_metadata()["AnalogueGain"]
        self.ui.lineEdit_Exposure.setText(str(self.thresholds["exposure"]))
        self.ui.lineEdit_Gain.setText(str(self.thresholds["gain"]))
        self.ui.actionAutoExposure_OFF.setChecked(False)

    def on_blobs_change(self):
        sender = self.sender()
        for idx, action in enumerate(self.blob_menu_items_list):
            checked = action == sender
            action.setChecked(checked)
            if checked:
                self.blob_detection = idx
        self.blobs_are_changing = True
        self.slider_event.set()

    def on_save(self):
        data = dict(self.thresholds)
        data.pop("demo", None)
        with open(self.filename, "w") as f:
            json.dump(data, f)
        self.persist_config()

    def on_save_as(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save thresholds", self.filename, "*.json")
        if file_name:
            self.filename = file_name
            self.on_save()

    def on_save_and_exit(self):
        self.on_save()
        self.on_quit()

    def on_device_selector(self):
        value = self.ui.comboBox.currentText()
        if value:
            self.current_device = value
            self.display_values()
            self.slider_event.set()

    def on_slider_move(self):
        self.slider_event.set()

    def on_reset_lab(self):
        self.thresholds[self.current_device]["th"] = [0, 100, -127, 127, -127, 127]
        self.display_values()
        self.slider_event.set()

    def on_load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open threshold file", self.filename, "*.json")
        if file_name:
            self.load_threshold_file(file_name)

    def on_start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        print("Start Camera pressed")
        self.stop_event.clear()
        self.camera_thread = CameraThread(self)
        self.camera_thread.frame_ready.connect(self.update_image)
        self.camera_thread.start()
        self.ui.pushButton_Start_Camera.setEnabled(False)

    def find_reference_blobs(self, image):
        key = self.DETECTION_KEYS.get(self.blob_detection)
        if not key or key not in self.thresholds:
            return
        threshold = self.thresholds[key]
        for blob in image.find_blobs(
            [threshold["th"]],
            pixels_threshold=threshold["pixel"],
            area_threshold=threshold["area"],
            merge=True,
        ):
            image.draw_rectangle(blob.rect(), color=(0, 0, 255))

    def camera_loop(self, parent):
        self.camera = Camera()
        self.camera.start(exposure=self.thresholds["exposure"], gain=self.thresholds["gain"])
        timer = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                frame, _ = self.camera.snapshot()
                color_image = Image(frame)

                if self.blob_detection == 1:
                    threshold = self.thresholds[self.current_device]
                    for blob in color_image.find_blobs(
                        [threshold["th"]],
                        pixels_threshold=threshold["pixel"],
                        area_threshold=threshold["area"],
                    ):
                        color_image.draw_rectangle(blob.rect(), color=(0, 0, 255))
                elif self.blob_detection > 1:
                    self.find_reference_blobs(color_image)

                if self.slider_event.is_set():
                    self.new_timer = time.perf_counter()
                    self.slider_event.clear()
                    self.thresholds_are_changing = True

                if self.thresholds_are_changing or self.blobs_are_changing:
                    if (time.perf_counter() - self.new_timer) > 0.02:
                        self.thresholds_are_changing = False
                        self.blobs_are_changing = False
                        self.thresholds[self.current_device]["th"] = [
                            self.ui.horizontalSlider_Lmin.value(),
                            self.ui.horizontalSlider_Lmax.value(),
                            self.ui.horizontalSlider_A_min.value(),
                            self.ui.horizontalSlider_A_max.value(),
                            self.ui.horizontalSlider_B_min.value(),
                            self.ui.horizontalSlider_B_max.value(),
                        ]

                binary_image = Image(frame, copy=False).binary(
                    self.thresholds[self.current_device]["th"],
                    to_three_channels=True,
                )
                color_image.img = cv2.resize(color_image.img, (976, 800), interpolation=cv2.INTER_LINEAR)
                binary_image = cv2.resize(binary_image, (976, 800), interpolation=cv2.INTER_LINEAR)
                self.bitmap_color = self.convert_cv_qt(color_image.img)
                self.bitmap_binary = self.convert_cv_qt(binary_image)
                parent.frame_ready.emit()
                timer = time.perf_counter()
        finally:
            if self.camera:
                self.camera.stop()
                self.camera = None
            self.ui.pushButton_Start_Camera.setEnabled(True)

    @pyqtSlot()
    def update_image(self):
        self.ui.label_Color_Frame.setPixmap(self.bitmap_color)
        self.ui.label_Binary_Frame.setPixmap(self.bitmap_binary)

    def convert_cv_qt(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        scaled = qt_image.scaled(800, 650, QtCore.Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled)

    def on_quit(self):
        self.persist_config()
        self.stop_event.set()
        if self.camera_thread:
            self.camera_thread.wait(1500)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.close()

    def closeEvent(self, event):
        self.stop_event.set()
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.wait(1500)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        event.accept()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdTunerWindow()
    window.resize(1840, 940)
    window.show()
    sys.exit(app.exec_())
