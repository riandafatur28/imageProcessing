# main.py (PERBAIKAN LENGKAP)
import sys
import math
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize, QEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QFileDialog,
    QMessageBox, QDialog, QSlider, QPushButton, QAction, QWidget, QLineEdit,
    QHBoxLayout, QCheckBox, QRubberBand, QGraphicsScene, QGraphicsView, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent
from core.image_processor import ImageProcessor
from ui.tugasBuQon_ui import Ui_MainWindow


# ---------- Slider Dialog ----------
class SliderDialog(QDialog):
    def __init__(self, title, min_val, max_val, init_val):
        super().__init__()
        self.setWindowTitle(title)
        self.value = init_val

        layout = QVBoxLayout(self)
        self.label = QLabel(f"Value: {self.value}")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)

    def update_label(self, val):
        self.value = val
        self.label.setText(f"Value: {self.value}")




# ---------- Average Filter Dialog ----------
class AverageFilterDialog(QDialog):
    def __init__(self, title="Average Filter", min_val=3, max_val=15, init_val=3):
        super().__init__()
        self.setWindowTitle(title)
        self.value = init_val
        self.keep_color = True

        layout = QVBoxLayout(self)
        self.label = QLabel(f"Kernel size: {self.value}")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(2)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider)

        self.checkbox = QCheckBox("Tetap Warna (RGB)")
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def on_slider_change(self, v):
        self.label.setText(f"Kernel size: {v}")

    def accept(self):
        self.value = self.slider.value()
        if self.value % 2 == 0:
            self.value += 1
        self.keep_color = self.checkbox.isChecked()
        super().accept()


# ---------- PIL → QImage ----------
def pil2qimage(pil_img):
    """Konversi PIL.Image ke QImage (menghitung bytesPerLine dengan benar)."""
    if pil_img is None:
        return None

    mode = pil_img.mode
    w, h = pil_img.size

    if mode == "RGB":
        data = pil_img.tobytes("raw", "RGB")
        bytes_per_line = 3 * w
        return QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

    elif mode == "L":
        data = pil_img.tobytes("raw", "L")
        bytes_per_line = w
        return QImage(data, w, h, bytes_per_line, QImage.Format_Grayscale8)

    elif mode == "RGBA":
        try:
            data = pil_img.tobytes("raw", "RGBA")
            bytes_per_line = 4 * w
            return QImage(data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        except Exception:
            img = pil_img.convert("RGBA")
            data = img.tobytes("raw", "RGBA")
            bytes_per_line = 4 * w
            return QImage(data, w, h, bytes_per_line, QImage.Format_RGBA8888)

    else:
        img = pil_img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        bytes_per_line = 3 * w
        return QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)


# ---------- Crop-capable QLabel (rubber-band selection) ----------
class CropLabel(QLabel):
    # emits QRect in label coordinates (x,y,width,height)
    selection_made = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = None
        self.origin = QPoint()
        self._display_info = None  # optional: store display mapping

    def set_display_info(self, img_size, pixmap_size, scale, offset_x, offset_y):
        self._display_info = {
            "img_size": img_size,
            "pixmap_size": pixmap_size,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y
        }

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            self.origin = event.pos()
            if self.rubberBand is None:
                self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubberBand is not None:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubberBand.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rubberBand is not None:
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            self.selection_made.emit(rect)
        super().mouseReleaseEvent(event)


# ---------- Aritmetical Operation Window ----------
class AritmeticalWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aritmetical Operation")
        self.resize(400, 200)

        layout = QVBoxLayout(self)
        input_layout = QHBoxLayout()
        self.input1 = QLineEdit()
        self.input1.setPlaceholderText("Input 1")
        self.input2 = QLineEdit()
        self.input2.setPlaceholderText("Input 2")
        input_layout.addWidget(self.input1)
        input_layout.addWidget(self.input2)
        layout.addLayout(input_layout)

        self.output = QLineEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Output")
        layout.addWidget(self.output)

        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self.add_values)
        layout.addWidget(btn_add)

    def add_values(self):
        try:
            a = float(self.input1.text())
            b = float(self.input2.text())
            self.output.setText(str(a + b))
        except ValueError:
            QMessageBox.warning(self, "Error", "Masukkan angka yang valid")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ---------- Tambah Menu Segmentation ----------
        seg_menu = self.menuBar().addMenu("Segmentation")

        # Tambah QAction
        self.actionView_All_Segmentations = QAction("View All Segmentations", self)
        self.actionGlobal_Thresholding = QAction("Global Thresholding", self)
        self.actionAdaptive_Thresholding = QAction("Adaptive Thresholding", self)
        self.actionK_Means = QAction("K-Means", self)
        self.actionWatershed = QAction("Watershed", self)
        self.actionRegion_Growing = QAction("Region Growing", self)

        # Masukkan ke menu
        seg_menu.addAction(self.actionGlobal_Thresholding)
        seg_menu.addAction(self.actionAdaptive_Thresholding)
        seg_menu.addAction(self.actionK_Means)
        seg_menu.addAction(self.actionWatershed)
        seg_menu.addAction(self.actionRegion_Growing)
        seg_menu.addSeparator()
        seg_menu.addAction(self.actionView_All_Segmentations)

        # Hubungkan aksi ke method
        self.actionGlobal_Thresholding.triggered.connect(self.apply_global_threshold)
        self.actionAdaptive_Thresholding.triggered.connect(self.apply_adaptive_threshold)
        self.actionK_Means.triggered.connect(self.apply_kmeans)
        self.actionWatershed.triggered.connect(self.apply_watershed)
        self.actionRegion_Growing.triggered.connect(self.apply_region_growing)

        self.actionFlip_H = QAction("Flip Horizontal", self)
        self.actionFlip_V = QAction("Flip Vertical", self)
        self.actionRotate = QAction("Rotate", self)
        self.actionTranslate = QAction("Translate", self)
        self.actionZoomIn = QAction("Zoom In", self)
        self.actionZoomOut = QAction("Zoom Out", self)
        self.actionCrop = QAction("Crop", self)

        self.menuView.addAction(self.actionFlip_H)
        self.menuView.addAction(self.actionFlip_V)
        self.menuView.addAction(self.actionRotate)
        self.menuView.addAction(self.actionTranslate)
        self.menuView.addAction(self.actionZoomIn)
        self.menuView.addAction(self.actionZoomOut)
        self.menuView.addAction(self.actionCrop)

        # ---------- State ----------
        self.original_image = None
        self.processed_image = None
        self.processor = ImageProcessor()
        self._zoom_factor = 1.0
        self._crop_mode = False

        # ---------- Layout kiri ----------
        self.image_label_left = CropLabel()
        self.image_label_left.setAlignment(Qt.AlignCenter)
        kiri_layout = QVBoxLayout(self.kiri)
        kiri_layout.setContentsMargins(0, 0, 0, 0)
        kiri_layout.addWidget(self.image_label_left)

        # ---------- Layout kanan ----------
        self.image_label_right = QLabel()
        self.image_label_right.setAlignment(Qt.AlignCenter)
        kanan_layout = QVBoxLayout(self.kanan)
        kanan_layout.setContentsMargins(0, 0, 0, 0)
        kanan_layout.addWidget(self.image_label_right)

        # hubungkan sinyal crop
        if hasattr(self.image_label_left, "selection_made"):
            self.image_label_left.selection_made.connect(self.on_crop_selection)

        # ---------- Menu / Actions ----------
        self.connect_actions()

    # Show All Segmentation
    def show_all_segmentations(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        seg_results = {}
        methods = [
            ("Global Thresholding", "global_threshold"),
            ("Adaptive Thresholding", "adaptive_threshold"),
            ("K-Means", "kmeans_clustering"),
            ("Watershed", "watershed_segmentation"),
            ("Region Growing", "region_growing")
        ]

        for name, method in methods:
            func = getattr(self.processor, method, None)
            if func:
                # Atur parameter default jika dibutuhkan
                if method == "kmeans_clustering":
                    seg_results[name] = func(self.original_image, K=3)
                elif method == "region_growing":
                    seg_results[name] = func(self.original_image, seed_point=(50, 50), threshold=15)
                else:
                    seg_results[name] = func(self.original_image)

        # Plot semua hasil
        import matplotlib.pyplot as plt
        total = len(seg_results) + 1  # termasuk original
        rows, cols = 3, 2  # 3 kebawah x 2 kesamping
        plt.figure(figsize=(6 * cols, 4 * rows))

        # Original
        plt.subplot(rows, cols, 1)
        if self.original_image.mode == "L":
            plt.imshow(self.original_image, cmap="gray")
        else:
            plt.imshow(self.original_image)
        plt.title("Original")
        plt.axis("off")

        # Segmentasi
        for i, (title, img) in enumerate(seg_results.items(), start=2):
            plt.subplot(rows, cols, i)
            if img.mode == "L":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ============================================================
    # ========== CONNECT ACTIONS ================================
    # ============================================================
    def connect_actions(self):
        def safe_connect(attr_name, slot):
            if hasattr(self, attr_name):
                try:
                    getattr(self, attr_name).triggered.connect(slot)
                except Exception:
                    pass

        # File menu
        safe_connect("actionBuka", self.open_image)
        safe_connect("actionSimpan_sebagai", self.save_image_as)
        safe_connect("actionKeluar", self.close)

        # View menu
        safe_connect("actionFlip_H", lambda: self.view_operation("flip_h"))
        safe_connect("actionFlip_V", lambda: self.view_operation("flip_v"))
        safe_connect("actionRotate", lambda: self.view_operation("rotate"))
        safe_connect("actionTranslate", lambda: self.view_operation("translate"))
        safe_connect("actionZoomIn", lambda: self.view_operation("zoom_in"))
        safe_connect("actionZoomOut", lambda: self.view_operation("zoom_out"))
        safe_connect("actionCrop", lambda: self.view_operation("crop"))

        # Filters / Colors mapping
        filter_map = {
            "actionKuning": ("to_yellow", None),
            "actionOrange": ("to_orange", None),
            "actionCyan": ("to_cyan", None),
            "actionPurple": ("to_purple", None),
            "actionGrey": ("to_gray", None),
            "actionCoklat": ("to_brown", None),
            "actionMerah": ("to_red", None),
            "actionAverage": ("to_grayscale_average", None),
            "actionLightness": ("to_grayscale_lightness", None),
            "actionLuminance": ("to_grayscale_luminance", None),
            "actionContrast": ("adjust_contrast", "slider_float"),
            "actionBrightness_Contrast": ("brightness_contrast", "dual_slider"),
            "actionInvers": ("invert", None),
            "actionLog_Brightness": ("log_brightness", "slider_int"),
            "actionGamma_Correction": ("gamma_correction", "slider_gamma"),
        }
        for act_name, (mname, kind) in filter_map.items():
            if hasattr(self, act_name):
                act = getattr(self, act_name)
                if kind is None:
                    act.triggered.connect(lambda checked=False, mn=mname: self.apply_by_name(mn))
                elif kind == "slider_float":
                    act.triggered.connect(lambda checked=False, mn=mname: self.apply_with_slider_float(mn))
                elif kind == "dual_slider":
                    act.triggered.connect(lambda checked=False, mn=mname: self.apply_brightness_contrast())
                elif kind == "slider_int":
                    act.triggered.connect(lambda checked=False, mn=mname: self.apply_with_slider_int(mn))
                elif kind == "slider_gamma":
                    act.triggered.connect(lambda checked=False, mn=mname: self.apply_with_slider_gamma(mn))

        # Bit depth actions
        for i in range(1, 8):
            act_name = f"action{i}_Bit"
            if hasattr(self, act_name):
                getattr(self, act_name).triggered.connect(lambda checked=False, n=i: self.apply_bit_depth(n))

        # Histogram / Fuzzy
        safe_connect("actionHistogram_Equalization", self.histogram_equalization)
        safe_connect("actionFuzzy_HE_RGB", self.fuzzy_rgb)
        safe_connect("actionFuzzy_Grayscale", self.fuzzy_grayscale)

        # Edge / Filter ops
        safe_connect("actionEdge_Detection_1", lambda: self.apply_by_name("edge_detection1"))
        safe_connect("actionEdge_Detection_2", lambda: self.apply_by_name("edge_detection2"))
        safe_connect("actionEdge_Detection_3", lambda: self.apply_by_name("edge_detection3"))
        safe_connect("actionSharpen", lambda: self.apply_by_name("sharpen"))
        safe_connect("actionUnsharp_Masking", lambda: self.apply_by_name("unsharp_masking"))
        safe_connect("actionAverage_Filter", lambda: self.apply_average_filter_dialog())
        safe_connect("actionLow_Pass_Filter", lambda: self.apply_by_name("low_pass_filter"))
        safe_connect("actionHigh_Pass_Filter", lambda: self.apply_by_name("high_pass_filter"))
        safe_connect("actionGaussian_Blur_3x3", lambda: self.apply_gaussian(3))
        safe_connect("actionGaussian_Blur_5x5", lambda: self.apply_gaussian(5))
        safe_connect("actionPrewitt", lambda: self.apply_by_name("edge_detection_prewitt"))
        safe_connect("actionSobel", lambda: self.apply_by_name("edge_detection_sobel"))

        # Histogram input/output
        safe_connect("actionInput", self.show_histogram_input)
        safe_connect("actionOutput", self.show_histogram_output)
        safe_connect("actionInput_Output", self.show_histogram_input_output)

        # Segmentation
        safe_connect("actionView_All_Segmentations", self.show_all_segmentations)
        safe_connect("actionGlobal_Thresholding", self.apply_global_threshold)
        safe_connect("actionAdaptive_Thresholding", self.apply_adaptive_threshold)
        safe_connect("actionK_Means", self.apply_kmeans)
        safe_connect("actionWatershed", self.apply_watershed)
        safe_connect("actionRegion_Growing", self.apply_region_growing)

        # Help
        safe_connect("actionAbout", self.about_dialog)

    # ============================================================
    # ========== DISPLAY & EVENTS ================================
    # ============================================================
    def show_image(self, pil_img, label):
        qimage = pil2qimage(pil_img)
        if qimage is None:
            return
        pixmap = QPixmap.fromImage(qimage)
        sw, sh = label.width(), label.height()
        w, h = pixmap.width(), pixmap.height()
        if w == 0 or h == 0 or sw == 0 or sh == 0:
            return
        scale = min(sw / w, sh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image:
            self.show_image(self.original_image, self.image_label_left)
        if self.processed_image:
            self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== FILE ============================================
    # ============================================================
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Buka Gambar", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_name:
            return
        with Image.open(file_name) as im:
            self.original_image = im.convert("RGB").copy()
        self.processed_image = None
        self._zoom_factor = 1.0
        self.show_image(self.original_image, self.image_label_left)
        self.image_label_right.clear()

    def save_image_as(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Tidak ada gambar hasil!")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "", "PNG (*.png);;JPEG (*.jpg)")
        if file_name:
            self.processed_image.save(file_name)

    # ---------- VIEW / TRANSFORM (lengkap) ----------
    def view_operation(self, op):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        img = self.processed_image if self.processed_image else self.original_image
        self.apply_operation(op, img)

    def apply_operation(self, op, img):
        self.processed_image = None

        if op == "flip_h":
            self.processed_image = img.transpose(Image.FLIP_LEFT_RIGHT)

        elif op == "flip_v":
            self.processed_image = img.transpose(Image.FLIP_TOP_BOTTOM)

        elif op == "rotate":
            dlg = SliderDialog("Rotate (derajat)", -180, 180, 0)
            if dlg.exec_():
                self.processed_image = img.rotate(dlg.value, expand=True)

        elif op == "translate":
            dlg_dx = SliderDialog("Geser Horizontal (px)", -2000, 2000, 0)
            dlg_dy = SliderDialog("Geser Vertikal (px)", -2000, 2000, 0)
            if dlg_dx.exec_() and dlg_dy.exec_():
                dx, dy = dlg_dx.value, dlg_dy.value
                w, h = img.size
                bg = Image.new("RGB", (w, h), (255, 255, 255))
                bg.paste(img, (dx, dy))
                self.processed_image = bg

        elif op == "zoom_in":
            dlg = SliderDialog("Zoom (%)", 10, 400, 150)
            if dlg.exec_():
                factor = dlg.value / 100.0
                w, h = img.size
                self.processed_image = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)

        elif op == "zoom_out":
            dlg = SliderDialog("Zoom (%)", 10, 400, 50)
            if dlg.exec_():
                factor = dlg.value / 100.0
                w, h = img.size
                self.processed_image = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)

        elif op == "crop":
            QMessageBox.information(self, "Crop", "Drag pada gambar kiri untuk memilih area crop.")
            self._crop_mode = True
            return

        if self.processed_image:
            self.show_image(self.processed_image, self.image_label_right)

    def wheelEvent(self, event):
        if self.original_image is None:
            return
        if not self.image_label_left.underMouse():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self._zoom_factor *= 1.1 if delta > 0 else 0.9
        self._zoom_factor = max(0.1, min(self._zoom_factor, 5.0))
        w, h = self.original_image.size
        new_size = (int(w * self._zoom_factor), int(h * self._zoom_factor))
        self.processed_image = self.original_image.resize(new_size, Image.LANCZOS)
        self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== FILTERS ========================================
    # ============================================================
    def apply_by_name(self, method_name):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        func = getattr(self.processor, method_name, None)
        if func is None:
            QMessageBox.warning(self, "Error", f"Method {method_name} tidak ditemukan di ImageProcessor")
            return
        try:
            result = func(self.original_image)
        except TypeError:
            QMessageBox.warning(self, "Error", f"Method {method_name} memerlukan argumen khusus")
            return
        if result is not None:
            self.processed_image = result
            self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_float(self, method_name):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        dlg = SliderDialog("Parameter (x100)", 10, 300, 150)
        if dlg.exec_():
            val = dlg.value / 100.0
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(self.original_image, val)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_int(self, method_name):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        dlg = SliderDialog("Parameter", 1, 100, 10)
        if dlg.exec_():
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(self.original_image, dlg.value)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_gamma(self, method_name):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        dlg = SliderDialog("Gamma x0.1", 1, 500, 100)
        if dlg.exec_():
            gamma = dlg.value / 10.0
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(self.original_image, gamma)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_bit_depth(self, n_bits):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        func = getattr(self.processor, "bit_depth", None)
        if func:
            self.processed_image = func(self.original_image, n_bits)
            self.show_image(self.processed_image, self.image_label_right)

    def apply_average_filter_dialog(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        dlg = AverageFilterDialog()
        if dlg.exec_():
            func = getattr(self.processor, "average_filter", None)
            if func:
                self.processed_image = func(self.original_image, size=dlg.value, keep_color=dlg.keep_color)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_gaussian(self, kernel_size):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        func = getattr(self.processor, "gaussian_blur", None)
        if func:
            self.processed_image = func(self.original_image, kernel_size=kernel_size)
            self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== HISTOGRAM & FUZZY ==============================
    # ============================================================
    def histogram_equalization(self):
        if self.original_image:
            func = getattr(self.processor, "histogram_equalization", None)
            if func:
                self.processed_image = func(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def fuzzy_rgb(self):
        if self.original_image:
            func = getattr(self.processor, "fuzzy_histogram_equalization_rgb", None)
            if func:
                self.processed_image = func(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def fuzzy_grayscale(self):
        if self.original_image:
            func = getattr(self.processor, "fuzzy_histogram_equalization_gray", None)
            if func:
                self.processed_image = func(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def show_histogram_input(self):
        if self.original_image:
            f = getattr(self.processor, "show_histogram_input", None)
            if f:
                f(self.original_image)

    def show_histogram_output(self):
        if self.processed_image:
            f = getattr(self.processor, "show_histogram_output", None)
            if f:
                f(self.processed_image)

    def show_histogram_input_output(self):
        if self.original_image and self.processed_image:
            f = getattr(self.processor, "show_histogram_input_output", None)
            if f:
                f(self.original_image, self.processed_image)

    # ============================================================
    # ========== SEGMENTATION ===================================
    # ============================================================
    def apply_global_threshold(self):
        if self.original_image:
            f = getattr(self.processor, "global_threshold", None)
            if f:
                self.processed_image = f(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_adaptive_threshold(self):
        if self.original_image:
            f = getattr(self.processor, "adaptive_threshold", None)
            if f:
                self.processed_image = f(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_kmeans(self):
        if self.original_image:
            dlg = SliderDialog("K (clusters)", 2, 10, 3)
            if dlg.exec_():
                f = getattr(self.processor, "kmeans_clustering", None)
                if f:
                    self.processed_image = f(self.original_image, K=dlg.value)
                    self.show_image(self.processed_image, self.image_label_right)

    def apply_watershed(self):
        if self.original_image:
            f = getattr(self.processor, "watershed_segmentation", None)
            if f:
                self.processed_image = f(self.original_image)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_region_growing(self):
        if self.original_image:
            x, ok1 = QInputDialog.getInt(self, "Seed Point", "Masukkan X:", 50, 0, 10000, 1)
            y, ok2 = QInputDialog.getInt(self, "Seed Point", "Masukkan Y:", 50, 0, 10000, 1)
            if ok1 and ok2:
                f = getattr(self.processor, "region_growing", None)
                if f:
                    self.processed_image = f(self.original_image, seed_point=(x, y), threshold=15)
                    self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== ABOUT ===========================================
    # ============================================================
    def about_dialog(self):
        QMessageBox.information(self, "About", "Aplikasi Image Processing Sederhana\nDibuat dengan PyQt5 + PIL\n")

    # ============================================================
    # ========== SLOT UNTUK CROP ================================
    # ============================================================
    def on_crop_selection(self, rect: QRect):
        if not self._crop_mode or self.original_image is None:
            return

        pixmap = self.image_label_left.pixmap()
        if pixmap is None:
            return

        img_w, img_h = self.original_image.size
        disp_w, disp_h = pixmap.width(), pixmap.height()

        if disp_w == 0 or disp_h == 0:
            self._crop_mode = False
            return

        label_w, label_h = self.image_label_left.width(), self.image_label_left.height()
        scale = min(label_w / img_w, label_h / img_h) if img_w and img_h else 1.0
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        rx1 = rect.left() - offset_x
        ry1 = rect.top() - offset_y
        rx2 = rect.right() - offset_x
        ry2 = rect.bottom() - offset_y

        rx1 = max(0, min(scaled_w, rx1))
        ry1 = max(0, min(scaled_h, ry1))
        rx2 = max(0, min(scaled_w, rx2))
        ry2 = max(0, min(scaled_h, ry2))

        if scaled_w == 0 or scaled_h == 0:
            self._crop_mode = False
            return
        x_ratio = img_w / scaled_w
        y_ratio = img_h / scaled_h

        x1 = int(rx1 * x_ratio)
        y1 = int(ry1 * y_ratio)
        x2 = int(rx2 * x_ratio)
        y2 = int(ry2 * y_ratio)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 > x1 and y2 > y1:
            cropped = self.original_image.crop((x1, y1, x2, y2))
            self.processed_image = cropped
            self.show_image(self.processed_image, self.image_label_right)

        self._crop_mode = False


# ---------- Main ----------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
