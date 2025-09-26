# main.py – Import PyQt5 yang sudah diperbaiki
import sys
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageFilter
from matplotlib import image

from core.arithmetic_dialog import ArithmeticDialog

# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize, QEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QFileDialog,
    QMessageBox, QDialog, QSlider, QPushButton, QAction, QWidget,
    QLineEdit, QHBoxLayout, QCheckBox, QRubberBand, QGraphicsScene,
    QGraphicsView, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent

# Module internal
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

    def apply_arithmetic(self, operation):
        img1 = self.get_current_image()  # 🔑 ambil gambar aktif (original/processed)
        img2 = self.another_image  # gambar kedua tetap

        if img1 and img2:
            dialog = ArithmeticDialog(img1, img2, operation, self)
            dialog.exec_()
        else:
            print("⚠️ Harap buka dua gambar terlebih dahulu.")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scale_factor = 1.0

        self.actionReset = QtWidgets.QAction("Reset", self)
        self.actionReset.setObjectName("actionReset")

        ## Buat action Tentang langsung
        self.actionTentang = QtWidgets.QAction("Tentang", self)
        self.actionTentang.triggered.connect(self.show_about_popup)

        # Tambahkan ke menubar
        self.menubar.addAction(self.actionTentang)

        # ---------- Tambah Menu Segmentation ----------
        seg_menu = self.menuBar().addMenu("Segmentation")

        # Tambah QAction
        self.actionView_All_Segmentations = QAction("View All Segmentations", self)
        self.actionGlobal_Thresholding = QAction("Global Thresholding", self)
        self.actionAdaptive_Thresholding = QAction("Adaptive Thresholding", self)
        self.actionK_Means = QAction("K-Means", self)
        self.actionWatershed = QAction("Watershed", self)
        self.actionRegion_Growing = QAction("Region Growing", self)
        self.menuAritmetical = self.menubar.addMenu("Aritmetical Operation")

        open_action = QAction("Open Arithmetic Dialog", self)
        open_action.triggered.connect(self.open_arithmetic)
        self.menuAritmetical.addAction(open_action)

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
        self.actionRemoveBG = QAction("Remove Background", self)
        self.actionRemoveBG.setObjectName("actionRemoveBG")

        self.menuView.addAction(self.actionFlip_H)
        self.menuView.addAction(self.actionFlip_V)
        self.menuView.addAction(self.actionRotate)
        self.menuView.addAction(self.actionTranslate)
        self.menuView.addAction(self.actionZoomIn)
        self.menuView.addAction(self.actionZoomOut)
        self.menuView.addAction(self.actionCrop)
        self.menuView.addAction(self.actionRemoveBG)
        self.actionRemoveBG.triggered.connect(self.apply_remove_bg)

        # ---------- State ----------
        self.original_image = None
        self.processed_image = None
        self.processor = ImageProcessor()
        self._zoom_factor = 1.0
        self._crop_mode = False

        # Erosion
        self.actionSquare_3.triggered.connect(lambda: self.apply_morphology("Erosion", "Square", 3))
        self.actionSquare_5.triggered.connect(lambda: self.apply_morphology("Erosion", "Square", 5))
        self.actionCross_3.triggered.connect(lambda: self.apply_morphology("Erosion", "Cross", 3))

        # Dilation
        self.actionSquare_4.triggered.connect(lambda: self.apply_morphology("Dilation", "Square", 4))
        self.actionSquare_6.triggered.connect(lambda: self.apply_morphology("Dilation", "Square", 6))
        self.actionCross_4.triggered.connect(lambda: self.apply_morphology("Dilation", "Cross", 4))

        # Opening
        self.actionSquare_9.triggered.connect(lambda: self.apply_morphology("Opening", "Square", 9))

        # Closing
        self.actionSquare_10.triggered.connect(lambda: self.apply_morphology("Closing", "Square", 10))

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

    def show_about_popup(self):
        # Bikin QDialog custom
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Tentang Aplikasi")
        dlg.resize(600, 400)  # ukuran popup lebih besar

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setSpacing(15)  # kasih jarak antar elemen

        # Judul utama
        label_title = QtWidgets.QLabel("Praktikum Workshop Pengolahan Citra Vision")
        label_title.setAlignment(QtCore.Qt.AlignCenter)
        label_title.setStyleSheet("font-size: 20pt; font-weight: bold;")

        # Sub Judul
        label_judul = QtWidgets.QLabel("Aplikasi Image Processing Sederhana")
        label_judul.setAlignment(QtCore.Qt.AlignCenter)
        label_judul.setStyleSheet("font-size: 16pt; font-style: italic;")

        # Versi
        label_version = QtWidgets.QLabel("Versi 1.0")
        label_version.setAlignment(QtCore.Qt.AlignCenter)
        label_version.setStyleSheet("font-size: 14pt;")

        # Creator
        label_creator = QtWidgets.QLabel("Rianda Faturrahman")
        label_creator.setAlignment(QtCore.Qt.AlignCenter)
        label_creator.setStyleSheet("font-size: 14pt;")

        # NIM
        label_nim = QtWidgets.QLabel("E41231605")
        label_nim.setAlignment(QtCore.Qt.AlignCenter)
        label_nim.setStyleSheet("font-size: 14pt;")

        # Tambahkan semua ke layout
        layout.addWidget(label_title)
        layout.addWidget(label_judul)
        layout.addWidget(label_version)
        layout.addWidget(label_creator)
        layout.addWidget(label_nim)

        # Tombol OK
        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.setFixedHeight(40)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-size: 14pt;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok, alignment=QtCore.Qt.AlignCenter)

        dlg.exec_()

    # Aritmetic Operational
    def apply_arithmetic(self, operation):
        img1 = self.get_current_image()  # ✅ ambil gambar aktif (processed kalau ada)
        img2 = self.another_image

        if img1 and img2:
            dialog = ArithmeticDialog(img1, img2, operation, self)
            dialog.exec_()
        else:
            print("⚠️ Harap buka dua gambar terlebih dahulu.")

    # Show All Segmentation
    def show_all_segmentations(self):
        img = self.get_current_image()  # ✅ ambil gambar terbaru
        if img is None:
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
                    seg_results[name] = func(img, K=3)
                elif method == "region_growing":
                    seg_results[name] = func(img, seed_point=(50, 50), threshold=15)
                else:
                    seg_results[name] = func(img)

        # misalnya nanti ditampilkan di tab/grid
        # self.show_segmentation_results(seg_results)

        # Plot semua hasil
        import matplotlib.pyplot as plt
        total = len(seg_results) + 1  # termasuk original
        rows, cols = 3, 2  # 3 kebawah x 2 kesamping
        plt.figure(figsize=(6 * cols, 4 * rows))

        # Original
        plt.subplot(rows, cols, 1)
        img = self.get_current_image()  # ✅ ambil gambar aktif (processed kalau ada)
        if img.mode == "L":
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        # Segmentasi
        for i, (title, img) in enumerate(seg_results.items(), start=2):
            if img is None:
                continue  # skip kalau hasil segmentasi kosong

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

        # reset
        safe_connect("actionReset", lambda: self.view_operation("reset"))

        # Aritmetic Operational
        safe_connect("actionAdd", lambda: self.apply_arithmetic("add"))
        safe_connect("actionSubtract", lambda: self.apply_arithmetic("subtract"))
        safe_connect("actionMultiply", lambda: self.apply_arithmetic("multiply"))

        # View menu
        safe_connect("actionFlip_H", lambda: self.view_operation("flip_h"))
        safe_connect("actionFlip_V", lambda: self.view_operation("flip_v"))
        safe_connect("actionRotate", lambda: self.view_operation("rotate"))
        safe_connect("actionTranslate", lambda: self.view_operation("translate"))
        safe_connect("actionZoomIn", lambda: self.view_operation("zoom_in"))
        safe_connect("actionZoomOut", lambda: self.view_operation("zoom_out"))
        safe_connect("actionCrop", lambda: self.view_operation("crop"))
        safe_connect("actionRemoveBG", self.apply_remove_bg)

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
                getattr(self, act_name).triggered.connect(
                    lambda checked=False, n=i: self.apply_bit_depth(n)
                )

        # Histogram / Fuzzy
        safe_connect("actionHistogram_Equalization", self.histogram_equalization)
        safe_connect("actionFuzzy_HE_RGB", self.fuzzy_rgb)
        safe_connect("actionFuzzy_Grayscale", self.fuzzy_grayscale)

        # Edge / Filter ops
        # di bagian inisialisasi menu / toolbar
        safe_connect("actionIdentify", lambda: self.view_operation("identify"))
        safe_connect("actionEdge_Detection_1", lambda: self.apply_by_name("edge_detection1"))
        safe_connect("actionEdge_Detection_2", lambda: self.apply_by_name("edge_detection2"))
        safe_connect("actionEdge_Detection_3", lambda: self.apply_by_name("edge_detection3"))
        safe_connect("actionSharpen", lambda: self.apply_by_name("sharpen"))
        safe_connect("actionUnsharp_Masking", lambda: self.apply_by_name("unsharp_masking"))
        safe_connect("actionAverage_Filter", lambda: self.apply_average_filter_dialog())
        safe_connect("actionLow_Pass_Filter", lambda: self.apply_by_name("low_pass_filter"))
        safe_connect("actionHigh_Pass_Filter", lambda: self.apply_by_name("high_pass_filter"))
        safe_connect("actionBandstop_Filter", lambda: self.apply_by_name("bandstop_filter"))
        safe_connect("actionGaussian_Blur_3x3", lambda: self.apply_gaussian(3))
        safe_connect("actionGaussian_Blur_5x5", lambda: self.apply_gaussian(5))
        safe_connect("actionPrewitt", lambda: self.apply_by_name("edge_detection_prewitt"))
        safe_connect("actionSobel", lambda: self.apply_by_name("edge_detection_sobel"))
        safe_connect("actionCanny", lambda: self.apply_by_name("edge_detection_canny"))

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
        if self.processed_image:
            # tampilkan hasil edit terbaru di kiri & kanan
            self.show_image(self.processed_image, self.image_label_left)
            self.show_image(self.processed_image, self.image_label_right)
        elif self.original_image:
            # kalau belum ada edit, tampilkan gambar asli di kiri & kanan
            self.show_image(self.original_image, self.image_label_left)
            self.show_image(self.original_image, self.image_label_right)

    # ============================================================
    # ========== FILE ============================================
    # ============================================================
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Buka Gambar", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_name:
            return

        with Image.open(file_name) as im:
            # simpan salinan gambar asli
            self.original_image = im.convert("RGB").copy()

        # set processed_image ke copy juga (supaya bisa diubah tanpa ganggu original)
        self.processed_image = self.original_image.copy()
        self._zoom_factor = 1.0

        # tampilkan ke kiri (asli)
        self.show_image(self.original_image, self.image_label_left)

        # kosongkan label kanan
        self.image_label_right.clear()

    def get_current_image(self):
        """Ambil gambar aktif: pakai processed_image jika ada, kalau tidak original_image."""
        return self.processed_image if self.processed_image else self.original_image

    def save_image_as(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Tidak ada gambar hasil!")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "", "PNG (*.png);;JPEG (*.jpg)")
        if file_name:
            self.processed_image.save(file_name)

    # Morpologi
    def apply_morphology(self, operation=None, kernel_type="Square", ksize=3):
        img = self.get_current_image()  # ✅ ambil gambar aktif (processed kalau ada)
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        f = getattr(self.processor, "morphology", None)
        if not f:
            return

        # Kalau operation None → tampilkan dialog pilihan
        if operation is None:
            from PyQt5.QtWidgets import QInputDialog
            options = [
                "Erosion - Square 3",
                "Erosion - Square 5",
                "Erosion - Cross 3",
                "Dilation - Square 3",
                "Dilation - Square 5",
                "Dilation - Cross 3",
                "Opening - Square 9",
                "Closing - Square 9"
            ]
            item, ok = QInputDialog.getItem(self, "Pilih Operasi Morfologi", "Operasi:", options, 0, False)
            if not ok:
                return

            # mapping string ke parameter
            if "Erosion" in item:
                operation = "Erosion"
            elif "Dilation" in item:
                operation = "Dilation"
            elif "Opening" in item:
                operation = "Opening"
            elif "Closing" in item:
                operation = "Closing"

            if "Square 3" in item:
                kernel_type, ksize = "Square", 3
            elif "Square 5" in item:
                kernel_type, ksize = "Square", 5
            elif "Cross 3" in item:
                kernel_type, ksize = "Cross", 3
            elif "Square 9" in item:
                kernel_type, ksize = "Square", 9

        # Jalankan proses morfologi
        self.processed_image = f(img, operation, kernel_type, ksize)
        self.show_image(self.processed_image, self.image_label_right)

    # ---------- VIEW / TRANSFORM (lengkap) ----------
    def view_operation(self, op):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        img = self.processed_image if self.processed_image else self.original_image
        self.apply_operation(op, img)

    def apply_operation(self, op, img):
        if op == "flip_h":
            self.processed_image = img.transpose(Image.FLIP_LEFT_RIGHT)

        elif op == "flip_v":
            self.processed_image = img.transpose(Image.FLIP_TOP_BOTTOM)

        elif op == "rotate":
            dlg = SliderDialog("Rotate (derajat)", -360, 360, 0)
            if dlg.exec_():
                angle = dlg.value
                # kalau RGBA pakai transparan, kalau RGB pakai putih
                fill = (255, 255, 255, 0) if img.mode == "RGBA" else (255, 255, 255)
                self.processed_image = img.rotate(angle, expand=True, fillcolor=fill)

        elif op == "translate":
            dlg_dx = SliderDialog("Geser Horizontal (px)", -360, 360, 0)
            dlg_dy = SliderDialog("Geser Vertikal (px)", -360, 360, 0)
            if dlg_dx.exec_() and dlg_dy.exec_():
                dx, dy = dlg_dx.value, dlg_dy.value
                w, h = img.size

                # hitung ukuran kanvas baru (biar gambar ga terpotong)
                new_w = w + abs(dx)
                new_h = h + abs(dy)

                # buat background (kalau RGBA → transparan, kalau RGB → putih)
                bg = Image.new("RGBA" if img.mode == "RGBA" else "RGB",
                               (new_w, new_h),
                               (255, 255, 255, 0) if img.mode == "RGBA" else (255, 255, 255))

                # posisi tempel gambar
                paste_x = max(dx, 0)
                paste_y = max(dy, 0)

                # tempelkan gambar
                if img.mode == "RGBA":
                    bg.paste(img, (paste_x, paste_y), img)
                else:
                    bg.paste(img, (paste_x, paste_y))
                self.processed_image = bg

        elif op == "zoom_in":
            dlg = SliderDialog("Zoom In (%)", 10, 200, 50)

            if dlg.exec_():
                factor = 1.0 + (dlg.value / 100.0)
                w, h = self.processed_image.size
                new_w = int(w * factor)
                new_h = int(h * factor)
                self.processed_image = self.processed_image.resize((new_w, new_h), Image.LANCZOS)
                self.show_image(self.processed_image, self.image_label_right)


        elif op == "zoom_out":
            dlg = SliderDialog("Zoom Out (%)", 10, 90, 50)

            if dlg.exec_():
                factor = 1.0 - (dlg.value / 100.0)
                factor = max(0.1, factor)
                w, h = self.processed_image.size
                new_w = int(w * factor)
                new_h = int(h * factor)
                self.processed_image = self.processed_image.resize((new_w, new_h), Image.LANCZOS)
                self.show_image(self.processed_image, self.image_label_right)

        elif op == "crop":
            QMessageBox.information(self, "Crop", "Drag pada gambar kiri untuk memilih area crop.")
            self._crop_mode = True
            return

        if self.processed_image:
            self.show_image(self.processed_image, self.image_label_right)

    def qimage_to_cv(self, qimage):
        """Konversi QImage ke numpy array OpenCV (BGR)."""
        qimage = qimage.convertToFormat(4)  # QImage.Format_RGBA8888
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def apply_operation(self, op, img):
        filter_ops = self.get_filter_ops()

        if op in filter_ops:
            self.processed_image = filter_ops[op](img)
            target_label = getattr(self, "image_label_right", None)
            if target_label is None:
                target_label = getattr(self, "image_label_left", None)
            self.show_image(self.processed_image, target_label)

        elif op == "identify":
            available = list(filter_ops.keys())
            msg = "Filter yang tersedia:\n" + "\n".join(available)
            QMessageBox.information(self, "Identify Filters", msg)


        elif op == "reset":

            if hasattr(self, "original_image"):
                self.processed_image = self.original_image.copy()

                self.show_image(self.processed_image, self.image_label_right)  # 👉 tampilkan di kanan

                QMessageBox.information(self, "Reset", "Gambar berhasil dikembalikan ke kondisi awal.")

    # ================= FILTER IDENTIFY =================
    def get_filter_ops(self):
        return {
            # Built-in filters
            "grayscale": lambda img: ImageOps.grayscale(img),
            "invert": lambda img: ImageOps.invert(img),
            "blur": lambda img: img.filter(ImageFilter.BLUR),
            "sharpen_builtin": lambda img: img.filter(ImageFilter.SHARPEN),
            "edge": lambda img: img.filter(ImageFilter.FIND_EDGES),

            # Custom filters
            "sharpen": lambda img: self.sharpen(img, factor=2),
            "unsharp_mask": lambda img: self.unsharp_masking(img),
            "low_pass": lambda img: self.low_pass_filter(img),
            "high_pass": lambda img: self.high_pass_filter(img),
            "bandstop": lambda img: self.bandstop_filter(img),
        }

    def pil_to_cv(self, pil_image):
        """Konversi PIL.Image ke numpy array OpenCV (BGR)."""
        import numpy as np
        import cv2

        # pastikan RGB
        rgb_image = pil_image.convert("RGB")
        np_image = np.array(rgb_image)

        # konversi ke BGR (format default OpenCV)
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    def apply_remove_bg(self):
        # tentukan gambar sumber: hasil terakhir kalau ada, kalau tidak pakai original
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        # konversi ke numpy (PIL → OpenCV BGR)
        img = self.pil_to_cv(src_img)

        # konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ambil warna background dari pojok kiri atas
        bg_color = img[0, 0].tolist()
        bg_color = np.uint8([[bg_color]])
        hsv_bg = cv2.cvtColor(bg_color, cv2.COLOR_BGR2HSV)[0][0]

        # toleransi default
        h_tol, s_tol, v_tol = 15, 50, 50

        lower = np.array([
            max(0, int(hsv_bg[0]) - h_tol),
            max(0, int(hsv_bg[1]) - s_tol),
            max(0, int(hsv_bg[2]) - v_tol)
        ], dtype=np.uint8)

        upper = np.array([
            min(179, int(hsv_bg[0]) + h_tol),
            min(255, int(hsv_bg[1]) + s_tol),
            min(255, int(hsv_bg[2]) + v_tol)
        ], dtype=np.uint8)

        # bikin mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)  # biar halus

        # bikin alpha channel (background jadi transparan)
        alpha = cv2.bitwise_not(mask)

        # gabungkan jadi RGBA (OpenCV pakai BGRA)
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, alpha])

        # konversi ke PIL supaya show_image bisa baca
        rgba_pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

        # simpan hasil
        self.processed_image = rgba_pil
        self.show_image(self.processed_image, self.image_label_right)

    def apply_operation(self, op, img):
        filter_ops = self.get_filter_ops()

        if op in filter_ops:  # ✅ apply filter
            self.processed_image = filter_ops[op](img)
            target_label = getattr(self, "image_label_right", None)
            if target_label is None:
                target_label = getattr(self, "image_label_left", None)
            self.show_image(self.processed_image, target_label)

        elif op == "identify":  # ✅ identify ditekan
            available = list(filter_ops.keys())
            msg = "Filter yang tersedia:\n" + "\n".join(available)
            QMessageBox.information(self, "Identify Filters", msg)

        else:
            # operasi lain (zoom, rotate, dsb.)
            pass

    def wheelEvent(self, event):
        # pilih gambar sumber → hasil terakhir kalau ada, kalau tidak original
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            return
        if not self.image_label_left.underMouse():
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        # hitung faktor zoom
        self._zoom_factor *= 1.1 if delta > 0 else 0.9
        self._zoom_factor = max(0.1, min(self._zoom_factor, 5.0))

        # resize gambar sumber
        w, h = src_img.size
        new_size = (int(w * self._zoom_factor), int(h * self._zoom_factor))
        self.processed_image = src_img.resize(new_size, Image.LANCZOS)

        # tampilkan hasil zoom di kanan
        self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== FILTERS ========================================
    # ============================================================
    def apply_by_name(self, method_name):
        # pilih gambar sumber
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        func = getattr(self.processor, method_name, None)
        if func is None:
            QMessageBox.warning(self, "Error", f"Method {method_name} tidak ditemukan di ImageProcessor")
            return

        try:
            result = func(src_img)
        except TypeError:
            QMessageBox.warning(self, "Error", f"Method {method_name} memerlukan argumen khusus")
            return

        if result is not None:
            self.processed_image = result
            self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_float(self, method_name):
        # pilih gambar sumber
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        dlg = SliderDialog("Parameter (x100)", 10, 300, 150)
        if dlg.exec_():
            val = dlg.value / 100.0
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(src_img, val)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_int(self, method_name):
        # pilih gambar sumber
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        dlg = SliderDialog("Parameter", 1, 100, 10)
        if dlg.exec_():
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(src_img, dlg.value)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_with_slider_gamma(self, method_name):
        # pilih gambar sumber
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        dlg = SliderDialog("Gamma x0.1", 1, 500, 100)
        if dlg.exec_():
            gamma = dlg.value / 10.0
            func = getattr(self.processor, method_name, None)
            if func:
                self.processed_image = func(src_img, gamma)
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
        # pilih gambar sumber
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        dlg = AverageFilterDialog()
        if dlg.exec_():
            func = getattr(self.processor, "average_filter", None)
            if func:
                self.processed_image = func(src_img, size=dlg.value, keep_color=dlg.keep_color)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_gaussian(self, kernel_size):
        # pilih gambar sumber (pakai hasil terakhir kalau ada)
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        func = getattr(self.processor, "gaussian_blur", None)
        if func:
            self.processed_image = func(src_img, kernel_size=kernel_size)
            self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== HISTOGRAM & FUZZY ==============================
    # ============================================================
    def histogram_equalization(self):
        # pakai hasil terakhir kalau ada, kalau belum ada pakai gambar awal
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        func = getattr(self.processor, "histogram_equalization", None)
        if func:
            self.processed_image = func(src_img)
            self.show_image(self.processed_image, self.image_label_right)

    def fuzzy_rgb(self):
        # ambil hasil terakhir kalau ada, kalau belum ada pakai original
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        func = getattr(self.processor, "fuzzy_histogram_equalization_rgb", None)
        if func:
            self.processed_image = func(src_img)
            self.show_image(self.processed_image, self.image_label_right)

    def fuzzy_grayscale(self):
        # ambil hasil terakhir kalau ada, kalau belum ada pakai original
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        func = getattr(self.processor, "fuzzy_histogram_equalization_gray", None)
        if func:
            self.processed_image = func(src_img)
            self.show_image(self.processed_image, self.image_label_right)

    def show_histogram_input(self):
        # gunakan hasil terakhir jika ada, kalau tidak pakai original
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        f = getattr(self.processor, "show_histogram_input", None)
        if f:
            f(src_img)

    def show_histogram_output(self):
        src_img = self.processed_image if self.processed_image else self.original_image

        if src_img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        f = getattr(self.processor, "show_histogram_output", None)
        if f:
            f(src_img)

    def show_histogram_input_output(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        img_out = self.processed_image if self.processed_image else self.original_image

        f = getattr(self.processor, "show_histogram_input_output", None)
        if f:
            f(self.original_image, img_out)

    # ============================================================
    # ========== SEGMENTATION ===================================
    # ============================================================
    def apply_global_threshold(self):
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return
        self.processed_image = self.processor.global_threshold(img)
        self.show_image(self.processed_image, self.image_label_right)

    def apply_adaptive_threshold(self):
        img = self.get_current_image()  # 🔑 ambil gambar aktif (processed kalau ada)
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        f = getattr(self.processor, "adaptive_threshold", None)
        if f:
            self.processed_image = f(img)
            self.show_image(self.processed_image, self.image_label_right)

    def apply_kmeans(self):
        img = self.get_current_image()  # 🔑 ambil gambar aktif
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        dlg = SliderDialog("K (clusters)", 2, 10, 3)
        if dlg.exec_():
            f = getattr(self.processor, "kmeans_clustering", None)
            if f:
                self.processed_image = f(img, K=dlg.value)
                self.show_image(self.processed_image, self.image_label_right)

    def apply_watershed(self):
        img = self.get_current_image()  # 🔑 ambil gambar aktif (processed kalau ada)
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        f = getattr(self.processor, "watershed_segmentation", None)
        if f:
            self.processed_image = f(img)
            self.show_image(self.processed_image, self.image_label_right)

    def apply_region_growing(self):
        img = self.get_current_image()  # 🔑 ambil gambar aktif (processed kalau ada)
        if img is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        x, ok1 = QInputDialog.getInt(self, "Seed Point", "Masukkan X:", 50, 0, 10000, 1)
        y, ok2 = QInputDialog.getInt(self, "Seed Point", "Masukkan Y:", 50, 0, 10000, 1)

        if ok1 and ok2:
            f = getattr(self.processor, "region_growing", None)
            if f:
                self.processed_image = f(img, seed_point=(x, y), threshold=15)
                self.show_image(self.processed_image, self.image_label_right)

    # ============================================================
    # ========== ABOUT ===========================================
    # ============================================================
    def about_dialog(self):
        QMessageBox.information(self, "About", "Aplikasi Image Processing Sederhana\nDibuat dengan PyQt5 + PIL\n")

    # ============================================================
    def apply_brightness_contrast(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Buka gambar dulu!")
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSlider, QPushButton
        import cv2
        import numpy as np
        from PIL import Image

        # Ambil gambar terakhir
        img_pil = self.processed_image if self.processed_image else self.original_image

        class BrightnessContrastDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Brightness & Contrast")
                self.brightness = 30
                self.contrast = 30

                layout = QVBoxLayout()

                layout.addWidget(QLabel("Brightness (-127 s/d 127)"))
                self.slider_b = QSlider()
                self.slider_b.setOrientation(Qt.Horizontal)
                self.slider_b.setRange(-127, 127)
                self.slider_b.setValue(self.brightness)
                layout.addWidget(self.slider_b)

                layout.addWidget(QLabel("Contrast (0 s/d 127)"))
                self.slider_c = QSlider()
                self.slider_c.setOrientation(Qt.Horizontal)
                self.slider_c.setRange(0, 127)
                self.slider_c.setValue(self.contrast)
                layout.addWidget(self.slider_c)

                self.btn_ok = QPushButton("Apply")
                layout.addWidget(self.btn_ok)
                self.setLayout(layout)

                self.btn_ok.clicked.connect(self.accept)

        dlg = BrightnessContrastDialog()
        if dlg.exec_():
            self.brightness = dlg.slider_b.value()
            self.contrast = dlg.slider_c.value()

            # Convert PIL → OpenCV BGR
            img = np.array(img_pil.convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Atur brightness & contrast
            beta = self.brightness
            alpha = self.contrast / 127 + 1.0

            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # Simpan hasil
            self.processed_image = Image.fromarray(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))

            # Tampilkan
            self.show_image(self.processed_image, self.image_label_right)

    def open_arithmetic(self):
        from core.arithmetic_dialog import ArithmeticDialog
        dialog = ArithmeticDialog(self)
        dialog.exec_()

    # ========== SLOT UNTUK CROP ================================
    # ============================================================
    def on_crop_selection(self, rect: QRect):
        if not self._crop_mode:
            return

        # Ambil gambar aktif: processed_image jika ada, kalau tidak pakai original_image
        img = self.processed_image if self.processed_image else self.original_image
        if img is None:
            return

        pixmap = self.image_label_left.pixmap()
        if pixmap is None:
            return

        img_w, img_h = img.size
        disp_w, disp_h = pixmap.width(), pixmap.height()

        if disp_w == 0 or disp_h == 0:
            self._crop_mode = False
            return

        # Hitung skala dan offset agar crop sesuai tampilan
        label_w, label_h = self.image_label_left.width(), self.image_label_left.height()
        scale = min(label_w / img_w, label_h / img_h) if img_w and img_h else 1.0
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        # Koordinat crop di label
        rx1 = rect.left() - offset_x
        ry1 = rect.top() - offset_y
        rx2 = rect.right() - offset_x
        ry2 = rect.bottom() - offset_y

        # Batasi koordinat agar tidak keluar dari label
        rx1 = max(0, min(scaled_w, rx1))
        ry1 = max(0, min(scaled_h, ry1))
        rx2 = max(0, min(scaled_w, rx2))
        ry2 = max(0, min(scaled_h, ry2))

        # Konversi ke koordinat asli gambar
        x_ratio = img_w / scaled_w
        y_ratio = img_h / scaled_h

        x1 = int(rx1 * x_ratio)
        y1 = int(ry1 * y_ratio)
        x2 = int(rx2 * x_ratio)
        y2 = int(ry2 * y_ratio)

        # Pastikan koordinat valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # Crop jika ukuran valid
        if x2 > x1 and y2 > y1:
            cropped = img.crop((x1, y1, x2, y2))
            self.processed_image = cropped
            self.show_image(self.processed_image, self.image_label_right)

        self._crop_mode = False


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()