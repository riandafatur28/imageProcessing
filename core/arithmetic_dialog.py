# core/arithmetic_dialog.py
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QGridLayout, QMessageBox, QComboBox, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage


class ArithmeticDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Operasi Aritmatika Citra")
        self.resize(1200, 800)

        # Layout utama
        layout = QVBoxLayout()

        # Grid untuk input dan output gambar
        grid = QGridLayout()

        # Label Input 1
        self.lbl_input1 = QLabel("Input 1 (Gambar)")
        self.img_input1 = QLabel("Belum ada gambar")
        self.img_input1.setStyleSheet("border: 1px solid black;")
        self.img_input1.setFixedSize(300, 300)
        grid.addWidget(self.lbl_input1, 0, 0)
        grid.addWidget(self.img_input1, 1, 0)

        # Label Input 2
        self.lbl_input2 = QLabel("Input 2 (Gambar)")
        self.img_input2 = QLabel("Belum ada gambar")
        self.img_input2.setStyleSheet("border: 1px solid black;")
        self.img_input2.setFixedSize(300, 300)
        grid.addWidget(self.lbl_input2, 0, 1)
        grid.addWidget(self.img_input2, 1, 1)

        # Label Output
        self.lbl_output = QLabel("Output (Hasil)")
        self.img_output = QLabel("Belum ada hasil")
        self.img_output.setStyleSheet("border: 1px solid black;")
        self.img_output.setFixedSize(300, 300)
        grid.addWidget(self.lbl_output, 0, 2)
        grid.addWidget(self.img_output, 1, 2)

        layout.addLayout(grid)

        # Tombol untuk memilih gambar input
        self.btn_load1 = QPushButton("Pilih Gambar 1")
        self.btn_load1.clicked.connect(self.load_image1)
        layout.addWidget(self.btn_load1)

        self.btn_load2 = QPushButton("Pilih Gambar 2")
        self.btn_load2.clicked.connect(self.load_image2)
        layout.addWidget(self.btn_load2)

        # Dropdown untuk memilih operasi
        self.cmb_operasi = QComboBox()
        self.cmb_operasi.addItems([
            "Penjumlahan Citra (+)",
            "Pengurangan Citra (-)",
            "Perkalian Citra (×)",
            "Pembagian Citra (÷)",
            "Operasi Bitwise OR",
            "Operasi Bitwise AND",
            "Blend Transparansi",
            "Blend Horizontal"   # 👈 Tambahan baru
        ])
        self.cmb_operasi.setMinimumHeight(50)
        layout.addWidget(self.cmb_operasi)

        # Tombol Hitung
        self.btn_hitung = QPushButton("Proses Operasi")
        self.btn_hitung.setMinimumHeight(50)
        self.btn_hitung.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-size: 16px;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.btn_hitung.clicked.connect(self.hitung)
        layout.addWidget(self.btn_hitung)

        self.setLayout(layout)

        # Variabel untuk menyimpan gambar
        self.image1 = None
        self.image2 = None

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 1", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image1 = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self.display_image(self.image1, self.img_input1)

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 2", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self.display_image(self.image2, self.img_input2)

    def display_image(self, img, widget):
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(widget.width(), widget.height())
        widget.setPixmap(pixmap)

    def hitung(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Error", "Harap pilih kedua gambar terlebih dahulu.")
            return

        # Ubah ukuran gambar agar sama tinggi
        h = min(self.image1.shape[0], self.image2.shape[0])
        img1_resized = cv2.resize(self.image1, (int(self.image1.shape[1] * h / self.image1.shape[0]), h))
        img2_resized = cv2.resize(self.image2, (int(self.image2.shape[1] * h / self.image2.shape[0]), h))

        op = self.cmb_operasi.currentText()

        if "Penjumlahan" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            hasil = cv2.add(img1_resized, img2_resized)

        elif "Pengurangan" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            hasil = cv2.subtract(img1_resized, img2_resized)

        elif "Perkalian" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            hasil = cv2.multiply(img1_resized, img2_resized)

        elif "Pembagian" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            img2_resized[img2_resized == 0] = 1
            hasil = cv2.divide(img1_resized, img2_resized)

        elif "Bitwise OR" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            hasil = cv2.bitwise_or(img1_resized, img2_resized)

        elif "Bitwise AND" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            hasil = cv2.bitwise_and(img1_resized, img2_resized)

        elif "Blend Transparansi" in op:
            w = min(img1_resized.shape[1], img2_resized.shape[1])
            img1_resized = cv2.resize(img1_resized, (w, h))
            img2_resized = cv2.resize(img2_resized, (w, h))
            alpha, beta = 0.5, 0.5
            hasil = cv2.addWeighted(img1_resized, alpha, img2_resized, beta, 0)

        elif "Blend Horizontal" in op:
            blend_width = 100  # lebar area overlap
            total_width = img1_resized.shape[1] + img2_resized.shape[1] - blend_width
            hasil = np.zeros((h, total_width, 3), dtype=np.uint8)

            # Tempel gambar 1
            hasil[:, :img1_resized.shape[1]] = img1_resized

            # Buat gradasi mask
            gradient = np.linspace(1, 0, blend_width)
            mask = np.tile(gradient, (h, 1))

            # Overlap
            overlap1 = img1_resized[:, -blend_width:]
            overlap2 = img2_resized[:, :blend_width]
            blended_overlap = (overlap1 * mask[..., None] + overlap2 * (1 - mask[..., None])).astype(np.uint8)
            hasil[:, img1_resized.shape[1] - blend_width:img1_resized.shape[1]] = blended_overlap

            # Sisanya dari gambar 2
            hasil[:, img1_resized.shape[1]:] = img2_resized[:, blend_width:]

        else:
            QMessageBox.warning(self, "Error", "Operasi tidak dikenal.")
            return

        self.display_image(hasil, self.img_output)
