# core/arithmetic_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton,
    QGridLayout, QMessageBox, QComboBox
)

class ArithmeticDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Aritmetical Operation")
        self.resize(1200, 800)

        # Layout utama
        layout = QVBoxLayout()

        # Grid untuk input dan output
        grid = QGridLayout()

        # Label Input 1
        self.lbl_input1 = QLabel("Input 1")
        self.txt_input1 = QTextEdit()
        grid.addWidget(self.lbl_input1, 0, 0)
        grid.addWidget(self.txt_input1, 1, 0)

        # Label Input 2
        self.lbl_input2 = QLabel("Input 2")
        self.txt_input2 = QTextEdit()
        grid.addWidget(self.lbl_input2, 0, 1)
        grid.addWidget(self.txt_input2, 1, 1)

        # Label Output
        self.lbl_output = QLabel("Output")
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        grid.addWidget(self.lbl_output, 0, 2)
        grid.addWidget(self.txt_output, 1, 2)

        layout.addLayout(grid)

        # Dropdown untuk memilih operasi
        self.cmb_operasi = QComboBox()
        self.cmb_operasi.addItems([
            "Penjumlahan (+)",
            "Pengurangan (-)",
            "Perkalian (×)",
            "Pembagian (÷)"
        ])
        self.cmb_operasi.setMinimumHeight(60)  # biar dropdown lebih tinggi
        layout.addWidget(self.cmb_operasi)

        # Tombol Hitung
        self.btn_hitung = QPushButton("Hitung")
        self.btn_hitung.setMinimumHeight(60)  # biar tombol lebih tinggi
        self.btn_hitung.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;   /* biru */
                color: white;               /* tulisan putih */
                font-size: 16px;            /* biar font lebih besar */
                border-radius: 8px;         /* sudut membulat */
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;  /* biru lebih gelap saat hover */
            }
        """)
        self.btn_hitung.clicked.connect(self.hitung)
        layout.addWidget(self.btn_hitung)

        self.setLayout(layout)

    def hitung(self):
        try:
            a = float(self.txt_input1.toPlainText())
            b = float(self.txt_input2.toPlainText())
            op = self.cmb_operasi.currentText()

            if "Penjumlahan" in op:
                hasil = a + b
            elif "Pengurangan" in op:
                hasil = a - b
            elif "Perkalian" in op:
                hasil = a * b
            elif "Pembagian" in op:
                if b == 0:
                    QMessageBox.warning(self, "Error", "Tidak bisa dibagi dengan nol")
                    return
                hasil = a / b
            else:
                hasil = "Operasi tidak dikenal"

            self.txt_output.setPlainText(str(hasil))

        except ValueError:
            QMessageBox.warning(self, "Error", "Masukkan angka valid di Input 1 dan Input 2")
