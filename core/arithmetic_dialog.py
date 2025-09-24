# core/arithmetic_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QGridLayout, QMessageBox
)

class ArithmeticDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Aritmetical Operation")
        self.setFixedSize(600, 300)

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

        # Tombol Hitung
        self.btn_hitung = QPushButton("Hitung Penjumlahan")
        self.btn_hitung.clicked.connect(self.hitung)
        layout.addWidget(self.btn_hitung)

        self.setLayout(layout)

    def hitung(self):
        try:
            a = float(self.txt_input1.toPlainText())
            b = float(self.txt_input2.toPlainText())
            hasil = a + b
            self.txt_output.setPlainText(str(hasil))
        except ValueError:
            QMessageBox.warning(self, "Error", "Masukkan angka valid di Input 1 dan Input 2")
