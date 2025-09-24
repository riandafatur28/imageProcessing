import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QDialog, QMessageBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import ImageFilter, ImageChops

class ImageProcessor:
    # ================= RGB Colors =================
    def to_yellow(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 2] = 0
        return Image.fromarray(np_img)

    def to_orange(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 1] = (np_img[:, :, 1] * 0.5).astype(np.uint8)
        np_img[:, :, 2] = 0
        return Image.fromarray(np_img)

    def to_cyan(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 0] = 0
        return Image.fromarray(np_img)

    def to_purple(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 1] = 0
        return Image.fromarray(np_img)

    def to_brown(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 0] = (np_img[:, :, 0] * 0.6).astype(np.uint8)
        np_img[:, :, 1] = (np_img[:, :, 1] * 0.4).astype(np.uint8)
        np_img[:, :, 2] = (np_img[:, :, 2] * 0.2).astype(np.uint8)
        return Image.fromarray(np_img)

    def to_red(self, image):
        np_img = np.array(image.convert("RGB"))
        np_img[:, :, 1] = 0
        np_img[:, :, 2] = 0
        return Image.fromarray(np_img)

    def to_gray(self, image):
        return ImageOps.grayscale(image)

    # ================= Grayscale =================
    def to_grayscale_average(self, image):
        np_img = np.array(image.convert("RGB"))
        avg = np.mean(np_img, axis=2).astype(np.uint8)
        return Image.fromarray(avg, mode="L")

    def to_grayscale_lightness(self, image):
        np_img = np.array(image.convert("RGB"))
        max_rgb = np.max(np_img, axis=2)
        min_rgb = np.min(np_img, axis=2)
        gray_np = ((max_rgb + min_rgb) / 2).astype(np.uint8)
        return Image.fromarray(gray_np, mode="L")

    def to_grayscale_luminance(self, image):
        np_img = np.array(image.convert("RGB"))
        lum = (0.2989 * np_img[:, :, 0] +
               0.5870 * np_img[:, :, 1] +
               0.1140 * np_img[:, :, 2]).astype(np.uint8)
        return Image.fromarray(lum, mode="L")

    # ================= Brightness / Contrast =================
    def adjust_contrast(self, image, factor=1.5):
        return ImageEnhance.Contrast(image).enhance(factor)

    def brightness_contrast(self, image, b_factor=1.2, c_factor=1.2):
        img = ImageEnhance.Brightness(image).enhance(b_factor)
        return ImageEnhance.Contrast(img).enhance(c_factor)

    # ================= Invers =================
    def invert(self, image):
        return ImageOps.invert(image.convert("RGB"))

    # ================= Log Brightness =================
    def log_brightness(self, image, factor=10):
        np_img = np.array(image.convert("L")).astype(np.float32)
        c = 255 / np.log(1 + np.max(np_img))
        log_img = c * np.log(1 + np_img) * (factor / 10.0)
        log_img = np.clip(log_img, 0, 255).astype(np.uint8)
        return Image.fromarray(log_img)

    # ================= Bit Depth =================
    def bit_depth(self, image, bits=1):
        np_img = np.array(image.convert("L"))
        levels = 2 ** bits
        factor = 256 // levels
        reduced = (np_img // factor) * factor
        return Image.fromarray(reduced.astype(np.uint8))

    # ================= Gamma Correction =================
    def gamma_correction(self, image, gamma=1.0):
        np_img = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        corrected = np.power(np_img, 1.0 / gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return Image.fromarray(corrected)

    # ================= Histogram Equalization =================
    def histogram_equalization(self, image):
        if image.mode == "L":
            np_img = np.array(image)
            hist, _ = np.histogram(np_img.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_masked = np.ma.masked_equal(cdf, 0)
            cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
            cdf = np.ma.filled(cdf_masked, 0).astype(np.uint8)
            img_eq = cdf[np_img]
            return Image.fromarray(img_eq)
        else:
            r, g, b = image.split()
            r_eq = self.histogram_equalization(r)
            g_eq = self.histogram_equalization(g)
            b_eq = self.histogram_equalization(b)
            return Image.merge("RGB", (r_eq, g_eq, b_eq))

    # ================= Fuzzy HE RGB =================
    def fuzzy_he_rgb(self, image):
        np_img = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        np_img = np.power(np_img, 0.5)
        np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    # ================= Fuzzy Grayscale =================
    def fuzzy_grayscale(self, image):
        gray = image.convert("L")
        np_img = np.array(gray).astype(np.float32) / 255.0
        np_img = np.power(np_img, 0.5)
        np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    # ================= EDGE DETECTION =================
    def edge_detection1(self, image):
        kernel = [-1, -1, -1,
                  -1, 8, -1,
                  -1, -1, -1]
        return image.filter(ImageFilter.Kernel((3, 3), kernel, 1, 0))

    def edge_detection2(self, image):
        kernel = [0, 1, 0,
                  1, -4, 1,
                  0, 1, 0]
        return image.filter(ImageFilter.Kernel((3, 3), kernel, 1, 0))

    def edge_detection3(self, image):
        kernel = [1, 0, -1,
                  0, 0, 0,
                  -1, 0, 1]
        return image.filter(ImageFilter.Kernel((3, 3), kernel, 1, 0))

    def edge_detection(self, image, type=1):
        if type == 1:
            return self.edge_detection1(image)
        elif type == 2:
            return self.edge_detection2(image)
        elif type == 3:
            return self.edge_detection3(image)
        else:
            return self.edge_detection1(image)

    # ================= GAUSSIAN BLUR =================
    def gaussian_blur(self, image, kernel_size=3):
        if kernel_size == 3:
            return image.filter(ImageFilter.GaussianBlur(radius=1))
        elif kernel_size == 5:
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        else:
            return image.filter(ImageFilter.GaussianBlur(radius=1))

    # ================= OTHER FILTERS =================
    def sharpen(self, image, factor=1):
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def unsharp_masking(self, image, radius=2):
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=150, threshold=3))

    def low_pass_filter(self, image):
        return image.filter(ImageFilter.BoxBlur(2))

    def high_pass_filter(self, image):
        kernel = ImageFilter.Kernel((3, 3),
                                    [-1, -1, -1,
                                     -1, 8, -1,
                                     -1, -1, -1], scale=1)
        return image.filter(kernel)

    def bandstop_filter(self, image):
        # Low-pass
        low = self.low_pass_filter(image)
        # High-pass
        high = self.high_pass_filter(image)
        # Gabungkan: bandstop = original - high + low
        temp = ImageChops.subtract(image, high)
        bandstop = ImageChops.add(temp, low)
        return bandstop

    # ================= HELP: 2D Convolution =================
    def convolve2d(self, image, kernel):
        from scipy.signal import convolve2d
        np_img = np.array(image, dtype=np.float32)

        if len(np_img.shape) == 2:  # Grayscale
            conv = convolve2d(np_img, kernel, mode='same', boundary='fill', fillvalue=0)
            conv = np.clip(conv, 0, 255).astype(np.uint8)
            return Image.fromarray(conv)

        elif len(np_img.shape) == 3:  # RGB
            channels = []
            for c in range(3):
                conv = convolve2d(np_img[:, :, c], kernel, mode='same', boundary='fill', fillvalue=0)
                conv = np.clip(conv, 0, 255).astype(np.uint8)
                channels.append(conv)
            conv = np.stack(channels, axis=-1)
            return Image.fromarray(conv)

    # ================= Average Filter =================
    def average_filter(self, image, size=3, keep_color=True):
        """Filter rata-rata (Mean Filter size x size)"""
        if size % 2 == 0:  # pastikan ukuran ganjil
            size += 1
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)

        if keep_color:
            return self.convolve2d(image, kernel)
        else:
            return self.convolve2d(image.convert("L"), kernel)

    # ================= Prewitt & Sobel =================
    def edge_detection_prewitt(self, image, threshold=100):
        import numpy as np
        from scipy import ndimage

        gray = image.convert("L")
        np_img = np.array(gray, dtype=np.float32)

        # Kernel Prewitt
        Kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=np.float32)

        Gx = ndimage.convolve(np_img, Kx)
        Gy = ndimage.convolve(np_img, Ky)

        mag = np.hypot(Gx, Gy)  # magnitude gradien
        mag = (mag / mag.max()) * 255  # normalisasi 0–255

        # Threshold dan pastikan tipe uint8
        edges = (mag > threshold) * 255
        return Image.fromarray(edges.astype(np.uint8))

    def edge_detection_sobel(self, image, threshold=100):
        import numpy as np
        from scipy import ndimage

        gray = image.convert("L")
        np_img = np.array(gray, dtype=np.float32)

        # Kernel Sobel
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)

        Gx = ndimage.convolve(np_img, Kx)
        Gy = ndimage.convolve(np_img, Ky)

        mag = np.hypot(Gx, Gy)  # magnitude gradien
        mag = (mag / mag.max()) * 255  # normalisasi 0-255

        # Threshold dan pastikan tipe uint8
        edges = (mag > threshold) * 255
        return Image.fromarray(edges.astype(np.uint8))

    # ================= Histogram Input =================
    def show_histogram_input(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Tidak ada gambar input!")
            return

        mode = image.mode
        np_in = np.array(image)

        if mode == "L":  # Grayscale
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            # Gambar input
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Gambar Input (Grayscale)", fontsize=14, fontweight="bold")
            axes[0].axis("off")

            # Histogram grayscale
            axes[1].hist(np_in.ravel(), bins=256, color="black", alpha=0.7)
            axes[1].set_xlim([0, 255])
            axes[1].set_title("Histogram Grayscale", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Intensitas")
            axes[1].set_ylabel("Jumlah Piksel")
            axes[1].grid(True, linestyle="--", alpha=0.6)

        else:  # RGB
            channels = ["Red", "Green", "Blue"]
            colors = ["red", "green", "blue"]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))

            # Baris 1: gambar input
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Gambar Input (RGB)", fontsize=14, fontweight="bold")
            axes[0, 0].axis("off")
            axes[0, 1].axis("off")
            axes[0, 2].axis("off")

            # Baris 2: histogram R, G, B
            for i in range(3):
                axes[1, i].hist(np_in[:, :, i].ravel(), bins=256, color=colors[i], alpha=0.7)
                axes[1, i].set_xlim([0, 255])
                axes[1, i].set_title(f"Histogram {channels[i]}")
                axes[1, i].grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        dialog = QDialog()
        dialog.setWindowTitle("Histogram Input")
        layout = QVBoxLayout(dialog)
        layout.addWidget(FigureCanvas(fig))
        dialog.resize(1000, 800)
        dialog.exec_()

    # ================= Histogram Output =================
    def show_histogram_output(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Tidak ada gambar hasil!")
            return

        mode = image.mode
        np_out = np.array(image)

        # --- Range Y fix ---
        y_min, y_max = 1000, 40000

        if mode == "L":  # Grayscale
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            # Gambar output
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Gambar Hasil (Grayscale)", fontsize=14, fontweight="bold")
            axes[0].axis("off")

            # Histogram grayscale
            axes[1].hist(np_out.ravel(), bins=256, color="black", alpha=0.7)
            axes[1].set_xlim([0, 255])
            axes[1].set_ylim([y_min, y_max])  # 🔹 fix range Y
            axes[1].set_title("Histogram Grayscale", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Intensitas")
            axes[1].set_ylabel("Jumlah Piksel")
            axes[1].grid(True, linestyle="--", alpha=0.6)

        else:  # RGB
            channels = ["Red", "Green", "Blue"]
            colors = ["red", "green", "blue"]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))

            # Baris 1: gambar output
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Gambar Hasil (RGB)", fontsize=14, fontweight="bold")
            axes[0, 0].axis("off")
            axes[0, 1].axis("off")
            axes[0, 2].axis("off")

            # Baris 2: histogram R, G, B
            for i in range(3):
                axes[1, i].hist(np_out[:, :, i].ravel(), bins=256, color=colors[i], alpha=0.7)
                axes[1, i].set_xlim([0, 255])
                axes[1, i].set_ylim([y_min, y_max])  # 🔹 fix range Y
                axes[1, i].set_title(f"Histogram {channels[i]}")
                axes[1, i].set_xlabel("Intensitas")
                axes[1, i].set_ylabel("Jumlah Piksel")
                axes[1, i].grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        dialog = QDialog()
        dialog.setWindowTitle("Histogram Output")
        layout = QVBoxLayout(dialog)
        layout.addWidget(FigureCanvas(fig))
        dialog.resize(1000, 800)
        dialog.exec_()

    # ================= Histogram Equalization =================
    def histogram_equalization_rgb(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Buka gambar dulu!")
            return None

        np_img = np.array(image.convert("RGB"))
        img_eq = np.zeros_like(np_img)

        for i in range(3):  # per channel
            img_eq[:, :, i] = cv2.equalizeHist(np_img[:, :, i])

        return Image.fromarray(img_eq)

    def histogram_equalization_gray(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Buka gambar dulu!")
            return None

        gray = np.array(image.convert("L"))
        eq = cv2.equalizeHist(gray)
        return Image.fromarray(eq)

    # ================= Fuzzy Histogram Equalization (RGB & Gray) =================
    def fuzzy_histogram_equalization_gray(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Buka gambar dulu!")
            return None

        gray = np.array(image.convert("L"))
        # normalisasi ke [0,1]
        norm = gray / 255.0

        # fungsi keanggotaan fuzzy (contoh pakai segitiga sederhana)
        fuzzy = np.sqrt(norm)  # membership function
        eq = (255 * fuzzy).astype(np.uint8)
        return Image.fromarray(eq)

    def fuzzy_histogram_equalization_rgb(self, image):
        if image is None:
            QMessageBox.warning(None, "Warning", "Buka gambar dulu!")
            return None

        np_img = np.array(image.convert("RGB"))
        img_fuzzy = np.zeros_like(np_img)

        for i in range(3):
            norm = np_img[:, :, i] / 255.0
            fuzzy = np.sqrt(norm)  # bisa diganti fungsi fuzzy lain
            img_fuzzy[:, :, i] = (255 * fuzzy).astype(np.uint8)

        return Image.fromarray(img_fuzzy)

    # ================= Histogram Input vs Output (adaptif RGB / Grayscale) =================
    def show_histogram_input_output(self, image_in, image_out):
        if image_in is None or image_out is None:
            QMessageBox.warning(None, "Warning", "Buka gambar dan lakukan proses dulu!")
            return

        # Deteksi mode gambar
        mode_in = "RGB" if image_in.mode == "RGB" else "L"
        mode_out = "RGB" if image_out.mode == "RGB" else "L"

        np_in = np.array(image_in.convert(mode_in))
        np_out = np.array(image_out.convert(mode_out))

        # --- Buat subplot: 2 kolom (input/output), baris menyesuaikan channel terbanyak ---
        max_channels = 3 if (mode_in == "RGB" or mode_out == "RGB") else 1
        fig, axes = plt.subplots(max_channels + 1, 2, figsize=(12, 3 * (max_channels + 1)))

        if max_channels == 1:
            axes = np.array([axes])  # supaya konsisten indexing

        # --- Baris 1: tampilkan gambar ---
        axes[0, 0].imshow(image_in, cmap="gray" if mode_in == "L" else None)
        axes[0, 0].set_title("Gambar Input", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(image_out, cmap="gray" if mode_out == "L" else None)
        axes[0, 1].set_title("Gambar Output", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # --- Range Y fix biar konsisten ---
        y_min, y_max = 0, None

        # --- Histogram untuk setiap channel ---
        if mode_in == "RGB":
            channels_in = ["Red", "Green", "Blue"]
            colors_in = ["red", "green", "blue"]
        else:
            channels_in = ["Grayscale"]
            colors_in = ["black"]

        if mode_out == "RGB":
            channels_out = ["Red", "Green", "Blue"]
            colors_out = ["red", "green", "blue"]
        else:
            channels_out = ["Grayscale"]
            colors_out = ["black"]

        for i in range(max_channels):
            # Input
            if i < len(channels_in):
                data_in = np_in[:, :, i] if mode_in == "RGB" else np_in
                axes[i + 1, 0].hist(data_in.ravel(), bins=256, color=colors_in[i], alpha=0.7)
                axes[i + 1, 0].set_xlim([0, 255])
                axes[i + 1, 0].set_ylim([y_min, y_max])
                axes[i + 1, 0].set_title(f"Input {channels_in[i]}", fontsize=11)
                axes[i + 1, 0].set_xlabel("Intensitas (0-255)")
                axes[i + 1, 0].set_ylabel("Jumlah Piksel")
                axes[i + 1, 0].grid(True, linestyle="--", alpha=0.5)
            else:
                axes[i + 1, 0].axis("off")

            # Output
            if i < len(channels_out):
                data_out = np_out[:, :, i] if mode_out == "RGB" else np_out
                axes[i + 1, 1].hist(data_out.ravel(), bins=256, color=colors_out[i], alpha=0.7)
                axes[i + 1, 1].set_xlim([0, 255])
                axes[i + 1, 1].set_ylim([y_min, y_max])
                axes[i + 1, 1].set_title(f"Output {channels_out[i]}", fontsize=11)
                axes[i + 1, 1].set_xlabel("Intensitas (0-255)")
                axes[i + 1, 1].set_ylabel("Jumlah Piksel")
                axes[i + 1, 1].grid(True, linestyle="--", alpha=0.5)
            else:
                axes[i + 1, 1].axis("off")

        # --- Judul besar & layout ---
        fig.suptitle("Histogram Input vs Output", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # --- Tampilkan di dialog PyQt5 ---
        dialog = QDialog()
        dialog.setWindowTitle("Histogram Input-Output")
        layout = QVBoxLayout(dialog)
        layout.addWidget(FigureCanvas(fig))
        dialog.resize(1000, 800)
        dialog.exec_()

    # ---------- REGION GROWING ----------
    def region_growing(self, img, seed_point, threshold=10):
        np_img = np.array(img.convert("L"))  # grayscale
        visited = np.zeros_like(np_img, dtype=np.uint8)
        h, w = np_img.shape
        stack = [seed_point]
        seed_value = np_img[seed_point[1], seed_point[0]]

        while stack:
            x, y = stack.pop()
            if visited[y, x] == 0:
                visited[y, x] = 255
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0:
                        if abs(int(np_img[ny, nx]) - int(seed_value)) < threshold:
                            stack.append((nx, ny))
        return Image.fromarray(visited)

    # ---------- K-MEANS SEGMENTATION ----------
    def kmeans_clustering(self, img, K=3):
        np_img = np.array(img)
        Z = np_img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        clustered = centers[labels.flatten()]
        clustered = clustered.reshape(np_img.shape)
        return Image.fromarray(clustered)

    # ---------- WATERSHED SEGMENTATION ----------
    def watershed_segmentation(self, img):
        np_img = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(np_img, markers)
        np_img[markers == -1] = [255, 0, 0]
        return Image.fromarray(np_img)

    # ---------- GLOBAL THRESHOLD ----------
    def global_threshold(self, img, thresh_val=127):
        np_img = np.array(img.convert("L"))
        _, binary = cv2.threshold(np_img, thresh_val, 255, cv2.THRESH_BINARY)
        return Image.fromarray(binary)

    # ---------- ADAPTIVE THRESHOLD ----------
    def adaptive_threshold(self, img):
        np_img = np.array(img.convert("L"))
        binary = cv2.adaptiveThreshold(np_img, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(binary)

    # ---------- PLOT SEMUA HASIL ----------
    def plot_all_segmentations(self, img, seed_point=(50, 50)):
        results = {
            "Original": img,
            "Region Growing": self.region_growing(img, seed_point),
            "K-Means": self.kmeans_clustering(img),
            "Watershed": self.watershed_segmentation(img),
            "Global Thresh": self.global_threshold(img),
            "Adaptive Thresh": self.adaptive_threshold(img),
        }

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for i, (title, image) in enumerate(results.items()):
            plt.subplot(2, 3, i + 1)
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def morphology(self, pil_img, operation="Erosion", kernel_type="Square", ksize=3):
        # Konversi PIL → OpenCV
        img = np.array(pil_img.convert("RGB"))

        # Tentukan kernel
        if kernel_type == "Square":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif kernel_type == "Cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Pisah channel RGB
        channels = cv2.split(img)
        processed = []

        for ch in channels:
            if operation == "Erosion":
                result = cv2.erode(ch, kernel, iterations=1)
            elif operation == "Dilation":
                result = cv2.dilate(ch, kernel, iterations=1)
            elif operation == "Opening":
                result = cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel)
            elif operation == "Closing":
                result = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
            else:
                result = ch
            processed.append(result)

        merged = cv2.merge(processed)
        return Image.fromarray(merged)

    def remove_background(self, pil_img, threshold=250):
        """
        Menghapus background putih dari gambar (bisa diatur threshold).
        """
        img = np.array(pil_img.convert("RGBA"))  # RGBA untuk transparansi

        # Pisahkan channel
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Buat mask background
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Invert mask (objek = 255, background = 0)
        mask_inv = cv2.bitwise_not(mask)

        # Tambahkan alpha channel (pakai mask)
        img[:, :, 3] = mask_inv

        return Image.fromarray(img, "RGBA")