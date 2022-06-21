import sys
import cv2
import numpy as np
import math
import scipy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from matplotlib import pyplot as plt


def konvolusi(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]
    F_height = F.shape[0]
    F_width = F.shape[1]
    H = (F_height) // 2
    W = (F_width) // 2
    out = np.zeros((X_height, X_width))
    for i in np.arange(H + 1, X_height - H):
        for j in np.arange(W + 1, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
                out[i, j] = sum

    return out


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        uic.loadUi('GUITUBES.ui', self)
        self.image = None

#       load,save,reset
        self.Button_loadCitra.clicked.connect(self.load)
        self.button_simpan.clicked.connect(self.save)
        self.Button_reset.clicked.connect(self.reset)

        self.actionLoad.triggered.connect(self.load)
        self.actionSave.triggered.connect(self.save)

#       Aesthetic Purposes
        self.groupBright.setEnabled(False)
        self.groupCont.setEnabled(False)
        self.Button_grayscale.setEnabled(False)
        self.Button_edgeDet.setEnabled(False)
        self.Button_reset.setEnabled(False)
        self.button_simpan.setEnabled(False)
        self.Button_gaussian.setEnabled(False)
        self.Button_contour.setEnabled(False)

#       Main
        self.Button_grayscale.clicked.connect(self.grayscale)
        self.Button_gaussian.clicked.connect(self.GaussianFilter)
        self.Button_edgeDet.clicked.connect(self.edgeDet)
        self.Button_contour.clicked.connect(self.kontur)

#       Dot Operation
        self.button_plusbright.clicked.connect(self.brightnessplus)
        self.button_minbright.clicked.connect(self.brightnessmin)
        self.button_pluscont.clicked.connect(self.contrastplus)
        self.button_mincont.clicked.connect(self.contrastmin)

        self.actionTranslasi.triggered.connect(self.translasi)
        self.actionCrop.triggered.connect(self.Crop)

#       Geometry Operation
        self.actionTranslasi.triggered.connect(self.translasi)
        self.actionZoom_In_2x.triggered.connect(self.ZoomIn2x)
        self.actionZoom_In_3x.triggered.connect(self.ZoomIn3x)
        self.actionZoom_In_4x.triggered.connect(self.ZoomIn4x)
        self.actionZoom_Out_1_2.triggered.connect(self.ZoomOut12)
        self.actionZoom_Out_1_4.triggered.connect(self.ZoomOut14)
        self.actionZoom_Out_3_5.triggered.connect(self.ZoomOut34)
        self.actionCrop.triggered.connect(self.Crop)

#       Spatial Operation
        self.actionGaussian_Filtering.triggered.connect(self.GaussianFilter)
        self.actionKFiltering1.triggered.connect(self.kfiltering1)
        self.actionKFiltering2.triggered.connect(self.kfiltering2)
        self.actionMFiltering1.triggered.connect(self.meanfilter1)
        self.actionMFiltering2.triggered.connect(self.meanfilter1)
        self.actionSharpening.triggered.connect(self.imagesharpen)
        self.actionLaplace.triggered.connect(self.sharpenlaplace)
        self.actionMedian_Filtering.triggered.connect(self.median)

#       Histogram
        self.actionHGrayscale.triggered.connect(self.grayHistogram)
        self.actionHRGB.triggered.connect(self.RGBHistogram)
        self.actionHEqualization.triggered.connect(self.EqualHistogram)

#   untuk meload citra dengan memilih file yang ada pada direktori
    def load(self):
        filename, filter = QFileDialog.getOpenFileName(
            self, 'Open File','',"Image Files(*.jpg / *.jpeg)")
            #self, 'Open File','F:\CodProj\Py\PCD\TuBesKul',"Image Files(*.jpg / *.jpeg)")

        if filename:
            self.image = cv2.imread(filename)
            self.displayImage(1)
            #menampilkan gambar
        else:
            print('Gagal Memuat')
            #jika gagal memuat

        self.labelNamaFile.setText(filename)

        self.opacityEffect = QGraphicsOpacityEffect()
        self.opacityEffect.setOpacity(1)
        self.Button_grayscale.setGraphicsEffect(self.opacityEffect)
        self.Button_grayscale.setEnabled(True)


    def save(self):
        # filename, filter = QFileDialog.getSaveFileName(self, 'Save File','C:\BAHAN\SEMESTER 4\PRAKTIKUM PENGOLAHAN CITRA DIGITAL\Program',"JPG Image (*.jpg)")
        filename, filter = QFileDialog.getSaveFileName(self, 'Save File','',"JPG Image (*.jpg)")
        if filename:
            cv2.imwrite(filename, self.image)
        else:
            print('Tidak Dapat Menyimpan')

    def reset(self):

        self.label_2.clear()
        self.labelHasil.clear()
        self.labelUmur.clear()
        self.label.clear()
        self.labelNamaFile.clear()

        self.Button_reset.setEnabled(False)
        self.Button_gaussian.setEnabled(False)
        self.Button_contour.setEnabled(False)
        self.Button_edgeDet.setEnabled(False)
        self.Button_grayscale.setEnabled(False)
        self.button_simpan.setEnabled(False)
        self.groupBright.setEnabled(False)
        self.groupCont.setEnabled(False)

#   Operasi Titik
    def grayscale(self):
        image = self.image #inisialisasi citra
        grayimage = image
        H,W,C = image.shape #inisialisasi bentuk citra
        for i in range(H):
            for j in range(W):
                grayimage[i, j] = 0.3 * image[i, j][0] + 0.59 * image[i, j][1] + 0.11 * image[i, j][2]
                #proses grayscale
        self.image = grayimage #memanggil hasil proses grayscale
        self.displayImage(2) #menampilkan pada window ke 2

        self.groupBright.setEnabled(True)
        self.groupCont.setEnabled(True)
        self.Button_edgeDet.setEnabled(True)
        self.Button_reset.setEnabled(True)
        self.button_simpan.setEnabled(True)
        self.Button_gaussian.setEnabled(True)


    def brightnessplus(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            #untuk error preventing, mengubah citra menjadi gray dari library opencv
        except:
            pass

        H, W = self.image.shape[:2] #mengambil bentuk citra
        brightness = 20 #nilai brightness
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a + brightness, 0, 255)
                #proses brightness
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                #nilai jika perhitungan sampai ke max/ terlalu minimum
                self.image.itemset((i, j), b)
                #memanggil proses brighness
        self.displayImage(2) #menampilkan pada window ke 2

    def brightnessmin(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            #untuk error preventing, mengubah citra menjadi gray dari library opencv
        except:
            pass

        H, W = self.image.shape[:2]#mengambil bentuk citra
        brightness = -20
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a + brightness, 0, 255)
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                self.image.itemset((i, j), b)
        self.displayImage(2)

    def contrastplus(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # untuk error preventing, mengubah citra menjadi gray dari library opencv
        except:
            pass

        H, W = self.image.shape[:2]#mengambil bentuk citra
        contrast = 1.6 #nilai contrast
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = math.ceil(a * contrast)
                #rumus contrast
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                # nilai jika perhitungan sampai ke max/ terlalu minimum
                self.image.itemset((i, j), b)#memanggil proses contrast
        self.displayImage(2)#menampilkan pada window ke 2

    def contrastmin(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contrast = -1.6
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = math.ceil(a * contrast)

                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b

                self.image.itemset((i, j), b)

        self.displayImage(1)

#   Main Operation
    def edgeDetbackup(self):
        self.Button_contour.setEnabled(True)
        img = self.image

        img_gaussian = cv2.GaussianBlur(img, (11, 11),0)
        imgd = cv2.Canny(img_gaussian, 30, 40, ) #ubah kesobel
        # closing = cv2.dilate(imgd, (1, 1), iterations=2)
        ret, thresh = cv2.threshold(imgd, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strel)
        self.image = closing
        self.displayImage(2)

    def edgeDet(self):
        self.Button_contour.setEnabled(True)
        img = self.image
        #proses pertama
        img_gaussian = cv2.GaussianBlur(img, (11, 11), 0)
        imgd = cv2.Canny(img_gaussian, 30, 40, )
        ret, thresh = cv2.threshold(imgd, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strel)

        #proses kedua agar ddilakukan closing
        img_gaussian2 = cv2.GaussianBlur(closing, (11, 11), 0)
        imgd2 = cv2.Canny(img_gaussian2, 30, 40, )
        ret, thresh = cv2.threshold(imgd2, 127, 255, cv2.THRESH_BINARY)
        strel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strel2)

        self.image = closing2
        self.displayImage(2)


    def GaussianFilter(self):
        self.image = cv2.GaussianBlur(self.image, (11, 11), 0)
        self.displayImage(2)

#   Kontur
    def kontur(self):
        img = self.image
        (cnt, _) = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        total = (len(cnt)/4)
        print(total)
        # teks = "Perkiraan usia {} tahun".format(total)
        teks = "Perkiraan usia {} tahun".format(str(round(total)))
        self.labelUmur.setText(teks)
        jumlahtot =str(round(total))
        self.labelHasil.setText("Jumlah kontur: " + str(len(cnt)))
        self.displayImage(2)


#   Operasi Geometri
    def translasi(self):
        h, w = self.image.shape[:2] #memanggil bentuk citra
        quarter_h, quarter_w = h / 8, w / 8
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        #proses translasi
        img = cv2.warpAffine(self.image, T, (w, h)) #memasukan proses translasi
        self.image = img #memanggil proses translasi
        self.displayImage(1) #menampilkan pada window 1


    def ZoomIn(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        #memanggil library untuk zoom
        cv2.imshow('Original', self.image)
        self.image = resize_image
        self.displayImage(1)
        #ganti selfdisplayiamge
        #cv2.imshow('Zoom In', resize_image)
        #cv2.waitKey()

    def ZoomIn2x(self):
        #untuk zoom 2x
        self.ZoomIn(2)

    def ZoomIn3x(self):
        self.ZoomIn(3)

    def ZoomIn4x(self):
        self.ZoomIn(4)


    def ZoomOut(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def ZoomOut12(self):
        self.ZoomOut(0.5)

    def ZoomOut14(self):
        self.ZoomOut(0.25)

    def ZoomOut34(self):
        self.ZoomOut(0.75)


    def Crop(self):
        x = -400 #keatas
        y = 400 #kebawah
        x1 = 600 #kiri samping
        y1 = 800 #kanan
        crop_img =self.image[x:y, x1:y1] #memanggil proses crop
        self.image = crop_img #menampilkan hasil proses crop
        self.displayImage(1)
        # cv2.imshow('Crop', crop_img) #menampilkan hasil crop
        #cv2.waitKey()

#   Operasi Spasial
    def kfiltering1(self):
        img = self.image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1], ], dtype='float')
        img_out = konvolusi(image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def kfiltering2(self):
        img = self.image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[6, 0, -6],
             [6, 1, -6],
             [6, 0, -6], ], dtype='float')
        img_out = konvolusi(image, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    @pyqtSlot()
    def meanfilter1(self):
        img = self.image #untuk memanggil gamabr
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #untuk memanggil gambar dan format warnanya (gray)
        mean = (1 / 9) * np.array( #rumus mean
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1], ], dtype='float') #kernel filtering
        img_out = konvolusi(image, mean) #memanggil fungsi konvolusi, hasil perhitungan mean
        plt.imshow(img_out, cmap='gray', interpolation='bicubic') #utk membuka window pyplot dengan interpolasi bicubic
        plt.xticks([]), plt.yticks([]) #untuk membaca nilai x dan y di plt
        plt.show() #menampilan plt

    def meanfilter2(self):
        img = self.image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = (1 / 4) * np.array(
            [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 0]], dtype='float')
        img_out = konvolusi(image, mean)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def imagesharpen(self):
        img = self.image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel3 = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0], ], dtype='float')
        img_out = konvolusi(image, kernel3)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def sharpenlaplace(self):
        img = self.image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplace = (1 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]], dtype='float')
        img_out = konvolusi(image, laplace)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def median(self):
        img = self.image #untuk memanggil gambar
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#untuk memanggil gambar dan format warnanya (gray)
        image_out = image.copy()#membuat fungsi image.copy
        h, w = image.shape[:2] #tinggi lebar gambar
        for i in np.arange(3, h - 3): #memanggil fungsi array range
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = image.item(i + k, j + l)
                        neighbors.append(a) #penutup list

                neighbors.sort() #untuk mensortir array
                median = neighbors[24] #listnya sebanya24
                b = median #membuat fungsi median
                image_out.itemset((i, j), b) #outputnya

        plt.imshow(image_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        print(img)

#   Histogram
    @pyqtSlot()
    def grayHistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)

        self.image = gray
        print(self.image)
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    # A10
    @pyqtSlot()
    def RGBHistogram(self):
        color = ('b', 'g', 'r')  # inisialisasi variabel warna
        for i, col in enumerate(color):  # memulai looping untuk membaca nilai rgb
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            # Membuat variabel untuk menghitung nilai r,g, dan b
            plt.plot(histo, color=col)  # menampilkan variabel histo ditambah dengan warna
            plt.xlim([0, 256])  # Merupakan format warnanya
            plt.show()  # untuk menampilkan grafik

    # A11
    @pyqtSlot()
    def EqualHistogram(self):  # nama prosedur
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        # inisialisasi hist dan bins untuk menampilkan histogram dengan skala maks 256
        cdf = hist.cumsum()  # inisialisasi cdf untuk membuat grafik cdf
        cdf_normalized = cdf * hist.max() / cdf.max()  # rumus proses cdf normalisasi
        cdf_m = np.ma.masked_equal(cdf, 0)  # proses masking
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # rumus perataan citra
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # menginputkan nilai cdf
        self.image = cdf[self.image]  # memasukan fungsi cdf ke self.image
        self.displayImage(2)  # menampilkan pada jendela kedua

        plt.plot(cdf_normalized, color='b')  # untuk normaliasisi warna biru
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')  # membuat histogram warna merah maks 256
        plt.xlim([0, 256])  # membuat nilai limit x 0 sampai 256
        plt.legend(('cdf', 'histogram'), loc='upper left')  # membuat window menunjukan keterangan
        plt.show()  # untuk menampilkan histogram

    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                     self.image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

        if windows == 3:
            self.label_3.setPixmap(QPixmap.fromImage(img))
            self.label_3.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_3.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Deteksi Umur Pohon by Kelompok B2')
window.show()
sys.exit(app.exec_())