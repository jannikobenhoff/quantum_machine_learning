import numpy as np
import scipy.signal
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import cv2

from new_try import quanv, MyModel


class Zahl():
    def __init__(self, imagearray):
        super(Zahl, self).__init__()
        self.imagearray = imagearray

    def flat(self):
        return self.imagearray.reshape(-1, 28*28)


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.ones(shape=(l, l))
    return kernel / np.sum(kernel)


def baw(imagearray):
    """BlackAndWhite"""
    maxi = imagearray.max()
    mini = imagearray.min()
    mean = np.mean(imagearray)

    imagearray2 = scipy.signal.convolve2d(imagearray, gkern(l=7), 'same')
    imagearray2 = scipy.signal.convolve2d(imagearray2, gkern(l=7), 'same')

    imagearray2 = imagearray2 - (25-mini)

    """black: 0, white: 255"""
    imagearray[imagearray > imagearray2] = 0
    imagearray[imagearray > 0] = 255

    return imagearray


def wab(imagearray):
    """WhiteAndBlack"""
    imagearray[imagearray < 250] = 0
    return imagearray


def scanning(imagearray):
    zahldrin = []
    zahlen = []
    thresh = 250
    '''Spalten'''
    start = 0
    end = 0
    for i in range(len(imagearray[0])-1):
        if sum(imagearray[:, i]) > thresh:
            if start == 0:
                start = i
            zahldrin.append(imagearray[:, i])
        elif start != 0:
            end = i
        if end != 0 and start != 0:
            zahlen.append(np.transpose(np.array(zahldrin)))
            zahldrin = []
            start = 0
            end = 0
    """Characters löschen, die aus weniger als 2 Spalten bestehen"""
    zahlen_aussortiert = []
    for zahl in zahlen:
        print(zahl.shape)
        if zahl.shape[1] > 2:
            zahlen_aussortiert.append(zahl)

    zahlen = zahlen_aussortiert
    zahlen2 = []
    endlist = []
    startlist = []
    start = 0
    end = 0
    for zahl in zahlen:
        for i in range(len(zahl)):
            if sum(zahl[i, :]) > thresh:
                if start == 0:
                    startlist.append(i)
                    start = i
            elif start != 0:
                end = i
                endlist.append(i)
            if end != 0 and start != 0:

                start = 0
                end = 0
    heightlist = [x - i for x, i in zip(endlist, startlist)]
    for zahl in zahlen:
        zahlen2.append(zahl[min(startlist):max(endlist), :])

    zahlen3 = []
    for zahl in zahlen2:
        start = 0
        end = 0
        for i in range(len(zahl)-1):
            if sum(zahl[i, :]) > thresh and sum(zahl[i+1, :]) > thresh:
                start = i
                break
        for i in range(len(zahl)-1, 0, -1):
            if sum(zahl[i, :]) > thresh and sum(zahl[i-1, :]) > thresh:
                end = i
                break
        zahlen3.append(zahl[start:end+1, :])

    delete = []
    for i in range(len(zahlen3)):
        zahl = zahlen3[i]
        if min(zahl.shape)/max(zahl.shape) < 0.1:
            delete.append(i)

    for i in reversed(delete):
        del zahlen3[i]
    return zahlen3


def scale(imagearray):
    breite = len(imagearray[0])
    höhe = len(imagearray)
    anzahl = int(np.ceil(breite / höhe))
    resized_image = cv2.resize(imagearray, (28*anzahl, 28))
    return resized_image


def addBorder(imagearray, reverse=False):
    sidelength = max(imagearray.shape)
    border = int(sidelength*0.4)
    sidelength = border + sidelength

    if sidelength < 10:
        return np.empty(shape=0)
    out = np.zeros([sidelength, sidelength], dtype=np.uint8)
    if reverse:
        out[out == 0] = 255

    x_start, y_start = int((sidelength-imagearray.shape[0])/2), int((sidelength-imagearray.shape[1])/2)
    out[x_start:x_start + imagearray.shape[0], y_start:y_start + imagearray.shape[1]] = imagearray

    return out


def scan_process(img_file, plot=True, save=False, whiteBG = False):
    """
    Input: .JPG File
    Output: List array mit Zahlen() Elementen
    """
    image = Image.open(img_file).convert('L')

    if save:
        image.save("screen.jpg")

    image = np.array(image)

    image = cv2.resize(image, (int(image.shape[1] / 8), int(image.shape[0] / 8)))

    image = baw(image)

    #plt.imshow(image)
    #plt.show()

    imageList = scanning(image)
    zahlenListe = []
    for image in imageList:

        image = addBorder(image)
        if image != np.empty(shape=0):

            image = scale(image)
            if whiteBG:
                '''Schwarze Zahl, weißer Hintergrund'''
                image = np.invert(image)

            zahlenListe.append(Zahl(image))

    return zahlenListe


def plot_histogram(image1, image2):
    histogram, bin_edges = np.histogram(image1)
    blue = "#6989ff"
    yellow = "#e0ea6c"
    plt.figure()
    plt.plot(bin_edges[0:-1], histogram, color=blue, label="Pre-Threshold", linewidth=3)
    histogram, bin_edges = np.histogram(image2)
    plt.plot(bin_edges[0:-1], histogram, color=yellow, label="After-Threshold", linewidth=3)
    plt.xlabel("Grayscale Value")
    plt.ylabel("Pixel Count")
    plt.xlim([0, 255])
    plt.legend()
    plt.savefig("histogram.pdf")


if __name__ == "__main__":
    zahlen = scan_process("__files/screen2.jpg")
    q_imgs = []
    for img in zahlen:
        print(img.imagearray)
        im = img.imagearray / 255
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))
        q_imgs.append(quanv(im))
    q_imgs = np.asarray(q_imgs)

    n_samples = 4
    n_channels = 4
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
    for k in range(n_samples):
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
            if k != 0:
                axes[c, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(q_imgs[k, :, :, c], cmap="gray")

    plt.tight_layout()
    plt.show()
    # fig, axes = plt.subplots(1, len(zahlen))
    #
    # for i in range(len(zahlen)):
    #     axes[i].imshow(zahlen[i].imagearray, cmap="gray")
    #     axes[i].get_yaxis().set_visible(False)
    #     axes[i].get_xaxis().set_visible(False)
    # plt.show()
    # plt.savefig("real_data_3.pdf")

    print("Quantum Prediction:")
    model = MyModel()
    model.load_weights('../checkpoints/my_checkpoint')
    q_imgs = []
    for img in zahlen:
        im = img.imagearray/255
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))
        q_imgs.append(quanv(im))
    q_imgs = np.asarray(q_imgs)
    predictions = np.argmax(model.predict(q_imgs), axis=1)
    print(predictions)
