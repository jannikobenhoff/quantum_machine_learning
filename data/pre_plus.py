from os import listdir
from os.path import isfile, join

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dashboard.scanning import baw, addBorder
from new_try import quanv


def stroking(image):
    """Adding line thickness to character"""
    '''255: weiß, 0: schwarz'''
    img = np.array(image)
    img = addBorder(img, reverse=True)

    for i in range(img.shape[0]-2):
        for ii in range(img.shape[1]-2):
            if img[i, ii] > 250 and img[i, ii+1] < 40:
                img[i, ii] = 0
                img[i, ii+1] = 0
                img[i, ii+2] = 0
                img[i, ii-2] = 0
                img[i, ii-1] = 0

    for i in range(img.shape[0]-2):
        for ii in range(img.shape[0]-2):
            if img[ii, i] > 250 and img[ii+1, i] < 40:
                img[ii, i] = 0
                img[ii-1, i] = 0
                img[ii-2, i] = 0
                img[ii+1, i] = 0
                img[ii+2, i] = 0

    return img
# 25,4 13

if __name__ =="__main__":
    new = False
    if new:
        print("NEW")
        list = []
        for i, f in enumerate(listdir("plus/+")):
            img = Image.open("plus/+/{}".format(f)).convert('L')

            im = stroking(img)
            im = cv2.resize(im, (28, 28))
            im = np.reshape(im, (28, 28, 1))
            im = np.invert(im)

            list.append(quanv(im))
            if i > 2000:
                break
        plus_images = np.asarray(list)

        # Save pre-processed images
        np.save("q_train_images_plus.npy", plus_images)

    plus_train_images = np.load("q_train_images_plus.npy")
    n_samples = 4
    n_channels = 4
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
    for k in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        print(plus_train_images[k, :, :, 0])
        axes[0, k].imshow(plus_train_images[k, :, :, 0], cmap="gray")

        # Plot all output channels
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
            if k != 0:
                axes[c, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(plus_train_images[k, :, :, c], cmap="gray")

    plt.tight_layout()
    plt.show()