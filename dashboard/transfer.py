import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

labels = {'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9, '4': 10,
          '5': 11, '6': 12, '7': 13, '8': 14, '9': 15, '=': 16, 'a': 17, 'C': 18, 'G': 19, 'H': 20,
          'M': 21, 'n': 22, 'R': 23, 'S': 24, 'T': 25, '[': 26, ']': 27, 'b': 28, 'd': 29,
          '/': 30, 'e': 31, 'f': 32, '/': 33, 'i': 34, 'infty': 35, 'int': 36,
          'l': 37, 'o': 38, 'p': 39, 'pi': 40, 'q': 41, 'sum': 42, '*': 43, 'y': 44,
          '{': 45, '}': 46}


def predict(img_list):
    model = torch.jit.load("/Users/jannikobenhoff/Documents/pythonProjects/formula_detection_cnn/utilities/model_thick.torch")
    predictions = []
    for img in img_list:
        img = img.imagearray
        img = np.invert(img)

        custom_image = torch.from_numpy(img).type(torch.float32) / 255
        custom_image = custom_image.unsqueeze(0)

        model.eval()
        with torch.inference_mode():
            custom_pred = model(custom_image.unsqueeze(dim=0))

        custom_image_pred_probs = torch.softmax(custom_pred, dim=1)

        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        custom_image_pred_label = list(labels.keys())[custom_image_pred_label.data]
        predictions.append(custom_image_pred_label)
    print("Prediction: ", predictions)
    # if "infinity" in predictions:
    #     i = predictions.index("infinity")
    #     predictions[i] = "i"
    #predictions = ['n', '*', '3', '+', '5']
    return "".join(predictions)
