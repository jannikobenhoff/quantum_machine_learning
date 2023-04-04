import torch
import numpy as np
from multi_model import Net
from new_try import MyModel, quanv
import tensorflow as tf


def quantum_predict(img_list):
    print("Quantum Prediction:")
    model = MyModel()
    model.load_weights('../checkpoints/my_checkpoint')
    q_imgs = []
    for img in img_list:
        im = img.imagearray/255
        images = np.array(im[..., tf.newaxis])
        q_imgs.append(quanv(images))
    q_imgs = np.asarray(q_imgs)
    predictions = np.argmax(model.predict(q_imgs), axis=1)
    print(predictions)
    return predictions

# def quantum_predict(img_list):
#     print("Quantum Prediction:")
#     model = Net(0)
#     model.load_state_dict(torch.load("../models/model0"))
#
#     predictions = []
#     with torch.no_grad():
#         for img in img_list:
#             img = img.imagearray
#             custom_image = torch.from_numpy(img).type(torch.float32) / 255
#
#             custom_image = custom_image.unsqueeze(0)
#
#             model.eval()
#             with torch.inference_mode():
#                 custom_pred = model(custom_image.unsqueeze(dim=0))
#                 print(custom_pred)
#                 pred = custom_pred.argmax(dim=1, keepdim=True)
#                 predictions.append(pred.numpy()[0][0])
#
#     return predictions

