import torch

from multi_model import Net


def quantum_predict(img_list):
    print("Quantum Prediction:")
    model = Net(0)
    model.load_state_dict(torch.load("../models/model0"))

    predictions = []
    with torch.no_grad():
        for img in img_list:
            img = img.imagearray
            custom_image = torch.from_numpy(img).type(torch.float32) / 255

            custom_image = custom_image.unsqueeze(0)

            model.eval()
            with torch.inference_mode():
                custom_pred = model(custom_image.unsqueeze(dim=0))
                print(custom_pred)
                pred = custom_pred.argmax(dim=1, keepdim=True)
                predictions.append(pred.numpy()[0][0])

    return predictions

