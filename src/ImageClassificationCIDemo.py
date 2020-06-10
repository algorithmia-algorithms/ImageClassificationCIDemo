import Algorithmia
import torch
from PIL import Image
import numpy as np
import json
from torchvision import models
from torchvision import transforms
CLIENT = Algorithmia.client()
SMID_ALGO = "algo://util/SmartImageDownloader/0.2.x"

def load_labels():
    with open('imagenet_class_index.json') as f:
        labels = json.load(f)
    labels = [labels[str(k)][1] for k in range(len(labels))]
    return labels

def load_model(name):
    if name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)
    return model.float().eval()

def load_image(image_url):
    input = {"image": image_url, "resize": {'width': 224, 'height': 224}}
    result = CLIENT.algo(SMID_ALGO).pipe(input).result["savePath"][0]
    local_path = CLIENT.file(result).getFile().name
    img_data = Image.open(local_path)
    return np.asarray(img_data, dtype=np.float)


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    img_tensor = load_image(input)
    transformed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_tensor = transformed(img_tensor).reshape(1, 3, 224, 224).float()
    infer = model.forward(img_tensor).squeeze()
    preds, indicies = torch.sort(torch.softmax(infer, dim=0), descending=True)
    predicted_values = preds.detach().numpy()
    indicies = indicies.detach().numpy()
    result = []
    for i in range(5):
        label = labels[indicies[i]]
        confidence = predicted_values[i]
        result.append({"label": label, "confidence": confidence})
    return result



model = load_model("alexnet")
labels = load_labels()

if __name__ == "__main__":
    input = "https://i.imgur.com/bXdORXl.jpeg"
    result = apply(input)
    print(result)