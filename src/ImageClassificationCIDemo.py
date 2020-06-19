import Algorithmia
import torch
from PIL import Image
import json
from torchvision import models
from torchvision import transforms

CLIENT = Algorithmia.client()
SMID_ALGO = "algo://util/SmartImageDownloader/0.2.x"
LABEL_PATH = "data://AlgorithmiaSE/image_cassification_demo/imagenet_class_index.json"

def load_labels():
    local_path = CLIENT.file(LABEL_PATH).getFile().name
    with open(local_path) as f:
        labels = json.load(f)
    labels = [labels[str(k)][1] for k in range(len(labels))]
    return labels


def load_model(name):
    if name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)
    return model.float().eval()


def load_image(image_url):
    input = {"image": image_url, "resize": {'width': 224, 'height': 224}}
    result = CLIENT.algo(SMID_ALGO).pipe(input).result["savePath"][0]
    local_path = CLIENT.file(result).getFile().name
    img_data = Image.open(local_path)
    return img_data


def infer_image(image_url, n):
    image_data = load_image(image_url)
    transformed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_tensor = transformed(image_data).unsqueeze(dim=0)
    infered = model.forward(img_tensor)
    preds, indicies = torch.sort(torch.softmax(infered.squeeze(), dim=0), descending=True)
    predicted_values = preds.detach().numpy()
    indicies = indicies.detach().numpy()
    result = []
    for i in range(n):
        label = labels[indicies[i]].lower().replace("_", " ")
        confidence = float(predicted_values[i])
        result.append({"label": label, "confidence": confidence})
    return result


def calculate_topn_accuracy(results):
    accuracy = 0.0
    for result in results:
        label = result['label']
        for pred in result['predictions']:
            if label == pred['label']:
                accuracy += 1.0
                break
    accuracy /= len(results)
    return accuracy


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    if isinstance(input, dict):
        if "n" in input:
            n = input['n']
        else:
            n = 3
        if "data" in input:
            if isinstance(input['data'], str):
                output = infer_image(input['data'], n)
            elif isinstance(input['data'], list):
                for row in input['data']:
                    row['predictions'] = infer_image(row['image_url'], n)
                output = input['data']
            else:
                raise Exception("'data' must be a image url or a list of image urls (with labels)")
            if "operation" in input:
                if input['operation'] == "benchmark":
                    accuracy = calculate_topn_accuracy(output)
                    return accuracy
                else:
                    return output
            else:
                return output

        else:
            raise Exception("'data' must be defined")
    else:
        raise Exception("input  must be a dictionary/json object")

model = load_model("squeezenet")
labels = load_labels()

if __name__ == "__main__":
    input = {"data": [
                 {"image_url": "https://i.imgur.com/bXdORXl.jpg", "label": "pomeranian"},
                 {"image_url": "https://i.imgur.com/YcAZMxM.jpg", "label": "pomeranian"},
                 {"image_url": "https://i.imgur.com/QMRNUMN.jpg", "label": "necklace"},
                 {"image_url": "https://i.imgur.com/o7WP6Px.jpg", "label": "necklace"},
                 {"image_url": "https://i.imgur.com/FzwSR.jpg", "label": "manhole cover"},
                 {"image_url": "https://i.imgur.com/EzllwpE.jpg", "label": "cardigan"},
                {"image_url": "https://i.imgur.com/HMvOHn7.jpg", "label": "cardigan"},
                {"image_url": "https://i.imgur.com/xcLDUQd.jpg", "label": "white wolf"},
                {"image_url": "https://i.imgur.com/gN6zgtN.jpg", "label": "lotion"},
                {"image_url": "https://i.imgur.com/MCt8OWb.jpg", "label": "burrito"},
                {"image_url": "https://i.imgur.com/lhWanDq.jpg", "label": "basketball"},
                {"image_url": "https://i.imgur.com/BZsMhIY.jpeg", "label": "lab coat"},
                     ], "n": 3, "operation": "benchmark"}
    # input = {"image_url": "https://i.imgur.com/bXdORXl.jpeg"}
    result = apply(input)
    print(result)
