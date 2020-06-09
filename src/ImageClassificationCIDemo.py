import Algorithmia
import torch
from PIL import Image
import numpy as np
from torchvision import models

CLIENT = Algorithmia.client()
SMID_ALGO = "algo://util/SmartImageDownloader/0.2.x"

def load_model(name):
    if name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)
    return model

def load_image(image_url):
    result = CLIENT.algo(SMID_ALGO).pipe(image_url).result["savePath"][0]
    local_path = CLIENT.file(result).getFile().name
    img_data = Image.open(local_path)
    return torch.from_numpy(np.asarray(img_data))


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    img_tensor = load_image(input)
    infer = model.forward(img_tensor)
    result = infer.detach().numpy()
    return result.tolist()



model = load_model("mobilenet")