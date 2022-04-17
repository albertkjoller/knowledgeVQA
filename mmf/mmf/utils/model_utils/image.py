from PIL import Image
import requests

def openImage(image_path):
    path = image_path if not image_path.startswith('http') else requests.get(image_path, stream=True).raw
    return Image.open(path).convert('RGB')
