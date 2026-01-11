from PIL import Image
import torch
from torchvision import transforms
from .model import AiSkinModel, CLASSES

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_image(image_path):
    model_object = AiSkinModel()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        output = model_object.model(img)
        pred_idx = output.argmax(dim=1).item()

    return CLASSES[pred_idx]
