import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from django.conf import settings


MODEL_PATH = os.path.join(settings.BASE_DIR, "classifier", "ai_model", "saved", "skin_model_full.pth")


CLASSES = ["acne", "atopic dermatitis", "eczema", "healthy", "melanoma", "psoriasis"]

CLASS_NAMES = {
    "acne": "Акне",
    "atopic dermatitis": "Атопічний дерматит",
    "eczema": "Екзема",
    "healthy": "Здорова шкіра",
    "melanoma": "Меланома",
    "psoriasis": "Псоріаз",
}

class AiSkinModel:
    def __init__(self):

        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(CLASSES))

        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Модель не знайдена: {MODEL_PATH}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            pred_idx = output.argmax(dim=1).item()
            pred_class = CLASSES[pred_idx]

        return CLASS_NAMES.get(pred_class, "Невідома хвороба")
