import torch
import torch.nn as nn
import numpy as np
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

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не знайдена: {MODEL_PATH}")

        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(CLASSES))

        # Завантажуємо що є — або state_dict, або повна модель
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint, dict):
            # це state_dict
            self.model.load_state_dict(checkpoint)
        else:
            # це повна модель
            self.model = checkpoint

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_class = CLASSES[pred_idx]

        return CLASS_NAMES.get(pred_class, "Невідома хвороба")

    def extract_features(self, image_path):
        # Реальні ознаки з фото через forward hook на передостанній шар
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)

        features = []

        def hook_fn(module, input, output):
            features.append(output.squeeze().detach().numpy())

        hook = self.model.features[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(img_tensor)

        hook.remove()

        if features:
            feat = features[0]
            if feat.ndim > 1:
                feat = feat.mean(axis=(-2, -1))
            return feat[:5]
        else:
            return np.zeros(5)