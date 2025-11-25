from django.db import models
from .ai_model.model import AiSkinModel
import os

class SkinImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    prediction = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        image_path = self.image.path

        model = AiSkinModel()

        self.prediction = predict_image(model, image_path)

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Image {self.id} - {self.prediction}"
