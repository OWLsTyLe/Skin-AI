from django.db import models
from .ai_model.model import AiSkinModel, CLASSES
from .ai_model.predict import predict_image
from .ai_model.statistics_analysis import StatisticalAnalysis
import numpy as np


class SkinImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    prediction = models.CharField(max_length=255, blank=True, null=True)
    stats_result = models.JSONField(null=True, blank=True)  # Сюди пишуться результати статистики
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        image_path = self.image.path

        model = AiSkinModel()
        self.prediction = predict_image(image_path)

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 2, 3, 5])

        stats_package = {
            "correlation": StatisticalAnalysis.correlation(x, y),
            "regression": StatisticalAnalysis.linear_regression(x, y),
            "anova": StatisticalAnalysis.anova([1, 2, 2], [2, 3, 3], [4, 5, 5]),
            "fisher": StatisticalAnalysis.fisher_test(x, y),
            "confidence": StatisticalAnalysis.confidence_interval(x, y),
        }

        self.stats_result = stats_package

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Image {self.id} - {self.prediction}"
