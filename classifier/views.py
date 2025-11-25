from django.shortcuts import render
import os
from django.conf import settings
from .ai_model.model import AiSkinModel

model = AiSkinModel()

def handle_uploaded_file(f):
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, f.name)

    with open(file_path, "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    return settings.MEDIA_URL + "uploads/" + f.name


def upload_image(request):
    prediction = None
    image_url = None

    if request.method == "POST" and request.FILES.get("image"):
        file = request.FILES["image"]

        # Зберігаємо файл
        image_url = handle_uploaded_file(file)

        # Шлях на диску для моделі
        image_path = os.path.join(settings.MEDIA_ROOT, "uploads", file.name)

        # Прогноз
        prediction = model.predict_image(image_path)

    return render(request, "classifier/upload.html", {
        "prediction": prediction,
        "image_url": image_url
    })
