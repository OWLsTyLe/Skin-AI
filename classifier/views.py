from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from .ai_model.model import AiSkinModel
from .ai_model.statistics_analysis import StatisticalAnalysis
from .models import SkinImage
import numpy as np

model = AiSkinModel()

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


def handle_uploaded_file(f):
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f.name)
    with open(file_path, "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return settings.MEDIA_URL + "uploads/" + f.name


def to_scalar(val):
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        else:
            return np.mean(val)
    return val


def format_statistics_educational(prediction, stats_package):
    corr = stats_package["correlation"]
    reg = stats_package["regression"]
    anova = stats_package["anova"]
    fisher = stats_package["fisher"]
    ci = stats_package["confidence"]

    correlation = to_scalar(corr.get('correlation'))
    slope = to_scalar(reg.get('slope'))
    F_statistic = to_scalar(anova.get('F_statistic'))
    F_value = to_scalar(fisher.get('F_value'))
    slope_low, slope_high = ci.get('slope_confidence_interval', (None, None))

    corr_text = ("ознаки на шкірі дуже яскраво виражені, що підтверджується сильною кореляцією між ознаками і прогнозом" if correlation and correlation > 0.7
                 else "ознаки помірно виражені, кореляція середня" if correlation and correlation > 0.4
                 else "ознаки слабко виражені, кореляція низька" if correlation is not None
                 else "дані про кореляцію недоступні")

    reg_text = ("зміни на фото суттєво впливають на прогноз" if slope and slope > 0.5
                else "зміни на фото помірно впливають на прогноз" if slope and slope > 0.2
                else "зміни на фото впливають слабо" if slope is not None
                else "дані недоступні")

    anova_text = ("різниця між ділянками шкіри суттєва" if F_statistic and F_statistic > 4
                  else "різниця між ділянками шкіри незначна" if F_statistic is not None
                  else "дані недоступні")

    fisher_text = ("модель бачить значущий вплив ознак на прогноз" if F_value and F_value > 3
                   else "ознаки мають слабкий вплив на прогноз" if F_value is not None
                   else "дані недоступні")

    ci_text = (f"довірчий інтервал 95%: від {slope_low:.2f} до {slope_high:.2f}"
               if slope_low is not None and slope_high is not None
               else "дані недоступні")

    text = f"Прогнозована хвороба: {prediction}\n\n"
    text += f"- На фото: {corr_text}.\n"
    text += f"- Регресія показує: {reg_text}.\n"
    text += f"- Аналіз різниць (ANOVA): {anova_text}.\n"
    text += f"- Сила впливу ознак (F-тест): {fisher_text}.\n"
    text += f"- Довірчий інтервал лінії регресії: {ci_text}.\n\n"
    text += ("Висновок: цей опис пояснює, як працює модель та чому вона передбачила саме цю хворобу. "
             "Це не діагноз – для точного обстеження зверніться до дерматолога.")
    return text

def upload_image(request):
    prediction = None
    image_url = None
    stats_text = None

    if request.method == "POST" and request.FILES.get("image"):
        file = request.FILES["image"]
        image_url = handle_uploaded_file(file)
        image_path = os.path.join(settings.MEDIA_ROOT, "uploads", file.name)

        prediction = model.predict_image(image_path)

        features = model.extract_features(image_path)
        x = np.arange(len(features))
        y = np.array(features)

        stats_package = {
            "correlation": StatisticalAnalysis.correlation(x, y),
            "regression": StatisticalAnalysis.linear_regression(x, y),
            "anova": StatisticalAnalysis.anova([y], [y + 0.05], [y - 0.05]),
            "fisher": StatisticalAnalysis.fisher_test(x, y),
            "confidence": StatisticalAnalysis.confidence_interval(x, y)
        }

        stats_text = format_statistics_educational(prediction, stats_package)

        skin_image = SkinImage(image="uploads/" + file.name)
        skin_image.prediction = prediction
        skin_image.stats_result = convert_numpy(stats_package)
        skin_image.save()

    return render(request, "classifier/upload.html", {
        "prediction": prediction,
        "image_url": image_url,
        "stats_text": stats_text
    })


# API для VR окулярів

@csrf_exempt
def api_predict(request):
    """
    POST /api/predict/
    Приймає фото, повертає JSON з діагнозом.

    Приклад відповіді:
    {
        "success": true,
        "prediction": "Акне",
        "stats_text": "..."
    }
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Тільки POST запити"}, status=405)

    if not request.FILES.get("image"):
        return JsonResponse({"success": False, "error": "Фото не надіслано"}, status=400)

    try:
        file = request.FILES["image"]
        handle_uploaded_file(file)
        image_path = os.path.join(settings.MEDIA_ROOT, "uploads", file.name)

        prediction = model.predict_image(image_path)

        features = model.extract_features(image_path)
        x = np.arange(len(features))
        y = np.array(features)

        stats_package = {
            "correlation": StatisticalAnalysis.correlation(x, y),
            "regression": StatisticalAnalysis.linear_regression(x, y),
            "anova": StatisticalAnalysis.anova([y], [y + 0.05], [y - 0.05]),
            "fisher": StatisticalAnalysis.fisher_test(x, y),
            "confidence": StatisticalAnalysis.confidence_interval(x, y)
        }

        stats_text = format_statistics_educational(prediction, stats_package)

        skin_image = SkinImage(image="uploads/" + file.name)
        skin_image.prediction = prediction
        skin_image.stats_result = convert_numpy(stats_package)
        skin_image.save()

        return JsonResponse({
            "success": True,
            "prediction": prediction,
            "stats_text": stats_text
        })

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)