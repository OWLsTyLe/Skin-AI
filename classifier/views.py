from django.shortcuts import render
import os
from django.conf import settings
from .ai_model.model import AiSkinModel
from .ai_model.statistics_analysis import StatisticalAnalysis
from .models import SkinImage
import numpy as np

model = AiSkinModel()

def convert_numpy(obj):
    """
    Рекурсивно перетворює всі numpy масиви та числа в стандартні Python типи.
    """
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
    """
    Конвертує numpy масив у скаляр.
    Якщо масив містить >1 елементів, повертає середнє.
    """
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        else:
            return np.mean(val)  # беремо середнє, щоб уникнути помилок
    return val
def format_statistics_educational(prediction, stats_package):
    """
    Формує зрозумілий та одночасно освітній опис для користувача про прогноз хвороби на фото.
    Пояснює, як статистичні методи (кореляція, регресія, ANOVA, F-тест, довірчі інтервали)
    вплинули на прогноз.
    """
    corr = stats_package["correlation"]
    reg = stats_package["regression"]
    anova = stats_package["anova"]
    fisher = stats_package["fisher"]
    ci = stats_package["confidence"]

    correlation = to_scalar(corr.get('correlation'))
    slope = to_scalar(reg.get('slope'))
    intercept = to_scalar(reg.get('intercept'))
    r_value = to_scalar(reg.get('r_value'))
    F_statistic = to_scalar(anova.get('F_statistic'))
    F_value = to_scalar(fisher.get('F_value'))
    slope_low, slope_high = ci.get('slope_confidence_interval', (None, None))

    # Кореляція
    if correlation is not None:
        if correlation > 0.7:
            corr_text = "ознаки на шкірі дуже яскраво виражені, що підтверджується сильною кореляцією між ознаками і прогнозом"
        elif correlation > 0.4:
            corr_text = "ознаки помірно виражені, кореляція середня"
        else:
            corr_text = "ознаки слабко виражені, кореляція низька"
    else:
        corr_text = "дані про кореляцію недоступні"

    # Регресія (метод найменших квадратів)
    if slope is not None:
        if slope > 0.5:
            reg_text = "зміни на фото суттєво впливають на прогноз. Метод найменших квадратів показує сильний зв'язок між ознаками та прогнозом"
        elif slope > 0.2:
            reg_text = "зміни на фото помірно впливають на прогноз. Лінія регресії відображає середній зв'язок між ознаками та прогнозом"
        else:
            reg_text = "зміни на фото впливають слабо. Лінія регресії майже не змінюється через окремі ознаки"
    else:
        reg_text = "дані про вплив змін на прогноз недоступні"

    # ANOVA (дисперсійний аналіз)
    if F_statistic is not None:
        if F_statistic > 4:
            anova_text = "різниця між ділянками шкіри суттєва, що підтверджується дисперсійним аналізом (ANOVA)"
        else:
            anova_text = "різниця між ділянками шкіри незначна, модель бачить шкіру як однорідну"
    else:
        anova_text = "дані про різницю між ділянками недоступні"

    # Критерій Фішера
    if F_value is not None:
        if F_value > 3:
            fisher_text = "модель бачить значущий вплив ознак на прогноз (F-тест). Тобто зміни, які ми спостерігаємо, реально важливі"
        else:
            fisher_text = "ознаки мають слабкий вплив на прогноз, значимість рівняння регресії низька"
    else:
        fisher_text = "дані про значимість впливу недоступні"

    # Довірчий інтервал
    if slope_low is not None and slope_high is not None:
        ci_text = (f"лінія регресії може коливатись у межах довірчого інтервалу 95%: "
                   f"від {slope_low:.2f} до {slope_high:.2f}, що показує можливі зміни прогнозу при нових вимірах")
    else:
        ci_text = "дані про довірчий інтервал відсутні"

    text = f"Прогнозована хвороба: {prediction}\n\n"
    text += f"- На фото: {corr_text}.\n"
    text += f"- Регресія показує: {reg_text}.\n"
    text += f"- Аналіз різниць (ANOVA): {anova_text}.\n"
    text += f"- Сила впливу ознак (F-тест): {fisher_text}.\n"
    text += f"- Довірчий інтервал лінії регресії: {ci_text}.\n\n"
    text += ("Висновок: цей опис пояснює, як працює модель та чому вона передбачила саме цю хворобу. "
             "Це допомагає закріпити основні статистичні принципи (метод найменших квадратів, кореляція, "
             "регресія, дисперсійний аналіз, критерій Фішера та довірчі інтервали). "
             "Це не діагноз – для точного обстеження зверніться до дерматолога.")

    return text


def upload_image(request):
    prediction = None
    image_url = None
    stats_text = None

    if request.method == "POST" and request.FILES.get("image"):
        file = request.FILES["image"]

        # Зберігаємо файл
        image_url = handle_uploaded_file(file)
        image_path = os.path.join(settings.MEDIA_ROOT, "uploads", file.name)

        # Прогноз AI
        prediction = model.predict_image(image_path)

        # Витягуємо реальні ознаки зі зображення для статистики
        features = model.extract_features(image_path)  # масив чисел ознак
        x = np.arange(len(features))
        y = np.array(features)

        stats_package = {
            "correlation": StatisticalAnalysis.correlation(x, y),
            "regression": StatisticalAnalysis.linear_regression(x, y),
            "anova": StatisticalAnalysis.anova([y], [y + 0.05], [y - 0.05]),  # приклад для різних ділянок
            "fisher": StatisticalAnalysis.fisher_test(x, y),
            "confidence": StatisticalAnalysis.confidence_interval(x, y)
        }

        # Формуємо текст словами
        stats_text = format_statistics_educational(prediction, stats_package)

        # Збереження в БД
        skin_image = SkinImage(image="uploads/" + file.name)
        skin_image.prediction = prediction
        skin_image.stats_result = convert_numpy(stats_package)
        skin_image.save()

    return render(request, "classifier/upload.html", {
        "prediction": prediction,
        "image_url": image_url,
        "stats_text": stats_text
    })
