from django.contrib import admin
from django.utils.html import format_html
from .models import SkinImage

@admin.register(SkinImage)
class SkinImageAdmin(admin.ModelAdmin):
    list_display = ("id", "preview", "prediction", "created_at")
    readonly_fields = ("preview", "prediction")

    def preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="120" height="120" style="object-fit: cover;"/>',
                obj.image.url
            )
        return "No image"

    preview.short_description = "Preview"
