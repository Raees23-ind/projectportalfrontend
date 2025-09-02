from django.contrib import admin
from .models import Project, UserProfile

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "owner", "cluster_id", "complexity_score", "duration_days", "created_at")
    search_fields = ("title", "description", "keywords")
    list_filter = ("cluster_id",)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "contact_number")
    search_fields = ("user__username", "contact_number")
