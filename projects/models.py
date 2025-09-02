from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

def project_zip_path(instance, filename):
    return f"projects/{instance.owner.id}/{filename}"

class Project(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="projects")
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    keywords = models.CharField(
        max_length=512,
        blank=True,
        help_text="Comma-separated keywords/tools (e.g. React, Django)"
    )
    complexity_score = models.IntegerField(default=1)  # 1..10
    duration_days = models.IntegerField(default=1)
    zip_file = models.FileField(upload_to=project_zip_path, null=True, blank=True)
    cluster_id = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def tools_list(self):
        if not self.keywords:
            return []
        return [t.strip().lower() for t in self.keywords.split(",") if t.strip()]

    def __str__(self):
        return f"{self.title} (id={self.id})"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    contact_number = models.CharField(max_length=30, blank=True, null=True)

    def __str__(self):
        return f"Profile of {self.user.username}"


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    else:
        # ensure profile exists on updates
        UserProfile.objects.get_or_create(user=instance)
