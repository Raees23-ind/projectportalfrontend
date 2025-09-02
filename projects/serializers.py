from rest_framework import serializers
from .models import Project, UserProfile
from django.contrib.auth.models import User

class ProjectSerializer(serializers.ModelSerializer):
    owner = serializers.StringRelatedField(read_only=True)
    zip_file = serializers.FileField(required=False, allow_null=True)

    class Meta:
        model = Project
        fields = ["id", "owner", "title", "description", "keywords", "complexity_score", "duration_days", "zip_file", "cluster_id", "created_at"]
        read_only_fields = ["cluster_id", "created_at", "owner"]


class SignupSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(write_only=True, required=True)
    contact_number = serializers.CharField(write_only=True, required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ("username", "email", "contact_number", "password")

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("This username is already taken.")
        return value

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("This email is already registered.")
        return value

    def create(self, validated_data):
        username = validated_data["username"]
        password = validated_data["password"]
        email = validated_data.get("email", "")
        contact_number = validated_data.get("contact_number", "")

        user = User(username=username, email=email)
        user.set_password(password)
        user.save()

        # create or update profile contact number
        profile, _ = UserProfile.objects.get_or_create(user=user)
        if contact_number:
            profile.contact_number = contact_number
            profile.save()

        return user
