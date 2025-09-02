from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from .models import Project
from .serializers import ProjectSerializer, SignupSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from ml.pipeline import ProjectClusteringPipeline
from django.contrib.auth.models import User
import numpy as np
from numpy.linalg import norm

# load pipeline at import time (if exists)
pipeline = ProjectClusteringPipeline.load_if_exists()

@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def signup(request):
    serializer = SignupSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    user = serializer.save()
    return Response({"username": user.username}, status=status.HTTP_201_CREATED)


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all().order_by("-created_at")
    serializer_class = ProjectSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        proj = serializer.save(owner=self.request.user)
        global pipeline
        if pipeline and pipeline.is_ready():
            data = {
                "title": proj.title,
                "description": proj.description,
                "keywords": proj.keywords,
                "complexity_score": proj.complexity_score,
                "duration_days": proj.duration_days,
            }
            try:
                cluster = pipeline.predict_single(data)
                proj.cluster_id = int(cluster)
                proj.save()
            except Exception as e:
                print("Pipeline prediction failed:", e)

    @action(detail=False, methods=["POST"], permission_classes=[permissions.IsAdminUser])
    def retrain(self, request):
        """
        Re-train pipeline from all projects in DB.
        Admin only.
        """
        from django.core.management import call_command
        call_command("train_pipeline")
        # reload pipeline
        global pipeline
        pipeline = ProjectClusteringPipeline.load_if_exists()
        return Response({"status": "retrained"})

    @action(detail=True, methods=["GET"], permission_classes=[permissions.AllowAny])
    def similar_grouped(self, request, pk=None):
        """
        Returns similar projects grouped into clusters (Cluster 0, Cluster 1, ...).
        Query params:
          - n: total number of similar candidates to consider (default 12)
        Response:
        {
          "project": { ... },
          "groups": {
            "Cluster 0": [ {project}, ... ],
            "Cluster 1": [...],
            ...
          }
        }
        Requires pipeline to be trained & saved.
        """
        global pipeline
        if not pipeline or not pipeline.is_ready():
            return Response({"detail": "Pipeline not ready. Train pipeline first."}, status=503)

        try:
            target = Project.objects.get(pk=pk)
        except Project.DoesNotExist:
            return Response({"detail": "Project not found."}, status=404)

        # parse params
        try:
            total_n = int(request.query_params.get("n", 12))
        except ValueError:
            total_n = 12

        # prepare candidates (exclude target)
        candidates_qs = Project.objects.exclude(pk=target.pk)
        candidates = list(candidates_qs)
        if not candidates:
            return Response({"project": ProjectSerializer(target, context={"request": request}).data, "groups": {}})

        # Build feature matrices for candidates and target
        texts = [(p.title or "") + " " + (p.description or "") for p in candidates]
        numeric = np.array([[float(p.complexity_score or 0), float(p.duration_days or 0)] for p in candidates])

        # Transform candidates using pipeline components
        try:
            txt_vec = pipeline.vectorizer.transform(texts)  # sparse
            txt_red = pipeline.svd.transform(txt_vec)
            tools_transformed = pipeline.mlb.transform([
                [t.strip().lower() for t in (p.keywords or "").split(",") if t.strip()]
                for p in candidates
            ])
            numeric_scaled = pipeline.scaler.transform(numeric)
            X_candidates = np.hstack([txt_red, tools_transformed, numeric_scaled])
        except Exception as e:
            return Response({"detail": f"Feature transform failed: {e}"}, status=500)

        # Transform target
        t_text = (target.title or "") + " " + (target.description or "")
        t_txt_vec = pipeline.vectorizer.transform([t_text])
        t_txt_red = pipeline.svd.transform(t_txt_vec)
        t_tools_v = pipeline.mlb.transform([
            [t.strip().lower() for t in (target.keywords or "").split(",") if t.strip()]
        ])
        t_numeric = np.array([[float(target.complexity_score or 0), float(target.duration_days or 0)]])
        t_num_scaled = pipeline.scaler.transform(t_numeric)
        x_target = np.hstack([t_txt_red, t_tools_v, t_num_scaled]).reshape(1, -1)

        # cosine similarity
        def cosine_sim_vector(target_vec, matrix):
            a = target_vec
            b = matrix
            a_norm = norm(a)
            b_norm = norm(b, axis=1)
            denom = a_norm * b_norm
            denom[denom == 0] = 1e-9
            sims = (b @ a.T).flatten() / denom
            return sims

        sims = cosine_sim_vector(x_target.flatten(), X_candidates)
        top_idx = np.argsort(-sims)[:min(total_n, len(candidates))]

        # prepare grouped result by cluster_id
        groups = {}

        for idx in top_idx:
            p = candidates[int(idx)]
            sim_score = float(sims[int(idx)])
            try:
                cat_cluster = pipeline.predict_single({
                    "title": p.title,
                    "description": p.description,
                    "keywords": p.keywords,
                    "complexity_score": p.complexity_score,
                    "duration_days": p.duration_days,
                })
                cat_name = f"Cluster {int(cat_cluster)}"
            except Exception:
                cat_name = "Unclustered"

            if cat_name not in groups:
                groups[cat_name] = []

            ser = ProjectSerializer(p, context={"request": request}).data
            ser["similarity"] = sim_score
            groups[cat_name].append(ser)

        return Response({
            "project": ProjectSerializer(target, context={"request": request}).data,
            "groups": groups
        }, status=200)
