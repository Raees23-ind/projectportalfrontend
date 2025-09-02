# train_pipeline.py
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
import os

# Import the necessary Django BaseCommand
from django.core.management.base import BaseCommand

# from ml.pipeline import ProjectClusteringPipeline
# Note: Re-enable this line when the import path issue is resolved.

class Command(BaseCommand):
    help = 'Runs the project clustering pipeline to train a new model and save it.'

    def handle(self, *args, **options):
        try:
            # Your existing code starts here
            self.stdout.write(self.style.SUCCESS("Starting the training pipeline..."))
            
            # Load your dataset
            # Use os.getcwd() to get the current working directory from where manage.py is run
            CSV_PATH = os.path.join(os.path.dirname(__file__),"projects.csv")
            
            # Since the project's root is the CWD, let's adjust the path to be relative
            # to the project's root if projects.csv is in the same directory as the script.
            
            # This path is more robust as it uses the absolute path of the command file itself
            # CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(_file_)), "projects.csv")
            
            df = pd.read_csv(CSV_PATH)
            self.stdout.write(self.style.SUCCESS("Dataset loaded successfully."))

            # Prepare features for silhouette
            texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
            keywords = df["keywords"].fillna("").tolist()
            numeric = df[["complexity_score", "duration_days"]].fillna(0).astype(float).values

            best_k = None
            best_score = -1
            best_pipeline = None

            # Temporarily import here to avoid the earlier import error
            from ml.pipeline import ProjectClusteringPipeline

            for k in range(2, 12):  # test k=2...11
                try:
                    self.stdout.write(f"--- Testing k={k} ---")
                    pipeline = ProjectClusteringPipeline().fit(df, n_clusters=k)

                    # Rebuild feature matrix
                    text_vec = pipeline.vectorizer.transform(texts).toarray()
                    tools_lists = [[t.strip().lower() for t in s.split(",") if t.strip()] for s in keywords]
                    tools_vec = pipeline.mlb.transform(tools_lists)
                    numeric_scaled = pipeline.scaler.transform(numeric)
                    X = np.hstack([text_vec, tools_vec, numeric_scaled])

                    score = silhouette_score(X, pipeline.kmeans.labels_)
                    self.stdout.write(f"k={k}, silhouette score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_pipeline = pipeline

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Skipping k={k}: {e}"))

            if best_pipeline:
                self.stdout.write(self.style.SUCCESS(f"\n✅ Best k found: {best_k} with silhouette score {best_score:.3f}"))
                
                # Save model
                best_pipeline.save()
                self.stdout.write(self.style.SUCCESS("Model saved successfully!"))

                # Print grouped projects
                labels = best_pipeline.kmeans.labels_
                df["cluster"] = labels
                self.stdout.write("\n--- Project Groups ---")
                for cluster_id in sorted(df["cluster"].unique()):
                    self.stdout.write(f"\nCluster {cluster_id}:")
                    cluster_projects = df[df["cluster"] == cluster_id]
                    for _, row in cluster_projects.iterrows():
                        self.stdout.write(f" - {row['title']}  ({row['keywords']})")
            else:
                self.stdout.write(self.style.WARNING("No valid model could be trained."))
        
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("projects.csv not found. Please check the file path."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An unexpected error occurred: {e}"))