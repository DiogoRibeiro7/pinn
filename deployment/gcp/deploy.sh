#!/usr/bin/env bash
# Deploy the PINN model server to Google Cloud Run.
set -euo pipefail
PROJECT_ID=${GCP_PROJECT:-$(gcloud config get-value project)}
IMAGE=gcr.io/$PROJECT_ID/pinn-server
docker build -t $IMAGE -f Dockerfile .

gcloud auth configure-docker

docker push $IMAGE
gcloud run deploy pinn-server --image $IMAGE --platform managed --region ${GCP_REGION:-us-central1} --allow-unauthenticated
