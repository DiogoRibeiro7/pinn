#!/usr/bin/env bash
# Deploy the PINN model server to AWS using Elastic Container Registry and ECS.
set -euo pipefail
IMAGE_NAME=${IMAGE_NAME:-pinn-server}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"

docker build -t "$IMAGE_NAME" -f Dockerfile .
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker tag "$IMAGE_NAME" "$ECR_URL"
docker push "$ECR_URL"
echo "Image pushed to $ECR_URL"
