#!/usr/bin/env bash
# Deploy the PINN model server to Azure Container Instances.
set -euo pipefail
IMAGE_NAME=${IMAGE_NAME:-pinn-server}
RESOURCE_GROUP=${AZ_RESOURCE_GROUP:-pinn-rg}
REGISTRY=${AZ_REGISTRY:-pinnregistry}
REGION=${AZ_REGION:-westeurope}

docker build -t "$IMAGE_NAME" -f Dockerfile .
az acr login --name "$REGISTRY"
docker tag "$IMAGE_NAME" "$REGISTRY.azurecr.io/$IMAGE_NAME"
docker push "$REGISTRY.azurecr.io/$IMAGE_NAME"
az container create --resource-group "$RESOURCE_GROUP" --name pinn-server \
  --image "$REGISTRY.azurecr.io/$IMAGE_NAME" --ports 8000 --dns-name-label pinn-demo --location "$REGION"
