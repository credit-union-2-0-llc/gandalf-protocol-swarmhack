#!/bin/bash
# deploy.sh — Push Gandalf Protocol to Azure Container Registry and deploy to ACI
# Prerequisites:
#   - Azure CLI installed (az --version)
#   - Logged in: az login
#   - Resource group created: az group create -n gandalf-rg -l westus2
#   - Container Registry created: az acr create -g gandalf-rg -n gandalfacr --sku Basic

set -e

# Configuration — EDIT THESE
REGISTRY_NAME="gandalfacr"
REGISTRY_URL="${REGISTRY_NAME}.azurecr.io"
IMAGE_NAME="gandalf-protocol"
IMAGE_TAG="latest"
RESOURCE_GROUP="gandalf-rg"
CONTAINER_NAME="gandalf-protocol-$(date +%s)"
REGION="westus2"

# Get Azure credentials for ACR
REGISTRY_USERNAME=$(az acr credential show -n $REGISTRY_NAME -g $RESOURCE_GROUP -o tsv --query 'username')
REGISTRY_PASSWORD=$(az acr credential show -n $REGISTRY_NAME -g $RESOURCE_GROUP -o tsv --query 'passwords[0].value')

echo "=== GiftGym Azure Deploy ==="
echo "Registry: $REGISTRY_URL"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"

# Build and push to ACR
echo "Building Docker image..."
az acr build -r $REGISTRY_NAME -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Get storage connection string (assumes storage account named gandalfstorage)
STORAGE_ACCOUNT="gandalfstorage"
STORAGE_KEY=$(az storage account keys list -n $STORAGE_ACCOUNT -g $RESOURCE_GROUP -o tsv --query '[0].value')
STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=${STORAGE_ACCOUNT};AccountKey=${STORAGE_KEY};EndpointSuffix=core.windows.net"

# Deploy to Azure Container Instances
echo "Deploying to ACI..."
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG} \
  --registry-login-server $REGISTRY_URL \
  --registry-username $REGISTRY_USERNAME \
  --registry-password $REGISTRY_PASSWORD \
  --environment-variables \
    ANTHROPIC_API_KEY="YOUR_API_KEY_HERE" \
    AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONNECTION_STRING" \
  --port 5000 \
  --protocol HTTP \
  --cpu 2 \
  --memory 2 \
  --restart-policy OnFailure

# Get the FQDN
echo "Container created: $CONTAINER_NAME"
az container show -g $RESOURCE_GROUP -n $CONTAINER_NAME --query ipAddress.fqdn -o tsv

echo "Deploy complete. Access at: http://$(az container show -g $RESOURCE_GROUP -n $CONTAINER_NAME --query ipAddress.fqdn -o tsv):5000"
