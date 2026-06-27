# Basic Dockerfile for serving PINN models with FastAPI
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[deploy]
ENV PINN_MODEL_PATH=/models/model.pt
EXPOSE 8000
CMD ["uvicorn", "pinn.deployment.server:app", "--host", "0.0.0.0", "--port", "8000"]
