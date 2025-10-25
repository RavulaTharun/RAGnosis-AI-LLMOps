## Parent image
FROM python:3.10-slim

## Essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## Work directory inside the docker container
WORKDIR /app

## Installing system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

## Copying all contents from local to container
COPY . .

## Install Python dependencies
RUN pip install --no-cache-dir -e .

## Expose only flask port
EXPOSE 5000

## Run the Flask app
CMD ["python", "app/application.py"]

