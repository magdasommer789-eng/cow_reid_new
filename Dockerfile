FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System dependencies for OpenCV and video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data volume — mount /data from host
VOLUME ["/data"]

CMD ["python", "-m", "scripts.train", "--all"]
