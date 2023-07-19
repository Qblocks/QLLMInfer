# Use the official CUDA-based PyTorch image as the base image
# Use the base image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

#### Install python 3.10 and set it as default python interpreter 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y&& \
    apt-get install -y git zip unzip gcc g++ software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y &&  apt update -y&& \
    apt install python3.10 -y && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    apt install python3.10-venv python3.10-dev -y && \
    curl -Ss https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Run Fast API app2.py file
CMD ["uvicorn", "app:app", "--port", "8501", "--host", "0.0.0.0"]

# docker run -dt --gpus all -p 8501:8501 --shm-size=12g monsterapi-llminfer