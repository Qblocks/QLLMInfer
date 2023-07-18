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

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


RUN python3 -m pip install --upgrade pip

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app using the InferenceModule
CMD streamlit run app.py --server.port 8501
