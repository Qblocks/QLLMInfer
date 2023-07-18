# MonsterAPI 
=====================================================
## LLMInfer Streamlit App with GPU Support
=====================================================

This repository contains a Streamlit app that uses a fine-tuned LORA weights and base model for generating responses based on user instructions. The app is designed to run with GPU support using Docker.

Prerequisites
-------------

To run the Streamlit app with GPU support, you will need:
- NVIDIA GPU drivers installed on your host machine.
- Docker runtime with GPU support.
- NVIDIA Container Toolkit (nvidia-docker2) for GPU support in Docker containers.

Build the Docker Image
----------------------

1. Clone this repository to your local machine:

```bash
git clone git@github.com:Qblocks/LLMInfer.git
cd LLMInfer/
```

2. Build the Docker image:

```bash
docker build -t monsterapi-llminfer .
```

3. Run the Docker Container

```bash
docker run --gpus all -p 8501:8501 monsterapi-llminfer
```

4. Open the Streamlit app in your browser

```bash
http://localhost:8501
```


