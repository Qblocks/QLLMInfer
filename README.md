# MonsterAPI 
=====================================================
## QLLMInfer Streamlit App with GPU Support
=====================================================

This repository contains a Streamlit app that uses a fine-tuned LORA weights and base model for generating responses based on user instructions. The app is designed to run with GPU support using Docker.

Prerequisites
-------------

To run the Streamlit app with GPU support, you will need:
- Machine with GPU on it. (Note: Choose GPU that can fit selected model.)
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

2. Craete config.py
create config.py to point to the correct basemodel and lora weights if fine-tuned using LoRA approach.

Sample:
```bash
base_model = "EleutherAI/gpt-j-6b" # # hugging face model name, url to zip file containing model
lora_weights = "Zangs3011/Gptj-6b-vicgalleGPT4-10epochs" # hugging face model name, url to zip file containing model, or None
prompt_structure =  """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{input_prompt}

### Response:
"""
# Here is a example for a prompt structure with a placeholder for the user input for instruction fine-tuning task
# In above prompt user input_prompt is inserted into the {input_prompt} placeholder
load_in_8bit = False # Set this to True to fit larger models into GPU memory sacrificing some speed.

```
More examples can be seen at [./config_examples/](./config_examples)

or 

Just copy and rename one frmo our config_examples to try out.

```bash
cp ./config_examples/config-gpt-j-code-alpaca-instruct.py config.py
```

3. Build the Docker image:

```bash
docker build -t monsterapi-llminfer .
```

4. Run the Docker Container

```bash
docker run -dt --gpus all -p 8501:8501 --shm-size=8g monsterapi-llminfer
```

5. Open the FastAPI app in your browser

```bash
http://localhost:8501/docs
```

### Note: By default base model is loaded in float16 precision.