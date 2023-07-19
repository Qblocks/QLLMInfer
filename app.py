import os
import time
import uuid
import shutil
import atexit
import zipfile
import logging
import tempfile
import requests
import warnings
from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks

import peft
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add description to the API
from config import base_model, lora_weights, prompt_structure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Using base model: {}".format(base_model))
logger.info("Using lora weights: {}".format(lora_weights))
logger.info("Using prompt structure: {}".format(prompt_structure))

try:
    from config import load_in_8bit
except ImportError:
    logger.warning("load_in_8bit not defined in config.py, defaulting to False")
    load_in_8bit = False

warnings.simplefilter('ignore')

class GPTInput(BaseModel):
    input_prompt: str
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.99
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 2
    max_new_tokens: Optional[int] = 400
    repetition_penalty: Optional[float] = 1.3

    class Config:
        schema_extra = {
            "example": {
                "input_prompt": "Write a response that appropriately completes the request.",
                "temperature": 0.4,
                "top_p": 0.99,
                "top_k": 40,
                "num_beams": 2,
                "max_new_tokens": 400,
                "repetition_penalty": 1.3
            }
        }

def download_model_and_unzip(url):

    # Create a temporary directory
    model_dir = tempfile.TemporaryDirectory()

    # Download the zip file
    r = requests.get(url, allow_redirects=True)
    zip_path = os.path.join(model_dir.name, 'model.zip')
    open(zip_path, 'wb').write(r.content)

    # Unzip the zip file into the temporary directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir.name)

    # Remove the zip file
    os.remove(zip_path)

    # Return the temporary directory
    return model_dir



class InferenceModule:
    def __init__(self, base_model: str, lora_weights: Optional[str]):
        # Check if base_model is a url and download and extract if not
        if base_model.startswith('http'):
            base_model_dir = download_model_and_unzip(base_model)
            base_model = base_model_dir.name
            atexit.register(shutil.rmtree, base_model_dir.name, ignore_errors=True) 
        
        if lora_weights is not None and lora_weights.startswith('http'):
            lora_weights_dir = download_model_and_unzip(lora_weights)            
            lora_weights = lora_weights_dir.name
            atexit.register(shutil.rmtree, lora_weights_dir.name, ignore_errors=True)
            assert os.path.exists(lora_weights)
            

        self.tokenizer = AutoTokenizer.from_pretrained(lora_weights)

        self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto', torch_dtype=torch.float16, load_in_8bit=load_in_8bit)
        if lora_weights is not None:
            self.model = peft.PeftModel.from_pretrained(self.model, lora_weights)

        self.generator = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        self.results = {}

    async def generate_response_async(self, input_data: GPTInput):
        response = self.generate_response(**input_data)
        pid = int(uuid.uuid4())
        self.results[pid] = response
        return pid

    def generate_response(self, input_prompt: str, temperature: float = 0.4, top_p: float = 0.99, 
                top_k: int = 40, num_beams: int = 2, max_new_tokens: int = 400, repetition_penalty: float = 1.3, prompt: Optional[str] = None):
        """
        Method to generate a response given an instruction using the fine-tuned model.

        Parameters:
        -----------
        input_prompt: str
            The input prompt to generate a response for.
        temperature: float
            The temperature to use for generation.
        top_p: float
            The top p value to use for generation.
        top_k: int
            The top k value to use for generation.
        num_beams: int
            The number of beams to use for generation.
        max_new_tokens: int
            The maximum number of tokens to generate.
        repetition_penalty: float
            The repetition penalty to use for generation.
        prompt: str = None
            Input prompt configuration.

        Returns:
        --------
        finetuned_generation: str
        """
        st = time.time()
        prompt = prompt_structure.format(input_prompt=input_prompt)

        generation_config = transformers.GenerationConfig(
            temperature=0.4,
            top_p=0.99,
            top_k=40,
            num_beams=2,
            max_new_tokens=400,
            repetition_penalty=1.3,
        )
        
        t = self.generator(prompt, generation_config=generation_config)
        finetuned_generation = t[0]['generated_text']
        et = time.time()
        generation_time = et - st
        return finetuned_generation, generation_time

app = FastAPI()  
inference_module = InferenceModule(base_model, lora_weights)

@app.get("/")
async def root():
    # Display the API description
    return {"message": "Welcome to the Monster LLM Serving Model API. \
            Please use the /docs endpoint to view the API documentation."}

@app.post("/generate_response/")
async def generate_response(input_data: GPTInput, background_tasks: BackgroundTasks):
    pid = await inference_module.generate_response_async(input_data.dict())
    response, exec_time = inference_module.results.get(pid)
    if response is None:
        return {"message": "No results found for the given PID"}

    del inference_module.results[pid]
    return {"result": response, "processing_time": exec_time}
