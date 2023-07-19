base_model = "EleutherAI/gpt-j-6b" # # hugging face model name, url to zip file containing model
lora_weights = "Zangs3011/Gptj-6b-vicgalleGPT4-10epochs" # hugging face model name, url to zip file containing model, or None
prompt_structure =  """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{input_prompt}

### Response:
"""
# Here is a example for a prompt structure with a placeholder for the user input for instruction fine-tuning task
# In above prompt user input_prompt is inserted into the {input_prompt} placeholder
load_in_8bit = True # optional set to True to load model in 8bit mode