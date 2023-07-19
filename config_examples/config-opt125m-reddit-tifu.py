base_model = "facebook/opt-125m" # # hugging face model name, url to zip file containing model
lora_weights = "https://finetuning-service.s3.us-east-2.amazonaws.com/finetune_outputs/a4140256-c891-4b75-b896-defa46902037/a4140256-c891-4b75-b896-defa46902037.zip" 
# hugging face model name, url to zip file containing model, or None
prompt_structure =  """Generate a summary for context provided
### Context:
{input_prompt}

### tldr:
"""
# Here is a example for a prompt structure with a placeholder for the user input for instruction fine-tuning task
# In above prompt user input_prompt is inserted into the {input_prompt} placeholder
load_in_8bit = False # optional set to True to load model in 8bit mode preferred for large models