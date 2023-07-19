base_model = "EleutherAI/gpt-j-6b"
lora_weights = "gvij/gpt-j-code-alpaca-instruct" # Finetuned using MonsterAPI no-code LLM finetuning service
prompt_structure =  """Below is an prompt that describes auto completion code to be generated. Write a completion that appropriately completes the request.
### Prompt:
{input_prompt}

### Completion:
"""
# Here is a example for a prompt structure with a placeholder for the user input for instruction fine-tuning task
# In above prompt user input_prompt is inserted into the {input_prompt} placeholder
# Also prompt structured to suit the usecase of generating code for a given prompt.