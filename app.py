import argparse
import streamlit as st
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft
import warnings
warnings.simplefilter('ignore')
import gc
import os


class InferenceModule:
    def __init__(self):
        base_model = os.environ.get('base_model', "EleutherAI/gpt-j-6b")
        lora_weights = os.environ.get('lora_weights', "Zangs3011/Gptj-6b-vicgalleGPT4-10epochs")
        self.tokenizer = AutoTokenizer.from_pretrained(lora_weights)
        self.model = AutoModelForCausalLM.from_pretrained(base_model,device_map='auto',torch_dtype=torch.float16)
        self.model = peft.PeftModel.from_pretrained(self.model, lora_weights)
        self.generator = transformers.pipeline("text-generation",model=self.model,tokenizer=self.tokenizer)
        print(100*"@")

    def generate_response(self, instruction: str, temperature: float = 0.4, top_p: float = 0.99, 
                top_k: int = 40, num_beams: int = 2, max_new_tokens: int = 400, repetition_penalty: float = 1.3):
        """
        Method to generate a response given an instruction using the fine-tuned model.

        Parameters:
        -----------
        instruction: str
            The instruction to generate a response for.
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

        Returns:
        --------
        finetuned_generation: str
        """
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        """.format(instruction=instruction)

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
        return finetuned_generation

    def initialize_streamlit_app(self):
        st.title('GPT-J-6B Demo (Fine-tuned Model from LORA Weights)')
        st.write("This is a demo of GPT-J-6B fine-tuned on the VicGalle/GPT4 dataset using LORA weights. The model was fine-tuned for 10 epochs and the best checkpoint was selected.")

        instruction = st.text_input('Enter an instruction to generate a response', value='Write a response that appropriately completes the request.')

        temperature = st.number_input('Temperature', value=0.4, step=0.1)
        top_p = st.number_input('Top P', value=0.99, step=0.01)
        top_k = st.number_input('Top K', value=40, step=1)
        num_beams = st.number_input('Num Beams', value=2, step=1)
        max_new_tokens = st.number_input('Max New Tokens', value=400, step=10)
        repetition_penalty = st.number_input('Repetition Penalty', value=1.3, step=0.1)

        assert type(instruction) == str, 'Please enter a string!'
        st.write('Instruction:', instruction)
        finetuned_generation = self.generate_response(instruction, temperature, top_p, top_k, 
            num_beams, max_new_tokens, repetition_penalty)

        st.write('Fine-tuned Model Output:')
        st.write(finetuned_generation)
    

if __name__ == "__main__":
    inference_module = InferenceModule()
    inference_module.initialize_streamlit_app()
