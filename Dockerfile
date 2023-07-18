# Use the official CUDA-based PyTorch image as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app using the InferenceModule
CMD python app.py --base_model "EleutherAI/gpt-j-6b" --lora_weights "Zangs3011/Gptj-6b-vicgalleGPT4-10epochs"
