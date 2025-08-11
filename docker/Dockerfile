# Use the official PyTorch image with CUDA 11.3 + cuDNN8
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a writable home directory
RUN mkdir /myhome && chmod 777 /myhome
ENV HOME=/myhome

# Ensure all files in /app are writable (optional)
RUN chmod -R 777 /app

# Set Python as the default entrypoint
ENTRYPOINT ["python"]

# Default command (can be overridden when running the container)
CMD ["model.py"]
