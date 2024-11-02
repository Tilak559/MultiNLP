# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set environment variable to force CPU-only for PyTorch
ENV CUDA_VISIBLE_DEVICES=""

# Run the application
CMD ["python", "src/main.py"]
