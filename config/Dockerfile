FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire repository
COPY . .

# Default command to run training; points to our simplified training script.
CMD ["python", "models/replicate/informer.py", "--train"]
