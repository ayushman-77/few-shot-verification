FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run the production backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
