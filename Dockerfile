# Base image
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run app (change if needed)
CMD ["python", "app.py"]