# Use the latest Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files into the container
COPY . .

# Expose the port used by the application
ENV LISTEN_PORT=5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py", "--host=0.0.0.0"]
