# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

COPY .env .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Run the backend and frontend in separate processes
CMD ["sh", "-c", "python api.py & streamlit run app.py"]