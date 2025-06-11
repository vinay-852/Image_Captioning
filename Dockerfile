FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and scripts
COPY . .

# Remove bash script, run Python directly
RUN python download_models.py

# Use exec form for ENTRYPOINT and CMD for arguments
ENTRYPOINT ["streamlit"]
CMD ["run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]