#!/bin/bash
# download_models.sh: Download pretrained models before running Streamlit

# Ensure cache and user data directories exist (these will be mounted as volumes)
mkdir -p /cache
mkdir -p /userdata

# Example: Download a model (replace with your actual model download commands)
# wget -O /app/models/model.pth https://example.com/path/to/model.pth
# python -c "from src.model import download_pretrained; download_pretrained()"

# Add your model download logic here

# Start Streamlit (replace app.py with your actual entrypoint if needed)
exec streamlit run app.py --server.fileWatcherType=none
# --server.fileWatcherType=none can help avoid issues with mounted volumes
