
# Start with a stable, lightweight Python environment
FROM python:3.11-slim

# Install system dependencies required for spatial data tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-privileged system user for Hugging Face security compliance
RUN useradd -u 1000 -m user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Set the working environment inside the container
WORKDIR /app

# Upgrade base package managers safely under user scope
RUN pip install --no-cache-dir --upgrade pip

# Copy your exact requirements file first to optimize building speed
COPY --chown=user requirements.txt .

# Install all your exact data science and spatial libraries
RUN pip install --no-cache-dir --user -r requirements.txt

# FORCE DOCKER TO PASS THE HUGGING FACE SECRET TO RUNTIME ENVIRONMENT
ENV EE_SERVICE_ACCOUNT_KEY=$EE_SERVICE_ACCOUNT_KEY

# Copy the rest of your assets and multi-page code directories
COPY --chown=user . .

# Hugging Face Spaces operate strictly on port 7860 by default
EXPOSE 7860

# Execute Solara targeting your modular pages directory directly
CMD ["solara", "run", "pages", "--host=0.0.0.0", "--port=7860", "--production"]