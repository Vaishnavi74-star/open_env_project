FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set correct working directory
WORKDIR /tmp/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and package setup first for caching
COPY requirements.txt setup.py ./

# Upgrade pip and install generic dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the application files (the entire repo)
COPY . .

# Install the package in editable mode as required by OpenEnv
RUN pip install -e .

# Verify the package is importable at build time
RUN python -c "from ev_charging_env import create_easy_task, create_medium_task, create_hard_task; print('✅ ev_charging_env imports OK')"
RUN python -c "from ev_charging_env.simple_tasks import SimpleTask, create_easy_task; print('✅ simple_tasks imports OK')"

# Expose port for HuggingFace Spaces (if needed)
EXPOSE 7860

# Default command
CMD ["python", "app.py"]
