FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire ev_charging_env package
COPY ev_charging_env/ ./ev_charging_env/

# Copy top-level scripts
COPY inference.py .
COPY openenv.yaml .

# Copy optional files (won't fail if missing during build)
COPY app.py . 2>/dev/null || true
COPY ui.py . 2>/dev/null || true
COPY benchmarks.py . 2>/dev/null || true
COPY documentation.py . 2>/dev/null || true
COPY README.md . 2>/dev/null || true

# Verify the package is importable at build time
RUN python -c "from ev_charging_env import create_easy_task, create_medium_task, create_hard_task; print('✅ ev_charging_env imports OK')"
RUN python -c "from ev_charging_env.simple_tasks import SimpleTask, create_easy_task; print('✅ simple_tasks imports OK')"

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Default command: run inference evaluation
CMD ["python", "inference.py"]
