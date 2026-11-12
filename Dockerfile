# Production Multi-Stage / Runtime Container for Physics-Informed D2NN
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Guarantee unbuffered logs, non-interactive apt, and headless matplotlib plotting
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKDIR=/workspace

# Install essential system utilities required for headless graphics and OpenCV/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository core modules, configs, tests, and entrypoint
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY run_experiment.py pytest.ini ./

# Create output and benchmark data directories with full permissions for unprivileged host mounts
RUN mkdir -p outputs/runs outputs/figures data/mnist && \
    chmod -R 777 /workspace

# Default entrypoint dispatcher
ENTRYPOINT ["python", "run_experiment.py"]
CMD ["--config", "configs/default.yaml"]
