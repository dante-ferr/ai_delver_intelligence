FROM tensorflow/tensorflow:2.15.0-gpu

# Set timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# Create user and set HOME
RUN adduser -u 5678 --disabled-password --gecos "" appuser
ENV HOME=/home/appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Python environment setup
COPY Pipfile Pipfile.lock ./

# Install Python dependencies system-wide
USER root
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --dev --deploy

# Switch to appuser for runtime
USER appuser
