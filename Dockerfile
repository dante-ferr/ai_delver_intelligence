FROM tensorflow/tensorflow:2.15.0-gpu

# Set timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# Accept UID/GID as build args and create user/group
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup && \
    useradd -m -u $UID -g $GID appuser

# System packages
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Install pipenv and project dependencies
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pipenv && pipenv install --system --dev --deploy

# Switch to the created user
USER appuser
ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1

CMD ["python3", "src/main.py"]
