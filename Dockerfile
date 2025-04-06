FROM ubuntu:22.04

# Set timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# System configuration
ENV DISPLAY=:99
ENV QT_X11_NO_MITSHM=1
ENV XAUTHORITY=/tmp/.docker.xauth
ENV TEMP_DIR=/tmp/app_temp
RUN mkdir -p /tmp/app_temp && chmod 777 /tmp/app_temp

ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV XLIB_SKIP_ARGB_VISUALS=1
ENV FREETYPE_PROPERTIES="truetype:interpreter-version=35"

# Create user and set HOME
RUN adduser -u 5678 --disabled-password --gecos "" appuser
ENV HOME=/home/appuser

# Create font and config directories with proper permissions
RUN mkdir -p /home/appuser/.fonts /home/appuser/.config/matplotlib && \
    chown -R appuser:appuser /home/appuser

# Set environment variables for font and matplotlib configuration
ENV FONTCONFIG_PATH=/home/appuser/.fonts
ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    tk-dev \
    dos2unix \
    fonts-dejavu \
    fontconfig \
    libcairo2-dev \
    libfreetype6-dev \
    libpng-dev \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Python environment setup
COPY Pipfile Pipfile.lock ./

# Install Python dependencies system-wide
USER root
RUN pip install --no-cache-dir --ignore-installed blinker==1.4 && \
    pip install --no-cache-dir pipenv && \
    pipenv install --system --dev --deploy

# Local packages
WORKDIR /app
COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir ./pytiling && \
    pip install --no-cache-dir ./pyglet_dragonbones

# Switch to appuser for runtime
USER appuser
