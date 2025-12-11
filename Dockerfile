FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# --- Install System Dependencies ---
# Merged your two blocks into one. Added python3-pip to ensure pip is available.
RUN apt-get update && apt-get install -y \
    openssh-server \
    openssh-client \
    openssh-sftp-server \
    curl \
    git \
    python3-pip \
    zsh \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# --- Install Ollama ---
ENV OLLAMA_MODELS=/workspace/.ollama/models
RUN mkdir -p /workspace/.ollama/models /workspace/.ollama/sentinel && \
    curl -fsSL https://ollama.com/install.sh | sh

# --- Python Setup ---
# Use "python3", which points to the system python (3.10)
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Install pip dependencies ---
WORKDIR /workspace
COPY requirements-docker.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# --- Set shell ---
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

COPY .docker.zshrc /root/.zshrc
RUN chsh -s /usr/bin/zsh root
# Set shell to zsh
# chsh -s /bin/zsh

# --- Create necessary folders ---
RUN mkdir -p output offsets models data

# --- Final Setup ---
COPY docker-start.sh /usr/local/bin/docker-start.sh
RUN chmod +x /usr/local/bin/docker-start.sh

# Expose SSH port (22) and Ollama port (11434)
EXPOSE 22 11434

ENTRYPOINT ["/usr/local/bin/docker-start.sh"]
CMD ["/bin/bash"]
