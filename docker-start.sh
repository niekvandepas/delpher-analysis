#!/bin/bash
set -e

# --- Fix SSH Latency ---
# Disable DNS lookups (prevents connection lag)
echo "UseDNS no" >> /etc/ssh/sshd_config
# Disable GSSAPI auth (prevents login/auth lag)
echo "GSSAPIAuthentication no" >> /etc/ssh/sshd_config
# Optimize QoS to prevent packet buffering
echo "IPQoS throughput" >> /etc/ssh/sshd_config

# --- Setup SSH Server ---
mkdir -p /var/run/sshd
/usr/sbin/sshd
echo "✅ SSH Server started"

# --- Start Ollama ---
echo "🚀 Starting Ollama..."
ollama serve &
sleep 5

# --- Smart Model Pulling ---
if ollama list | grep -q "llama3:8b"; then
    echo "✅ Model llama3:8b already exists. Skipping pull."
else
    echo "📥 Model not found. Pulling llama3:8b..."
    ollama pull llama3:8b
fi

# --- Smart Git Clone ---
cd /workspace
REPO_DIR="delpher-analysis"

if [ -d "$REPO_DIR" ]; then
    echo "🔄 Repository exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull || echo "⚠️  Git pull failed (likely local changes), continuing..."
else
    echo "Cloning repository..."
    git clone https://github.com/niekvandepas/delpher-analysis.git
fi

# --- Keep Alive (Crucial for RunPod) ---
echo "✨ Setup complete. Keeping container alive..."
sleep infinity
