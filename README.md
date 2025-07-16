# Llama Combine

<div align="center">
  <img src="images/llama-combine.png" alt="Llama Combine" width="400">
</div>

A Gradio web application for fine-tuning language models using LoRA (Low-Rank Adaptation). Supports both local development on ARM64 machines and, in the future, production deployment with CUDA GPUs.

## Why "Llama Combine"?

Just like a combine harvester efficiently processes wheat in the field, Llama Combine helps you harvest the power of Large Language Models (LLMs) - specifically the Llama family and other models. It combines (pun intended) the essential tools for fine-tuning: data preparation, model training, and comparison testing all in one streamlined interface. Whether you're cultivating a specialized model for your domain or just experimenting with AI farming, Llama Combine makes the process as smooth as a well-oiled tractor! 🚜🦙

## Quick Start

1. **Set up environment**:
   ```bash
   # Run the setup script
   ./setup_env.sh
   # OR manually copy the template
   cp env-template.txt .env
   ```

2. **Add your tokens** to `.env`:
   - `HF_TOKEN` - Required for Llama models (see Hugging Face Model Access section)
   - `OPENAI_API_KEY` - Required for data generation features

3. **Install and run**:
   ```bash
   pip install -r requirements-dev.txt
   python app.py --dev
   ```

4. **Load a model**:
   - For quick start: Use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (no auth needed)
   - For Llama models: See the Hugging Face Model Access section below

## Features

- Load and fine-tune models from Hugging Face
- Compare before/after model outputs
- Export models to various formats (local download, Hugging Face, Cloudflare R2)
- Data conversion tools for creating training datasets
- Support for both CPU (ARM64) and GPU (CUDA) execution

## Running Modes

### Development Mode (ARM64/CPU)

For local development on ARM64 machines (e.g., Apple Silicon Macs):

```bash
# Copy the example env file and add your keys
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements-dev.txt

# For CPU-only PyTorch on ARM64
pip install torch torchvision torchaudio

# Run in dev mode
python app.py --dev
```

### Production Mode (CUDA GPU)

For deployment on GPU-enabled servers (default mode):

```bash
# Install dependencies
pip install -r requirements.txt

# Run in production mode (default)
python app.py
```

## Environment Variables

Create a `.env` file or set these environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
R2_BUCKET=your_r2_bucket
R2_ENDPOINT=your_r2_endpoint
R2_ACCESS_KEY=your_r2_access_key
R2_SECRET_KEY=your_r2_secret_key
```

## Hugging Face Model Access

### Getting Started with Hugging Face

Some models (like Meta's Llama) require both authentication AND explicit access approval:

1. **Create a Hugging Face Account**
   - Sign up at https://huggingface.co/join
   - Verify your email address

2. **Generate an Access Token**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "llama-combine")
   - Select "read" permissions (minimum required)
   - Copy the token that starts with `hf_...`
   - Add it to your `.env` file: `HF_TOKEN=hf_your_token_here`

3. **Request Access to Gated Models** (e.g., Llama)
   - Visit the model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B)
   - Click the "Request access" button
   - Fill out the form and accept the license terms
   - **Wait for approval** (usually automatic, but can take a few hours)
   - You'll receive an email when approved

### Model Recommendations

**No Authentication Required:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Great for testing, no token needed

**Requires Token Only:**
- Most community models on Hugging Face

**Requires Token + Access Request:**
- `meta-llama/Llama-3.2-1B` - Meta's official Llama models
- Other gated models (will show "Gated model" on their page)

### Troubleshooting Access Issues

**401 Error**: Your token is missing or invalid
- Solution: Check your HF_TOKEN in .env file

**403 Error**: Token works but you don't have access yet
- Solution: Request access on the model page and wait for approval

**Still having issues?**
- Make sure you're logged into Hugging Face when requesting access
- Check your email for approval notifications
- Try TinyLlama while waiting for access to other models

## Deployment to Fly.io

The included `dockerfile` is configured for CUDA GPU deployment:

```bash
fly deploy
```

## Authentication

The app uses password authentication in production mode:
- Default password: `llamafarm321`
- In development mode (`--dev`), authentication is disabled for convenience
- To change the password, modify the `check_auth` function in `app.py`

## Notes

- In DEV mode, training will be slower due to CPU-only execution
- The app automatically detects and displays which mode is active
- Data is stored in `/data` directory (create locally if needed) 