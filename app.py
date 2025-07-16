import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset, Dataset
import torch
import zipfile
import os
from huggingface_hub import login, HfApi
from openai import OpenAI  # For ChatGPT integration
import boto3  # For R2/S3
import json  # For data validation
import argparse  # For command-line arguments
import sys
from dotenv import load_dotenv  # For loading .env file
from datetime import datetime  # For timestamps

# Parse command-line arguments
parser = argparse.ArgumentParser(description='AI Model Studio')
parser.add_argument('--dev', action='store_true', help='Run in development mode (ARM64/CPU)')
args = parser.parse_args()

# Load .env file in development mode
if args.dev:
    load_dotenv()

# Set device based on mode
if args.dev:
    DEVICE = "cpu"
    DEVICE_MAP = "cpu"
    print("Running in DEV mode: Using CPU (ARM64 compatible)")
else:
    DEVICE = "cuda"
    DEVICE_MAP = "cuda"
    print("Running in PROD mode: Using CUDA GPU")

# Check if CUDA is available when not in dev mode
if not args.dev and not torch.cuda.is_available():
    print("WARNING: CUDA not available but running in PROD mode. This may cause errors.")

# Persistent paths on Fly volume (or local in dev mode)
DATA_DIR = "./data" if args.dev else "/data"
ORIGINAL_MODEL_DIR = os.path.join(DATA_DIR, "original_models")  # New: For before model
MODEL_DIR = os.path.join(DATA_DIR, "models")  # For after/fine-tuned
MERGED_MODEL_DIR = os.path.join(DATA_DIR, "merged_models")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
os.makedirs(ORIGINAL_MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch secrets from Fly.io environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
R2_BUCKET = os.getenv('R2_BUCKET')
R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')

def load_model(model_id):
    # Pass token if available for models that require authentication
    kwargs = {}
    if HF_TOKEN:
        kwargs['token'] = HF_TOKEN
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    except OSError as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            return f"""Error loading model: Authentication required.

For Llama models, you need to:
1. Create a Hugging Face account at https://huggingface.co
2. Go to https://huggingface.co/{model_id} and accept the license
3. Create an access token at https://huggingface.co/settings/tokens
4. Add HF_TOKEN to your .env file (dev) or Fly secrets (prod)

Current HF_TOKEN status: {'Set' if HF_TOKEN else 'Not set'}

Try using TinyLlama instead - it doesn't require authentication."""
        elif "403" in str(e) or "restricted" in str(e):
            return f"""Error loading model: Access not granted yet.

✅ Your HF_TOKEN is working!
❌ But you need to request access to this model:

1. Visit: https://huggingface.co/{model_id}
2. Click "Request access" button
3. Accept the license terms
4. Wait a few minutes for approval (usually automatic)
5. Try loading again

Alternative: Use "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - no access required!"""
        else:
            return f"Error loading model: {str(e)}"
    
    # Save original
    model.save_pretrained(ORIGINAL_MODEL_DIR)
    tokenizer.save_pretrained(ORIGINAL_MODEL_DIR)
    # Copy to fine-tune dir
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return "Model loaded successfully! Original preserved for comparison."

# Old chat_fn removed - using new chat_with_model function in Chat & Test tab

def generate_data_suggestions(description, num_suggestions=10):
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set in Fly secrets."
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""Generate exactly {num_suggestions} JSONL lines for fine-tuning data based on: {description}. 
    
    IMPORTANT: Return ONLY the JSON objects, one per line, with no additional text, markdown, or explanations.
    Each line must be a valid JSON object in this exact format:
    {{"prompt": "user input", "completion": "AI response"}}
    
    Do not include ```json or ``` markers. Do not include any explanatory text."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Clean up the response to ensure proper JSONL format
    content = response.choices[0].message.content.strip()
    
    # Remove any markdown code blocks if present
    if content.startswith("```"):
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip():
                cleaned_lines.append(line.strip())
        content = '\n'.join(cleaned_lines)
    
    # Parse and validate each line, then properly format as JSONL
    valid_lines = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            # Try to parse as JSON to validate
            parsed = json.loads(line)
            # Ensure it has the required fields
            if 'prompt' in parsed and 'completion' in parsed:
                # Re-serialize to ensure consistent formatting
                valid_lines.append(json.dumps(parsed))
        except json.JSONDecodeError:
            # Skip invalid lines
            continue
    
    if valid_lines:
        return '\n'.join(valid_lines)
    else:
        return "Error: Could not generate valid JSONL data. Please try again with a clearer description."

def convert_any_to_jsonl(any_data):
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set in Fly secrets."
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    Convert the following raw data into strict JSONL format for fine-tuning. Each line must be a valid JSON object: {{"prompt": "user input or question", "completion": "AI response or answer"}}.
    Rules:
    - Infer pairs from the text: e.g., if it's a conversation, split user messages as 'prompt' and AI as 'completion'.
    - If it's unstructured text, create logical Q&A pairs.
    - If it's a list or table, treat each item/row as a pair.
    - Output ONLY the JSONL lines, no extra text, explanations, or headers.
    - Ensure each line is valid JSON; no trailing commas or invalid syntax.
    - Aim for 5-20 lines depending on input size; keep prompts/completions concise.
    Raw data: {any_data}
    """
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def validate_jsonl(data):
    try:
        lines = data.strip().split('\n')
        for line in lines:
            if line.strip():
                json.loads(line)
        return True, "Valid JSONL!"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSONL: {str(e)}"

def add_data(pasted_data, uploaded_file, append=False):
    mode = "a" if append and os.path.exists(DATASET_PATH) else "w"
    if uploaded_file:
        with open(DATASET_PATH, "wb") as f:
            f.write(uploaded_file)
    elif pasted_data:
        with open(DATASET_PATH, mode) as f:
            if mode == "a" and pasted_data:
                f.write("\n" + pasted_data.strip())
            else:
                f.write(pasted_data.strip())
    return "Data added to persistent volume! Ready for fine-tuning."

def view_dataset():
    """View the current dataset"""
    if not os.path.exists(DATASET_PATH):
        return "No dataset found. Please add data first.", 0
    
    try:
        with open(DATASET_PATH, "r") as f:
            data = f.read()
        
        # Count lines
        lines = [line.strip() for line in data.split('\n') if line.strip()]
        
        # Validate and show sample
        valid_lines = []
        for line in lines[:10]:  # Show first 10 lines as sample
            try:
                json.loads(line)
                valid_lines.append(line)
            except:
                pass
        
        sample = '\n'.join(valid_lines) if valid_lines else "No valid JSONL lines found"
        return f"Dataset contains {len(lines)} lines.\n\nFirst few lines:\n{sample}", len(lines)
    except Exception as e:
        return f"Error reading dataset: {str(e)}", 0

def clear_dataset():
    """Clear the current dataset"""
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
        return "Dataset cleared successfully."
    return "No dataset to clear."

def fine_tune(lora_rank, epochs, model_name=None):
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        return "Error: No dataset found. Please add data in the 'Add Data' tab first."
    
    # Validate dataset first
    try:
        with open(DATASET_PATH, "r") as f:
            data = f.read()
    except Exception as e:
        return f"Error reading dataset: {str(e)}"
    
    valid, msg = validate_jsonl(data)
    if not valid:
        return msg
    
    # Create model directory with name if provided
    if model_name:
        model_save_dir = os.path.join(DATA_DIR, "models", model_name)
    else:
        # Use timestamp if no name provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = os.path.join(DATA_DIR, "models", f"model_{timestamp}")
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map=DEVICE_MAP)  # Start from copy of original
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Ensure model is in training mode
    model.train()
    
    # Enable gradients for LoRA training
    for param in model.parameters():
        param.requires_grad = False
    
    lora_config = LoraConfig(r=lora_rank, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # This will show which parameters are trainable
    
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        # Concatenate prompt and completion for causal LM training
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Format as a conversation
            text = f"{prompt}\n\n{completion}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize with padding and truncation
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(DATA_DIR, "ft_output"), 
        num_train_epochs=epochs, 
        per_device_train_batch_size=1 if args.dev else 4,  # Smaller batch for CPU
        save_steps=100,
        fp16=False if args.dev else True,  # Disable mixed precision on CPU
        dataloader_pin_memory=False if args.dev else True,  # Disable pin memory for CPU
        gradient_checkpointing=False,  # Disable gradient checkpointing to avoid grad issues
        logging_steps=10,
        eval_strategy="no",  # Disable evaluation to speed up training
        save_strategy="no",  # Don't save intermediate checkpoints
        report_to="none",  # Disable reporting to wandb, etc.
        remove_unused_columns=False,  # Keep all columns
    )
    # Try new API first, fall back to old API for compatibility
    try:
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, processing_class=tokenizer)
    except TypeError:
        # Fall back to old API if processing_class not supported
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
    
    # Start training with console reminder
    print("\n" + "="*50)
    print("🚀 TRAINING STARTED!")
    print("📊 Check the console below for real-time progress...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save to named directory
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    # Also save as "latest" for easy access
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Save metadata
    import json
    metadata = {
        "name": model_name or "Unnamed",
        "timestamp": datetime.now().isoformat(),
        "dataset_lines": len(dataset),
        "epochs": epochs,
        "lora_rank": lora_rank
    }
    with open(os.path.join(model_save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return f"""✅ **Fine-tuning complete!**

Model saved as: **{model_name or 'latest'}**

**Training Summary:**
- Dataset: {len(dataset)} examples
- Epochs: {epochs}
- LoRA Rank: {lora_rank}

**Next Steps:**
1. Go to the "Chat & Test" tab to compare models
2. Your fine-tuned model will appear in the dropdown
3. Test with different prompts and temperatures

💡 **Tip**: Check your console/terminal for the training logs to see how loss decreased during training!"""

def merge_lora():
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map=DEVICE_MAP)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    peft_model = PeftModel.from_pretrained(model, MODEL_DIR)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    return "LoRA merged into full model for export compatibility!"

def export_model(export_type, hf_repo):
    # Merge before export for compatibility
    merge_lora()
    export_dir = MERGED_MODEL_DIR
    zip_path = os.path.join(DATA_DIR, "model.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), export_dir))
    
    if export_type == "Download":
        if args.dev:
            return f"Model exported to: {zip_path}\nLocal development mode - file saved locally."
        else:
            return f"Model exported to: {zip_path}\nNote: Direct download not available in web UI. Use Fly SSH to retrieve the file."
    elif export_type == "Hugging Face":
        if not HF_TOKEN:
            return "HF_TOKEN not set. Please add it to your .env file (dev) or Fly secrets (prod)."
        if not hf_repo:
            return "Provide HF Repo Name."
        login(HF_TOKEN)
        api = HfApi()
        api.create_repo(repo_id=hf_repo, exist_ok=True)
        api.upload_folder(folder_path=export_dir, repo_id=hf_repo, repo_type="model")
        return f"Uploaded full model to https://huggingface.co/{hf_repo}\nPull with Ollama: ollama pull {hf_repo} (may need GGUF conversion locally via llama.cpp)\nFor Hugging Face TGI: Use docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference --model-id {hf_repo}"
    elif export_type == "Cloudflare R2":
        if args.dev:
            return "R2 export is disabled in development mode. Use Download or Hugging Face instead."
        if not (R2_BUCKET and R2_ENDPOINT and R2_ACCESS_KEY and R2_SECRET_KEY):
            return "R2 secrets (R2_BUCKET, R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY) not fully set in Fly."
        s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY)
        key = "models/model.zip"  # Customize if needed
        s3.upload_file(zip_path, R2_BUCKET, key)
        presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': R2_BUCKET, 'Key': key}, ExpiresIn=3600)
        return f"Model ZIP uploaded to R2! Download link (expires in 1 hour): {presigned_url}\nUnzip locally, then use with Ollama (convert to GGUF if needed) or TGI (load from folder). For direct pull, prefer HF export."

# Simple authentication: Password-only (any username accepted)
def check_auth(username, password):
    return password == "llamafarm321"

def get_model_choices():
    """Get list of models for dropdown"""
    choices = ["Original Model"]
    
    # Add latest fine-tuned if exists
    if os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
        choices.append("Latest Fine-tuned")
    
    # Add named models
    models_dir = os.path.join(DATA_DIR, "models")
    if os.path.exists(models_dir):
        for model_dir in sorted(os.listdir(models_dir)):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                choices.append(model_dir)
    
    return choices

with gr.Blocks() as demo:
    mode_text = "DEV Mode (ARM64/CPU)" if args.dev else "PROD Mode (CUDA GPU)"
    gr.Markdown(f"# AI Model Studio - {mode_text}\nChat with before/after models! Exports merge LoRA for Ollama/TGI compatibility.")
    
    with gr.Tab("Load Model"):
        model_id = gr.Dropdown(choices=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "meta-llama/Llama-3.2-1B"], label="Select Model (or enter custom ID)")
        custom_id = gr.Textbox(label="Custom HF Model ID (optional)")
        load_btn = gr.Button("Load Model")
        load_status = gr.Textbox(label="Status")
        def load_wrapper(dropdown, custom):
            id = custom if custom else dropdown
            return load_model(id)
        load_btn.click(load_wrapper, inputs=[model_id, custom_id], outputs=load_status)
    
    with gr.Tab("Chat & Test"):
        gr.Markdown("### Compare Models Side-by-Side")
        
        # System prompt at the top - using State to persist
        system_prompt_state = gr.State("You are a helpful assistant.")
        system_prompt = gr.Textbox(
            label="System Prompt (applies to both models)",
            placeholder="You are a helpful assistant...",
            value="You are a helpful assistant.",
            lines=2
        )
        
        # Single chat input - using State to persist
        chat_input_state = gr.State("")
        with gr.Row():
            chat_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2,
                scale=4
            )
            send_btn = gr.Button("Send to Both", variant="primary", scale=1)
        
        # Model selection dropdowns
        with gr.Row():
            before_model_choice = gr.Dropdown(
                choices=get_model_choices(),
                label="🔵 Before Model",
                value="Original Model",
                scale=1
            )
            after_model_choice = gr.Dropdown(
                choices=get_model_choices(),
                label="🟢 After Model", 
                value="Latest Fine-tuned" if "Latest Fine-tuned" in get_model_choices() else "Original Model",
                scale=1
            )
            refresh_models_btn = gr.Button("🔄", scale=1)
        
        # Model controls
        with gr.Row():
            model_selection = gr.Radio(
                ["Before Only", "After Only", "Both Models"],
                label="Send to:",
                value="Both Models",
                scale=2
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature (higher = more creative)",
                scale=1
            )
        
        # Side-by-side responses
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 Before (Original Model)")
                before_response = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("### 🟢 After (Fine-tuned Model)")
                after_response = gr.Textbox(
                    label="Response", 
                    lines=10,
                    interactive=False
                )
        
        # Status indicator
        status = gr.Textbox(label="Status", interactive=False, visible=True, lines=8)
        
        # Help section
        with gr.Accordion("📚 Help: Understanding the Metrics", open=False):
            gr.Markdown("""
### 🎯 How to Use This Interface

1. **System Prompt**: Sets the model's behavior/personality. This is prepended to every message.
   - Example: "You are a helpful landscaping expert who gives concise, practical advice."

2. **Temperature**: Controls randomness/creativity (0.1 to 2.0)
   - **0.1-0.5**: Very consistent, factual responses
   - **0.6-0.9**: Balanced creativity (recommended)
   - **1.0-1.5**: More creative, varied responses
   - **1.6-2.0**: Very creative, may be unpredictable

3. **Send To Options**:
   - **Both Models**: Compare original vs fine-tuned side-by-side
   - **Before Only**: Test just the original model
   - **After Only**: Test just your fine-tuned model

### 📊 Understanding the Status Metrics

**🔵 Before Model**: The original base model before fine-tuning
**🟢 After Model**: Your fine-tuned model with LoRA adaptations

**⏱️ Timing Information**:
- Shows how long each model takes to generate
- Helps identify if fine-tuning affects speed
- "% faster/slower" compares the models

**📝 Token Estimates**:
- **Tokens**: Basic units of text (roughly 0.75 words each)
- **Input Tokens**: How much text you sent
- **Output Tokens**: How much text was generated
- More tokens = more processing time

### 💡 Tips for Testing

1. **Start with low temperature (0.5)** to see consistent behavior differences
2. **Test the same prompts** that match your training data
3. **Try edge cases** your training didn't cover
4. **Compare response quality**, not just speed
5. **Note which model stays more on-topic**

### 🔍 What to Look For

- **Accuracy**: Does the fine-tuned model give better domain-specific answers?
- **Style**: Has the writing style changed to match your training data?
- **Speed**: Is there a significant performance difference?
- **Consistency**: Which model gives more reliable responses?
""")
        
        # Chat history
        with gr.Accordion("Chat History", open=False):
            chat_history = gr.Textbox(
                label="History",
                lines=10,
                interactive=False,
                value=""
            )
        
        # Clear button
        clear_btn = gr.Button("Clear Responses", variant="secondary")
        
        # Update chat function to handle both models with status updates
        def chat_both_models(message, system, selection, history_text, temp, before_model, after_model):
            if not message:
                return "", "", history_text, ""
            
            import time
            
            before_resp = ""
            after_resp = ""
            before_time = 0
            after_time = 0
            
            # Send to models sequentially (especially important for local CPU)
            if selection in ["Before Only", "Both Models"]:
                status_msg = f"🔵 Generating response from {before_model}...\n📊 Temperature: {temp}\n💭 Processing your message..."
                yield "", "", history_text, status_msg
                start_time = time.time()
                before_resp = chat_with_model(message, system, before_model, temp)
                before_time = time.time() - start_time
            
            if selection in ["After Only", "Both Models"]:
                if selection == "After Only":
                    status_msg = f"🟢 Generating response from {after_model}...\n📊 Temperature: {temp}\n💭 Processing your message..."
                else:
                    status_msg = f"🟢 Now generating from {after_model}...\n📊 Temperature: {temp}\n⏱️ {before_model} took: {before_time:.2f}s"
                
                # Show before response if we have it
                before_display = f"[{before_time:.2f}s] {before_resp}" if before_time > 0 else ""
                yield before_display, "", history_text, status_msg
                
                start_time = time.time()
                after_resp = chat_with_model(message, system, after_model, temp)
                after_time = time.time() - start_time
            
            # Update history with timing info
            new_history = f"{history_text}\n\n{'='*50}\nUser: {message}\n"
            if before_resp:
                new_history += f"\nBefore ({before_time:.2f}s): {before_resp}\n"
            if after_resp:
                new_history += f"\nAfter ({after_time:.2f}s): {after_resp}\n"
            
            # Add timing to response display
            if before_time > 0:
                before_resp = f"[{before_time:.2f}s] {before_resp}"
            if after_time > 0:
                after_resp = f"[{after_time:.2f}s] {after_resp}"
            
            # Final status with detailed metrics
            total_time = before_time + after_time
            status_lines = ["✅ Generation Complete!"]
            status_lines.append(f"📊 Temperature: {temp}")
            status_lines.append(f"⏱️ Total Time: {total_time:.2f}s")
            
            if before_time > 0 and after_time > 0:
                status_lines.append(f"  • Before Model: {before_time:.2f}s")
                status_lines.append(f"  • After Model: {after_time:.2f}s")
                if after_time < before_time:
                    speedup = ((before_time - after_time) / before_time) * 100
                    status_lines.append(f"  • After is {speedup:.1f}% faster! 🚀")
                elif after_time > before_time:
                    slowdown = ((after_time - before_time) / before_time) * 100
                    status_lines.append(f"  • After is {slowdown:.1f}% slower")
            
            # Token estimates (rough approximation)
            prompt_tokens = len(message.split()) * 1.3  # Rough estimate
            before_tokens = len(before_resp.split()) * 1.3 if before_resp else 0
            after_tokens = len(after_resp.split()) * 1.3 if after_resp else 0
            
            status_lines.append(f"📝 Token Estimates:")
            status_lines.append(f"  • Input: ~{int(prompt_tokens)} tokens")
            if before_tokens > 0:
                status_lines.append(f"  • Before Output: ~{int(before_tokens)} tokens")
            if after_tokens > 0:
                status_lines.append(f"  • After Output: ~{int(after_tokens)} tokens")
            
            status_msg = "\n".join(status_lines)
            yield before_resp, after_resp, new_history, status_msg
        
        # New chat function that includes system prompt
        def chat_with_model(message, system, model_choice, temp):
            try:
                if model_choice == "Original Model":
                    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_DIR, device_map=DEVICE_MAP)
                    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_DIR)
                elif model_choice == "Latest Fine-tuned":
                    if not os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
                        return "No fine-tuned model found. Please fine-tune first."
                    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_DIR, device_map=DEVICE_MAP)
                    model = PeftModel.from_pretrained(model, MODEL_DIR)
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                else:
                    # Named model
                    model_path = os.path.join(DATA_DIR, "models", model_choice)
                    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
                        return f"Model '{model_choice}' not found."
                    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_DIR, device_map=DEVICE_MAP)
                    model = PeftModel.from_pretrained(model, model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Set pad token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Format with system prompt
                full_prompt = f"{system}\n\nUser: {message}\nAssistant:"
                
                inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=temp,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        top_p=0.95,  # Add nucleus sampling for better quality
                        repetition_penalty=1.1  # Reduce repetition
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the assistant's response
                response = response.split("Assistant:")[-1].strip()
                return response
            except Exception as e:
                return f"Error: {str(e)}"
        
        def clear_responses():
            return "", "", "", ""
        
        # Update dropdowns when refresh is clicked
        def refresh_model_choices():
            choices = get_model_choices()
            return gr.update(choices=choices), gr.update(choices=choices)
        
        refresh_models_btn.click(
            refresh_model_choices,
            outputs=[before_model_choice, after_model_choice]
        )
        
        # Persist system prompt and chat input
        system_prompt.change(lambda x: x, inputs=system_prompt, outputs=system_prompt_state)
        chat_input.change(lambda x: x, inputs=chat_input, outputs=chat_input_state)
        
        # Connect the interface
        send_btn.click(
            chat_both_models,
            inputs=[chat_input, system_prompt, model_selection, chat_history, temperature, before_model_choice, after_model_choice],
            outputs=[before_response, after_response, chat_history, status]
        ).then(
            lambda: "",
            outputs=chat_input
        )
        
        # Also allow Enter key to send
        chat_input.submit(
            chat_both_models,
            inputs=[chat_input, system_prompt, model_selection, chat_history, temperature, before_model_choice, after_model_choice],
            outputs=[before_response, after_response, chat_history, status]
        ).then(
            lambda: "",
            outputs=chat_input
        )
        
        clear_btn.click(clear_responses, outputs=[before_response, after_response, chat_history, status])
    
    with gr.Tab("Add Data"):
        gr.Markdown("### Easy Data Entry Options")
        with gr.Accordion("Data Format Help", open=False):
            gr.Markdown("""
            **What's Needed**: The app requires data in JSONL format (one JSON object per line), where each line is a dictionary with exactly two keys—`"prompt"` (the input/query the model should respond to) and `"completion"` (the desired output/response). This is a standard format for instruction-tuning LLMs like Llama. If you don't have JSONL ready, the "Add Data" tab lets you paste raw text (e.g., conversations, lists, or unstructured notes), and it uses OpenAI to convert it automatically into valid JSONL pairs. You can also upload a pre-made JSONL file or generate suggestions via ChatGPT based on a description.
            
            **How Much**: A minimum of 5-10 lines (examples) can work for basic fine-tuning on small models like TinyLlama-1.1B, especially for simple adaptations (e.g., teaching a specific style or domain knowledge). However, for better results, aim for 50-500+ lines—more data leads to more robust learning and less overfitting. With just 10 lines, the model might memorize them but generalize poorly; it's great for quick experiments but not production-quality. The app validates the JSONL before training to ensure it's error-free.
            
            **What Happens During Training**: The data is loaded as a Hugging Face Dataset, tokenized (converted to numerical inputs the model understands), and batched. It's fed into the training loop to update the LoRA adapters, teaching the model patterns from your examples (e.g., how to respond to similar prompts).
            
            **Key Rules for the Format**:
            - **JSONL Structure**: Each line is valid JSON (parsable independently). Use double quotes for strings, and escape any internal quotes if needed (e.g., \" for nested quotes).
            - **Content Guidelines**:
              - `"prompt"`: Keep it concise (e.g., 10-200 words)—this is what the user might say or ask.
              - `"completion"`: The ideal response (e.g., 50-500 words)—make it informative, accurate, and aligned with your goals.
            - **Minimum Requirements**: At least 5-10 lines for basic fine-tuning, but more (50+) yields better results.
            - **Encoding**: UTF-8 text; no binary data.
            - **Common Pitfalls to Avoid**: Invalid JSON (e.g., missing quotes, extra commas), empty lines, or mismatched keys—the app validates before training and will error if invalid.
            
            **Example JSONL Format**:
           {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
{"prompt": "Explain quantum computing simply.", "completion": "Quantum computing uses quantum bits (qubits) that can be in multiple states at once, allowing faster solving of complex problems than classical computers."}
{"prompt": "Write a haiku about the ocean.", "completion": "Waves crash on the shore,\nBlue depths hide ancient secrets,\nTides whisper softly."}
{"prompt": "How do I make coffee?", "completion": "Grind beans, boil water, pour over grounds in a filter, and let it drip for 3-4 minutes."}
{"prompt": "Who won the 2022 World Cup?", "completion": "Argentina won the 2022 FIFA World Cup, defeating France in the final."}
                                     """)

        with gr.Accordion("Paste Any Raw Data & Convert", open=False):
            any_data = gr.Textbox(label="Paste Any Data (text, convo, list, etc.)", lines=10)
            convert_btn = gr.Button("Convert to JSONL via OpenAI")
            converted_data = gr.Textbox(label="Converted JSONL (edit if needed)", lines=10)
            with gr.Row():
                copy_converted_btn = gr.Button("📋 Copy Converted Data", scale=1)
            convert_btn.click(convert_any_to_jsonl, inputs=[any_data], outputs=converted_data)

        pasted_data = gr.Textbox(label="Paste/Edit JSONL Data (or copy from above)", lines=10, value='Example:\n{"prompt": "Hello", "completion": "Hi there!"}\n{"prompt": "What is AI?", "completion": "Artificial Intelligence."}')
        data_file = gr.File(label="Or Upload JSONL")
        append_checkbox = gr.Checkbox(label="Append to Existing Dataset (if any)", value=True)

        with gr.Row():
            description = gr.Textbox(label="Describe for ChatGPT Suggestions (e.g., 'Q&A on cats')", scale=4)
            num_suggestions = gr.Number(label="Number of Suggestions", value=10, minimum=1, maximum=50, scale=1)
        suggest_btn = gr.Button("Generate Suggestions via ChatGPT")
        suggestions = gr.Textbox(label="Generated Data (copy to paste above)", lines=10)
        with gr.Row():
            copy_suggestions_btn = gr.Button("📋 Copy Suggestions", scale=1)
        suggest_btn.click(generate_data_suggestions, inputs=[description, num_suggestions], outputs=suggestions)

        # Copy button handlers
        copy_converted_btn.click(lambda x: x, inputs=converted_data, outputs=pasted_data)
        copy_suggestions_btn.click(lambda x: x, inputs=suggestions, outputs=pasted_data)

        validate_btn = gr.Button("Validate JSONL")
        validate_status = gr.Textbox(label="Validation Status")
        
        def validate_and_analyze(data):
            valid, msg = validate_jsonl(data)
            if valid:
                # Additional quality checks
                lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
                warnings = []
                
                for i, line in enumerate(lines):
                    try:
                        obj = json.loads(line)
                        completion = obj.get('completion', '')
                        
                        # Check for narrative patterns
                        narrative_patterns = [
                            "assistant says", "user says", "the assistant", "the user",
                            "responds:", "replied:", "answered:", "asked:",
                            "seems satisfied", "feels", "thinks"
                        ]
                        
                        for pattern in narrative_patterns:
                            if pattern.lower() in completion.lower():
                                warnings.append(f"Line {i+1}: Contains narrative text '{pattern}' - consider removing")
                                break
                        
                        # Check for excessive length
                        if len(completion) > 500:
                            warnings.append(f"Line {i+1}: Very long response ({len(completion)} chars) - consider shortening")
                        
                        # Check for dialog formatting
                        if "\n\nUser:" in completion or "\n\nAssistant:" in completion:
                            warnings.append(f"Line {i+1}: Contains dialog formatting - remove User:/Assistant: markers")
                    except:
                        pass
                
                if warnings:
                    return msg + "\n\n⚠️ Quality Warnings:\n" + "\n".join(warnings)
                else:
                    return msg + "\n\n✅ No quality issues detected!"
            return msg

        validate_btn.click(validate_and_analyze, inputs=pasted_data, outputs=validate_status)

        add_btn = gr.Button("Add Data", variant="primary")
        add_status = gr.Textbox(label="Status")
        gr.Markdown("---")
        gr.Markdown("### Current Dataset Management")
        
        with gr.Row():
            view_btn = gr.Button("View Current Dataset")
            clear_btn = gr.Button("Clear Dataset", variant="stop")
        
        view_output = gr.Textbox(label="Dataset Content", lines=10, interactive=False)
        with gr.Row():
            copy_dataset_btn = gr.Button("📋 Copy Current Dataset", scale=1)
        with gr.Row():
            count_output = gr.Number(label="Total Lines", interactive=False)
            clear_status = gr.Textbox(label="Clear Status", visible=False)
        
        # Copy dataset handler
        copy_dataset_btn.click(lambda x: x, inputs=view_output, outputs=pasted_data)
        
        # Define button click handlers after all components are created
        def add_and_refresh(pasted_data, uploaded_file, append):
            status = add_data(pasted_data, uploaded_file, append)
            view_text, count = view_dataset()
            return status, view_text, count
        
        add_btn.click(add_and_refresh, inputs=[pasted_data, data_file, append_checkbox], 
                     outputs=[add_status, view_output, count_output])
        
        view_btn.click(view_dataset, outputs=[view_output, count_output])
        
        def clear_and_refresh():
            status = clear_dataset()
            view_text, count = view_dataset()
            return status, view_text, count
            
        clear_btn.click(clear_and_refresh, outputs=[clear_status, view_output, count_output])
        
        gr.Markdown("---")
        gr.Markdown("### ✍️ View & Edit Full Dataset")
        gr.Markdown("Load the full dataset to manually delete lines or make corrections. **Warning**: Large datasets may be slow to load.")
        
        load_full_dataset_btn = gr.Button("Load Full Dataset for Editing")
        
        with gr.Group(visible=False) as editor_group:
            full_dataset_editor = gr.Textbox(
                label="Full Dataset Content (edit directly and save)",
                lines=20,
                interactive=True,
            )
            with gr.Row():
                save_dataset_btn = gr.Button("Save Changes", variant="primary")
                copy_full_dataset_btn = gr.Button("📋 Copy to Paste Area", scale=1)
                cancel_edit_btn = gr.Button("Cancel")

        edit_status = gr.Textbox(label="Edit Status", visible=False, lines=3)
        
        # Copy full dataset handler
        copy_full_dataset_btn.click(lambda x: x, inputs=full_dataset_editor, outputs=pasted_data)
        
        def load_full_dataset():
            """Loads the entire dataset into the editor and shows the editor group."""
            if not os.path.exists(DATASET_PATH):
                return "No dataset to load.", gr.update(visible=False), ""
            with open(DATASET_PATH, "r") as f:
                content = f.read()
            return content, gr.update(visible=True), ""

        def save_dataset(content):
            """Saves the edited content back to the dataset file."""
            try:
                # Validate that the content is still valid JSONL
                lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    json.loads(line)

                # Overwrite the original file
                with open(DATASET_PATH, "w") as f:
                    f.write(content.strip())
                
                # Refresh the main view
                view_text, count = view_dataset()
                
                return gr.update(visible=True, value=f"✅ Saved successfully! {len(lines)} lines."), gr.update(visible=False), view_text, count

            except json.JSONDecodeError as e:
                return gr.update(visible=True, value=f"❌ Save Failed: Invalid JSON on line {i+1}. Fix and try again.\nError: {e}"), gr.update(visible=True), None, None
            except Exception as e:
                return gr.update(visible=True, value=f"❌ An unexpected error occurred: {e}"), gr.update(visible=True), None, None

        def cancel_edit():
            """Hides the editor without saving."""
            return gr.update(visible=False), ""
        
        load_full_dataset_btn.click(
            load_full_dataset, 
            outputs=[full_dataset_editor, editor_group, edit_status]
        )
        save_dataset_btn.click(
            save_dataset, 
            inputs=[full_dataset_editor],
            outputs=[edit_status, editor_group, view_output, count_output]
        )
        cancel_edit_btn.click(cancel_edit, outputs=[editor_group, edit_status])
        
        # Auto-load dataset on tab load
        view_output.value, count_output.value = view_dataset()

    with gr.Tab("Fine-Tune"):
        # Show current dataset info
        gr.Markdown("### Current Dataset")
        dataset_info = gr.Textbox(label="Dataset Status", interactive=False, value=view_dataset()[0])
        refresh_btn = gr.Button("Refresh Dataset Info")
        refresh_btn.click(lambda: view_dataset()[0], outputs=dataset_info)
        
        gr.Markdown("### Model Configuration")
        model_name = gr.Textbox(
            label="Model Name (optional)", 
            placeholder="e.g., 'landscaping-v1', 'customer-service-v2'",
            value=""
        )
        
        with gr.Accordion("LoRA Rank Help", open=False):
            gr.Markdown("""
**What's Needed**: This is a slider input (range 4-64, default 8) specifying the "rank" (or dimensionality) of the low-rank matrices in LoRA. No additional setup—just choose a value based on your needs.

**How Much**: Lower ranks (e.g., 4-16) are efficient and often sufficient for most tasks, using less memory and training faster. Higher ranks (32-64) capture more complex adaptations but require more GPU resources and data. For small datasets (like 10 lines), stick to low ranks (8-16) to avoid overfitting; higher ranks need hundreds of examples to shine.

**What Happens During Training**: LoRA adds small, trainable matrices to the model's layers (targeting attention modules like q_proj and v_proj). The rank determines the size of these matrices—e.g., rank 8 means 8-dimensional updates. During training, only these adapters are updated (not the full model), making it efficient (e.g., 1-5% of parameters). This fine-tunes the model on your data without changing the original weights much.
""")

        lora_rank = gr.Slider(4, 64, value=8, label="LoRA Rank")
        
        with gr.Accordion("⚠️ Troubleshooting Weird Responses", open=True):
            gr.Markdown("""
### If your fine-tuned model gives strange responses:

**1. Too Few Training Examples**
- With only 10 examples, the model can memorize rather than learn patterns
- **Fix**: Add more examples (aim for 30-50 minimum)

**2. Training Data Issues**
- Check if your completions contain extra text or narrative
- **Bad**: "The assistant responds: [answer]. The user seems satisfied."
- **Good**: Just the direct answer

**3. Overfitting from Too Many Epochs**
- Small datasets + many epochs = weird outputs
- **For 10 examples**: Use 1 epoch only
- **For 50+ examples**: Can use 2-3 epochs

**4. Temperature Settings**
- High temperature (>1.0) amplifies weirdness
- **Test at 0.5-0.7** for more stable outputs

**5. Data Contamination**
- The model might have picked up patterns from unrelated text
- Review your dataset for any narrative or meta-text
- Keep responses focused and professional
""")

        with gr.Accordion("Epochs Help", open=False):
            gr.Markdown("""
**What's Needed**: This is a slider (range 1-10, default 3) for the number of full passes over your dataset during training. No extras—just select based on data size and desired learning depth.

**How Much**: 1-3 epochs work well for small datasets (e.g., 10 lines) to avoid overfitting (where the model memorizes but doesn't generalize). For larger datasets (100+ lines), 3-5 epochs are ideal. More than 5 can lead to diminishing returns or degradation if data is limited. The "Quick Fine-Tune" button uses 1 epoch for fast tests.

**What Happens During Training**: Each epoch is one complete cycle through all your data examples. The Trainer optimizes the LoRA adapters using gradient descent (with a batch size of 4), computing loss (how wrong the model's predictions are) and adjusting weights to minimize it. Multiple epochs reinforce learning, but too many can cause the model to overfit to noise in small datasets.
""")

        epochs = gr.Slider(1, 10, value=3, label="Epochs")

        with gr.Accordion("📊 Understanding Training Metrics (Check Console)", open=True):
            gr.Markdown("""
### Training Progress in Console

**⚠️ IMPORTANT**: During training, check your terminal/console for real-time progress updates!

### What the Numbers Mean:

**📉 Loss** (e.g., `'loss': 13.6248`)
- **What it is**: Measures how "wrong" the model's predictions are compared to your training data
- **Lower is better**: Loss should decrease during training (e.g., 13.6 → 5.7)
- **Typical range**: Starts at 10-20, should drop to 1-5 by the end
- **If stuck high**: Your data might be too complex or contradictory

**📐 Gradient Norm** (e.g., `'grad_norm': 59.729`)
- **What it is**: The "strength" of weight updates being applied
- **Normal range**: 10-200 is typical
- **Too high (>500)**: Model is struggling, might need lower learning rate
- **Too low (<1)**: Model isn't learning much, might need higher learning rate

**🎯 Learning Rate** (e.g., `'learning_rate': 4.1e-05`)
- **What it is**: How big the steps are when updating the model
- **Format**: `4.1e-05` means 0.000041 (scientific notation)
- **Decreases over time**: Starts higher, gradually reduces for fine-tuning
- **Too high**: Model overshoots, loss jumps around
- **Too low**: Model learns very slowly

**🔄 Epoch** (e.g., `'epoch': 0.6`)
- **What it is**: Progress through your dataset (0.6 = 60% through first pass)
- **1.0 = one complete pass** through all training examples
- **Multiple epochs**: Let the model see examples multiple times

### Example Good Training:
```
{'loss': 13.6248, 'grad_norm': 59.729, 'learning_rate': 4.1e-05, 'epoch': 0.2}
{'loss': 11.436, 'grad_norm': 141.146, 'learning_rate': 3.1e-05, 'epoch': 0.4}
{'loss': 7.3914, 'grad_norm': 144.198, 'learning_rate': 2.1e-05, 'epoch': 0.6}
{'loss': 5.7764, 'grad_norm': 139.783, 'learning_rate': 1.1e-05, 'epoch': 0.8}
```
✅ Loss decreasing steadily (13.6 → 5.7)
✅ Gradient norm stable (59-144 range)
✅ Learning rate decreasing as planned

### Signs of Problems:
- ❌ Loss increasing or stuck
- ❌ Gradient norm > 1000 or < 0.1
- ❌ Loss jumping up and down wildly
- ❌ Training seems frozen (no updates)

### Where to Look:
In your terminal where you started the app, you'll see updates every 10 steps!
""")

        with gr.Accordion("Overall Training Process Help", open=False):
            gr.Markdown("""
When you hit "Fine-Tune" (or "Quick Fine-Tune"):
1. **Validation**: The app checks your JSONL data for validity (e.g., proper JSON structure)—if invalid, it aborts with an error.
2. **Setup**: Loads the base model (e.g., TinyLlama) from Hugging Face, applies LoRA configuration (with your chosen rank), and prepares the dataset by tokenizing prompts/completions.
3. **Training Loop**: Uses Hugging Face's Trainer to run the specified epochs. It processes data in batches, computes forward passes (model predictions), calculates loss, backpropagates errors, and updates LoRA adapters. This happens on Fly.io's GPU for speed (e.g., 5-30 minutes for small models/datasets).
4. **Saving**: Saves the adapted (LoRA-only) model to the volume. No full retraining—LoRA keeps it lightweight (~10-100MB added).
5. **Outcomes**: The model becomes better at tasks in your data (e.g., domain-specific responses). With 10 lines, expect subtle improvements; more data/epochs yield stronger personalization. After, test in the "Chat & Test" tab, then export (merging LoRA for compatibility with Ollama/TGI).
""")

        with gr.Accordion("Tips & Best Practices", open=False):
            gr.Markdown("""
### Training Best Practices

**🔄 Should I train on a trained model or start fresh?**

**Always start from the base model!** Here's why:

1. **Avoid Overfitting**: Training on top of a fine-tuned model can lead to:
   - Loss of general capabilities
   - Overfitting to specific patterns
   - Degraded performance on unrelated tasks

2. **Better Approach - Combine Datasets**:
   - If you want to add new capabilities, combine your datasets
   - Example: Have "landscaping-v1" data? Add new customer service data to it
   - Train once on the combined dataset from the base model

3. **Iterative Improvement Process**:
   - Train Model v1 → Test → Identify gaps
   - Add new data to address gaps
   - Train Model v2 from base with ALL data (v1 + new)
   - Name models clearly: "landscaping-v1", "landscaping-v2", etc.

### Quick Tips

1. **Start Small**: Begin with 10-50 examples, 1-3 epochs
2. **Name Your Models**: Use descriptive names like "customer-service-v1"
3. **Keep Dataset History**: Save different versions of your training data
4. **Test Thoroughly**: Use the side-by-side chat to compare versions
5. **Document Changes**: Note what each version improves

### Example Workflow

1. Create "landscaping-v1" with 10 Q&A pairs → Train
2. Test and find it needs better pricing responses
3. Add 5 pricing Q&A pairs to your original 10 (total 15)
4. Train "landscaping-v2" from base model with all 15 pairs
5. Compare v1 vs v2 in chat interface
""")

        gr.Markdown("### 🚀 Start Training")
        gr.Markdown("**Note:** During training, check your terminal/console for real-time progress updates (loss, gradient norm, etc.)")
        
        ft_btn = gr.Button("Fine-Tune on GPU")
        quick_ft_btn = gr.Button("Quick Fine-Tune (Defaults: Rank=8, Epochs=1)")
        ft_status = gr.Textbox(label="Status")
        ft_btn.click(fine_tune, inputs=[lora_rank, epochs, model_name], outputs=ft_status)
        quick_ft_btn.click(lambda name: fine_tune(8, 1, name), inputs=[model_name], outputs=ft_status)
        
        # Model management section
        gr.Markdown("### Saved Models")
        
        def list_saved_models():
            models_list = ["• Original Model (Base)"]
            
            # Check if we have a fine-tuned model in the default location
            if os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
                models_list.append("• Latest Fine-tuned Model")
            
            # Check for named models
            models_dir = os.path.join(DATA_DIR, "models")
            if os.path.exists(models_dir):
                for model_dir in sorted(os.listdir(models_dir)):
                    model_path = os.path.join(models_dir, model_dir)
                    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                        metadata_path = os.path.join(model_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                                    models_list.append(f"• {metadata['name']} - {metadata['timestamp'][:10]} (Rank: {metadata['lora_rank']}, Epochs: {metadata['epochs']}, Lines: {metadata['dataset_lines']})")
                            except:
                                models_list.append(f"• {model_dir}")
                        else:
                            models_list.append(f"• {model_dir}")
            
            return "\n".join(models_list) if len(models_list) > 1 else "No saved models yet. Train a model first!"
        
        saved_models_list = gr.Textbox(label="Available Models", value=list_saved_models(), interactive=False, lines=5)
        refresh_models_btn = gr.Button("Refresh Model List")
        refresh_models_btn.click(list_saved_models, outputs=saved_models_list)

    with gr.Tab("Export"):
        export_type = gr.Dropdown(["Download", "Hugging Face", "Cloudflare R2"], label="Export Type")
        hf_repo = gr.Textbox(label="HF Repo Name (for HF; required if Hugging Face selected)")
        exp_btn = gr.Button("Export Merged Model")
        exp_output = gr.Textbox(label="Export Status/Link")
        exp_btn.click(export_model, inputs=[export_type, hf_repo], outputs=exp_output)

demo.launch(
    server_name="127.0.0.1" if args.dev else "0.0.0.0", 
    server_port=7860, 
    auth=None if args.dev else check_auth,  # Disable auth in dev mode
    share=False,  # Don't try to create a public share link
    root_path=None if args.dev else "/",  # No root path in dev mode
)