#!/bin/bash

# ================= Configuration =================
# Get the script directory and project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# LLaMA-Factory directory (relative to project root)
LLAMA_FACTORY_DIR="${PROJECT_ROOT}/LLaMA-Factory"
DATA_DIR="${LLAMA_FACTORY_DIR}/data"
DATASET_INFO_FILE="${DATA_DIR}/dataset_info.json"
FINAL_OUTPUT_FILE="exp_rl_coldstart_multi.json"
COLD_START_NAME="exp_rl_coldstart_multi"
SFT_CONFIG="examples/train_lora/qwen2_5vl_lora_sft.yaml"

# ================= Functions =================
print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

print_success() {
    echo "[SUCCESS] $1"
}

# ================= Main Script =================
print_info "Starting SFT training pipeline..."

# Step 1: Check if LLaMA-Factory directory exists
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    print_error "LLaMA-Factory directory not found: $LLAMA_FACTORY_DIR"
    exit 1
fi
print_success "LLaMA-Factory directory found"

# Step 2: Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi
print_success "Data directory found"

# Step 3: Check if cold start data file exists
COLD_START_DATA_FILE="${DATA_DIR}/${FINAL_OUTPUT_FILE}"
if [ ! -f "$COLD_START_DATA_FILE" ]; then
    print_error "Cold start data file not found: $COLD_START_DATA_FILE"
    print_error "Please run hotpotqa_processor.py first to generate the cold start data."
    exit 1
fi
print_success "Cold start data file found: $COLD_START_DATA_FILE"

# Step 4: Check if dataset_info.json exists and contains the dataset
if [ ! -f "$DATASET_INFO_FILE" ]; then
    print_error "Dataset info file not found: $DATASET_INFO_FILE"
    print_error "Please run hotpotqa_processor.py first to generate the dataset info."
    exit 1
fi

# Check if dataset is registered in dataset_info.json
if ! grep -q "\"$COLD_START_NAME\"" "$DATASET_INFO_FILE"; then
    print_error "Dataset '$COLD_START_NAME' not found in $DATASET_INFO_FILE"
    print_error "Please run hotpotqa_processor.py first to register the dataset."
    exit 1
fi
print_success "Dataset '$COLD_START_NAME' is registered in dataset_info.json"

# Step 5: Check if SFT config file exists
cd "$LLAMA_FACTORY_DIR" || exit 1
if [ ! -f "$SFT_CONFIG" ]; then
    print_error "SFT config file not found: $SFT_CONFIG"
    print_error "Please ensure the config file exists in LLaMA-Factory directory."
    exit 1
fi
print_success "SFT config file found: $SFT_CONFIG"

# Step 6: Display dataset statistics
print_info "Dataset statistics:"
DATA_COUNT=$(python3 -c "import json; data = json.load(open('$COLD_START_DATA_FILE')); print(len(data))" 2>/dev/null)
if [ $? -eq 0 ]; then
    print_info "  - Number of samples: $DATA_COUNT"
else
    print_info "  - Could not determine sample count"
fi

# Step 7: Run SFT training
print_info "Starting SFT training..."
print_info "Config: $SFT_CONFIG"
print_info "Dataset: $COLD_START_NAME"
print_info "Working directory: $LLAMA_FACTORY_DIR"
print_info "Python interpreter: $(which python)"
echo ""

# Run the training command using python -m to ensure using current environment
# This is more reliable than llamafactory-cli which might use wrong Python
python -m llamafactory.cli train "$SFT_CONFIG"

# Check if training was successful
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    print_success "SFT training completed successfully!"
else
    print_error "SFT training failed with exit code $TRAIN_EXIT_CODE"
    exit 1
fi

print_info "SFT training pipeline completed."

