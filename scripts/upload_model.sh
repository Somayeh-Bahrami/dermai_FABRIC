#!/usr/bin/env bash
# Upload a fine-tuned checkpoint to the Hugging Face Hub.
#
# Prerequisite (one time): hf auth login   (paste a write token)
#
# Usage:
#   scripts/upload_model.sh efficientnet-b0
#   scripts/upload_model.sh vit-base-patch16-224 myusername
#
# The first argument is the checkpoint folder name under outputs/, which also
# becomes the repo name. The username defaults to the logged-in Hugging Face
# account and can be overridden with an optional second argument. Public model.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <model-name> [hf-username]" >&2
    exit 1
fi

model_name="$1"
hf_user="${2:-$(hf auth whoami | sed -n 's/^user=//p')}"

if [ -z "$hf_user" ]; then
    echo "could not determine Hugging Face username; run 'hf auth login' first" >&2
    exit 1
fi

checkpoint_dir="outputs/${model_name}"
repo_id="${hf_user}/dermai-${model_name}"

if [ ! -d "$checkpoint_dir" ]; then
    echo "checkpoint not found: ${checkpoint_dir}" >&2
    echo "train the model first so it is saved under outputs/" >&2
    exit 1
fi

base_model="google/${model_name}"

cat > "${checkpoint_dir}/README.md" <<EOF
---
base_model: ${base_model}
library_name: transformers
license: apache-2.0
tags:
  - image-classification
  - skin-lesion
  - ham10000
---

# dermai-${model_name}

Fine-tuned [${base_model}](https://huggingface.co/${base_model}) for 7-class
skin lesion classification on HAM10000. Part of the DermAI explainability
project comparing CNN and Vision Transformer explanations.
EOF

echo "uploading ${checkpoint_dir} to ${repo_id}"
hf upload "$repo_id" "$checkpoint_dir" .
echo "done: https://huggingface.co/${repo_id}"
