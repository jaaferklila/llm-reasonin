#!/bin/bash
#SBATCH --output=logs_LLM_Only/o%x.out
#SBATCH --error=logs_LLM_Only/e%x.err
#SBATCH --mem=30000
#SBATCH --time=2-00:00:00
#SBATCH -p gpuv100,gpup100,gpup6000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=llmOnly

# ================= Environment ================= #
source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker

echo "Début du job sur l'hôte : $(hostname)"
echo "Démarrage du traitement..."

# ================= Offline HF Settings ================= #
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
#unset HF_HUB_OFFLINE  # à décommenter si vous voulez revenir en ligne

# ================= Run Python Script ================= #
# Définir les modèles et datasets (Bash array)
MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b")
#DATASET_NAMES=("MQuAKE-CF-3k-v2" "strategyqa")
DATASET_NAMES=("MQuAKE-CF-3k-v2" "strategyqa")

# Boucle sur tous les modèles et datasets
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Evaluating model $MODEL_NAME on dataset $DATASET_NAME"
        srun python LLM_Only.py \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME"
    done
done

echo "Done. Job finished."
