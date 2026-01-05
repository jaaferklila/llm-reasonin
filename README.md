# Model Results

## StrategyQA

Accuracy Comparison: LLM only vs LLM + MASK  

![Accuracy StrategyQA](images/StrategyQA.png)

## MQuAKE-CF-3k-v2

Comparison: LLM only vs LLM + MASK  

![MQuAKE-CF-3k-v2](images/MQuAKE-CF-3k-v2.png)

## Reproducing the Results

If you want to reproduce the results from this project, follow these steps:

### 1️⃣ Create the Conda environment

```bash
conda create -n llmreasoning python=3.10
conda activate llmreasoning
pip install -r requirements.txt
