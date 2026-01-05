# LLM Reasoning Experiments

This repository contains experiments comparing LLM strategies on different datasets.

---

## Model Results

### StrategyQA

Accuracy Comparison: LLM only vs LLM + MASK  

![Accuracy StrategyQA](images/StrategyQA.png)

### MQuAKE-CF-3k-v2

Accuracy Comparison: LLM only vs LLM + MASK  

![MQuAKE-CF-3k-v2](images/MQuAKE-CF-3k-v2.png)

---
## Example: LLM-only vs LLM + MASK reasoning (MQuAKE-CF-3k-v2)

**Question:** "Where did the author of the Earthsea series pass away?"

- **LLM-only approach:**  
```bash
question = "Where did the author of the Earthsea series pass away?"
answer = generate_answer_with_reasoning(question, llmtokenizer, model, max_new_tokens=200)
print("Answer:", answer)
Output:
Answer: Seattle, Washington, USA
Evidence: Ursula K. Le Guin, the American author of the Earthsea series, passed away on January 22, 2018, at the age of 88, in Portland, Oregon, but she had lived in Portland for many years, and had also lived in Seattle.
```
❌ Wrong answer: The model gave the place where the author lived rather than where she actually passed away.
´´´
LLM + MASK (compositional reasoning):
```bash
Step 1: "Who is the author of the Earthsea series?" → Ursula K. Le Guin

python
Copier le code
question = "Who is the author of the Earthsea series?"
answer = generate_answer_with_reasoning(question, llmtokenizer, model, max_new_tokens=200)
print("Answer:", answer)
Output:
Answer: Ursula K. Le Guin
Evidence: Ursula Kroeber Le Guin was an American novelist, critic, essayist, and poet best known for her works of speculative fiction. Her notable works include the Earthsea series.

Step 2: "Where did Ursula K. Le Guin pass away?" → Portland, Oregon

question = "Where did Ursula K. Le Guin pass away?"
answer = generate_answer_with_reasoning(question, llmtokenizer, model, max_new_tokens=200)
print("Answer:", answer)
Output:
Answer: Portland, Oregon
Evidence: Ursula Kroeber Le Guin (1929-2018) was an American novelist. Le Guin died on January 22, 2018, at the age of 88, in her home in Portland, Oregon.
```
✅ Correct answer: By decomposing the question, the model identifies the correct death place.


LLM-only may fail on multi-hop questions even if the information exists in pretraining.

LLM + MASK solves complex questions by decomposing them into sub-questions, producing correct step-by-step reasoning.

## Reproducing the Results

Follow these steps to reproduce the results:

```bash
# Clone the repository
git clone https://github.com/jaaferklila/-llm-reasonin.git
cd llm-reasonin

# 1️Create the Conda environment
conda create -n llmreasoning python=3.10
conda activate llmreasoning
pip install -r requirements.txt
```
# 2️ Run the experiments
For the LLM only run:
```bash
python LLM_Only.py --model_name <model_name> --dataset_name <dataset_name>
```

Example:
For the StrategyQA dataset with the LLaMA model:
```bash
python LLM_Only.py --model_name llama3.1-8b --dataset_name strategyqa
```

Results:

Printed in the terminal

Saved into a folder called output/
