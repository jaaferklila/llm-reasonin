# =========================================================
# Imports
# =========================================================
import json
import os
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import llm_only_answer, contains_correct_answer, convert_yes_no_to_bool


def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b"]
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MQuAKE-CF-3k-v2",
        choices=["MQuAKE-CF-3k-v2", "strategyqa"],
        help="Dataset to evaluate"
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--edit_num", type=int, default=0)
    parser.add_argument("--nsample", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    edit_num = args.edit_num
    nsample = args.nsample

    print("#" * 60)
    print(f"Using model: {model_name}")
    print("#" * 60)
    print(f"Dataset: {dataset_name}")
    print("#" * 60)
    print(f"CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")

    # =========================================================
    # Load dataset
    # =========================================================

    if dataset_name.lower() == "strategyqa":
        dataset_path = f"./datasets/{dataset_name}/strategyqa_with_masks.json"
    else:
       dataset_path = f"./datasets/{dataset_name}/{dataset_name}.json"

# Load dataset
    with open(dataset_path, "r") as f:
        all_dataset = json.load(open(f"{dataset_path}", "r"))

    if edit_num == 0:
        edit_num = len(all_dataset)
    #all_dataset=all_dataset[:1]
    all_dataset_list = [all_dataset[i:i+edit_num] for i in range(0, len(all_dataset), edit_num)]
    print(f"Dataset size: {len(all_dataset)}")
    print("#" * 60)
    # =========================================================
    # Load model
    # =========================================================
    MODEL_NAME_TO_PATH = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    }

    model_path = MODEL_NAME_TO_PATH[model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if model_name == "qwen2.5-7b" else torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    torch.set_grad_enabled(False)

    model_class_name = model.__class__.__name__.lower()
    if "llama" in model_class_name:
        end_token_ids = [128001, 128009]
    elif "qwen" in model_class_name:
        end_token_ids = [151645, 151643]
    else:
        end_token_ids = [tokenizer.eos_token_id]

    # =========================================================
    # Load prompts
    # =========================================================
    mask_prompts = json.load(open("./prompt/llm_only.json"))
    strategy_prompts = json.load(open("./prompt/strategyqa_problems.json"))

    # =========================================================
    # Evaluation loop
    # =========================================================
    tot = 0
    correct = 0
    results = []

    output_path = f"./output/LLM_Only/{model_name}_{dataset_name}"
    os.makedirs(output_path, exist_ok=True)

    for dataset in all_dataset_list:
        for d in dataset:

            if dataset_name.lower() == "strategyqa":
                question = d.get("question")
                true_answer = d.get("answer")
                #facts = d.get("facts", [])
                print(f'The question is: {question}')
                print(f'The answer is: {true_answer}')
                correct_answers=[d.get("answer")]
                model_answer = llm_only_answer(
                question,
                model,
                tokenizer,
                end_token_ids,
                mask_prompts,
                strategy_prompts,
                dataset=dataset_name
            )
                pred=convert_yes_no_to_bool(model_answer)
                print(f'The model answer is: {pred}')
                tot += 1
                if(pred==true_answer):
                    correct += 1
                print(f'correct/total: {correct}/{tot}')
                accuracy = correct / tot if tot > 0 else 0.0
                print(f"Accuracy: {accuracy:.4f}")
                

            else:
                question = d["questions"][0]
                correct_answers = (
                    [d["answer"].lower()]
                    + [a.lower() for a in d.get("answer_alias", [])]
                )
                

                model_answer = llm_only_answer(
                    question,
                    model,
                    tokenizer,
                    end_token_ids,
                    mask_prompts,
                    strategy_prompts,
                    dataset=dataset_name
                )
                print(f'The correct answer is: {correct_answers}')
                print(f'The model answer is: {model_answer}')
                if contains_correct_answer(model_answer, correct_answers):
                    correct += 1
                tot += 1
                print(f'correct/total: {correct}/{tot}')
                results.append({
                    "question": question,
                    "correct_answers": correct_answers,
                    "model_answer": model_answer
                })

                if nsample > 0 and tot >= nsample:
                  break

                if nsample > 0 and tot >= nsample:
                   break

    # =========================================================
    # Save results
    # =========================================================
    final_accuracy = correct / tot if tot > 0 else 0.0

    with open(f"{output_path}/result.json", "w") as f:
            json.dump(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "accuracy": final_accuracy,
                                "total": tot,
                                "correct": correct,
                                "results": results
                            },
                            f,
                            indent=4
                        )

    print(f"\nFinal Accuracy: {final_accuracy:.4f}")
    print(f"Results saved to {output_path}/result.json")


# =========================================================
# Run main
# =========================================================
if __name__ == "__main__":
    main()

