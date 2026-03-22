import json
import subprocess

JSON_PATH = "prompt_pairs.json"
DEVICE = "cuda"

def run_prompt(prompt):
    cmd = f'audioldm2 -t "{prompt}" -d cuda'
    print(f"\n▶ Running: {prompt}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        prompt_pairs = json.load(f)

    for concept, pairs in prompt_pairs.items():
        print(f"\n===== Concept: {concept} =====")

        if concept.lower() == "tempo":
            continue

        for pair in pairs:
            pair_id = pair["id"]
            prompt_a = pair["prompt_a"]
            prompt_b = pair["prompt_b"]
            print(f"\n--- Pair: {pair_id} ---")
            run_prompt(prompt_a)
            run_prompt(prompt_b)            

if __name__ == "__main__":
    main()