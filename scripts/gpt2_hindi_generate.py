import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
INPUT_CSV = "Hindi_Dataset.csv"
OUTPUT_CSV = "gpt2_hindi_outputs.csv"

HINDI_COL = "Hindi"       
ID_COL = "PromptID"       

MODEL_NAME = "gpt2"       
MAX_NEW_TOKENS = 120
TEMPERATURE = 0.9
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3
# ---------------------------------------

def build_prompt(hindi_text: str) -> str:
    # Simple continuation prompt that works better than "User/Assistant" for GPT-2
    return f"हिंदी पाठ:\n{hindi_text}\n\nउत्तर:\n"

def main():
    df = pd.read_csv(INPUT_CSV)

    if HINDI_COL not in df.columns:
        raise ValueError(f"Column '{HINDI_COL}' not found. Found columns: {list(df.columns)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # GPT-2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    results = []

    for i, row in df.iterrows():
        pid = row[ID_COL] if ID_COL in df.columns else i
        hindi_text = str(row[HINDI_COL])

        prompt = build_prompt(hindi_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if response == "":
            response = "[EMPTY_OUTPUT]"

        results.append({
            "PromptID": pid,
            "hindi_prompt": hindi_text,
            "gpt2_response": response
        })

        if (i + 1) % 25 == 0:
            print(f"Done {i+1}/{len(df)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
