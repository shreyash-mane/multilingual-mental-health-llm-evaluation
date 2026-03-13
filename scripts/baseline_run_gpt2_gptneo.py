import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== CONFIG =====================
INPUT_XLSX = r"C:\Users\shrey\OneDrive\Desktop\GPT_3\Dataset_100.xlsx"
OUTPUT_XLSX = r"C:\Users\shrey\OneDrive\Desktop\GPT_3\baseline_outputs_gpt2_gptneo_100.xlsx"

PROMPT_ID_COL = "PromptID"
HINDI_COL = "Hindi"

MODELS = [
    ("gpt2-medium", "GPT2_Medium"),
    ("EleutherAI/gpt-neo-1.3B", "GPTNeo_1_3B"),
]

MAX_NEW_TOKENS = 70
TEMPERATURE = 0.5
TOP_P = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 4

# ==================================================

def build_prompt(hindi_text: str) -> str:
    return (
    "2-3 वाक्यों में सरल और सहानुभूतिपूर्ण उत्तर लिखें।\n"
    "कोई चिकित्सीय सलाह न दें।\n\n"
    f"वर्णन: {hindi_text}\n"
    "उत्तर:"
)


def clean_response(text: str) -> str:
    if text is None:
        return "[EMPTY_OUTPUT]"
    t = text.strip()

    for stop in ["वर्णन:", "प्रतिक्रिया:", "नीचे एक व्यक्ति", "मानसिक स्थिति"]:
        if stop in t:
            t = t.split(stop, 1)[0].strip()

    t = "\n".join([line.strip() for line in t.splitlines() if line.strip()])

    if len(t.split()) < 5:
        return "[LOW_QUALITY_OUTPUT]"

    return t

def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model

def generate_one(model, tokenizer, device, prompt: str) -> str:
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
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True, errors="ignore")
    return clean_response(raw)

def main():
    df = pd.read_excel(INPUT_XLSX)

    if HINDI_COL not in df.columns:
        raise ValueError(f"Column '{HINDI_COL}' not found. Found columns: {list(df.columns)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    out = df[[PROMPT_ID_COL, HINDI_COL]].copy() if PROMPT_ID_COL in df.columns else df[[HINDI_COL]].copy()

    for model_name, col_name in MODELS:
        print("\nLoading model:", model_name)
        tokenizer, model = load_model(model_name, device)

        responses = []
        for i, row in df.iterrows():
            pid = row[PROMPT_ID_COL] if PROMPT_ID_COL in df.columns else i
            hindi_prompt = str(row[HINDI_COL])

            prompt = build_prompt(hindi_prompt)
            resp = generate_one(model, tokenizer, device, prompt)
            responses.append(resp)

            if (i + 1) % 10 == 0:
                print(f"{col_name}: Done {i+1}/{len(df)}")

        out[col_name] = responses

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    out.to_excel(OUTPUT_XLSX, index=False)
    print("\nSaved:", OUTPUT_XLSX)

if __name__ == "__main__":
    main()
