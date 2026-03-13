import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


INPUT_XLSX = r"C:\Users\dell\Heriot-Watt University\Dissertation Datasets\Sample_3.xlsx"
OUTPUT_XLSX = r"C:\Users\dell\Heriot-Watt University\Dissertation Datasets\Sample3_translated_indictrans2.xlsx"


SOURCE_COL = "Orignal"   
TARGET_COL = "Hindi"     


MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_TAG = "eng_Latn"
TGT_TAG = "hin_Deva"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

model.config.use_cache = False


def split_into_chunks(text: str, max_chars: int = 900):
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"\n\s*\n", text)  
    chunks, buf = [], ""

    for p in parts:
        p = p.strip()
        if not p:
            continue

        if len(p) > max_chars:
            sents = re.split(r"(?<=[.!?])\s+", p)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s
        else:
            if not buf:
                buf = p
            elif len(buf) + len(p) + 2 <= max_chars:
                buf = buf + "\n\n" + p
            else:
                chunks.append(buf)
                buf = p

    if buf:
        chunks.append(buf)

    return chunks


def translate_batch(chunks):
    tagged = [f"{SRC_TAG} {TGT_TAG} {t}" for t in chunks]

    inputs = tokenizer(
        tagged,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            use_cache=False,   
            num_beams=2,       
            max_new_tokens=256
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_long_text(text: str):
    chunks = split_into_chunks(text, max_chars=900)
    if not chunks:
        return ""

    out = []
    batch_size = 2

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        out.extend(translate_batch(batch))
        print(f"  chunks {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)} done")

    return "\n\n".join(out)


df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

if SOURCE_COL not in df.columns:
    raise ValueError(f"Column '{SOURCE_COL}' not found. Available columns: {list(df.columns)}")

df[TARGET_COL] = ""

total = len(df)
for idx, text in enumerate(df[SOURCE_COL].astype(str).tolist(), start=1):
    print(f"Row {idx}/{total}")
    df.at[idx - 1, TARGET_COL] = translate_long_text(text)

df.to_excel(OUTPUT_XLSX, index=False)
print("Saved:", OUTPUT_XLSX)
