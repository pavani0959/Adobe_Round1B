import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# === Load SentenceTransformer Model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Persona and Job ===
def load_input_details(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    persona = input_data.get("persona", {}).get("role", "").lower()
    job_task = input_data.get("job_to_be_done", {}).get("task", "")
    return persona, job_task

# === Score a Single Block ===
def score_block(text, persona_keywords):
    if not text.strip():
        return 0.0
    text_embedding = model.encode(text, convert_to_tensor=True)
    keyword_embeddings = model.encode(persona_keywords, convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_embedding, keyword_embeddings)[0]
    return float(cosine_scores.max().item())

# === Extract and Score All Blocks from All PDFs ===
def extract_and_score_all(input_folder, persona_keywords):
    all_scored_blocks = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"📄 Processing: {filename}")
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + " "

                    text = text.strip()
                    score = score_block(text, persona_keywords)

                    all_scored_blocks.append({
                        "text": text,
                        "page": page_num + 1,
                        "score": score,
                        "source_file": filename
                    })

    return all_scored_blocks

# === Save Final Merged Output ===
def save_merged_output(scored_blocks, output_path):
    scored_blocks.sort(key=lambda x: x["score"], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_blocks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Merged output saved to: {output_path}")
    print(f"🔢 Total blocks: {len(scored_blocks)}")

# === Main Entry ===
if __name__ == "__main__":
    input_folder = "input"
    input_json_path = os.path.join(input_folder, "input.json")
    output_path = "merged_output.json"

    persona, job_task = load_input_details(input_json_path)
    keywords = persona.split() + job_task.split()

    print(f"\n👤 Persona: {persona}")
    print(f"🛠️  Task: {job_task}")
    print(f"🔍 Keywords: {keywords}\n")

    scored_blocks = extract_and_score_all(input_folder, keywords)
    save_merged_output(scored_blocks, output_path)
