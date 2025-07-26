import os
import json
import fitz
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    role = data.get("persona", {}).get("role", "").lower()
    task = data.get("job_to_be_done", {}).get("task", "")
    return role, task

def get_score(txt, kw):
    if not txt.strip():
        return 0.0
    txt_emb = model.encode(txt, convert_to_tensor=True)
    kw_emb = model.encode(kw, convert_to_tensor=True)
    scores = util.cos_sim(txt_emb, kw_emb)[0]
    return float(scores.max().item())

def is_heading(blk, txt):
    wc = len(txt.split())
    if wc > 10:
        return False
    for line in blk.get("lines", []):
        for span in line["spans"]:
            if span["size"] > 16 or "bold" in span["font"].lower():
                return True
    return False

def get_blocks(pdf_dir, kw):
    all_blks = []
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            doc = fitz.open(path)
            for pno in range(len(doc)):
                page = doc[pno]
                blocks = page.get_text("dict")["blocks"]
                for blk in blocks:
                    if "lines" not in blk:
                        continue
                    txt = ""
                    for line in blk["lines"]:
                        for span in line["spans"]:
                            txt += span["text"] + " "
                    txt = txt.strip()
                    if not txt:
                        continue
                    sc = get_score(txt, kw)
                    hd = is_heading(blk, txt)
                    all_blks.append({
                        "text": txt,
                        "page": pno + 1,
                        "score": sc,
                        "file": fname,
                        "is_heading": hd
                    })
    return all_blks

def top_headings(blks, n=5):
    hd_blks = [b for b in blks if b.get("is_heading")]
    hd_blks.sort(key=lambda x: x["score"], reverse=True)
    top = hd_blks[:n]
    return [{
        "document": b["file"],
        "section_title": b["text"],
        "importance_rank": i + 1,
        "page_number": b["page"]
    } for i, b in enumerate(top)]

def top_subsections(blks, n=5):
    nh_blks = [b for b in blks if not b.get("is_heading")]
    nh_blks.sort(key=lambda x: x["score"], reverse=True)
    top = nh_blks[:n]
    return [{
        "document": b["file"],
        "refined_text": b["text"],
        "page_number": b["page"]
    } for b in top]

def save_output(blks, out_path, docs, role, task):
    blks.sort(key=lambda x: x["score"], reverse=True)
    meta = {
        "input_documents": docs,
        "persona": role,
        "job_to_be_done": task,
        "processing_timestamp": datetime.now().isoformat()
    }
    output = {
        "metadata": meta,
        "extracted_sections": top_headings(blks, 5),
        "subsection_analysis": top_subsections(blks, 5)
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    in_dir = "input"
    in_json = os.path.join(in_dir, "input.json")
    out_json = "challenge1b_output.json"
    role, task = load_input(in_json)
    keywords = role.split() + task.split()
    pdfs = sorted([f for f in os.listdir(in_dir) if f.endswith(".pdf")])
    blocks = get_blocks(in_dir, keywords)
    save_output(blocks, out_json, pdfs, role, task)