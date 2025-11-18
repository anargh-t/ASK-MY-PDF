"""
Utility script to create lightweight documentation assets (PNG + PDFs).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF, XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def create_architecture_png() -> None:
    width, height = 1200, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    heading_font = font

    draw.text((40, 20), "ASK MY PDF - System Architecture", fill="black", font=heading_font)

    boxes = [
        (50, 80, 350, 200, "PDF Upload\n(Streamlit UI)"),
        (420, 80, 720, 200, "Validation & Storage\n(FastAPI)"),
        (790, 80, 1090, 200, "Extraction Layer\n(pdfplumber)"),
        (50, 260, 350, 380, "Chunking + Cleaning\n(500 tokens, overlap 100)"),
        (420, 260, 720, 380, "Embeddings\n(all-MiniLM-L6-v2)"),
        (790, 260, 1090, 380, "FAISS Vector Store\n(IndexFlatL2)"),
        (50, 440, 350, 560, "Retriever\nTop-k search"),
        (420, 440, 720, 560, "LLM Reasoning\nGemini 1.5 Flash"),
        (790, 440, 1090, 560, "API Response\nHistory + Metrics"),
    ]

    def measure(text: str):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="black", width=3)
        text_lines = label.split("\n")
        total_height = len(text_lines) * 14
        for idx, line in enumerate(text_lines):
            w, h = measure(line)
            draw.text(((x1 + x2 - w) / 2, (y1 + y2 - total_height) / 2 + idx * 14), line, fill="black", font=font)

    arrows = [
        ((350, 140), (420, 140)),
        ((720, 140), (790, 140)),
        ((350, 320), (420, 320)),
        ((720, 320), (790, 320)),
        ((350, 500), (420, 500)),
        ((720, 500), (790, 500)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        draw.line([x1, y1, x2, y2], fill="black", width=3)
        draw.polygon([(x2, y2), (x2 - 12, y2 - 8), (x2 - 12, y2 + 8)], fill="black")

    img.save(DOCS_DIR / "architecture.png")


def build_pdf(filename: str, title: str, sections: dict) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Arial", size=11)
    content_width = pdf.w - 2 * pdf.l_margin
    for heading, bullets in sections.items():
        pdf.ln(6)
        pdf.set_font("Arial", "B", 13)
        pdf.multi_cell(content_width, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", size=11)
        for bullet in bullets:
            pdf.multi_cell(content_width, 6, f"- {bullet}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(DOCS_DIR / filename))


def create_threat_model_pdf() -> None:
    sections = {
        "STRIDE Threat Model": [
            "Spoofing: Fake PDF uploads mitigated via MIME/type validation.",
            "Tampering: Vector index writes restricted to server-side storage.",
            "Repudiation: Query history stored with timestamps.",
            "Information disclosure: API keys isolated in .env and never sent to UI.",
            "Denial of Service: File size limits (20MB) + chunk constraints.",
            "Elevation of privilege: PDFs treated as data, no execution paths.",
        ],
        "Controls": [
            "Input validation, sanitization, and exception handling at each endpoint.",
            "Server-side logging with latency metrics for anomaly detection.",
            "Use of environment variables + secrets scanning guidance.",
        ],
    }
    build_pdf("threat_model.pdf", "ASK MY PDF - Threat Model", sections)


def create_sprint_plan_pdf() -> None:
    sections = {
        "Sprint 1 - Model Development": [
            "Deliver extraction, chunking, embeddings, FAISS index, and evaluation metrics.",
        ],
        "Sprint 2 - API + UI": [
            "Expose /upload, /extract-text, /query, /history endpoints.",
            "Streamlit UI for upload, preview, chat, and history.",
        ],
        "Sprint 3 - Security & Robustness": [
            "Validation, threat model, resilience testing, secrets management.",
        ],
        "Sprint 4 - Packaging & Release": [
            "Documentation, README, demo checklist, folder structure verification.",
        ],
    }
    build_pdf("sprint_plan.pdf", "ASK MY PDF - Sprint Plan", sections)


def main():
    create_architecture_png()
    create_threat_model_pdf()
    create_sprint_plan_pdf()


if __name__ == "__main__":
    main()

