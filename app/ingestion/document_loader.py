import csv
import os

from app.ingestion.pdf_loader import clean_text, load_pdf


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}


def load_document(file_path, original_filename="", source_extension=""):
    extension = (source_extension or os.path.splitext(file_path)[1]).lower()
    source_name = original_filename or os.path.basename(file_path)

    if extension == ".pdf":
        docs = load_pdf(file_path)
    elif extension == ".txt":
        docs = _load_txt(file_path)
    elif extension == ".csv":
        docs = _load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    for doc in docs:
        doc["source_name"] = source_name
        doc["source_type"] = extension.lstrip(".") or "document"
    return docs


def _load_txt(file_path):
    text = _read_text_file(file_path)
    cleaned = clean_text(text)
    return [{"text": cleaned, "page": 1}] if cleaned else []


def _load_csv(file_path):
    text = _read_text_file(file_path)
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                cells = [cell.strip() for cell in row if cell and cell.strip()]
                if cells:
                    rows.append(" | ".join(cells))
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                cells = [cell.strip() for cell in row if cell and cell.strip()]
                if cells:
                    rows.append(" | ".join(cells))
    except csv.Error:
        rows = [text]

    cleaned = clean_text("\n".join(rows))
    return [{"text": cleaned, "page": 1}] if cleaned else []


def _read_text_file(file_path):
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "rb") as handle:
        return handle.read().decode("utf-8", errors="ignore")
