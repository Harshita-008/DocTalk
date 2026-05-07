import re

from pypdf import PdfReader


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def repair_spacing_artifacts(text):
    """Repair common missing-space artifacts produced by PDF text extraction."""
    if not text:
        return ""

    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n+\s*(?=(?:\d{1,3}[\).]|[-*]))", "\n", text)
    text = re.sub(r"\s*\n+\s*(?=[A-Z][A-Za-z0-9 /-]{2,40}:)", "\n", text)
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"(?<=[A-Za-z])-(?=(?:how|when|where|why|what|who|which)\b)", " - ", text)

    patterns = [
        (
            r"\b(It|This|That|These|Those|They|There|Here|We|You|He|She)"
            r"(is|are|was|were|will|would|can|could|should|has|have|had|"
            r"does|do|did|became|becomes?|seems?|appears?|helps?|provides?|"
            r"emphasizes?|emphasises?|highlights?|explores?|introduces?|"
            r"discusses?|includes?|shows?|means|refers|serves?|deals?|"
            r"doesn't|isn't|wasn't|won't|can't)\b",
            r"\1 \2",
        ),
        (
            r"\b(The|A|An)"
            r"(chapter|section|focus|goal|term|concept|process|method|methods|"
            r"source|sources|record|records|document|documents|benefit|benefits|"
            r"problem|problems|challenge|challenges|result|results|role|roles|"
            r"purpose|importance|following|first|second|next|same|main)\b",
            r"\1 \2",
        ),
        (
            r"\b(In|On|At|By|To|For|From|With|Without|Before|After|During|"
            r"Because|Although|While|When|Where|What|How|Why|Who|Which)"
            r"(the|a|an|this|that|these|those|their|its|his|her|our|your)\b",
            r"\1 \2",
        ),
        (r"\b(all)(about|the)\b", r"\1 \2"),
        (r"\b(historical)(inquiry|narratives?|events?)\b", r"\1 \2"),
        (r"\b(historical)(sources?)\b", r"\1 \2"),
        (r"\b(official)(records?|government)\b", r"\1 \2"),
        (r"\b(reliability)(and)\b", r"\1 \2"),
        (r"\b(their)(ability|beliefs)\b", r"\1 \2"),
        (r"\b(soil)(quality)\b", r"\1 \2"),
        (r"\b(in)(history)\b", r"\1 \2"),
        (
            r"\b(to)(better|create|cover|map|preserve|provide|study|assess|"
            r"understand|reconstruct|convey|disseminate|determine|gather|"
            r"fully)\b",
            r"\1 \2",
        ),
        (r"\b(will)(never|learn|discover|comprehend|understand)\b", r"\1 \2"),
        (r"\b(fully)(comprehend|understand)\b", r"\1 \2"),
    ]

    previous = None
    while previous != text:
        previous = text
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\bAn other\b", "Another", text)
    text = re.sub(r"\ban other\b", "another", text)
    text = re.sub(
        r"\b(How|What|Why|When|Where|Who|Which)"
        r"(do|does|did|is|are|was|were|can|will|would|should)\b",
        r"\1 \2",
        text,
    )
    text = re.sub(
        r"\b(what|why|when|where|who|which)\s+(is|are|was|were)([A-Za-z]{4,})\b",
        r"\1 \2 \3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(how)\s+(do|does|did)(we|you|they|historians|people)\b",
        r"\1 \2 \3",
        text,
        flags=re.IGNORECASE,
    )

    return text


def _build_replacements():
    """Return replacement pairs as (old, new) with full Unicode support.

    Using a function lets us build the strings programmatically so no
    non-ASCII character ever appears as a raw literal inside a "..." string
    in the source code -- avoiding any parser or editor confusion.
    """
    pairs = []

    # Non-breaking space
    pairs.append((" ", " "))

    # Single-level mojibake: UTF-8 bytes of a typographic char misread via Windows-1252.
    # Each sequence is built from the individual Unicode code points so the
    # source file stays entirely ASCII-safe.
    ldq = "“"   # LEFT DOUBLE QUOTATION MARK
    rdq = "”"   # RIGHT DOUBLE QUOTATION MARK
    lsq = "‘"   # LEFT SINGLE QUOTATION MARK
    rsq = "’"   # RIGHT SINGLE QUOTATION MARK
    emd = "—"   # EM DASH
    end = "–"   # EN DASH
    ell = "…"   # HORIZONTAL ELLIPSIS

    # a-umlaut + euro + specific Win-1252 char = mojibake for each typographic char
    a_um  = "â"   # a with circumflex (byte E2 misread as Latin-1)
    euro  = "€"   # euro sign (byte 80 misread as Win-1252)
    oe    = "œ"   # oe ligature (byte 9C as Win-1252)
    u009d = ""   # control char (byte 9D -- undefined in Win-1252)
    tilde = "˜"   # small tilde (byte 98 as Win-1252)
    tm    = "™"   # trade mark (byte 99 as Win-1252)

    pairs += [
        (a_um + euro + "¦", "..."),  # ellipsis mojibake (byte A6)
        (a_um + euro + oe,       '"'),    # left double quote mojibake
        (a_um + euro + u009d,    '"'),    # right double quote mojibake
        (a_um + euro + tilde,    "'"),    # left single quote mojibake
        (a_um + euro + tm,       "'"),    # right single quote mojibake
        (a_um + euro + rdq,      "-"),    # em dash mojibake (94 -> rdq in Win-1252)
        (a_um + euro + ldq,      "-"),    # en dash mojibake (93 -> ldq in Win-1252)
    ]

    # Direct Unicode typographic characters -> plain ASCII equivalents
    pairs += [
        (ldq, '"'),
        (rdq, '"'),
        (lsq, "'"),
        (rsq, "'"),
        (emd, "-"),
        (end, "-"),
        (ell, "..."),
    ]

    return pairs


_REPLACEMENTS = _build_replacements()


def clean_text(text):
    if not text:
        return ""

    text = CONTROL_CHARS.sub(" ", text)

    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)

    text = repair_spacing_artifacts(text)

    # Remove dot-leader and underline artifacts common in textbook PDFs.
    text = re.sub(r"[.]{2,}", " ", text)
    text = re.sub(r"[-_]{2,}", " ", text)

    text = re.sub(r"\s+", " ", text)

    # Strip References / Bibliography only when it is a standalone section
    # heading (own line) -- avoids cutting paragraphs that merely contain
    # the word "references" in passing.
    text = re.sub(
        r"\n\s*(?:References?|Bibliography)\s*\n.*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


def _clean_page_noise(text):
    lines = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lowered = line.lower()
        if re.search(r"\bcontents lists available at\b|\bjournal homepage\b|\bwww\.", lowered):
            continue
        if re.search(r"^\d+\s+of\s+\d+$", lowered):
            continue
        if re.search(r"\b(?:received|accepted|available online)\b", lowered):
            continue
        if re.search(r"^https?://", lowered):
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_with_pymupdf(file_path):
    try:
        import fitz
    except Exception:
        return []

    docs = []
    try:
        pdf = fitz.open(file_path)
    except Exception:
        return []

    for index, page in enumerate(pdf):
        blocks = []
        try:
            raw_blocks = page.get_text("blocks")
        except Exception:
            raw_blocks = []

        for block in raw_blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, block_text = block[:5]
            block_text = _clean_page_noise(block_text)
            if not block_text:
                continue
            blocks.append({
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "text": block_text,
            })

        if not blocks:
            continue

        width = float(page.rect.width)
        split = width / 2.0
        full_width = []
        left = []
        right = []

        for block in blocks:
            center = (block["x0"] + block["x1"]) / 2.0
            if block["x0"] < width * 0.18 and block["x1"] > width * 0.82:
                full_width.append(block)
            elif center < split:
                left.append(block)
            else:
                right.append(block)

        ordered = []
        for group in (full_width, left, right):
            ordered.extend(
                block["text"]
                for block in sorted(group, key=lambda item: (item["y0"], item["x0"]))
            )

        text = clean_text("\n".join(ordered))
        if text:
            docs.append({"text": text, "page": index + 1})

    try:
        pdf.close()
    except Exception:
        pass

    return docs


def extract_page_text(page):
    candidates = []

    for kwargs in ({}, {"extraction_mode": "layout"}):
        try:
            extracted = page.extract_text(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        cleaned = clean_text(extracted)
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return ""

    return max(candidates, key=text_quality_score)


def text_quality_score(text):
    words = text.split()
    if not words:
        return 0

    long_glued_words = sum(
        1
        for word in words
        if len(word) > 18
        and re.search(r"[a-z]{6,}(?:the|and|of|to|for|with|from)[a-z]{4,}", word.lower())
    )
    readable_words = sum(1 for word in words if re.search(r"[A-Za-z]", word))

    return readable_words - (long_glued_words * 12)


def load_pdf(file_path):
    pymupdf_docs = _extract_with_pymupdf(file_path)
    if pymupdf_docs:
        return pymupdf_docs

    reader = PdfReader(file_path)
    docs = []

    for i, page in enumerate(reader.pages):
        text = extract_page_text(page)

        if text:
            docs.append({
                "text": text,
                "page": i + 1
            })

    return docs
