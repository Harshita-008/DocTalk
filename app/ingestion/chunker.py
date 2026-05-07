import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CONTEXT_WINDOW_SIZE

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None


MIN_CHUNK_WORDS = 18
MIN_FALLBACK_WORDS = 5

SECTION_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+)+\s+[A-Z][A-Za-z0-9,;:'\"()&/ -]{2,120}$"
)
SHORT_HEADING_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9,'\"()&/ -]{2,70})\s*:?\s*$"
)
LIST_ITEM_RE = re.compile(r"^\s*(?:\d{1,3}[\).]|[-*]|\u2022|[a-z]\))\s+")
LABEL_ITEM_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9'\"/&(). -]{2,72}:\s+\S+")
LOW_VALUE_LINE_RE = re.compile(
    r"^\s*(?:self assessment|review questions?|further readings?|keywords?|"
    r"objectives?|contents?|chapter overview|source:|task|caution|did u know\?|"
    r"learning outcomes?|key terms?|summary questions?|check your understanding)\b",
    re.IGNORECASE,
)

BOILERPLATE_PHRASES = {
    "here we have provided",
    "to better comprehend the ideas",
    "students should review the chapter",
    "students should review",
    "please note that",
}


def is_valid_chunk(text):
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        return False
    if sum(1 for word in words if re.search(r"[A-Za-z]", word)) < 10:
        return False
    if _is_mostly_boilerplate(text):
        return False
    return True


def normalize_chunk_text(text):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = _restore_structure(text)
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    lines = _merge_orphan_list_markers(lines)
    return "\n".join(lines).strip()


def _merge_orphan_list_markers(lines):
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\d{1,3}[\).]$", line) and i + 1 < len(lines):
            merged.append(f"{line} {lines[i + 1]}")
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def _restore_structure(text):
    """Recover generic PDF structure before chunking.

    PDF extractors often flatten headings, numbered lists, and label-style
    textbook items into one long paragraph. Retrieval works much better when
    those boundaries are visible to both the embedder and the generator.
    """
    text = re.sub(r"\s*/circle\s*6\s+", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\n+\s*", "\n", text)

    text = re.sub(
        r"(?<!^)(?<!\n)\s+(\d+(?:\.\d+)+\s+[A-Z][A-Za-z0-9,;:'\"()&/ -]{2,120})",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!^)(?<!\n)\s+(\d{1,3}[\).]\s+(?=[A-Z]))",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!^)(?<!\n)\s+([-*]\s+(?=[A-Z]))",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!\d\. )(?<!\d\) )(?<!^)(?<!\n)\s+([A-Z][A-Za-z0-9'\"/&(). -]{2,72}:\s+)",
        r"\n\1",
        text,
    )

    return text


def _clean_line(line):
    line = re.sub(r"\s+", " ", line or "").strip()
    line = re.sub(r"^Page\s+\d+\s*:?\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\d+\s+of\s+\d+\s+", "", line, flags=re.IGNORECASE)
    return line


def _is_list_like_line(line):
    return bool(LIST_ITEM_RE.match(line))


def _is_heading_like_line(line):
    if SECTION_HEADING_RE.match(line):
        return True
    if line.endswith(":") and len(line.split()) <= 10:
        return True
    return False


def _is_labeled_item(line):
    return bool(LABEL_ITEM_RE.match(line))


def _is_low_value_line(line):
    lowered = (line or "").lower()
    if bool(LOW_VALUE_LINE_RE.match(line)):
        return True
    return any(lowered.startswith(phrase) for phrase in BOILERPLATE_PHRASES)


def _is_mostly_boilerplate(text):
    lowered = (text or "").lower()
    hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lowered)
    return hits >= 2 and len(lowered.split()) < 180


def _split_units(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    units = []
    sentence_re = re.compile(r"(?<=[.!?])\s+")

    for line in lines:
        if _is_low_value_line(line):
            continue
        if _is_heading_like_line(line) or _is_list_like_line(line) or _is_labeled_item(line):
            units.append(line)
            continue
        units.extend([part.strip() for part in sentence_re.split(line) if part.strip()])

    return units


def _unit_words(unit):
    return len((unit or "").split())


def _split_long_unit(unit, chunk_size):
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(chunk_size * 6, 500),
            chunk_overlap=max(CHUNK_OVERLAP * 6, 80),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = [piece.strip() for piece in splitter.split_text(unit) if piece.strip()]
        if pieces:
            return pieces

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", unit) if part.strip()]
    if len(sentences) > 1 and max(_unit_words(sentence) for sentence in sentences) < chunk_size:
        return sentences

    words = unit.split()
    slices = []
    start = 0
    while start < len(words):
        slices.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += max(chunk_size - CHUNK_OVERLAP, 1)
    return slices


def _chunk_text_from_units(units):
    return "\n".join(unit.strip() for unit in units if unit.strip()).strip()


def _heading_title(unit):
    if not unit:
        return ""
    if SECTION_HEADING_RE.match(unit):
        return unit
    if unit.endswith(":") and len(unit.split()) <= 10:
        return unit.rstrip(":")
    return ""


def _overlap_units(units, overlap):
    if overlap <= 0:
        return []

    selected = []
    count = 0
    for unit in reversed(units):
        if _is_heading_like_line(unit):
            continue
        selected.insert(0, unit)
        count += _unit_words(unit)
        if count >= overlap:
            break

    return selected


def chunk_text(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    chunk_index = 0

    for doc in docs:
        text = normalize_chunk_text(doc["text"])
        page = doc["page"]
        source_name = doc.get("source_name", "")
        source_type = doc.get("source_type", "document")
        before_doc_count = len(chunks)
        units = _split_units(text)

        if not units:
            if _has_minimum_fallback_text(text, source_type):
                chunks.append({
                    "text": text,
                    "page": page,
                    "chunk_index": chunk_index,
                    "section_title": "",
                    "source_name": source_name,
                    "source_type": source_type,
                })
                chunk_index += 1
            continue

        current_units = []
        current_words = 0
        current_heading = ""

        def flush(keep_overlap=True):
            nonlocal chunk_index, current_units, current_words

            chunk = _chunk_text_from_units(current_units)
            if is_valid_chunk(chunk):
                chunks.append({
                    "text": chunk,
                    "page": page,
                    "chunk_index": chunk_index,
                    "section_title": current_heading,
                    "source_name": source_name,
                    "source_type": source_type,
                })
                chunk_index += 1

            if keep_overlap:
                tail = _overlap_units(current_units, overlap)
                if current_heading and tail and tail[0] != current_heading:
                    tail.insert(0, current_heading)
                current_units = tail
                current_words = sum(_unit_words(unit) for unit in current_units)
            else:
                current_units = []
                current_words = 0

        for unit in units:
            if _is_low_value_line(unit):
                continue

            heading = _heading_title(unit)
            if heading and current_units and current_words >= MIN_CHUNK_WORDS:
                flush(keep_overlap=False)

            if heading:
                current_heading = heading

            unit_len = _unit_words(unit)
            if unit_len == 0:
                continue

            if unit_len > chunk_size:
                for piece in _split_long_unit(unit, chunk_size):
                    piece_len = _unit_words(piece)
                    if current_units and current_words + piece_len > chunk_size:
                        flush()
                    current_units.append(piece)
                    current_words += piece_len
                continue

            if current_units and current_words + unit_len > chunk_size:
                flush()

            current_units.append(unit)
            current_words += unit_len

        if current_units:
            flush(keep_overlap=False)

        if len(chunks) == before_doc_count and _has_minimum_fallback_text(text, source_type):
            chunks.append({
                "text": text,
                "page": page,
                "chunk_index": chunk_index,
                "section_title": "",
                "source_name": source_name,
                "source_type": source_type,
            })
            chunk_index += 1

    return _attach_context_windows(chunks)


def _has_minimum_fallback_text(text, source_type="document"):
    words = (text or "").split()
    if source_type in {"csv", "txt"}:
        return len(words) >= 2 and any(re.search(r"[A-Za-z0-9]", word) for word in words)

    return (
        len(words) >= MIN_FALLBACK_WORDS
        and sum(1 for word in words if re.search(r"[A-Za-z]", word)) >= MIN_FALLBACK_WORDS
    )


def _attach_context_windows(chunks, radius=CONTEXT_WINDOW_SIZE):
    if radius <= 0:
        for chunk in chunks:
            chunk["window_text"] = chunk["text"]
        return chunks

    for index, chunk in enumerate(chunks):
        start = max(0, index - radius)
        end = min(len(chunks), index + radius + 1)
        window_parts = []
        for neighbor in chunks[start:end]:
            if abs(int(neighbor.get("page", 0)) - int(chunk.get("page", 0))) > 1:
                continue
            text = neighbor.get("text", "")
            if text:
                window_parts.append(text)
        chunk["window_text"] = _dedupe_window_lines("\n".join(window_parts))

    return chunks


def _dedupe_window_lines(text):
    seen = set()
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        key = re.sub(r"\W+", "", line.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines).strip()
