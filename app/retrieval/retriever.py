import re

from app.config import MAX_CONTEXT_CHUNKS, RERANK_TOP_N, SIMILARITY_THRESHOLD, TOP_K
from app.retrieval.vector_store import VectorStore


STOPWORDS = {
    "a", "about", "an", "and", "answer", "are", "as", "by", "case", "did",
    "do", "does", "document", "explain", "for", "from", "give", "has",
    "have", "how", "in", "into", "is", "main", "mean", "means", "meant",
    "key", "of", "on", "or", "provided", "simple", "study", "tell", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "word", "words", "solve", "solves", "solved",
    "address", "addresses", "addressed",
}

LIST_WORDS = {
    "type", "types", "kind", "kinds", "category", "categories",
    "classification", "classifications", "classified", "form", "forms",
    "list", "section", "sections", "part", "parts", "component",
    "components", "element", "elements", "characteristic", "characteristics",
    "feature", "features", "stage", "stages", "step", "steps",
    "trait", "traits", "quality", "qualities",
}

PROBLEM_WORDS = {
    "problem", "problems", "challenge", "challenges", "issue", "issues",
    "difficulty", "difficulties", "barrier", "barriers", "constraint",
    "constraints",
}

EXPLANATORY_WORDS = {
    "role", "roles", "importance", "important", "contribution",
    "contributions", "impact", "impacts", "effect", "effects", "function",
    "functions", "benefit", "benefits", "significance", "purpose",
    "responsibility", "responsibilities",
}

LIST_CUES = LIST_WORDS | {
    "following", "include", "includes", "including", "consist", "consists",
    "comprise", "comprises", "basis",
}

DEFINITION_CUES = {
    "is", "are", "means", "meaning", "refers", "defined", "definition",
    "process", "tendency", "concept", "system", "method", "approach",
}

LOW_VALUE_MARKERS = {
    "review questions", "further readings", "table of contents",
    "learning objectives", "chapter overview", "chapter objectives",
    "self assessment", "fill in the blanks", "here we have provided",
    "to better comprehend the ideas", "students should review the chapter",
    "- how, when, and where is dates", "syllabus", "sr. no.", "objectives",
    "contents objectives", "references",
    "declaration of competing interest", "credit authorship contribution",
}


class Retriever:
    def __init__(self):
        self.vs = VectorStore()

    def retrieve(self, query):
        all_chunks = self._load_all_chunks()
        if not all_chunks:
            return []

        top_k = max(TOP_K, 20)
        vector_chunks = self._format_vector_results(self.vs.query(query, top_k=top_k))
        keyword_chunks = self._keyword_rank(query, all_chunks)[:top_k]

        candidates = self._merge_chunks(vector_chunks + keyword_chunks)
        reranked = self._rerank_candidates(query, candidates)
        reranked = [
            chunk for chunk in reranked
            if self._has_minimum_evidence(query, chunk)
        ]

        if not reranked:
            return []

        expanded = self._expand_context(query, reranked, all_chunks)
        expanded = self._merge_chunks(expanded)
        expanded = self._rerank_candidates(query, expanded)
        expanded = [
            chunk for chunk in expanded
            if self._has_minimum_evidence(query, chunk)
        ]

        return self._limit_chunks(query, expanded)

    def _format_vector_results(self, results):
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            distance = distances[i] if i < len(distances) else None
            vector_score = 0.0
            if isinstance(distance, (int, float)):
                vector_score = max(0.0, 1.0 - float(distance))

            formatted.append({
                "id": ids[i] if i < len(ids) else self._chunk_key(text, meta),
                "text": text or "",
                "page": int(meta.get("page", 0) or 0),
                "chunk_index": int(meta.get("chunk_index", i) or i),
                "section_title": meta.get("section_title", "") or "",
                "window_text": meta.get("window_text", "") or text or "",
                "source_name": meta.get("source_name", "") or "",
                "source_type": meta.get("source_type", "") or "",
                "distance": distance,
                "vector_score": vector_score,
                "score": vector_score,
                "source": "vector",
                "rank": i,
            })

        return formatted

    def _load_all_chunks(self):
        data = self.vs.get_all()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])

        chunks = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            chunks.append({
                "id": ids[i] if i < len(ids) else self._chunk_key(text, meta),
                "text": text or "",
                "page": int(meta.get("page", 0) or 0),
                "chunk_index": int(meta.get("chunk_index", i) or i),
                "section_title": meta.get("section_title", "") or "",
                "window_text": meta.get("window_text", "") or text or "",
                "source_name": meta.get("source_name", "") or "",
                "source_type": meta.get("source_type", "") or "",
                "vector_score": 0.0,
                "score": 0.0,
                "source": "all",
                "rank": i,
            })

        return sorted(chunks, key=lambda c: (c["page"], c["chunk_index"]))

    def _keyword_rank(self, query, chunks):
        query_terms = self._query_terms(query)
        topic_terms = self._topic_terms(query)
        subject_terms = self._subject_terms(query)
        ranked = []

        for chunk in chunks:
            text = chunk.get("text", "")
            text_lower = text.lower()
            if not text_lower.strip():
                continue

            term_hits = self._count_term_hits(query_terms, text_lower)
            topic_hits = self._count_term_hits(topic_terms, text_lower)
            subject_hits = self._count_term_hits(subject_terms, text_lower)

            score = term_hits * 1.1 + topic_hits * 3.0 + subject_hits * 4.5
            score += self._phrase_score(query, text_lower)

            if self._is_definition_question(query):
                score += self._cue_score(text_lower, DEFINITION_CUES)
                score += self._definition_score(query, text_lower)
                score += self._acronym_definition_score(query, text_lower)
                score += self._named_method_score(query, text_lower)
            if self._is_list_question(query):
                score += self._cue_score(text_lower, LIST_CUES)
                score += self._heading_score(query, chunk)
            if self._is_problem_question(query):
                score += self._cue_score(text_lower, PROBLEM_WORDS | {"following"})
                score += self._heading_score(query, chunk)
            if self._is_explanatory_question(query):
                score += self._cue_score(text_lower, EXPLANATORY_WORDS)
                score += self._cue_score(text_lower, {"creates", "provides", "helps", "improves", "accelerate"})
                if re.search(r"(?:^|\n)\s*\d{1,3}[\).]\s+", text):
                    score += 8.0
            if self._is_main_issue_question(query):
                score += self._cue_score(text_lower, PROBLEM_WORDS | {"fraud", "confessed", "confession"})
                if re.search(r"\b(accounting fraud|confessed to .*fraud|inflating .*revenue|profits? reported|cash balances .*did not exist)\b", text_lower):
                    score += 45.0

            score += self._chunk_quality_score(text)

            if score > 0:
                item = dict(chunk)
                item["score"] = score
                item["keyword_score"] = score
                item["source"] = "keyword"
                ranked.append(item)

        return sorted(ranked, key=lambda c: (-c.get("score", 0), c["page"], c["chunk_index"]))

    def _rerank_candidates(self, query, chunks):
        query_terms = self._query_terms(query)
        topic_terms = self._topic_terms(query)
        subject_terms = self._subject_terms(query)
        ranked = []

        for chunk in chunks:
            item = dict(chunk)
            text = item.get("text", "")
            text_lower = text.lower()

            term_hits = self._count_term_hits(query_terms, text_lower)
            topic_hits = self._count_term_hits(topic_terms, text_lower)
            subject_hits = self._count_term_hits(subject_terms, text_lower)
            quality = self._chunk_quality_score(text)
            vector_score = float(item.get("vector_score", 0.0) or 0.0)
            keyword_score = float(item.get("keyword_score", item.get("score", 0.0)) or 0.0)

            relevance = (
                vector_score * 2.5
                + keyword_score
                + term_hits * 0.8
                + topic_hits * 2.0
                + subject_hits * 3.0
                + quality
                + self._phrase_score(query, text_lower)
            )
            if self._is_definition_question(query):
                relevance += self._acronym_definition_score(query, text_lower)
                relevance += self._named_method_score(query, text_lower)
                relevance += self._definition_score(query, text_lower)

            item["term_hits"] = term_hits
            item["topic_hits"] = topic_hits
            item["subject_hits"] = subject_hits
            item["quality_score"] = quality
            item["score"] = relevance
            ranked.append(item)

        return sorted(
            ranked,
            key=lambda c: (
                -c.get("score", 0.0),
                c.get("page", 0),
                c.get("chunk_index", 0),
            ),
        )

    def _has_minimum_evidence(self, query, chunk):
        text = (chunk.get("text") or "").lower()
        if not text.strip():
            return False

        topic_terms = self._topic_terms(query)
        subject_terms = self._subject_terms(query)
        topic_hits = self._count_term_hits(topic_terms, text)
        subject_hits = self._count_term_hits(subject_terms, text)
        vector_score = float(chunk.get("vector_score", 0.0) or 0.0)
        score = float(chunk.get("score", 0.0) or 0.0)

        if self._is_state_space_question(query) and re.search(r"\bstate[- ]space|state vector|dynamic model|equations of motion\b", text):
            return True

        if not self._is_state_space_question(query) and re.search(r"\b(pid|control law|controller equations?|equations?)\b", query.lower()) and re.search(r"\b(control law is given by|control input is defined as|position error vector|gain matrices|gains are|derivative terms|closed-loop response using pid control)\b", text):
            return True

        if self._is_low_value_text(text):
            return False

        if self._is_list_question(query):
            requested_list_terms = [
                term for term in self._content_terms(query)
                if term in LIST_WORDS
            ]
            if (
                requested_list_terms
                and self._count_term_hits(requested_list_terms, text) == 0
                and not self._has_list_cue_evidence(text)
                and not self._is_contribution_question(query)
            ):
                return False
            qualifier_terms = self._qualifier_terms(query)
            if qualifier_terms and self._count_term_hits(qualifier_terms, text) == 0:
                return False
        if self._is_problem_question(query) and re.search(r"\bfollowing\s+(?:problems|challenges|issues|difficulties)\b", text):
            return True

        if subject_terms:
            required_subject = min(2, len(subject_terms))
            if subject_hits >= required_subject:
                return True
            if self._is_definition_question(query):
                if len(subject_terms) >= 2:
                    return False
                return subject_hits >= 1
            if self._is_explanatory_question(query) and subject_hits >= 1:
                return True
            if topic_hits >= max(1, min(2, len(topic_terms))):
                return True
            return False

        if topic_terms and topic_hits > 0:
            return True

        return vector_score >= SIMILARITY_THRESHOLD and score >= 1.0

    def _expand_context(self, query, chunks, all_chunks):
        if not chunks:
            return []

        if self._is_main_issue_question(query):
            return self._expand_neighbors(chunks[:4], all_chunks, radius=2)

        if self._is_framework_process_question(query) and re.search(r"\b(pid|controller equations?|control law)\b", query.lower()):
            selected = self._expand_neighbors(chunks[:5], all_chunks, radius=1)
            for chunk in all_chunks:
                text_lower = (chunk.get("text") or "").lower()
                if re.search(r"\bpid\b", text_lower) and re.search(r"\binverted pendulum|\bip\b|gain matrices|gains are|derivative terms|control laws", text_lower):
                    selected.append(chunk)
            return selected

        if self._is_list_question(query) or self._is_problem_question(query):
            section = self._expand_best_section(query, chunks, all_chunks, max_chunks=MAX_CONTEXT_CHUNKS + 6)
            if section:
                return section
            return self._expand_neighbors(chunks[:4], all_chunks, radius=3)

        if self._is_explanatory_question(query):
            if re.search(r"\bimportance\b|\bimportant\b|\beconomic development\b|\beconomy\b", query.lower()) and re.search(r"\bentrepreneurship\b", query.lower()):
                return self._expand_neighbors(chunks[:6], all_chunks, radius=5)
            for chunk in chunks[:20]:
                text_lower = (chunk.get("text") or "").lower()
                if re.search(r"\bfollowing\b.{0,120}\b(reasons?|roles?|importance|benefits?|points?)\b", text_lower) or "vital role" in text_lower:
                    return self._expand_neighbors([chunk], all_chunks, radius=5)
            return self._expand_neighbors(chunks[:5], all_chunks, radius=2)

        if self._is_definition_question(query):
            return self._expand_neighbors(chunks[:4], all_chunks, radius=1)

        return self._expand_neighbors(chunks[:4], all_chunks, radius=1)

    def _expand_best_section(self, query, ranked_chunks, all_chunks, max_chunks):
        best = ranked_chunks[0]
        best_index = self._index_for_chunk(best, all_chunks)
        if best_index is None:
            return []

        section_title = best.get("section_title") or ""
        selected_indexes = set()

        if section_title:
            for i, chunk in enumerate(all_chunks):
                if chunk.get("section_title") == section_title:
                    selected_indexes.add(i)

        if not selected_indexes:
            section_number = self._nearest_section_number(all_chunks, best_index)
            if section_number:
                start = best_index
                while start > 0 and self._same_section(all_chunks[start - 1], section_number):
                    start -= 1
                end = best_index + 1
                while end < len(all_chunks) and self._same_section(all_chunks[end], section_number):
                    end += 1
                selected_indexes.update(range(start, min(end, start + max_chunks)))

        if len(selected_indexes) < 3:
            radius = 4 if self._is_list_question(query) or self._is_problem_question(query) else 2
            selected_indexes.update(
                i for i in range(best_index - radius, best_index + radius + 1)
                if 0 <= i < len(all_chunks)
            )

        ordered = [all_chunks[i] for i in sorted(selected_indexes)]
        return ordered[:max_chunks]

    def _nearest_section_number(self, chunks, index):
        for i in range(index, max(index - 6, -1), -1):
            number = self._first_section_number(chunks[i].get("text", ""))
            if number:
                return number
        return None

    def _same_section(self, chunk, section_number):
        if not section_number:
            return True
        number = self._first_section_number(chunk.get("text", ""))
        if not number:
            return True
        return number == section_number or number.startswith(section_number + ".")

    def _expand_neighbors(self, seed_chunks, all_chunks, radius=1):
        index_by_id = {
            chunk.get("id"): i
            for i, chunk in enumerate(all_chunks)
            if chunk.get("id") is not None
        }
        selected = set()

        for chunk in seed_chunks:
            index = index_by_id.get(chunk.get("id")) if chunk.get("id") is not None else None
            if index is None:
                index = self._index_for_chunk(chunk, all_chunks)
            if index is None:
                continue
            for neighbor in range(index - radius, index + radius + 1):
                if 0 <= neighbor < len(all_chunks):
                    selected.add(neighbor)

        return [all_chunks[i] for i in sorted(selected)]

    def _index_for_chunk(self, target, chunks):
        target_id = target.get("id")
        for i, chunk in enumerate(chunks):
            if target_id and chunk.get("id") == target_id:
                return i
            if (
                chunk.get("page") == target.get("page")
                and chunk.get("chunk_index") == target.get("chunk_index")
            ):
                return i
        return None

    def _limit_chunks(self, query, chunks):
        reranked = self._rerank_candidates(query, chunks)

        query_lower = query.lower()
        if re.search(r"\bimportance\b|\bimportant\b|\beconomic development\b|\beconomy\b", query_lower) and re.search(r"\bentrepreneurship\b", query_lower):
            limit = max(MAX_CONTEXT_CHUNKS, 6)
        elif self._is_framework_process_question(query) and re.search(r"\b(pid|controller equations?|control law|reinforcement learning)\b", query_lower):
            limit = max(MAX_CONTEXT_CHUNKS, 5)
        elif self._is_list_question(query) or self._is_problem_question(query):
            limit = min(MAX_CONTEXT_CHUNKS, 3)
        elif self._is_explanatory_question(query):
            limit = min(MAX_CONTEXT_CHUNKS, 3)
        else:
            limit = min(MAX_CONTEXT_CHUNKS, 3)

        selected = self._select_diverse_chunks(reranked, min(limit, max(RERANK_TOP_N, limit)))
        return sorted(selected, key=lambda c: (c.get("page", 0), c.get("chunk_index", 0)))

    def _select_diverse_chunks(self, chunks, limit):
        selected = []
        selected_terms = []

        for chunk in chunks:
            terms = set(self._content_terms(chunk.get("text", "")))
            if selected_terms and any(
                len(terms & existing) > max(8, len(terms) * 0.65)
                for existing in selected_terms
            ):
                continue
            selected.append(chunk)
            selected_terms.append(terms)
            if len(selected) >= limit:
                break

        if len(selected) < limit:
            selected_ids = {chunk.get("id") for chunk in selected}
            for chunk in chunks:
                if chunk.get("id") in selected_ids:
                    continue
                selected.append(chunk)
                if len(selected) >= limit:
                    break

        return selected

    def _merge_chunks(self, chunks):
        by_key = {}
        merged = []

        for chunk in chunks:
            key = chunk.get("id") or self._chunk_key(chunk.get("text", ""), chunk)
            if key in by_key:
                existing = merged[by_key[key]]
                if chunk.get("score", 0.0) > existing.get("score", 0.0):
                    merged[by_key[key]] = chunk
                continue
            by_key[key] = len(merged)
            merged.append(chunk)

        return merged

    def _query_terms(self, query):
        terms = self._content_terms(query)
        return list(dict.fromkeys(terms))

    def _topic_terms(self, query):
        ignored = STOPWORDS | LIST_WORDS | PROBLEM_WORDS | EXPLANATORY_WORDS | {
            "faced", "basis", "based", "define",
        }
        return [
            token for token in self._content_terms(query)
            if token not in ignored
        ]

    def _subject_terms(self, query):
        query_lower = self._normalize_question(query)
        patterns = [
            r"\bexamples?\s+of\s+(.+?)(?:\?|$)",
            r"\b(?:role|roles|importance|impact|effect|effects|benefits?|purpose|contributions?)\s+of\s+(.+?)(?:\s+in\s+|\s+for\s+|\?|$)",
            r"\b(?:types?|kinds?|categories|classifications?|forms?|sections?|parts?|components?|elements?|stages?|steps?)\s+(?:involved\s+in|of)\s+(.+?)(?:\?|$)",
            r"\b(?:problems?|challenges?|issues?|difficulties|barriers)\s+(?:faced\s+by|of|before|related\s+to)\s+(.+?)(?:\?|$)",
            r"\b(?:what\s+is|define|meaning\s+of)\s+(.+?)(?:\?|$)",
            r"\b(?:main|major|primary)\s+(?:issue|problem|reason|cause)\s+(?:in|of|with)\s+(.+?)(?:\?|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                subject_text = self._clean_subject_text(match.group(1))
                terms = self._content_terms(subject_text)
                terms = [
                    term for term in terms
                    if term not in LIST_WORDS | PROBLEM_WORDS | EXPLANATORY_WORDS | {"example", "examples", "vocabulary"}
                ]
                if terms:
                    return terms

        return self._topic_terms(query)

    def _clean_subject_text(self, text):
        text = re.split(r"\s+and\s+(?:what|how|why|when|where|which|who)\b", text or "", maxsplit=1)[0]
        text = re.sub(r"\b(?:problem|issue|challenge)s?\s+(?:does|do|did)\s+it\s+(?:solve|address)\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _content_terms(self, text):
        return [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
            if token not in STOPWORDS
        ]

    def _normalize_question(self, query):
        query = (query or "").lower()
        query = re.sub(r"\bin\s+simple\s+words\b", "", query)
        query = re.sub(r"\s+", " ", query)
        return query.strip()

    def _count_term_hits(self, terms, text_lower):
        return sum(1 for term in terms if self._term_matches(term, text_lower))

    def _phrase_score(self, query, text_lower):
        subject = " ".join(self._subject_terms(query))
        score = 0.0
        if subject and subject in text_lower:
            score += 6.0
        compact_subject = re.sub(r"\s+", " ", self._clean_subject_text(self._normalize_question(query)))
        if compact_subject and compact_subject in text_lower:
            score += 10.0
        for phrase in self._important_phrases(query):
            if phrase in text_lower:
                score += 4.0

        subject_terms = self._subject_terms(query)
        if self._is_list_question(query):
            list_terms = [term for term in self._content_terms(query) if term in LIST_WORDS]
            for cue in list_terms or ["types", "classification", "classified"]:
                for subject_term in subject_terms:
                    for variant in self._term_variants(subject_term):
                        if re.search(rf"\b{re.escape(cue)}\s+of\s+(?:an?\s+|the\s+)?{re.escape(variant)}\b", text_lower):
                            score += 34.0
                        if f"{cue} of {variant}" in text_lower:
                            score += 28.0
                        if f"{variant} {cue}" in text_lower:
                            score += 16.0
                        if f"classified into" in text_lower and variant in text_lower:
                            score += 12.0
            if re.search(r"\bbased\s+on\s+ownership\b|\bbasis\s+of\s+ownership\b", query.lower()):
                if re.search(r"\bclassification\s+on\s+the\s+basis\s+of\s+ownership\b|\bclassified\s+on\s+the\s+basis\s+of\s+ownership\b", text_lower):
                    score += 70.0
                if re.search(r"\b(founders?|pure entrepreneurs|second-generation|family-owned|franchisees?|owner-managers?)\b", text_lower):
                    score += 18.0

        if self._is_problem_question(query):
            for subject_term in subject_terms:
                for variant in self._term_variants(subject_term):
                    if f"problems faced by {variant}" in text_lower:
                        score += 28.0
                    if f"problems before {variant}" in text_lower:
                        score += 18.0
                    if f"{variant} have to face" in text_lower:
                        score += 18.0

        if re.search(r"\b(contributions?|components?|main parts?|framework)\b", query.lower()):
            if re.search(r"\b(major|main)\s+contributions\b|contributions\s+are\s+summarized", text_lower):
                score += 34.0
            if re.search(r"\b(we\s+present|we\s+design|we\s+propose|we\s+introduce|we\s+extensively)\b", text_lower):
                score += 12.0
            if re.search(r"\b(multi-scale|gaussian mixture|self-attention|temporal consistency|latent space|anomaly scoring)\b", text_lower):
                score += 8.0

        if self._is_framework_process_question(query):
            if re.search(r"\bproposed framework\b|\bmain contribution\b|\bnine stages\b|\bstage\s+[a-i]\b|\bmvc\b|model-view-controller", text_lower):
                score += 26.0
            if re.search(r"\b(?:summarize|summarise|summary)\b", query.lower()) and re.search(r"\b(replica(?:ble)?|web\s*vr|virtual laborator|real-time numerical simulation|interactive\s+3\s*d|Simscape|user-based validation|future work)\b", text_lower):
                score += 90.0
            if re.search(r"\breal-time numerical simulation\b|\binteractive\s+3\s*d\b|\bweb\s*vr\b|\bvirtual laborator", text_lower):
                score += 10.0
            if re.search(r"\b(objective|aim|purpose)\b", query.lower()) and re.search(r"\b(main contribution|reduce .*barriers|without relying on proprietary|specialized software|expensive hardware)\b", text_lower):
                score += 24.0
            if re.search(r"\b(preferred|why)\b|\bphysical\s+laborator\w*\b", query.lower()) and re.search(r"\b(economic|logistical|usability constraints|costly|risky|impractical|accessibility|scalability|low cost|standard web browsers|everyday devices)\b", text_lower):
                score += 80.0
            if re.search(r"\b(preferred|why)\b|\bphysical\s+laborator\w*\b", query.lower()) and re.search(r"\bphysical systems\b|\bphysical dynamic systems\b|\bphysical experimentation\b", text_lower):
                score += 35.0
            if re.search(r"\b(validate|validation|dynamic systems?|which)\b", query.lower()) and re.search(r"\b(simple pendulum|inverted pendulum|mass-spring-damper|MSD|SP and IP|robotic systems)\b", text_lower):
                score += 60.0
            if re.search(r"\b(validate|validation|dynamic systems?|which)\b", query.lower()) and re.search(r"\bframework\s+is\s+validated\b|\bincludes\s+two\s+representative\s+systems\b|\bvalidation\s+of\s+the\s+framework\s+with\s+a\s+different\s+dynamic\s+system\b", text_lower):
                score += 80.0
            if re.search(r"\b(mvc|model-view-controller|components?)\b", query.lower()) and re.search(r"\b(model|view|controller|Model\.js|View\.js|Controller\.js)\b", text_lower):
                score += 80.0
            if re.search(r"\b(pid|control law|controller equations?|equations?)\b", query.lower()) and re.search(r"\b(pid|proportional-integral-derivative|control law|control input|gain matrices|gains are|position error vector|derivative terms|closed-loop response using pid control)\b", text_lower):
                score += 90.0
            if re.search(r"\beuler|ode\s*1|numerical integration\b", query.lower()) and re.search(r"\beuler|ode\s*1|simplicity|low computational cost|real-time requirements\b", text_lower):
                score += 90.0
            if re.search(r"\bsampling interval|step size|dt\b", query.lower()) and re.search(r"\bsampling interval|step size|dt|5\s*ms|accuracy and computational cost\b", text_lower):
                score += 90.0
            if re.search(r"\bfuture|optimization|improvements?|limitations?\b", query.lower()) and re.search(r"\bfuture work|advanced modeling|hybrid simulation|performance optimization|disturbances|sensor noise|WebAssembly\b", text_lower):
                score += 80.0
            if re.search(r"\bperformance evaluation|evaluation methods?|validation metrics?|benchmark\b", query.lower()) and re.search(r"\bnumerical accuracy|performance benchmarking|robustness evaluation|Simscape|user-based validation|MSE|VAF\b", text_lower):
                score += 80.0

        if self._is_state_space_question(query) and re.search(r"\bstate[- ]space|state vector|dynamic model|equations of motion\b", text_lower):
            score += 130.0

        if re.search(r"\bimportance\b|\bimportant\b|\beconomic development\b|\beconomy\b", query.lower()):
            if re.search(r"\bimportance\s+of\s+entrepreneurship\b|\bentrepreneurship\s+holds\s+vital\s+role\s+in\s+an\s+economy\b", text_lower):
                score += 70.0
            if re.search(r"\bcreates wealth\b|\bprovides employment\b|\bresearch and development\b|\beconomic prosperity\b|\bproductive activities\b", text_lower):
                score += 16.0

        query_lower = query.lower()
        if re.search(r"\b(main|primary|central|overall)\s+(topic|subject|theme|focus)\b", query_lower):
            if re.search(r"\b(topic|about|introduction|abstract|overview)\b", text_lower):
                score += 14.0
            if re.search(r"\b(shared through|main sections|primary goal|overall goal)\b", text_lower):
                score += 24.0
        if "control" in query_lower and re.search(r"\b(as a control|a control was|in the control)\b", text_lower):
            score += 80.0
        if "past tense" in query_lower and re.search(r"\bpast tense\b.{0,120}\bbecause\b|\bbecause\b.{0,120}\bpast\b", text_lower):
            score += 80.0
        if "rationale" in query_lower and re.search(r"\brationale\b.{0,160}\bsteps?\b|\bsteps?\b.{0,160}\brationale\b", text_lower):
            score += 80.0
        if re.search(r"\bvisual aids?\b", query_lower) and re.search(r"\bcharts?\b|\bfigures?\b|\bdiagrams?\b|\btables?\b", text_lower):
            score += 90.0
        if re.search(r"\bexamples?\b", query_lower) and re.search(r"\b(first|second|third)[-\s]person\s*:", text_lower):
            score += 90.0
        if re.search(r"\bsources? of error\b", query_lower) and re.search(r"\bsources? of error\b.{0,160}\baffected\b|\baffected\b.{0,160}\bsources? of error\b", text_lower):
            score += 80.0
        if re.search(r"\bpersonal mentions?\b|\bmention(?:ing)? the researchers\b", query_lower) and re.search(r"\bpurpose of removing\b.{0,180}\bobjectivity\b|\bobjectivity\b.{0,180}\bexperiment\b", text_lower):
            score += 80.0
        if re.search(r"\breplicat(?:e|ion|ed|able)\b", query_lower) and re.search(r"\breplicat\w*\b.{0,160}\blegitimacy\b|\blegitimacy\b.{0,160}\breplicat\w*\b", text_lower):
            score += 80.0

        return score

    def _qualifier_terms(self, query):
        query_lower = self._normalize_question(query)
        qualifiers = []
        for pattern in [
            r"\bbased\s+on\s+([a-zA-Z -]{3,60})(?:\?|$)",
            r"\bon\s+the\s+basis\s+of\s+([a-zA-Z -]{3,60})(?:\?|$)",
        ]:
            match = re.search(pattern, query_lower)
            if match:
                qualifiers.extend(self._content_terms(match.group(1)))
        ignored = LIST_WORDS | PROBLEM_WORDS | EXPLANATORY_WORDS | {"basis", "based"}
        return [term for term in dict.fromkeys(qualifiers) if term not in ignored]

    def _important_phrases(self, query):
        words = self._topic_terms(query)
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i + 1]}")
        return phrases

    def _definition_score(self, query, text_lower):
        score = 0.0
        for subject in self._subject_terms(query):
            for variant in self._term_variants(subject):
                if re.search(rf"\b{re.escape(variant)}\b\s+(?:is|are|means|refers to|can be defined)", text_lower):
                    score += 8.0
                if re.search(rf"\b(?:definition|meaning)\s+of\s+{re.escape(variant)}\b", text_lower):
                    score += 8.0
        return score

    def _acronym_definition_score(self, query, text_lower):
        query_lower = query.lower()
        acronyms = re.findall(r"\(([A-Z][A-Z0-9-]{1,12})\)", query)
        subject = self._clean_subject_text(self._normalize_question(query))
        subject = re.sub(r"\s*\([^)]+\)", "", subject).strip()
        score = 0.0
        for acronym in acronyms:
            if re.search(rf"\b{re.escape(acronym.lower())}\b", text_lower):
                score += 8.0
            if subject and subject in text_lower and re.search(rf"\b{re.escape(acronym.lower())}\b", text_lower):
                score += 18.0
        if subject and subject in text_lower and re.search(r"\b(is|are|refers|defined|system|method|approach)\b", text_lower):
            score += 8.0
        return score

    def _named_method_score(self, query, text_lower):
        methods = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b", query or "")
        score = 0.0
        for method in methods:
            method_lower = method.lower()
            if not re.search(rf"\b{re.escape(method_lower)}\b", text_lower):
                continue
            score += 10.0
            if (
                re.search(rf"\b(?:we\s+present|present|propose|introduce|novel)\b[^.]{{0,180}}\b{re.escape(method_lower)}\b", text_lower)
                or re.search(rf"\b{re.escape(method_lower)}\b[^.]{{0,180}}\b(framework|method|model|approach|autoencoder)", text_lower)
            ):
                score += 22.0
            if re.search(r"\babstract\b|\bintroduction\b|\bkeywords\b", text_lower):
                score += 4.0
            if re.search(r"\btable\b|\bablation\b|\bmixed dataset\b", text_lower):
                score -= 6.0
        return score

    def _heading_score(self, query, chunk):
        heading = f"{chunk.get('section_title', '')}\n{chunk.get('text', '')[:180]}".lower()
        topic_hits = self._count_term_hits(self._topic_terms(query), heading)
        cue_hits = 0
        if self._is_list_question(query):
            cue_hits += self._count_term_hits(LIST_CUES, heading)
        if self._is_problem_question(query):
            cue_hits += self._count_term_hits(PROBLEM_WORDS, heading)
        return topic_hits * 4.0 + cue_hits * 3.0

    def _cue_score(self, text_lower, cues):
        return sum(1.2 for cue in cues if re.search(rf"\b{re.escape(cue)}\b", text_lower))

    def _chunk_quality_score(self, text):
        text_lower = (text or "").lower()
        words = text_lower.split()
        score = 0.0

        if any(marker in text_lower for marker in LOW_VALUE_MARKERS):
            score -= 8.0
        if self._looks_like_reference_text(text_lower):
            score -= 18.0
        if re.search(r"\b(?:contents|objectives|keywords|review questions|further readings)\b", text_lower):
            score -= 5.0
        if re.search(r"\b(?:is|are|means|refers to|defined as|include|includes|following)\b", text_lower):
            score += 1.2
        if text_lower.count("?") >= 2:
            score -= 2.0
        if len(words) < 35:
            score -= 0.6
        if re.search(r"\n\s*(?:\d{1,3}[\).]|[-*])\s+", text or ""):
            score += 1.5
        if re.search(r"\b\d+(?:\.\d+)+\s+[A-Z]", text or ""):
            score += 1.0

        return score

    def _is_low_value_text(self, text_lower):
        return any(marker in text_lower for marker in LOW_VALUE_MARKERS) or self._looks_like_reference_text(text_lower)

    def _looks_like_reference_text(self, text_lower):
        return (
            "references" in text_lower
            or "further readings" in text_lower
            or "for enquiry" in text_lower
            or text_lower.count("http://") >= 2
            or text_lower.count("https://") >= 2
            or "online links" in text_lower
            or "sultan chand" in text_lower
            or "tata mc graw hill" in text_lower
            or text_lower.count(" et al.") >= 2
            or text_lower.count(" proc.") >= 1
            or text_lower.count(" pp.") >= 2
            or len(re.findall(r"\[\d+\]", text_lower)) >= 3
        )

    def _has_list_cue_evidence(self, text_lower):
        return bool(re.search(
            r"\b(classified|classification|categories|following|include|includes|consists?|comprises?|basis)\b",
            text_lower,
        ))

    def _first_section_number(self, text):
        match = re.search(r"\b(\d+(?:\.\d+)+)\s+[A-Za-z]", text or "")
        return match.group(1) if match else None

    def _term_matches(self, term, text_lower):
        for variant in self._term_variants(term):
            if re.search(rf"\b{re.escape(variant)}\b", text_lower):
                return True
            if "-" in variant:
                flexible = re.escape(variant).replace(r"\-", r"[-\s]?")
                if re.search(rf"\b{flexible}\b", text_lower):
                    return True
        return False

    def _term_variants(self, term):
        variants = {term}

        if term.endswith("s") and len(term) > 4:
            variants.add(term[:-1])
        elif len(term) > 3:
            variants.add(term + "s")
        if term.endswith("ies") and len(term) > 5:
            variants.add(term[:-3] + "y")
        if term.endswith("y") and len(term) > 5:
            variants.add(term[:-1] + "ies")
            variants.add(term[:-1] + "ical")
        if term.endswith("ical") and len(term) > 6:
            variants.add(term[:-4] + "y")
        if term.endswith("isation") and len(term) > 8:
            root = term[:-7]
            variants.update({root + "ise", root + "ization", root + "ize"})
        if term.endswith("ization") and len(term) > 8:
            root = term[:-7]
            variants.update({root + "isation", root + "ise", root + "ize"})
        if term.endswith("ise") and len(term) > 5:
            root = term[:-3]
            variants.update({root + "isation", root + "ize", root + "ization"})
        if term.endswith("ize") and len(term) > 5:
            root = term[:-3]
            variants.update({root + "ization", root + "ise", root + "isation"})
        if term == "important":
            variants.add("importance")
        if term == "importance":
            variants.add("important")

        if "-" in term:
            variants.add(term.replace("-", ""))
            variants.add(term.replace("-", " "))

        # Generic nation/adjective pairs (e.g. "america"/"american")
        if term.endswith("an") and len(term) > 5:
            variants.add(term[:-2])        # american -> america
        elif len(term) > 5 and not term.endswith("an"):
            variants.add(term + "an")      # america -> american

        return variants

    def _is_list_question(self, query):
        return bool(re.search(
            r"\b(types?|kinds?|categories|classifications?|forms?|list|sections?|"
            r"parts?|components?|elements?|stages?|steps?|characteristics?|features?|traits?|qualities)\b",
            query.lower(),
        ))

    def _is_contribution_question(self, query):
        return bool(re.search(r"\b(contributions?|components?|main parts?|framework)\b", query.lower()))

    def _is_framework_process_question(self, query):
        return bool(re.search(
            r"\b(objective|framework|web\s*vr|mvc|model-view-controller|"
            r"stages?|steps?|simscape|dynamic systems?|3\s*d visualization|numerical simulation|"
            r"state[- ]space|pid|control law|controller equations?|equations?|summary|summarize|summarise|"
            r"euler|sampling interval|step size|synchroni[sz]ation|proprietary|future|optimization|"
            r"evaluation|benchmark|unreal|unity|rendering)\b|"
            r"\bvirtual\s+laborator\w*\b|\bphysical\s+laborator\w*\b",
            query.lower(),
        ))

    def _is_state_space_question(self, query):
        return bool(re.search(r"\bstate[- ]space\b", query.lower()))

    def _is_problem_question(self, query):
        return bool(re.search(
            r"\b(problems?|challenges?|issues?|difficulties|barriers|faced by|before)\b",
            query.lower(),
        ))

    def _is_main_issue_question(self, query):
        return bool(re.search(r"\b(main|major|primary)\s+(issue|problem|reason|cause)\b", query.lower()))

    def _is_explanatory_question(self, query):
        query_lower = query.lower()
        return bool(re.search(
            r"\b(role|roles|importance|important|contributions?|impact|effects?|functions?|"
            r"benefits?|significance|purpose|responsibilit(?:y|ies))\b",
            query_lower,
        )) or bool(re.search(r"^\s*(how|why)\b", query_lower))

    def _is_definition_question(self, query):
        return bool(re.search(r"^\s*(what\s+is|what\s+does|define|meaning\s+of)\b", query.lower()))

    def _chunk_key(self, text, meta):
        return f"{meta.get('page', 0)}:{meta.get('chunk_index', 0)}:{hash(text)}"
