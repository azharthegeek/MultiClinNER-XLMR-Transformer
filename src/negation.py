"""Rule-based NegEx post-processing for clinical NER inference.

Detects negated entity mentions by scanning for negation cue words in a token
window before each detected B- entity. Negated spans are relabelled to "O"
so they are not reported as false positive entities.

Apply ONLY at inference time — never during training or test-set evaluation,
which uses gold-standard BIO labels.
"""

NEGATION_CUES = {
    "en": [
        "no evidence of",
        "no history of",
        "negative for",
        "ruled out",
        "rules out",
        "absence of",
        "free of",
        "unremarkable for",
        "without",
        "denies",
        "denied",
        "never",
        "non",
        "not",
        "no",
    ],
    "es": [
        "sin evidencia de",
        "no presenta",
        "sin antecedentes de",
        "no hay",
        "negativo para",
        "descartó",
        "descarta",
        "ausencia de",
        "libre de",
        "no se observa",
        "nunca",
        "niega",
        "negó",
        "sin",
        "no",
    ],
}

NEGATION_WINDOW = 6


def is_negated(entity_start_idx, tokens, lang="en"):
    """Return True if a negation cue appears within NEGATION_WINDOW tokens before entity.

    Multi-word cues (e.g. "no evidence of") are matched as exact phrase sequences.
    Single-word cues (e.g. "no") are matched anywhere in the window.

    Args:
        entity_start_idx: Word index of the B- token in the sentence.
        tokens:           Word-level token list for the full sentence.
        lang:             "en" or "es" — selects negation cue list.
    """
    window_start = max(0, entity_start_idx - NEGATION_WINDOW)
    window = [t.lower() for t in tokens[window_start:entity_start_idx]]
    cues = NEGATION_CUES.get(lang, NEGATION_CUES["en"])
    for cue in cues:
        cue_words = cue.split()
        for i in range(len(window) - len(cue_words) + 1):
            if window[i : i + len(cue_words)] == cue_words:
                return True
    return False


def apply_negex(entities, tokens, lang="en"):
    """Relabel negated entity spans to "O" in the prediction list.

    When a B-X token is preceded by a negation cue within NEGATION_WINDOW
    tokens, that B-X and all immediately following I-X tokens (the full span)
    are changed to "O".

    Args:
        entities: List of {"word": str, "label": str} in word order.
                  Must be the same length as tokens and correspond 1-to-1.
        tokens:   Original word-level token list (for cue window lookup).
        lang:     "en" or "es".

    Returns:
        modified:       Same-length list with negated spans set to "O".
        negated_spans:  List of {"span": str, "type": str} for error analysis.
    """
    modified = [dict(e) for e in entities]
    negated_spans = []
    i = 0
    while i < len(modified):
        label = modified[i]["label"]
        if label.startswith("B-") and is_negated(i, tokens, lang):
            entity_type = label[2:]
            span_words = [modified[i]["word"]]
            modified[i]["label"] = "O"
            j = i + 1
            while j < len(modified) and modified[j]["label"] == f"I-{entity_type}":
                span_words.append(modified[j]["word"])
                modified[j]["label"] = "O"
                j += 1
            negated_spans.append({"span": " ".join(span_words), "type": entity_type})
            i = j
        else:
            i += 1
    return modified, negated_spans
