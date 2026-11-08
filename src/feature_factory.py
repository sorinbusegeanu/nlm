"""Versioned fold-local TF-IDF and stylometry feature views."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION = "features-v1"
STYLOMETRY_VERSION = "stylometry-v2"
BASE_STYLOMETRY_GROUPS = ("length", "case", "punctuation", "function_words")
STYLOMETRY_GROUPS = ("L", "V", "C", "P", "W", "F", "R", "X", "M", "U")
DEFAULT_STYLOMETRY_RECIPE = {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": []}
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+")
_BASE_FUNCTION_WORDS = ("the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "for", "with", "that", "this", "is", "are")
_EXTRA_FUNCTION_WORDS = (
    "as", "at", "by", "from", "into", "onto", "upon", "than", "then", "when", "while", "where", "which", "who", "whom", "whose", "why", "how",
    "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "he", "her", "hers", "him", "his", "i", "it", "its", "me", "my", "mine", "our", "ours", "she", "their", "theirs", "them", "they", "us", "we", "you", "your", "yours",
    "because", "although", "however", "therefore", "also", "just", "very", "not", "no", "yes", "so", "such", "each", "every", "any", "some", "all", "both", "either", "neither",
)


def _percentile(values, q):
    return float(np.quantile(values, q)) if values else 0.0


def _mtld(tokens, factor=0.72):
    """Approximate MTLD with deterministic forward and reverse passes."""
    if not tokens:
        return 0.0
    def pass_length(items):
        types = set()
        count = 0
        factors = 0.0
        for token in items:
            count += 1
            types.add(token)
            if count and len(types) / count <= factor:
                factors += 1.0
                count = 0
                types.clear()
        if count:
            ttr = len(types) / count
            if ttr < 1.0:
                factors += (1.0 - ttr) / (1.0 - factor)
        return factors
    factors = (pass_length(tokens) + pass_length(list(reversed(tokens)))) / 2.0
    return float(len(tokens) / factors) if factors else float(len(tokens))


def _stylometry_record(raw):
    text = str(raw)
    words = _WORD_RE.findall(text)
    lower = [word.casefold() for word in words]
    sentences = [sentence for sentence in _SENTENCE_RE.split(text) if sentence.strip()]
    sentence_words = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
    n_words = max(1, len(words))
    n_chars = max(1, len(text))
    base = {}
    base.update(dict(zip(("char_count", "word_count", "sentence_count", "chars_per_word", "words_per_sentence", "mean_word_length"), (
        len(text), len(words), len(sentences), len(text) / n_words,
        len(words) / max(1, len(sentences)), sum(map(len, words)) / n_words))))
    base.update({"uppercase_char_fraction": sum(ch.isupper() for ch in text) / n_chars,
                 "digit_char_fraction": sum(ch.isdigit() for ch in text) / n_chars})
    punct = [",", ";", ":", "—", "–", "-", "'", '"', "(", "?", "!"]
    base.update({f"base_punct_{i:02d}": text.count(ch) / n_chars for i, ch in enumerate(punct)})
    base.update({f"base_fw_{word}": sum(token == word for token in lower) / n_words for word in _BASE_FUNCTION_WORDS})

    word_lengths = [len(word) for word in words]
    para = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    para_words = [len(_WORD_RE.findall(part)) for part in para]
    extra = {}
    for prefix, values in (("word_len", word_lengths), ("sentence_words", sentence_words), ("paragraph_words", para_words)):
        extra[f"L_{prefix}_median"] = float(np.median(values)) if values else 0.0
        extra[f"L_{prefix}_std"] = float(np.std(values)) if values else 0.0
        for q, label in ((0.1, "p10"), (0.9, "p90")):
            extra[f"L_{prefix}_{label}"] = _percentile(values, q)
    extra.update({"L_short_sentence_fraction": sum(n < 5 for n in sentence_words) / max(1, len(sentence_words)),
                  "L_long_sentence_fraction": sum(n > 30 for n in sentence_words) / max(1, len(sentence_words)),
                  "L_paragraph_count": len(para),
                  "L_paragraph_char_mean": float(np.mean([len(p) for p in para])) if para else 0.0,
                  "L_paragraph_char_std": float(np.std([len(p) for p in para])) if para else 0.0})

    counts = {}
    for token in lower:
        counts[token] = counts.get(token, 0) + 1
    freq = list(counts.values())
    extra.update({"V_type_token_ratio": len(counts) / n_words,
                  "V_mtld": _mtld(lower),
                  "V_hapax_fraction": sum(count == 1 for count in freq) / max(1, len(counts)),
                  "V_repeated_token_fraction": sum(count for count in freq if count > 1) / n_words,
                  "V_token_frequency_mean": float(np.mean(freq)) if freq else 0.0,
                  "V_token_frequency_std": float(np.std(freq)) if freq else 0.0,
                  "V_token_frequency_max": max(freq, default=0)})

    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    extra.update({"C_lower_word_fraction": sum(word.islower() for word in alpha_words) / max(1, len(alpha_words)),
                  "C_upper_word_fraction": sum(word.isupper() for word in alpha_words) / max(1, len(alpha_words)),
                  "C_title_word_fraction": sum(word.istitle() for word in alpha_words) / max(1, len(alpha_words)),
                  "C_all_caps_fraction": sum(len(word) > 1 and word.isupper() for word in alpha_words) / max(1, len(alpha_words)),
                  "C_internal_capital_fraction": sum(any(ch.isupper() for ch in word[1:]) for word in alpha_words) / max(1, len(alpha_words)),
                  "C_sentence_initial_cap_fraction": sum(bool((match := re.search(r"\b[A-Za-z]", sentence)) and match.group().isupper()) for sentence in sentences) / max(1, len(sentences)),
                  "C_cap_after_punctuation_fraction": len(re.findall(r"[.!?:]\s+[A-Z]", text)) / max(1, len(sentences))})

    typography_chars = (".", ",", ";", ":", "!", "?", "…", "-", "–", "—", "−", "'", "’", "‘", '"', "“", "”", "«", "»", "(", ")", "[", "]", "{", "}", "/", "\\")
    extra.update({f"P_char_{i:02d}": text.count(ch) / n_chars for i, ch in enumerate(typography_chars)})
    punct_total = sum(ch in typography_chars for ch in text)
    extra.update({"P_punctuation_per_sentence": punct_total / max(1, len(sentences)),
                  "P_repeated_punctuation_runs": len(re.findall(r"[.!?,;:!?…—–-]{2,}", text)) / n_chars,
                  "P_ellipsis_count": text.count("…") + len(re.findall(r"\.{3,}", text)),
                  "P_curly_quote_count": sum(text.count(ch) for ch in ("‘", "’", "“", "”", "«", "»"))})

    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    extra.update({"W_line_count": len(lines), "W_newline_rate": text.count("\n") / n_chars,
                  "W_blank_line_count": sum(not line.strip() for line in lines),
                  "W_tab_count": text.count("\t"),
                  "W_repeated_space_count": len(re.findall(r" {2,}", text)),
                  "W_leading_space_lines": sum(line[:1].isspace() for line in nonempty_lines),
                  "W_trailing_space_lines": sum(line != line.rstrip() for line in nonempty_lines),
                  "W_indented_line_fraction": sum(bool(re.match(r"^(?: {4}|\t)", line)) for line in nonempty_lines) / max(1, len(nonempty_lines))})
    extra.update({"M_bullet_line_fraction": sum(bool(re.match(r"^\s*(?:[-*+]\s+|•\s+)", line)) for line in nonempty_lines) / max(1, len(nonempty_lines)),
                  "M_numbered_line_fraction": sum(bool(re.match(r"^\s*\d+[.)]\s+", line)) for line in nonempty_lines) / max(1, len(nonempty_lines)),
                  "M_markdown_heading_fraction": sum(bool(re.match(r"^\s{0,3}#{1,6}\s", line)) for line in nonempty_lines) / max(1, len(nonempty_lines)),
                  "M_code_fence_count": text.count("```")})
    extra.update({"U_non_ascii_fraction": sum(ord(ch) > 127 for ch in text) / n_chars,
                  "U_nfkc_change_fraction": sum(unicodedata.normalize("NFKC", ch) != ch for ch in text) / n_chars})

    extra.update({f"F_{word}": lower.count(word) / n_words for word in _EXTRA_FUNCTION_WORDS})
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", word.casefold()))) for word in words)
    avg_sentence = len(words) / max(1, len(sentences))
    avg_syllables = syllables / n_words
    extra.update({"R_sentence_word_std": float(np.std(sentence_words)) if sentence_words else 0.0,
                  "R_sentence_word_p90": _percentile(sentence_words, 0.9),
                  "R_comma_per_sentence": text.count(",") / max(1, len(sentences)),
                  "R_clause_connector_fraction": sum(token in {"because", "although", "which", "while", "since", "unless", "whereas"} for token in lower) / n_words,
                  "R_modal_fraction": sum(token in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"} for token in lower) / n_words,
                  "R_passive_proxy_fraction": sum(token.endswith(("ed", "en")) and i > 0 and lower[i-1] in {"am", "is", "are", "was", "were", "be", "been", "being"} for i, token in enumerate(lower)) / n_words,
                  "R_syllables_per_word": avg_syllables,
                  "R_flesch_reading_ease": 206.835 - 1.015 * avg_sentence - 84.6 * avg_syllables,
                  "R_flesch_kincaid_grade": 0.39 * avg_sentence + 11.8 * avg_syllables - 15.59})
    extra.update({"X_punctuation_per_sentence": punct_total / max(1, len(sentences)),
                  "X_long_word_fraction": sum(length > 6 for length in word_lengths) / n_words,
                  "X_paragraphs_per_1000_words": len(para) * 1000 / n_words,
                  "X_comma_to_sentence_ratio": text.count(",") / max(1, len(sentences))})
    return base, extra


def stylometry(texts, recipe=None, *, return_feature_names=False):
    """Extract deterministic row-local CatBoost style features for a recipe.

    Recipe keys are baseline (one or more of length/case/punctuation/function_words),
    exclude (baseline groups to omit), and groups (any of L,V,C,P,W,F,R,X).
    The default exactly preserves the historical baseline feature vector.
    """
    recipe = {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": [], **(recipe or {})}
    baseline_groups = list(recipe.get("baseline", BASE_STYLOMETRY_GROUPS))
    excluded = set(recipe.get("exclude", []))
    baseline_groups = [name for name in baseline_groups if name not in excluded]
    groups = list(recipe.get("groups", []))
    unknown_base = set(baseline_groups) - set(BASE_STYLOMETRY_GROUPS)
    unknown_groups = set(groups) - set(STYLOMETRY_GROUPS)
    unknown_excluded = excluded - set(BASE_STYLOMETRY_GROUPS)
    if unknown_base or unknown_groups or unknown_excluded:
        raise ValueError(f"unknown stylometry recipe entries: baseline={sorted(unknown_base)}, groups={sorted(unknown_groups)}, exclude={sorted(unknown_excluded)}")
    if len(baseline_groups) != len(set(baseline_groups)) or len(groups) != len(set(groups)):
        raise ValueError("stylometry recipe groups must not be repeated")
    rows, names = [], None
    for text in texts:
        base, extra = _stylometry_record(text)
        chosen = []
        for group in baseline_groups:
            if group == "length": chosen.extend(k for k in base if k in {"char_count", "word_count", "sentence_count", "chars_per_word", "words_per_sentence", "mean_word_length"})
            elif group == "case": chosen.extend(k for k in base if k in {"uppercase_char_fraction", "digit_char_fraction"})
            elif group == "punctuation": chosen.extend(k for k in base if k.startswith("base_punct_"))
            elif group == "function_words": chosen.extend(k for k in base if k.startswith("base_fw_"))
        for group in groups:
            chosen.extend(k for k in extra if k.startswith(group + "_"))
        if names is None:
            names = chosen
        rows.append([base[k] if k in base else extra[k] for k in chosen])
    width = len(names or [])
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), width)
    if not np.isfinite(matrix).all():
        raise ValueError("stylometry produced non-finite features")
    return (matrix, list(names or [])) if return_feature_names else matrix


def _normalize(text: str) -> str:
    """A derived view only; canonical source text is never modified."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _shape(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[a-z]", "x", text)
    return re.sub(r"x{5,}", "xxxxx", text)


def _config_key(config):
    payload = json.dumps({"version": FEATURE_VERSION, **config}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class FoldFeatures:
    """All vocabulary/scaling state is fitted using the supplied training texts."""
    def __init__(self, config=None):
        self.config = {"views": ["char", "word"], **(config or {})}
        self.vectorizers = {}
        self.scaler = None
        self.stylometry_feature_names = []

    def fit_transform(self, train_text):
        texts = [str(t) for t in train_text]
        matrices = []
        for view in self.config["views"]:
            if view == "char":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
                source = texts
            elif view == "word":
                vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, strip_accents="unicode", sublinear_tf=True)
                source = texts
            elif view == "punctuation":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True, token_pattern=None)
                source = [re.sub(r"[\w\s]", " ", t) for t in texts]
            elif view == "shape":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
                source = [_shape(t) for t in texts]
            elif view == "stylometry":
                self.scaler = StandardScaler()
                matrices.append(csr_matrix(self.scaler.fit_transform(stylometry(texts))))
                continue
            elif view == "stylometry_raw":
                matrix, names = stylometry(texts, self.config.get("stylometry_recipe"),
                                           return_feature_names=True)
                self.stylometry_feature_names = names
                matrices.append(csr_matrix(matrix))
                continue
            else:
                raise ValueError(f"unknown feature view: {view}")
            matrices.append(vec.fit_transform(source))
            self.vectorizers[view] = vec
        if not matrices:
            raise ValueError("at least one feature view is required")
        return hstack(matrices, format="csr")

    def transform(self, texts):
        texts = [str(t) for t in texts]
        matrices = []
        for view in self.config["views"]:
            if view == "stylometry":
                matrices.append(csr_matrix(self.scaler.transform(stylometry(texts))))
            elif view == "stylometry_raw":
                matrices.append(csr_matrix(stylometry(
                    texts, self.config.get("stylometry_recipe"))))
            else:
                source = [_shape(t) for t in texts] if view == "shape" else [re.sub(r"[\w\s]", " ", t) for t in texts] if view == "punctuation" else texts
                matrices.append(self.vectorizers[view].transform(source))
        return hstack(matrices, format="csr")


def build_features(train_text, other_text=None, *, ngram_char=(3, 5), ngram_word=(1, 2), max_features=None, views=None):
    """Compatibility helper for the original char+word SVC interface."""
    cfg = {"views": views or ["char", "word"]}
    features = FoldFeatures(cfg)
    train = features.fit_transform(train_text)
    if other_text is None:
        return train, features
    return train, features.transform(other_text), features


def load_or_fit(train_text, other_text, *, config=None, cache_dir=None, fold_token=""):
    """Cache only fold-fitted objects/features; cache key includes fit rows and recipe."""
    config = {"views": ["char", "word"], **(config or {})}
    train_text = [str(t) for t in train_text]
    other_text = [str(t) for t in other_text]
    payload = "\0".join(train_text).encode()
    validation_payload = "\0".join(other_text).encode()
    key = hashlib.sha256(payload + b"\1" + validation_payload + _config_key(config).encode() + str(fold_token).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.joblib" if cache_dir else None
    if path and path.exists():
        import joblib
        obj = joblib.load(path)
        return obj["train"], obj["other"], obj["features"], key
    features = FoldFeatures(config)
    train = features.fit_transform(train_text)
    other = features.transform(other_text)
    if path:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"train": train, "other": other, "features": features}, path)
    return train, other, features, key
