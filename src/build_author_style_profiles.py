import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.turkish_model import (
    RAW_DIR,
    REPORT_DIR,
    clean_text_preserve_case,
    make_sentence_chunks,
    filter_authors_with_enough_books,
    MIN_WORDS,
    MAX_WORDS,
)
from src.turkish_features import extract_turkish_features


PROFILE_FEATURES = [
    "avg_word_length",
    "avg_sentence_length",
    "type_token_ratio",
    "stopword_ratio",
    "word_count",
    "verb_suffix_ratio",
    "comma_rate",
    "semicolon_rate",
    "exclamation_rate",
    "question_rate",
]


def load_profile_records():
    dataset = []

    for author_dir in sorted(RAW_DIR.iterdir()):
        if not author_dir.is_dir():
            continue

        author = author_dir.name

        for path in sorted(author_dir.glob("*.txt")):
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            clean_text = clean_text_preserve_case(raw_text)
            chunks = make_sentence_chunks(
                clean_text,
                min_words=MIN_WORDS,
                max_words=MAX_WORDS,
            )

            for chunk in chunks:
                dataset.append((chunk, author, path.name))

    dataset, valid_authors, excluded_authors, _ = filter_authors_with_enough_books(dataset)
    return dataset, valid_authors, excluded_authors


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    dataset, valid_authors, excluded_authors = load_profile_records()

    by_author = defaultdict(list)
    for text, author, _ in dataset:
        by_author[author].append(extract_turkish_features(text))

    profiles = {}

    for author, rows in sorted(by_author.items()):
        profile = {
            "sample_count": len(rows),
            "features": {},
        }

        for feature in PROFILE_FEATURES:
            values = [row.get(feature, 0) for row in rows]
            profile["features"][feature] = summarize(values)

        profiles[author] = profile

    output = {
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "valid_authors": valid_authors,
        "excluded_authors": excluded_authors,
        "profiles": profiles,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "turkish_author_style_profiles.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Yazar stil profilleri kaydedildi: {output_path}")
    for author, profile in profiles.items():
        print(f"{author}: {profile['sample_count']} örnek")


if __name__ == "__main__":
    main()