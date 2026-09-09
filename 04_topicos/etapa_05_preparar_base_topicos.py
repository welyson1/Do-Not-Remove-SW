from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    ENGLISH_COMMENTS_PERIOD_CSV,
    LANGUAGE_FILTER_COLUMNS,
    MIN_TOPIC_COMMENT_LENGTH,
    TOPIC_BASE_COMMENTS_CSV,
    ensure_directories,
    load_language_comment_file,
    normalize_series,
)


def main() -> None:
    ensure_directories()

    comments_period, repaired = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    if repaired:
        comments_period.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    comments_period = comments_period.copy()
    comments_period["comment_normalized"] = normalize_series(comments_period["comment"])
    comments_period["comment_length"] = comments_period["comment_normalized"].str.len()

    prepared = comments_period.loc[comments_period["comment_normalized"] != ""].copy()
    prepared = prepared.loc[prepared["comment_length"] >= MIN_TOPIC_COMMENT_LENGTH].copy()
    prepared = prepared.drop_duplicates(subset=["comment_normalized"]).copy()
    prepared.to_csv(TOPIC_BASE_COMMENTS_CSV, index=False)

    print(f"Comentarios no periodo: {len(comments_period)}")
    print(f"Base preparada para topicos: {len(prepared)}")
    print(f"Arquivo salvo em: {TOPIC_BASE_COMMENTS_CSV}")


if __name__ == "__main__":
    main()
