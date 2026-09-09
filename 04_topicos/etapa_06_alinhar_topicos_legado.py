from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from pipeline_00_config import (
    ENGLISH_COMMENTS_CSV,
    ENGLISH_COMMENTS_PERIOD_CSV,
    LANGUAGE_FILTER_COLUMNS,
    LEGACY_TOPIC_COMMENTS_CSV,
    LEGACY_TOPIC_DESCRIPTIONS_CSV,
    TOPIC_ALIGNED_COMMENTS_CSV,
    TOPIC_BASE_COMMENTS_CSV,
    TOPIC_COVERAGE_REPORT_TXT,
    TOPIC_DESCRIPTIONS_CSV,
    ensure_directories,
    load_language_comment_file,
    normalize_series,
    read_csv_with_fallback,
)


def main() -> None:
    ensure_directories()

    comments_full, repaired_full = load_language_comment_file(ENGLISH_COMMENTS_CSV, LANGUAGE_FILTER_COLUMNS)
    comments_period, repaired_period = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    if repaired_full:
        comments_full.to_csv(ENGLISH_COMMENTS_CSV, index=False)
    if repaired_period:
        comments_period.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    base = read_csv_with_fallback(TOPIC_BASE_COMMENTS_CSV, low_memory=False)
    legacy_comments = read_csv_with_fallback(LEGACY_TOPIC_COMMENTS_CSV, sep=";", engine="python")
    legacy_descriptions = read_csv_with_fallback(LEGACY_TOPIC_DESCRIPTIONS_CSV, sep=";", engine="python")

    comments_period = comments_period.copy()
    comments_period["comment_normalized"] = normalize_series(comments_period["comment"])
    comments_period["comment_length"] = comments_period["comment_normalized"].str.len()
    blank_removed = int((comments_period["comment_normalized"] == "").sum())
    short_removed = int(((comments_period["comment_normalized"] != "") & (comments_period["comment_length"] < 15)).sum())
    duplicate_removed = int(
        comments_period.loc[comments_period["comment_length"] >= 15, "comment_normalized"].duplicated().sum()
    )

    legacy_comments = legacy_comments.copy()
    legacy_comments["comment_normalized"] = normalize_series(legacy_comments["comment"])
    legacy_comments["topic_id"] = pd.to_numeric(legacy_comments["topic_id"], errors="coerce")
    conflict_count = int((legacy_comments.groupby("comment_normalized")["topic_id"].nunique() > 1).sum())
    legacy_comments = legacy_comments.sort_values(["comment_normalized", "comment_id"]).drop_duplicates(
        subset=["comment_normalized"], keep="first"
    )

    aligned = base.merge(
        legacy_comments[["comment_normalized", "topic_id", "Name", "Representation"]],
        on="comment_normalized",
        how="left",
    )
    aligned = aligned.rename(
        columns={
            "Name": "topic_name",
            "Representation": "topic_representation",
        }
    )
    aligned.to_csv(TOPIC_ALIGNED_COMMENTS_CSV, index=False)

    descriptions = legacy_descriptions.rename(
        columns={
            "Topic": "topic_id",
            "Count": "topic_count",
            "Name": "topic_name",
            "Representation": "topic_representation",
            "Representative_Docs": "representative_docs",
        }
    )
    descriptions["topic_id"] = pd.to_numeric(descriptions["topic_id"], errors="coerce")
    descriptions = descriptions.dropna(subset=["topic_id"]).drop_duplicates(subset=["topic_id"]).sort_values("topic_id")
    descriptions.to_csv(TOPIC_DESCRIPTIONS_CSV, index=False)

    matched = int(aligned["topic_id"].notna().sum())
    unmatched = int(aligned["topic_id"].isna().sum())

    lines = [
        "Relatorio de cobertura da etapa de topicos",
        "=" * 72,
        f"Comentarios no corpus final em ingles: {len(comments_full)}",
        f"Comentarios no corpus do periodo analitico: {len(comments_period)}",
        f"Comentarios vazios removidos: {blank_removed}",
        f"Comentarios curtos removidos (<15 caracteres): {short_removed}",
        f"Comentarios duplicados por texto removidos: {duplicate_removed}",
        f"Comentarios na base de topicos: {len(base)}",
        f"Comentarios unicos no arquivo legado de topicos: {legacy_comments['comment_normalized'].nunique()}",
        f"Textos com conflito de topic_id no legado: {conflict_count}",
        f"Comentarios alinhados com topic_id: {matched}",
        f"Comentarios sem correspondencia no legado: {unmatched}",
        "",
        "Observacao: a base de topicos usa texto normalizado, comentario minimo de 15 caracteres e deduplicacao por texto.",
    ]
    TOPIC_COVERAGE_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Base de topicos: {len(base)}")
    print(f"Alinhados com topicos: {matched}")
    print(f"Sem correspondencia: {unmatched}")
    print(f"Relatorio salvo em: {TOPIC_COVERAGE_REPORT_TXT}")


if __name__ == "__main__":
    main()
