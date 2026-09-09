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
    ENGLISH_FILTER_REPORT_TXT,
    ENGLISH_VIDEOS_CSV,
    FILTERED_THEME_VIDEOS_CSV,
    LANGUAGE_COLUMNS,
    LANGUAGE_COMPARISON_CSV,
    LANGUAGE_VIDEO_THRESHOLD,
    ensure_directories,
    load_language_comment_file,
    parse_utc_datetime,
    read_csv_with_fallback,
    study_period_mask,
)


def main() -> None:
    ensure_directories()

    comments, repaired = load_language_comment_file(LANGUAGE_COMPARISON_CSV, LANGUAGE_COLUMNS)
    videos = read_csv_with_fallback(FILTERED_THEME_VIDEOS_CSV, low_memory=False, on_bad_lines="skip")

    comments = comments.copy()
    comments["en_agreement_count"] = (
        (comments["lang_fasttext"] == "en").astype(int)
        + (comments["lang_langdetect"] == "en").astype(int)
        + (comments["lang_langid"] == "en").astype(int)
    )
    comments["is_majority_en"] = comments["en_agreement_count"] >= 2

    if repaired:
        comments.to_csv(LANGUAGE_COMPARISON_CSV, index=False)

    video_stats = comments.groupby("video_id").agg(
        total_comments=("comment_id", "count"),
        english_comments=("is_majority_en", "sum"),
    ).reset_index()
    video_stats["percent_en"] = (video_stats["english_comments"] / video_stats["total_comments"].replace(0, 1)) * 100

    selected_video_ids = set(video_stats.loc[video_stats["percent_en"] > LANGUAGE_VIDEO_THRESHOLD, "video_id"])
    videos_final = videos.loc[videos["video_id"].isin(selected_video_ids)].copy()
    comments_final = comments.loc[comments["video_id"].isin(selected_video_ids) & comments["is_majority_en"]].copy()

    comments_final["published_at"] = parse_utc_datetime(comments_final["published_at"])
    period_mask = study_period_mask(comments_final["published_at"])
    comments_period = comments_final.loc[period_mask].copy()

    videos_final.to_csv(ENGLISH_VIDEOS_CSV, index=False)
    comments_final.to_csv(ENGLISH_COMMENTS_CSV, index=False)
    comments_period.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    before_start = int((comments_final["published_at"] < pd.Timestamp("2018-01-01")).sum())
    after_end = int((comments_final["published_at"] >= pd.Timestamp("2025-10-01")).sum())
    day_30_recovered = int(
        ((comments_final["published_at"] >= pd.Timestamp("2025-09-30")) & (comments_final["published_at"] < pd.Timestamp("2025-10-01"))).sum()
    )

    report_lines = [
        "Relatorio de filtragem do corpus em ingles",
        "=" * 72,
        f"Comentarios de entrada na comparacao: {len(comments)}",
        f"Videos de entrada apos filtragem tematica: {len(videos)}",
        f"Comentarios com maioria em ingles (2 de 3): {int(comments['is_majority_en'].sum())}",
        f"Videos analisados com comentarios: {len(video_stats)}",
        f"Limiar operacional aplicado por video: > {LANGUAGE_VIDEO_THRESHOLD:.0f}%",
        f"Videos finais no corpus em ingles: {len(videos_final)}",
        f"Comentarios finais no corpus em ingles: {len(comments_final)}",
        f"Comentarios no periodo analitico [2018-01-01, 2025-10-01): {len(comments_period)}",
        f"Comentarios antes de 2018-01-01: {before_start}",
        f"Comentarios em 2025-10 ou depois: {after_end}",
        f"Comentarios do dia 2025-09-30 preservados com a nova regra: {day_30_recovered}",
        f"CSV de comparacao precisou de reparo em leitura: {'sim' if repaired else 'nao'}",
        "",
        f"Arquivo de videos: {ENGLISH_VIDEOS_CSV.name}",
        f"Arquivo de comentarios: {ENGLISH_COMMENTS_CSV.name}",
        f"Arquivo de comentarios no periodo: {ENGLISH_COMMENTS_PERIOD_CSV.name}",
    ]
    ENGLISH_FILTER_REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Videos finais: {len(videos_final)}")
    print(f"Comentarios finais: {len(comments_final)}")
    print(f"Comentarios no periodo: {len(comments_period)}")
    print(f"Relatorio salvo em: {ENGLISH_FILTER_REPORT_TXT}")


if __name__ == "__main__":
    main()
