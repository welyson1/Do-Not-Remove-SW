from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    CLASS_AUDIT_TEMPLATE_CSV,
    ENGLISH_COMMENTS_PERIOD_CSV,
    LANGUAGE_FILTER_COLUMNS,
    ensure_directories,
    load_language_comment_file,
)
from etapa_10_testar_sanidade_classes import (
    CLASS_COLUMNS,
    CLASS_DEFINITIONS,
    classify_comments,
)

DEFAULT_AUDIT_SAMPLE_SIZE = 50
DEFAULT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara uma amostra de auditoria manual para estimar precisao dos dicionarios.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_AUDIT_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def backup_existing_audit() -> Path | None:
    if not CLASS_AUDIT_TEMPLATE_CSV.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CLASS_AUDIT_TEMPLATE_CSV.with_name(
        f"{CLASS_AUDIT_TEMPLATE_CSV.stem}_baseline_{timestamp}{CLASS_AUDIT_TEMPLATE_CSV.suffix}"
    )
    shutil.copy2(CLASS_AUDIT_TEMPLATE_CSV, backup_path)
    return backup_path


def sample_predicted_positives(
    frame: pd.DataFrame,
    class_code: str,
    sample_size: int,
    random_state: int,
    used_comment_texts: set[str],
) -> pd.DataFrame:
    subset = frame.loc[frame[class_code]].copy()
    subset = subset.loc[~subset["comment_normalized"].isin(used_comment_texts)].copy()
    if subset.empty:
        return subset

    n = min(sample_size, len(subset))
    if len(subset) > n:
        subset = subset.sample(n=n, random_state=random_state).copy()

    subset["predicted_class"] = class_code
    subset["predicted_class_label"] = CLASS_DEFINITIONS[class_code]["label"]
    subset["predicted_class_matches"] = subset[f"{class_code}_matches"].apply(lambda value: " | ".join(value))
    subset["all_predicted_labels"] = subset["label_combo"]
    subset["sample_rank_within_class"] = range(1, len(subset) + 1)
    subset["manual_is_true"] = ""
    subset["manual_primary_label"] = ""
    subset["manual_should_be_other_class"] = ""
    subset["manual_notes"] = ""
    subset["auditor"] = ""
    subset["audit_date"] = ""
    return subset


def build_audit_template(frame: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    frames = []
    used_comment_texts: set[str] = set()
    for index, class_code in enumerate(CLASS_COLUMNS):
        sampled = sample_predicted_positives(
            frame,
            class_code=class_code,
            sample_size=sample_size,
            random_state=random_seed + index,
            used_comment_texts=used_comment_texts,
        )
        if sampled.empty:
            continue
        used_comment_texts.update(sampled["comment_normalized"].tolist())
        frames.append(
            sampled[
                [
                    "predicted_class",
                    "predicted_class_label",
                    "sample_rank_within_class",
                    "comment_id",
                    "video_id",
                    "published_at",
                    "is_reply",
                    "like_count",
                    "all_predicted_labels",
                    "predicted_class_matches",
                    "comment",
                    "manual_is_true",
                    "manual_primary_label",
                    "manual_should_be_other_class",
                    "manual_notes",
                    "auditor",
                    "audit_date",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    output.insert(0, "audit_batch", f"rq3_expanded_dedup_seed_{random_seed}_n_{sample_size}")
    return output


def main() -> None:
    args = parse_args()
    ensure_directories()

    comments, repaired = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    if repaired:
        comments.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    classified, dedup_stats = classify_comments(comments)
    audit_template = build_audit_template(
        classified,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    backup_path = backup_existing_audit()
    audit_template.to_csv(CLASS_AUDIT_TEMPLATE_CSV, index=False, encoding="utf-8")

    if backup_path is not None:
        print(f"Backup da auditoria anterior salvo em: {backup_path}")
    print(f"Comentarios antes da deduplicacao: {dedup_stats['total_before_dedup']}")
    print(f"Comentarios duplicados removidos: {dedup_stats['duplicates_removed']}")
    print(f"Comentarios apos deduplicacao: {dedup_stats['total_after_dedup']}")
    print(f"Template de auditoria salvo em: {CLASS_AUDIT_TEMPLATE_CSV}")
    print(f"Linhas geradas: {len(audit_template)}")


if __name__ == "__main__":
    main()
