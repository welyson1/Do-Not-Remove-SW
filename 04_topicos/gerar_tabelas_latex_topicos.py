from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    ENGLISH_VIDEOS_CSV,
    STAGE_04_RESULTS_DIR,
    TOPICS_TO_REMOVE,
    TOPIC_ALIGNED_COMMENTS_CSV,
    TOPIC_DESCRIPTIONS_CSV,
    ensure_directories,
    read_csv_with_fallback,
)

OUTPUT_TEX = STAGE_04_RESULTS_DIR / "07_tabelas_topicos_metadados.tex"


def latex_escape(text: object) -> str:
    escaped = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def format_int(value: object) -> str:
    number = int(round(float(value)))
    return f"{number:,}".replace(",", r"\,")


def parse_keywords(value: object, max_keywords: int = 6) -> str:
    if pd.isna(value):
        return ""

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = stripped
        else:
            parsed = stripped

    if isinstance(parsed, (list, tuple)):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return ", ".join(cleaned[:max_keywords])

    return str(parsed).strip()


def load_topic_metrics() -> pd.DataFrame:
    comments = read_csv_with_fallback(TOPIC_ALIGNED_COMMENTS_CSV, low_memory=False)
    videos = read_csv_with_fallback(ENGLISH_VIDEOS_CSV, low_memory=False)

    comments = comments.copy()
    comments["topic_id"] = pd.to_numeric(comments["topic_id"], errors="coerce")
    comments["like_count"] = pd.to_numeric(comments["like_count"], errors="coerce").fillna(0)
    comments["is_reply"] = comments["is_reply"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    comments = comments.dropna(subset=["topic_id"]).copy()
    comments["topic_id"] = comments["topic_id"].astype(int)
    comments = comments.loc[~comments["topic_id"].isin(set(TOPICS_TO_REMOVE + [-1]))].copy()

    videos = videos.copy()
    videos["view_count"] = pd.to_numeric(videos["view_count"], errors="coerce").fillna(0)
    joined = comments.merge(
        videos[["video_id", "channel_id", "view_count"]],
        on=["video_id", "channel_id"],
        how="left",
    )

    metrics = (
        joined.groupby("topic_id")
        .agg(
            videos=("video_id", "nunique"),
            channels=("channel_id", "nunique"),
            likes=("like_count", "sum"),
            comments=("is_reply", lambda values: int((~values).sum())),
            replies=("is_reply", lambda values: int(values.sum())),
        )
        .reset_index()
    )

    views = (
        joined[["topic_id", "video_id", "view_count"]]
        .drop_duplicates(subset=["topic_id", "video_id"])
        .groupby("topic_id", as_index=False)["view_count"]
        .sum()
        .rename(columns={"view_count": "views"})
    )

    metrics = metrics.merge(views, on="topic_id", how="left")
    metrics["views"] = metrics["views"].fillna(0)
    return metrics.sort_values("topic_id").reset_index(drop=True)


def load_topic_descriptions() -> pd.DataFrame:
    descriptions = read_csv_with_fallback(TOPIC_DESCRIPTIONS_CSV, low_memory=False)
    descriptions = descriptions.copy()
    descriptions["topic_id"] = pd.to_numeric(descriptions["topic_id"], errors="coerce")
    descriptions = descriptions.dropna(subset=["topic_id"]).copy()
    descriptions["topic_id"] = descriptions["topic_id"].astype(int)
    descriptions = descriptions.loc[~descriptions["topic_id"].isin(set(TOPICS_TO_REMOVE + [-1]))].copy()
    descriptions["keywords"] = descriptions["topic_representation"].apply(parse_keywords)
    return descriptions.sort_values("topic_id").reset_index(drop=True)


def render_topic_overview_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\small",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.08\textwidth}>{\raggedright\arraybackslash}p{0.70\textwidth}>{\raggedleft\arraybackslash}p{0.12\textwidth}}",
        r"\caption{Thematic clusters identified by BERTopic, representative keywords, and the number of videos associated with each topic.}\label{tab:topic-clusters}\\",
        r"\toprule",
        r"ID & Keywords & \# Videos \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Thematic clusters identified by BERTopic, representative keywords, and the number of videos associated with each topic.}\\",
        r"\toprule",
        r"ID & Keywords & \# Videos \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for row in frame.itertuples(index=False):
        lines.append(
            f"{format_int(row.topic_id)} & {latex_escape(row.keywords)} & {format_int(row.videos)} \\\\"
        )

    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def render_metadata_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\footnotesize",
        r"\begin{longtable}{>{\raggedleft\arraybackslash}p{0.07\textwidth}>{\raggedleft\arraybackslash}p{0.19\textwidth}>{\raggedleft\arraybackslash}p{0.14\textwidth}>{\raggedleft\arraybackslash}p{0.13\textwidth}>{\raggedleft\arraybackslash}p{0.13\textwidth}>{\raggedleft\arraybackslash}p{0.13\textwidth}}",
        r"\caption{Metadata and engagement statistics by topic. Views are summed once per unique video inside each topic, comments refer to top-level comments, and replies refer to reply comments.}\label{tab:topic-metadata}\\",
        r"\toprule",
        r"Topic & Views & Likes & Comments & Channels & Replies \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Metadata and engagement statistics by topic.}\\",
        r"\toprule",
        r"Topic & Views & Likes & Comments & Channels & Replies \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for row in frame.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    format_int(row.topic_id),
                    format_int(row.views),
                    format_int(row.likes),
                    format_int(row.comments),
                    format_int(row.channels),
                    format_int(row.replies),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def build_latex_document(topic_table: str, metadata_table: str) -> str:
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{array}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\setlength{\LTpre}{0pt}",
            r"\setlength{\LTpost}{12pt}",
            "",
            r"\begin{document}",
            "%",
            "% Generated from 04_topicos/02_comentarios_com_topicos.csv,",
            "% 04_topicos/03_descricoes_topicos.csv, and 03_idioma/06_videos_corpus_ingles.csv.",
            "%",
            topic_table,
            metadata_table,
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    ensure_directories()

    descriptions = load_topic_descriptions()
    metrics = load_topic_metrics()
    merged = descriptions.merge(metrics, on="topic_id", how="inner")

    topic_table = render_topic_overview_table(merged)
    metadata_table = render_metadata_table(merged)
    latex_document = build_latex_document(topic_table, metadata_table)

    OUTPUT_TEX.write_text(latex_document, encoding="utf-8")

    print(f"Topicos tabulados: {len(merged)}")
    print(f"Arquivo salvo em: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
