from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline_00_config import (
    ENGLISH_COMMENTS_PERIOD_CSV,
    PLATFORM_PATTERNS,
    STUDY_END,
    STUDY_START,
    TOPICS_TO_REMOVE,
    TOPIC_ALIGNED_COMMENTS_CSV,
    TOPIC_ANALYSIS_REPORT_TXT,
    TOPIC_BASE_COMMENTS_CSV,
    TOPIC_DESCRIPTIONS_CSV,
    TOPIC_PLATFORM_HEATMAP_PDF,
    TOPIC_TEMPORAL_PDF,
    TOPIC_COVERAGE_REPORT_TXT,
    ensure_directories,
    extract_matching_terms,
    parse_utc_datetime,
    read_csv_with_fallback,
)


PEAK_ZSCORE_THRESHOLD = 3.0


def zscore_by_row(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1), axis=0)
    row_std = frame.std(axis=1).replace(0, 1)
    return centered.div(row_std, axis=0)


def zscore_by_column(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=0), axis=1)
    column_std = frame.std(axis=0).replace(0, 1)
    return centered.div(column_std, axis=1)


def generate_temporal_heatmap(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = analysis_df.groupby(["mes_ano", "topic_id"]).size().reset_index(name="count")
    pivot = counts.pivot_table(index="topic_id", columns="mes_ano", values="count", fill_value=0)
    full_range = pd.period_range(start=STUDY_START, end=STUDY_END, freq="M")
    pivot = pivot.reindex(columns=full_range, fill_value=0).sort_index()
    heatmap_data = zscore_by_row(pivot)

    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, vmin=-2, vmax=2, cbar_kws={"label": "Z-score"})
    plt.xlabel("Linha do tempo")
    plt.ylabel("Topic ID")
    tick_positions = np.arange(0, len(pivot.columns), 6)
    tick_labels = [str(pivot.columns[index]) for index in tick_positions]
    plt.xticks(tick_positions + 0.5, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(TOPIC_TEMPORAL_PDF, bbox_inches="tight")
    plt.close()

    return pivot, heatmap_data


def detect_temporal_peaks(
    pivot: pd.DataFrame,
    zscores: pd.DataFrame,
    threshold: float = PEAK_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for topic_id, row in zscores.iterrows():
        for month, zscore in row.items():
            if zscore > threshold:
                rows.append(
                    {
                        "topic_id": topic_id,
                        "mes_ano": month,
                        "count": int(pivot.loc[topic_id, month]),
                        "zscore": float(zscore),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["topic_id", "mes_ano", "count", "zscore"])

    return (
        pd.DataFrame(rows)
        .sort_values(["zscore", "topic_id", "mes_ano"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def generate_platform_heatmap(analysis_df: pd.DataFrame):
    rows = []
    for row in analysis_df.itertuples(index=False):
        found_terms = extract_matching_terms(row.comment, PLATFORM_PATTERNS)
        for term in found_terms:
            rows.append({"topic_id": row.topic_id, "topic_name": row.topic_name, "platform": term})

    if not rows:
        return None

    platform_mentions = pd.DataFrame(rows)
    top_platforms = platform_mentions["platform"].value_counts().head(10).index
    filtered = platform_mentions.loc[platform_mentions["platform"].isin(top_platforms)].copy()
    matrix = pd.crosstab(filtered["topic_id"], filtered["platform"])
    matrix = zscore_by_column(matrix)

    plt.figure(figsize=(10.5, 4.6))
    sns.heatmap(matrix.T, cmap="RdBu_r", center=0, vmin=-2, vmax=2, cbar_kws={"label": "Z-score"})
    plt.xlabel("Topic ID")
    plt.ylabel("Plataforma")
    plt.tight_layout()
    plt.savefig(TOPIC_PLATFORM_HEATMAP_PDF, bbox_inches="tight")
    plt.close()

    return filtered["platform"].value_counts()


def main() -> None:
    ensure_directories()

    period_comments = read_csv_with_fallback(ENGLISH_COMMENTS_PERIOD_CSV, low_memory=False)
    topic_base = read_csv_with_fallback(TOPIC_BASE_COMMENTS_CSV, low_memory=False)
    aligned = read_csv_with_fallback(TOPIC_ALIGNED_COMMENTS_CSV, low_memory=False)
    descriptions = read_csv_with_fallback(TOPIC_DESCRIPTIONS_CSV, low_memory=False)

    aligned = aligned.copy()
    aligned["published_at"] = parse_utc_datetime(aligned["published_at"])
    aligned["topic_id"] = pd.to_numeric(aligned["topic_id"], errors="coerce")
    aligned = aligned.dropna(subset=["topic_id", "published_at"]).copy()
    aligned["topic_id"] = aligned["topic_id"].astype(int)

    excluded_topics = set(TOPICS_TO_REMOVE + [-1])
    analysis_df = aligned.loc[~aligned["topic_id"].isin(excluded_topics)].copy()
    analysis_df["mes_ano"] = analysis_df["published_at"].dt.to_period("M")

    descriptions = descriptions.copy()
    descriptions["topic_id"] = pd.to_numeric(descriptions["topic_id"], errors="coerce")
    descriptions = descriptions.dropna(subset=["topic_id"]).copy()
    descriptions["topic_id"] = descriptions["topic_id"].astype(int)
    descriptions = descriptions.loc[~descriptions["topic_id"].isin(excluded_topics)].copy()

    pivot, temporal_zscores = generate_temporal_heatmap(analysis_df)
    platform_counts = generate_platform_heatmap(analysis_df)
    temporal_peaks = detect_temporal_peaks(pivot, temporal_zscores)
    topic_names = descriptions.set_index("topic_id")["topic_name"].to_dict()

    lines = [
        "Relatorio de analise de topicos",
        "=" * 72,
        f"Comentarios no corpus do periodo: {len(period_comments)}",
        f"Comentarios na base de topicos: {len(topic_base)}",
        f"Comentarios alinhados com topic_id: {len(aligned)}",
        f"Comentarios analisados apos excluir topicos de ruido: {len(analysis_df)}",
        f"Relatorio de cobertura usado como referencia: {TOPIC_COVERAGE_REPORT_TXT.name}",
        "",
        "Topicos analisados",
        "ID | Nome do topico | Principais termos",
    ]

    for row in descriptions.itertuples(index=False):
        representation = str(row.topic_representation)
        if len(representation) > 80:
            representation = representation[:77] + "..."
        lines.append(f"{row.topic_id} | {row.topic_name} | {representation}")

    lines.extend(
        [
            "",
            "Picos temporais por topico",
            "Criterio operacional: z_{t,m} = (n_{t,m} - media_t) / s_t",
            "n_{t,m}: comentarios do topico t no mes m; media_t e s_t: media e desvio padrao da serie mensal do topico t.",
            f"Limiar de pico temporal: z > {PEAK_ZSCORE_THRESHOLD:g}",
            f"Celulas topico-mes com pico temporal: {len(temporal_peaks)}",
            f"Topicos com ao menos um pico temporal: {temporal_peaks['topic_id'].nunique()}",
            "Topic | Nome do topico | Mes | Comentarios | z-score",
        ]
    )
    for row in temporal_peaks.itertuples(index=False):
        topic_name = topic_names.get(row.topic_id, "desconhecido")
        lines.append(f"{row.topic_id} | {topic_name} | {row.mes_ano} | {row.count} | {row.zscore:.2f}")

    lines.append("")
    lines.append("Plataformas mais citadas nos comentarios")
    if platform_counts is None:
        lines.append("Nenhuma plataforma foi detectada nos comentarios analisados.")
    else:
        for platform, count in platform_counts.head(10).items():
            lines.append(f"{platform}: {count}")

    TOPIC_ANALYSIS_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Comentarios analisados: {len(analysis_df)}")
    print(f"Relatorio salvo em: {TOPIC_ANALYSIS_REPORT_TXT}")


if __name__ == "__main__":
    main()
