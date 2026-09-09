from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline_00_config import (
    CHARACTERIZATION_METRICS_PDF,
    CHARACTERIZATION_PLATFORM_HEATMAP_PDF,
    CHARACTERIZATION_REPORT_TXT,
    ENGLISH_COMMENTS_CSV,
    ENGLISH_COMMENTS_PERIOD_CSV,
    ENGLISH_VIDEOS_CSV,
    LANGUAGE_FILTER_COLUMNS,
    PLATFORM_PATTERNS,
    STUDY_END,
    STUDY_START,
    ensure_directories,
    load_language_comment_file,
    parse_utc_datetime,
    read_csv_with_fallback,
)


def zscore_by_column(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=0), axis=1)
    column_std = frame.std(axis=0).replace(0, 1)
    return centered.div(column_std, axis=1)


def style_axes(axis) -> None:
    axis.grid(axis="y", linestyle="--", alpha=0.6)
    axis.grid(axis="x", alpha=0)


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")

    comments_full, repaired_full = load_language_comment_file(ENGLISH_COMMENTS_CSV, LANGUAGE_FILTER_COLUMNS)
    comments_period, repaired_period = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    if repaired_full:
        comments_full.to_csv(ENGLISH_COMMENTS_CSV, index=False)
    if repaired_period:
        comments_period.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    videos = read_csv_with_fallback(ENGLISH_VIDEOS_CSV, low_memory=False)

    videos["published_at"] = parse_utc_datetime(videos["published_at"])
    comments_period["published_at"] = parse_utc_datetime(comments_period["published_at"])

    videos = videos.dropna(subset=["published_at"]).copy()
    comments_period = comments_period.dropna(subset=["published_at"]).copy()

    videos["view_count"] = pd.to_numeric(videos["view_count"], errors="coerce").fillna(0)
    comments_period["like_count"] = pd.to_numeric(comments_period["like_count"], errors="coerce").fillna(0)

    videos["month_year"] = videos["published_at"].dt.to_period("M")
    comments_period["month_year"] = comments_period["published_at"].dt.to_period("M")

    full_month_range = pd.period_range(start=STUDY_START, end=STUDY_END, freq="M")

    video_monthly = videos.groupby("month_year").agg(
        videos=("video_id", "nunique"),
        canais=("channel_id", "nunique"),
        visualizacoes=("view_count", "sum"),
    ).reindex(full_month_range, fill_value=0)

    comment_monthly = comments_period.groupby("month_year").agg(
        comentarios=("comment_id", "count"),
        likes=("like_count", "sum"),
        usuarios_unicos=("author_channel_id", "nunique"),
    ).reindex(full_month_range, fill_value=0)

    monthly = pd.concat([video_monthly, comment_monthly], axis=1).fillna(0)
    monthly["timestamp"] = monthly.index.to_timestamp()

    metrics = [
        ("videos", "Vídeos"),
        ("canais", "Canais"),
        ("comentarios", "Comentários"),
        ("visualizacoes", "Visualizações"),
        ("likes", "Curtidas"),
        ("usuarios_unicos", "Usuários únicos"),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5.8), sharex=True)
    axes = axes.flatten()
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    year_locator = mdates.YearLocator(2)
    year_formatter = mdates.DateFormatter("%Y")
    x_min = monthly["timestamp"].min()
    x_max = monthly["timestamp"].max()

    for index, (metric, title) in enumerate(metrics):
        axis = axes[index]
        max_value = float(monthly[metric].max())
        exponent = 0 if max_value < 1000 else int(math.floor(math.log10(max_value) / 3) * 3)
        scale = 1 if exponent == 0 else 10**exponent
        axis.plot(monthly["timestamp"], monthly[metric] / scale, color="black", linewidth=1.0)
        ylabel = title if exponent == 0 else f"{title} (x10^{exponent})"
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Ano" if index >= 3 else "")
        axis.text(-0.18, 1.05, letters[index], transform=axis.transAxes, fontsize=9, fontweight="bold")
        axis.set_xlim(x_min, x_max)
        axis.xaxis.set_major_locator(year_locator)
        axis.xaxis.set_major_formatter(year_formatter)
        axis.tick_params(axis="x", labelsize=8, rotation=0, pad=2)
        if index < 3:
            axis.tick_params(axis="x", labelbottom=False)
        style_axes(axis)

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(CHARACTERIZATION_METRICS_PDF, bbox_inches="tight")
    plt.close(fig)

    videos["texto_busca"] = videos["title"].fillna("") + " " + videos["description"].fillna("")
    platform_counts = {}
    for term, pattern in PLATFORM_PATTERNS.items():
        count = int(videos["texto_busca"].str.contains(pattern, na=False).sum())
        if count > 0:
            platform_counts[term] = count

    top_platforms = sorted(platform_counts, key=platform_counts.get, reverse=True)[:30]
    quarter_range = pd.period_range(start=STUDY_START, end=STUDY_END, freq="Q")
    quarter_matrix = pd.DataFrame(0, index=quarter_range, columns=top_platforms)
    videos["quarter"] = videos["published_at"].dt.to_period("Q")

    for term in top_platforms:
        pattern = PLATFORM_PATTERNS[term]
        mentions = videos.loc[videos["texto_busca"].str.contains(pattern, na=False)].groupby("quarter").size()
        for quarter, count in mentions.items():
            if quarter in quarter_matrix.index:
                quarter_matrix.loc[quarter, term] = int(count)

    quarter_heatmap = zscore_by_column(quarter_matrix)
    xtick_labels = [str(period.year) if period.quarter == 1 else "" for period in quarter_heatmap.index]

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        quarter_heatmap.T,
        cmap="vlag",
        center=0,
        vmin=-2,
        vmax=2,
        cbar_kws={"label": "Z-score"},
        xticklabels=xtick_labels,
        yticklabels=True,
    )
    plt.xlabel("Ano")
    plt.ylabel("Plataformas")
    plt.tight_layout()
    plt.savefig(CHARACTERIZATION_PLATFORM_HEATMAP_PDF, bbox_inches="tight")
    plt.close()

    comments_outside_period = len(comments_full) - len(comments_period)
    report_lines = [
        "Relatorio de caracterizacao do corpus",
        "=" * 72,
        f"Periodo de estudo: {STUDY_START.date()} a {STUDY_END.date()}",
        f"Videos no corpus final em ingles: {videos['video_id'].nunique()}",
        f"Comentarios no corpus final em ingles: {len(comments_full)}",
        f"Comentarios no corpus analitico do periodo: {len(comments_period)}",
        f"Comentarios fora do periodo analitico: {comments_outside_period}",
        f"Canais no corpus: {videos['channel_id'].nunique()}",
        f"Visualizacoes somadas dos videos: {int(videos['view_count'].sum())}",
        f"Curtidas somadas dos comentarios no periodo: {int(comments_period['like_count'].sum())}",
        f"Usuarios unicos nos comentarios do periodo: {comments_period['author_channel_id'].nunique()}",
        "",
        "Top 30 plataformas em titulo e descricao dos videos",
    ]
    for position, platform in enumerate(top_platforms, start=1):
        report_lines.append(f"{position:>2}. {platform}: {platform_counts[platform]}")

    CHARACTERIZATION_REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Figura de metricas salva em: {CHARACTERIZATION_METRICS_PDF}")
    print(f"Relatorio salvo em: {CHARACTERIZATION_REPORT_TXT}")


if __name__ == "__main__":
    main()
