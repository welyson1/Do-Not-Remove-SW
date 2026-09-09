from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from pipeline_00_config import (
    LANGUAGE_COLUMNS,
    LANGUAGE_COMPARISON_CSV,
    LANGUAGE_THRESHOLD_PLOT_PNG,
    LANGUAGE_THRESHOLD_REPORT_TXT,
    LANGUAGE_VIDEO_THRESHOLD,
    ensure_directories,
    load_language_comment_file,
)


def calculate_video_stats(comments):
    comments = comments.copy()
    comments["is_majority_en"] = (
        (comments["lang_fasttext"] == "en").astype(int)
        + (comments["lang_langdetect"] == "en").astype(int)
        + (comments["lang_langid"] == "en").astype(int)
    ) >= 2
    video_stats = comments.groupby("video_id").agg(
        total_comments=("comment_id", "count"),
        english_comments=("is_majority_en", "sum"),
    ).reset_index()
    video_stats["percent_en"] = (video_stats["english_comments"] / video_stats["total_comments"].replace(0, 1)) * 100
    return video_stats


def build_sensitivity_table(video_stats):
    total_videos = len(video_stats)
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    rows = []
    for threshold in thresholds:
        kept = int((video_stats["percent_en"] > threshold).sum())
        rows.append(
            {
                "threshold": threshold,
                "videos_kept": kept,
                "videos_removed": total_videos - kept,
                "retention_pct": (kept / total_videos) * 100,
            }
        )
    return rows


def write_report(video_stats, sensitivity, repaired: bool) -> None:
    total_videos = len(video_stats)
    low_zone = int((video_stats["percent_en"] <= 20).sum())
    mixed_zone = int(((video_stats["percent_en"] > 20) & (video_stats["percent_en"] < 80)).sum())
    high_zone = int((video_stats["percent_en"] >= 80).sum())

    lines = [
        "Relatorio de sensibilidade do limiar de idioma",
        "=" * 72,
        f"Total de videos analisados: {total_videos}",
        f"Media de ingles por video: {video_stats['percent_en'].mean():.2f}%",
        f"Mediana de ingles por video: {video_stats['percent_en'].median():.2f}%",
        f"Limiar operacional atual do pipeline: > {LANGUAGE_VIDEO_THRESHOLD:.0f}%",
        f"CSV de comparacao precisou de reparo em leitura: {'sim' if repaired else 'nao'}",
        "",
        "Tabela de sensibilidade",
        "Limiar | Videos mantidos | Videos removidos | Retencao (%)",
    ]
    for row in sensitivity:
        lines.append(
            f"{int(row['threshold']):>6}% | {int(row['videos_kept']):>15} | "
            f"{int(row['videos_removed']):>15} | {row['retention_pct']:>11.2f}"
        )

    lines.extend(
        [
            "",
            "Distribuicao por zonas",
            f"Zona 0-20%: {low_zone} videos ({(low_zone / total_videos) * 100:.2f}%)",
            f"Zona 20-80%: {mixed_zone} videos ({(mixed_zone / total_videos) * 100:.2f}%)",
            f"Zona 80-100%: {high_zone} videos ({(high_zone / total_videos) * 100:.2f}%)",
            "",
            "Leitura metodologica",
            "O limiar de 50% preserva o corpus operacional atual.",
            "O limiar de 70% continua disponivel como corte mais conservador, mas nao e imposto pelo script.",
        ]
    )
    LANGUAGE_THRESHOLD_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cdf(video_stats) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    ax = sns.ecdfplot(data=video_stats, x="percent_en", linewidth=2.5, label="Distribuicao dos videos")
    line = ax.lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    def get_y_for_x(target_x: float) -> float:
        index = int(np.abs(x_data - target_x).argmin())
        return float(y_data[index])

    for threshold, color in zip([50, 70, 90], ["#e67e22", "#e74c3c", "#27ae60"], strict=False):
        y_value = get_y_for_x(threshold)
        kept_pct = 100 - (y_value * 100)
        plt.axvline(x=threshold, color=color, linestyle="--", alpha=0.7)
        plt.hlines(y=y_value, xmin=0, xmax=threshold, color=color, linestyle=":", alpha=0.5)
        plt.text(
            threshold + 1,
            max(y_value - 0.05, 0.02),
            f"Limiar {threshold}%\nRetem: {kept_pct:.1f}%",
            color=color,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": color, "boxstyle": "round,pad=0.3"},
        )
        plt.plot(threshold, y_value, marker="o", color=color, markersize=6)

    plt.title("Sensibilidade do percentual de comentarios em ingles por video")
    plt.xlabel("Percentual de comentarios em ingles (%)")
    plt.ylabel("Proporcao acumulada de videos")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.tight_layout()
    plt.savefig(LANGUAGE_THRESHOLD_PLOT_PNG, dpi=300)
    plt.close()


def main() -> None:
    ensure_directories()
    comments, repaired = load_language_comment_file(LANGUAGE_COMPARISON_CSV, LANGUAGE_COLUMNS)
    video_stats = calculate_video_stats(comments)
    sensitivity = build_sensitivity_table(video_stats)
    write_report(video_stats, sensitivity, repaired)
    plot_cdf(video_stats)

    print(f"Relatorio salvo em: {LANGUAGE_THRESHOLD_REPORT_TXT}")
    print(f"Grafico salvo em: {LANGUAGE_THRESHOLD_PLOT_PNG}")


if __name__ == "__main__":
    main()
