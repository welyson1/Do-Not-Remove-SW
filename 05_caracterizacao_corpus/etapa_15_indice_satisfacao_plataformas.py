"""etapa_15_indice_satisfacao_plataformas.py

Script standalone que usa os caminhos de dados da pasta 'files/'
(onde os dados reais estão), independente do pipeline_00_config.py do github_repo.

Execução:
    python etapa_15_indice_satisfacao_plataformas.py

Outputs em: d:\mestrado-temporario\DadosWelysonColetaNova\files\05_caracterizacao_corpus\
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pasta raiz dos dados (onde estão as pastas 01_, 02_, 03_, ...)
DATA_ROOT = Path(r"D:\mestrado-temporario\DadosWelysonColetaNova\files")

# Adicionar o pipeline_00_config.py do files/ ao path
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

# Adicionar etapa_10 do github_repo
GITHUB_REPO_STAGE05 = DATA_ROOT / "github_repo" / "05_caracterizacao_corpus"
if str(GITHUB_REPO_STAGE05) not in sys.path:
    sys.path.insert(0, str(GITHUB_REPO_STAGE05))

GITHUB_REPO = DATA_ROOT / "github_repo"
if str(GITHUB_REPO) not in sys.path:
    sys.path.insert(0, str(GITHUB_REPO))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Caminhos diretos (sem depender de pipeline_00_config.py) ──────────────────

INPUT_COMMENTS_CSV = DATA_ROOT / "03_idioma" / "08_comentarios_corpus_ingles_periodo.csv"
OUTPUT_DIR = DATA_ROOT / "05_caracterizacao_corpus"
OUTPUT_CSV = OUTPUT_DIR / "15_indice_satisfacao_plataformas.csv"
OUTPUT_HEATMAP_PNG = OUTPUT_DIR / "15_figura_satisfacao_plataformas.png"
OUTPUT_HEATMAP_PDF = OUTPUT_DIR / "15_figura_satisfacao_plataformas.pdf"
OUTPUT_REPORT_TXT = OUTPUT_DIR / "15_relatorio_satisfacao_plataformas.txt"

STUDY_START = pd.Timestamp("2018-01-01")
STUDY_END = pd.Timestamp("2025-09-30")

# ── Parâmetros ─────────────────────────────────────────────────────────────────

MIN_EVALUATIVE_COMMENTS = 10  # mínimo de C3+C4 por célula plataforma×trimestre
TOP_N_PLATFORMS = 15

# ── Importações do pipeline ───────────────────────────────────────────────────

try:
    # Tenta usar o pipeline_00_config.py local (files/)
    from pipeline_00_config import (
        PLATFORM_PATTERNS,
        extract_matching_terms,
        normalize_series,
    )
    print("pipeline_00_config.py carregado de files/")
except ImportError:
    # Fallback: importa do github_repo
    from pipeline_00_config import (
        PLATFORM_PATTERNS,
        extract_matching_terms,
        normalize_series,
    )
    print("pipeline_00_config.py carregado de github_repo/")

# Classificador léxico (C1-C6) de etapa_10
from etapa_10_testar_sanidade_classes import classify_comments


# ── Funções auxiliares ────────────────────────────────────────────────────────


def load_comments() -> pd.DataFrame:
    """Carrega o CSV de comentários em inglês do período de estudo."""
    print(f"Lendo: {INPUT_COMMENTS_CSV}")
    # Tenta leitura simples primeiro
    try:
        df = pd.read_csv(INPUT_COMMENTS_CSV, low_memory=False)
        print(f"  {len(df)} linhas carregadas.")
        return df
    except Exception as exc:
        print(f"  Leitura padrão falhou ({exc}), tentando com engine=python...")
        df = pd.read_csv(INPUT_COMMENTS_CSV, low_memory=False, engine="python", on_bad_lines="skip")
        print(f"  {len(df)} linhas carregadas (com engine=python).")
        return df


def parse_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def tag_platforms(comments: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'platform_tags' com lista de plataformas no texto."""
    comments = comments.copy()
    comments["comment_norm"] = normalize_series(comments["comment"])
    print("  Detectando menções de plataforma em cada comentário...")
    comments["platform_tags"] = comments["comment_norm"].apply(
        lambda text: extract_matching_terms(text, PLATFORM_PATTERNS)
    )
    return comments


def explode_platform_tags(comments: pd.DataFrame) -> pd.DataFrame:
    """Transforma cada (comentário, lista_de_plataformas) em (comentário, plataforma)."""
    exploded = comments.explode("platform_tags").dropna(subset=["platform_tags"]).copy()
    exploded = exploded[exploded["platform_tags"].str.strip() != ""].copy()
    return exploded


def compute_satisfaction_index(
    exploded: pd.DataFrame,
    min_evaluative: int = MIN_EVALUATIVE_COMMENTS,
) -> pd.DataFrame:
    """Calcula ratio C3/(C3+C4) por plataforma × trimestre."""
    grouped = (
        exploded.groupby(["quarter", "platform_tags"])
        .agg(
            n_c3=("C3", "sum"),
            n_c4=("C4", "sum"),
            n_total=("comment_id", "count"),
        )
        .reset_index()
    )
    grouped["n_evaluative"] = grouped["n_c3"] + grouped["n_c4"]
    # Descarta células com poucos comentários avaliativos
    grouped = grouped[grouped["n_evaluative"] >= min_evaluative].copy()
    grouped["satisfaction_ratio"] = grouped["n_c3"] / grouped["n_evaluative"]
    return grouped.sort_values(["platform_tags", "quarter"])


def select_top_platforms(satisfaction: pd.DataFrame, top_n: int = TOP_N_PLATFORMS) -> list[str]:
    """Seleciona as top-N plataformas por volume total de comentários avaliativos."""
    totals = (
        satisfaction.groupby("platform_tags")["n_evaluative"]
        .sum()
        .sort_values(ascending=False)
    )
    return totals.head(top_n).index.tolist()


def build_pivot(
    satisfaction: pd.DataFrame,
    top_platforms: list[str],
    full_quarter_range: pd.PeriodIndex,
) -> pd.DataFrame:
    """Tabela pivot: plataformas × trimestres com ratio de satisfação."""
    filtered = satisfaction[satisfaction["platform_tags"].isin(top_platforms)].copy()
    pivot = filtered.pivot_table(
        index="platform_tags", columns="quarter", values="satisfaction_ratio"
    )
    pivot = pivot.reindex(columns=full_quarter_range)
    # Ordena por satisfação média descendente
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    return pivot


def generate_heatmap(pivot: pd.DataFrame) -> None:
    """Gera e salva heatmap de satisfação."""
    xtick_labels = [str(p.year) if p.quarter == 1 else "" for p in pivot.columns]
    mask = pivot.isna()

    plt.figure(figsize=(16, max(6, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        mask=mask,
        linewidths=0.3,
        linecolor="lightgray",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Índice de satisfação C3/(C3+C4)"},
        xticklabels=xtick_labels,
        yticklabels=True,
    )
    plt.xlabel("Ano")
    plt.ylabel("Plataforma")
    plt.title(
        "Índice de satisfação percebida por plataforma LCDP ao longo do tempo\n"
        f"(ratio = C3 / (C3+C4)  |  mín. {MIN_EVALUATIVE_COMMENTS} comentários avaliativos por célula)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_HEATMAP_PDF, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap salvo: {OUTPUT_HEATMAP_PNG}")
    print(f"  Heatmap salvo: {OUTPUT_HEATMAP_PDF}")


def write_report(
    satisfaction: pd.DataFrame,
    top_platforms: list[str],
    n_total: int,
    n_with_platform: int,
) -> None:
    """Escreve relatório de achados."""
    lines = [
        "Relatório: Índice de satisfação percebida por plataforma LCDP",
        "=" * 72,
        "",
        "Metodologia",
        f"  Índice: ratio = C3 / (C3 + C4) por plataforma × trimestre",
        f"  C3 = Elogio ou testemunho positivo (classificador léxico)",
        f"  C4 = Crítica, frustração ou objeção (classificador léxico)",
        f"  Mínimo de comentários avaliativos (C3+C4) por célula: {MIN_EVALUATIVE_COMMENTS}",
        "",
        "Cobertura",
        f"  Comentários analisados: {n_total}",
        f"  Comentários com menção de plataforma: {n_with_platform}",
        f"  Plataformas com dados suficientes para análise: {satisfaction['platform_tags'].nunique()}",
        f"  Células plataforma×trimestre com dados: {len(satisfaction)}",
        "",
        f"Top {TOP_N_PLATFORMS} plataformas por volume avaliativo",
        f"{'Plataforma':<32} {'C3+C4':>7} {'Ratio médio':>12} {'N trimestres':>13}",
        "-" * 70,
    ]
    for platform in top_platforms:
        sub = satisfaction[satisfaction["platform_tags"] == platform]
        lines.append(
            f"  {platform:<30} {int(sub['n_evaluative'].sum()):>7} "
            f"{float(sub['satisfaction_ratio'].mean()):>12.3f} "
            f"{len(sub):>13}"
        )

    # Achados principais
    lines.extend(["", "Achados principais (top plataformas)"])
    platform_means = (
        satisfaction[satisfaction["platform_tags"].isin(top_platforms)]
        .groupby("platform_tags")["satisfaction_ratio"]
        .mean()
        .sort_values(ascending=False)
    )
    lines.append(
        f"  Maior satisfação média : {platform_means.index[0]} (ratio={platform_means.iloc[0]:.3f})"
    )
    lines.append(
        f"  Menor satisfação média : {platform_means.index[-1]} (ratio={platform_means.iloc[-1]:.3f})"
    )

    # Tendência recente
    lines.extend(["", "Variação recente: último trimestre vs. 4 trimestres antes"])
    recent = satisfaction["quarter"].max()
    ref = recent - 4
    for platform in top_platforms:
        sub = satisfaction[satisfaction["platform_tags"] == platform]
        r_row = sub[sub["quarter"] == recent]["satisfaction_ratio"]
        ref_row = sub[sub["quarter"] == ref]["satisfaction_ratio"]
        if not r_row.empty and not ref_row.empty:
            delta = float(r_row.values[0]) - float(ref_row.values[0])
            direction = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
            lines.append(
                f"  {platform:<30} {direction} Δ={delta:+.3f} "
                f"({float(ref_row.values[0]):.3f} → {float(r_row.values[0]):.3f})"
            )

    OUTPUT_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Relatório salvo: {OUTPUT_REPORT_TXT}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar corpus
    comments = load_comments()

    # 2. Classificar (C1–C6)
    print("Classificando comentários C1-C6...")
    classified, _ = classify_comments(comments)
    classified["published_at"] = parse_utc_datetime(classified["published_at"])
    classified = classified.dropna(subset=["published_at"]).copy()
    classified["quarter"] = classified["published_at"].dt.to_period("Q")
    print(f"  {len(classified)} comentários classificados.")

    # Estatísticas de cobertura do classificador
    n_c3 = int(classified["C3"].sum())
    n_c4 = int(classified["C4"].sum())
    print(f"  C3 (elogios): {n_c3} ({n_c3/len(classified):.1%})")
    print(f"  C4 (críticas): {n_c4} ({n_c4/len(classified):.1%})")

    # 3. Tag de plataformas
    tagged = tag_platforms(classified)
    n_with_platform = int(tagged["platform_tags"].apply(len).gt(0).sum())
    print(f"  Comentários com menção de plataforma: {n_with_platform} ({n_with_platform/len(tagged):.1%})")

    # 4. Explodir por plataforma
    exploded = explode_platform_tags(tagged)
    print(f"  Pares comentário×plataforma: {len(exploded)}")

    # 5. Calcular índice
    print("Calculando índice de satisfação por plataforma × trimestre...")
    satisfaction = compute_satisfaction_index(exploded)
    print(f"  Células com dados suficientes: {len(satisfaction)}")
    print(f"  Plataformas cobertas: {satisfaction['platform_tags'].nunique()}")

    # 6. Selecionar top-N e gerar visualizações
    top_platforms = select_top_platforms(satisfaction)
    print(f"  Top {TOP_N_PLATFORMS}: {top_platforms}")

    full_quarter_range = pd.period_range(start=STUDY_START, end=STUDY_END, freq="Q")
    pivot = build_pivot(satisfaction, top_platforms, full_quarter_range)

    print("Gerando heatmap...")
    generate_heatmap(pivot)

    # 7. Salvar CSV e relatório
    print("Salvando resultados...")
    satisfaction.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  CSV salvo: {OUTPUT_CSV}")
    write_report(satisfaction, top_platforms, n_total=len(classified), n_with_platform=n_with_platform)

    print("\n✓ Análise concluída com sucesso.")


if __name__ == "__main__":
    main()
