from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_00_config import (
    ENGLISH_COMMENTS_PERIOD_CSV,
    MACROTHEME_LATEX_TABLE,
    MACROTHEME_MAPPING_CSV,
    MACROTHEME_ORDER,
    MACROTHEME_REPORT_TXT,
    MACROTHEME_TEMPORAL_PDF,
    MACROTHEME_TEMPORAL_PNG,
    STUDY_END,
    STUDY_START,
    TOPICS_TO_REMOVE,
    TOPIC_ALIGNED_COMMENTS_CSV,
    TOPIC_BASE_COMMENTS_CSV,
    TOPIC_DESCRIPTIONS_CSV,
    ensure_directories,
    parse_utc_datetime,
    read_csv_with_fallback,
)


REQUIRED_MAPPING_COLUMNS = [
    "topic_id",
    "macrotema",
    "criterio_principal",
    "justificativa_curta",
    "termos_representativos",
    "n_comentarios_topico",
]

MACROTHEME_JUSTIFICATIONS = {
    "Plataformas e ecossistemas": "Tópicos centrados em ferramentas ou ecossistemas nomeados.",
    "Desenvolvimento de aplicações, UI e lógica": "Tópicos sobre construção de telas, componentes, controles e lógica de aplicação.",
    "Funcionalidades e casos de uso": "Tópicos que descrevem funcionalidades concretas ou domínios de aplicação.",
    "Dados, arquivos e artefatos": "Tópicos sobre arquivos, documentos, mídia, templates ou artefatos manipulados pelas plataformas.",
    "Integração, backend e infraestrutura": "Tópicos sobre APIs, autenticação, bancos de dados, hospedagem, segurança e infraestrutura.",
    "Automação e workflows": "Tópicos sobre fluxos, gatilhos, orquestração e automação de processos.",
    "IA, agentes e conversação": "Tópicos sobre IA generativa, agentes, prompts, chatbots, copilots e modelos conversacionais.",
    "Aprendizagem, tutoriais e capacitação": "Tópicos sobre tutoriais, explicações, certificação, idioma e materiais de aprendizagem.",
    "Comunidade, engajamento e interação": "Tópicos sobre agradecimentos, respostas, sugestões, inscrições e interação comunitária.",
    "Suporte, erros e limitações": "Tópicos sobre problemas, falhas, limitações, tentativas de correção e pedidos de diagnóstico.",
    "Governança, custos, licenças e disponibilidade": "Tópicos sobre preços, licenças, planos, disponibilidade, preview e restrições de acesso.",
}


def format_percent(part: int, total: int) -> str:
    if total == 0:
        return "0,00%"
    return f"{(part / total) * 100:.2f}%".replace(".", ",")


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def fail_if_invalid(errors: list[str]) -> None:
    if errors:
        details = "\n- ".join(errors)
        raise ValueError(f"Mapeamento de macrotemas invalido:\n- {details}")


def load_macrotheme_mapping(descriptions: pd.DataFrame) -> pd.DataFrame:
    mapping = read_csv_with_fallback(MACROTHEME_MAPPING_CSV, low_memory=False)
    errors: list[str] = []

    missing_columns = [column for column in REQUIRED_MAPPING_COLUMNS if column not in mapping.columns]
    if missing_columns:
        errors.append(f"Colunas obrigatorias ausentes: {missing_columns}")
        fail_if_invalid(errors)

    mapping = mapping[REQUIRED_MAPPING_COLUMNS].copy()
    mapping["topic_id"] = pd.to_numeric(mapping["topic_id"], errors="coerce")
    invalid_topic_rows = mapping.loc[mapping["topic_id"].isna()]
    if not invalid_topic_rows.empty:
        errors.append(f"Linhas com topic_id invalido: {invalid_topic_rows.index.tolist()}")

    mapping = mapping.dropna(subset=["topic_id"]).copy()
    mapping["topic_id"] = mapping["topic_id"].astype(int)
    mapping["n_comentarios_topico"] = pd.to_numeric(mapping["n_comentarios_topico"], errors="coerce")
    invalid_count_rows = mapping.loc[mapping["n_comentarios_topico"].isna()]
    if not invalid_count_rows.empty:
        errors.append(f"Linhas com n_comentarios_topico invalido: {invalid_count_rows['topic_id'].tolist()}")

    duplicate_ids = sorted(mapping.loc[mapping["topic_id"].duplicated(), "topic_id"].unique().tolist())
    if duplicate_ids:
        errors.append(f"topic_id duplicados no mapeamento: {duplicate_ids}")

    for column in ["macrotema", "criterio_principal", "justificativa_curta", "termos_representativos"]:
        empty_ids = mapping.loc[mapping[column].fillna("").astype(str).str.strip() == "", "topic_id"].tolist()
        if empty_ids:
            errors.append(f"Campo vazio em {column} para topic_id: {empty_ids}")

    descriptions = descriptions.copy()
    descriptions["topic_id"] = pd.to_numeric(descriptions["topic_id"], errors="coerce")
    descriptions = descriptions.dropna(subset=["topic_id"]).copy()
    descriptions["topic_id"] = descriptions["topic_id"].astype(int)
    descriptions["topic_count"] = pd.to_numeric(descriptions["topic_count"], errors="coerce")

    excluded_topics = set(TOPICS_TO_REMOVE + [-1])
    described_ids = set(descriptions["topic_id"].tolist())
    substantive_ids = described_ids - excluded_topics
    mapped_ids = set(mapping["topic_id"].tolist())

    excluded_mapped = sorted(mapped_ids & excluded_topics)
    if excluded_mapped:
        errors.append(f"Topicos excluidos indevidamente mapeados: {excluded_mapped}")

    unknown_ids = sorted(mapped_ids - described_ids)
    if unknown_ids:
        errors.append(f"Topicos inexistentes no arquivo de descricoes: {unknown_ids}")

    missing_ids = sorted(substantive_ids - mapped_ids)
    if missing_ids:
        errors.append(f"Topicos substantivos sem macrotema: {missing_ids}")

    extra_ids = sorted(mapped_ids - substantive_ids)
    if extra_ids:
        errors.append(f"Topicos extras fora do conjunto substantivo: {extra_ids}")

    invalid_macrothemes = sorted(set(mapping["macrotema"].tolist()) - set(MACROTHEME_ORDER))
    if invalid_macrothemes:
        errors.append(f"Macrotemas fora da ordem oficial: {invalid_macrothemes}")

    count_reference = descriptions.set_index("topic_id")["topic_count"].to_dict()
    count_mismatches = []
    for row in mapping.itertuples(index=False):
        expected_count = count_reference.get(row.topic_id)
        if pd.notna(expected_count) and pd.notna(row.n_comentarios_topico):
            if int(expected_count) != int(row.n_comentarios_topico):
                count_mismatches.append((row.topic_id, int(row.n_comentarios_topico), int(expected_count)))
    if count_mismatches:
        errors.append(f"n_comentarios_topico divergente das descricoes: {count_mismatches}")

    fail_if_invalid(errors)

    return mapping.merge(
        descriptions[["topic_id", "topic_name", "topic_count"]],
        on="topic_id",
        how="left",
        validate="one_to_one",
    )


def generate_latex_table(mapping: pd.DataFrame) -> None:
    lines = [
        "% Tabela gerada automaticamente por 04_topicos/etapa_08_analisar_macrotemas.py",
        r"\begingroup",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{longtable}{@{}p{0.24\textwidth}p{0.27\textwidth}p{0.41\textwidth}@{}}",
        r"\caption{Mapeamento curatorial dos tópicos em macrotemas.}",
        r"\label{tab:mapeamento_macrotemas}\\",
        r"\toprule",
        r"Macrotema & Tópicos associados & Critério sintético \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Macrotema & Tópicos associados & Critério sintético \\",
        r"\midrule",
        r"\endhead",
    ]

    for macrotheme in MACROTHEME_ORDER:
        subset = mapping.loc[mapping["macrotema"] == macrotheme].sort_values("topic_id")
        topic_ids = ", ".join(str(topic_id) for topic_id in subset["topic_id"].tolist())
        justification = MACROTHEME_JUSTIFICATIONS[macrotheme]
        lines.append(
            f"{latex_escape(macrotheme)} & {latex_escape(topic_ids)} & {latex_escape(justification)} " + r"\\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    MACROTHEME_LATEX_TABLE.write_text("\n".join(lines), encoding="utf-8")


def write_macrotheme_report(
    period_comments_count: int,
    topic_base_count: int,
    aligned_count: int,
    analysis_df: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    mapped_count = len(analysis_df)
    distribution = (
        analysis_df.groupby("macrotema")
        .agg(comentarios=("topic_id", "size"), topicos=("topic_id", "nunique"))
        .reindex(MACROTHEME_ORDER)
        .fillna(0)
        .astype(int)
    )

    lines = [
        "Relatorio de macrotemas",
        "=" * 72,
        "Metodo: agregacao curatorial/manual dos topicos BERTopic em macrotemas de ordem superior.",
        "Uso de LLM generativa: nao houve uso de modelo de chat, prompt ou LLM generativa nesta etapa.",
        "Criterio: inspecao dos termos c-TF-IDF representativos e de comentarios exemplares de cada topico.",
        "Regra: cada topico substantivo deve pertencer a exatamente um macrotema.",
        "",
        f"Topicos removidos como ruido ou fora de escopo: {[-1] + TOPICS_TO_REMOVE}",
        f"Topicos substantivos mapeados: {mapping['topic_id'].nunique()}",
        "Duplicatas no mapeamento: 0",
        "Topicos substantivos sem macrotema: 0",
        "Topicos excluidos indevidamente mapeados: 0",
        "",
        "Cobertura",
        f"Comentarios no corpus do periodo analitico: {period_comments_count}",
        f"Comentarios na base de topicos: {topic_base_count}",
        f"Comentarios alinhados com topic_id: {aligned_count}",
        f"Comentarios analisados apos excluir topicos de ruido: {mapped_count}",
        f"Comentarios com macrotema atribuido: {mapped_count}",
        f"Cobertura sobre comentarios analisados apos exclusao de ruido: {format_percent(mapped_count, mapped_count)}",
        f"Cobertura sobre corpus do periodo analitico: {format_percent(mapped_count, period_comments_count)}",
        f"Cobertura sobre base de topicos: {format_percent(mapped_count, topic_base_count)}",
        "",
        "Distribuicao por macrotema",
        "Macrotema | Topicos | Comentarios | % dos comentarios analisados",
    ]

    for macrotheme, row in distribution.iterrows():
        comments = int(row["comentarios"])
        topics = int(row["topicos"])
        lines.append(f"{macrotheme} | {topics} | {comments} | {format_percent(comments, mapped_count)}")

    lines.extend(["", "Mapeamento topico -> macrotema", "topic_id | topic_name | macrotema | criterio"])
    for row in mapping.sort_values("topic_id").itertuples(index=False):
        lines.append(f"{row.topic_id} | {row.topic_name} | {row.macrotema} | {row.criterio_principal}")

    MACROTHEME_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_directories()

    period_comments = read_csv_with_fallback(ENGLISH_COMMENTS_PERIOD_CSV, low_memory=False)
    topic_base = read_csv_with_fallback(TOPIC_BASE_COMMENTS_CSV, low_memory=False)
    aligned = read_csv_with_fallback(TOPIC_ALIGNED_COMMENTS_CSV, low_memory=False)
    descriptions = read_csv_with_fallback(TOPIC_DESCRIPTIONS_CSV, low_memory=False)
    mapping = load_macrotheme_mapping(descriptions)

    aligned = aligned.copy()
    aligned["topic_id"] = pd.to_numeric(aligned["topic_id"], errors="coerce")
    aligned["published_at"] = parse_utc_datetime(aligned["published_at"])
    aligned = aligned.dropna(subset=["topic_id", "published_at"]).copy()
    aligned["topic_id"] = aligned["topic_id"].astype(int)

    excluded_topics = set(TOPICS_TO_REMOVE + [-1])
    analysis_df = aligned.loc[~aligned["topic_id"].isin(excluded_topics)].copy()
    analysis_df = analysis_df.merge(
        mapping[["topic_id", "macrotema"]],
        on="topic_id",
        how="left",
        validate="many_to_one",
    )

    missing_observed_ids = sorted(analysis_df.loc[analysis_df["macrotema"].isna(), "topic_id"].unique().tolist())
    fail_if_invalid([f"Topicos observados sem macrotema: {missing_observed_ids}"] if missing_observed_ids else [])

    analysis_df["mes_ano"] = analysis_df["published_at"].dt.to_period("M")
    counts = analysis_df.groupby(["mes_ano", "macrotema"]).size().reset_index(name="count")
    pivot = counts.pivot_table(index="mes_ano", columns="macrotema", values="count", fill_value=0)
    full_range = pd.period_range(start=STUDY_START, end=STUDY_END, freq="M")
    pivot = pivot.reindex(index=full_range, fill_value=0)
    pivot = pivot.reindex(columns=MACROTHEME_ORDER, fill_value=0)
    proportions = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)

    plt.figure(figsize=(12, 6.4))
    for macrotheme in MACROTHEME_ORDER:
        plt.plot(proportions.index.astype(str), proportions[macrotheme], label=macrotheme, linewidth=1.8)

    tick_positions = np.arange(0, len(proportions.index), 6)
    plt.xticks(tick_positions, proportions.index.astype(str)[tick_positions], rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Mês", fontsize=12)
    plt.ylabel("Proporção de comentários analisados", fontsize=12)
    plt.grid(axis="y", alpha=0.2)
    plt.legend(ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(MACROTHEME_TEMPORAL_PNG, dpi=300)
    plt.savefig(MACROTHEME_TEMPORAL_PDF)
    plt.close()

    generate_latex_table(mapping)
    write_macrotheme_report(
        period_comments_count=len(period_comments),
        topic_base_count=len(topic_base),
        aligned_count=len(aligned),
        analysis_df=analysis_df,
        mapping=mapping,
    )

    print(f"Topicos substantivos mapeados: {mapping['topic_id'].nunique()}")
    print(f"Comentarios com macrotema: {len(analysis_df)}")
    print(f"Grafico salvo em: {MACROTHEME_TEMPORAL_PNG}")
    print(f"Relatorio salvo em: {MACROTHEME_REPORT_TXT}")
    print(f"Tabela LaTeX salva em: {MACROTHEME_LATEX_TABLE}")


if __name__ == "__main__":
    main()
