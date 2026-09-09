from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import CLASS_AGREEMENT_METRICS_REPORT_TXT, CLASS_AGREEMENT_SAMPLE_CSV

CLASS_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6"]
TRUE_VALUES = {"1", "true", "t", "yes", "y", "sim", "s"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "nao", "não", "nÃ£o"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula concordancia interanotador para a amostra de teste da RQ3."
    )
    parser.add_argument("--input", type=Path, default=CLASS_AGREEMENT_SAMPLE_CSV)
    parser.add_argument("--output", type=Path, default=CLASS_AGREEMENT_METRICS_REPORT_TXT)
    return parser.parse_args()


def parse_binary_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isclose(float(value), 1.0):
            return 1
        if math.isclose(float(value), 0.0):
            return 0
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return 1
    if normalized in FALSE_VALUES:
        return 0
    return None


def find_annotation_columns(columns: list[str]) -> dict[str, list[str]]:
    pattern = re.compile(r"^ann(\d+)_(C[1-6])$")
    by_class = {class_code: [] for class_code in CLASS_COLUMNS}
    for column in columns:
        match = pattern.match(column)
        if not match:
            continue
        by_class[match.group(2)].append(column)

    for class_code, class_columns in by_class.items():
        class_columns.sort(key=lambda name: int(pattern.match(name).group(1)))  # type: ignore[union-attr]
    return by_class


def annotator_ids(annotation_columns: dict[str, list[str]]) -> list[str]:
    ids: set[str] = set()
    pattern = re.compile(r"^(ann\d+)_C[1-6]$")
    for columns in annotation_columns.values():
        for column in columns:
            match = pattern.match(column)
            if match:
                ids.add(match.group(1))
    return sorted(ids, key=lambda value: int(value.replace("ann", "")))


def build_rows(frame: pd.DataFrame, columns: list[str]) -> list[list[int | None]]:
    rows: list[list[int | None]] = []
    for _, row in frame.iterrows():
        rows.append([parse_binary_label(row[column]) for column in columns])
    return rows


def krippendorff_alpha_nominal(rows: list[list[int | None]]) -> float | None:
    total_annotations = 0
    total_counts = {0: 0, 1: 0}
    observed_disagreement = 0.0

    for row in rows:
        valid = [value for value in row if value in (0, 1)]
        unit_n = len(valid)
        if unit_n < 2:
            continue

        unit_counts = {0: valid.count(0), 1: valid.count(1)}
        observed_disagreement += sum(count * (unit_n - count) for count in unit_counts.values()) / (unit_n - 1)
        for value in valid:
            total_counts[value] += 1
            total_annotations += 1

    if total_annotations <= 1:
        return None

    observed = observed_disagreement / total_annotations
    expected = sum(
        count * (total_annotations - count) for count in total_counts.values()
    ) / (total_annotations * (total_annotations - 1))

    if math.isclose(expected, 0.0):
        return None
    return 1 - (observed / expected)


def fleiss_kappa_binary(rows: list[list[int | None]]) -> float | None:
    valid_rows = [[value for value in row if value in (0, 1)] for row in rows]
    if not valid_rows:
        return None

    valid_counts = {len(row) for row in valid_rows}
    if len(valid_counts) != 1:
        return None
    raters = valid_counts.pop()
    if raters < 2:
        return None

    unit_count = len(valid_rows)
    category_totals = {0: 0, 1: 0}
    unit_agreements: list[float] = []

    for row in valid_rows:
        counts = {0: row.count(0), 1: row.count(1)}
        for label, count in counts.items():
            category_totals[label] += count
        unit_agreements.append(
            sum(count * (count - 1) for count in counts.values()) / (raters * (raters - 1))
        )

    observed = sum(unit_agreements) / unit_count
    expected = sum((category_totals[label] / (unit_count * raters)) ** 2 for label in (0, 1))
    if math.isclose(1 - expected, 0.0):
        return None
    return (observed - expected) / (1 - expected)


def summarize_rows(rows: list[list[int | None]]) -> dict[str, int]:
    valid_counts = [sum(value in (0, 1) for value in row) for row in rows]
    return {
        "units": len(rows),
        "with_any": sum(count > 0 for count in valid_counts),
        "with_two_or_more": sum(count >= 2 for count in valid_counts),
        "complete": sum(count == len(rows[0]) for count in valid_counts) if rows else 0,
        "valid_assignments": sum(valid_counts),
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "nao calculado"
    return f"{value:.4f}"


def build_report(frame: pd.DataFrame, input_path: Path) -> str:
    annotation_columns = find_annotation_columns(list(frame.columns))
    detected_annotators = annotator_ids(annotation_columns)
    populated_classes = {
        class_code: columns for class_code, columns in annotation_columns.items() if columns
    }

    lines = [
        "Relatorio de concordancia interanotador da RQ3",
        "=" * 72,
        f"Arquivo analisado: {input_path}",
        f"Linhas analisadas: {len(frame)}",
        "Padrao esperado de colunas: ann*_C1, ann*_C2, ..., ann*_C6",
        f"Anotadores detectados: {', '.join(detected_annotators) if detected_annotators else 'nenhum'}",
        "",
    ]

    if not populated_classes:
        lines.extend(
            [
                "Status",
                "Pendente: nenhuma coluna de anotador foi detectada.",
                "Crie ou preencha colunas no padrao ann1_C1 ... ann3_C6 com valores 0/1.",
            ]
        )
        return "\n".join(lines) + "\n"

    alpha_values: list[float] = []
    fleiss_values: list[float] = []
    pooled_rows: list[list[int | None]] = []

    lines.append("Resultados por classe")
    for class_code in CLASS_COLUMNS:
        columns = annotation_columns[class_code]
        if not columns:
            lines.append(f"{class_code}: nenhuma coluna detectada.")
            lines.append("")
            continue

        rows = build_rows(frame, columns)
        pooled_rows.extend(rows)
        summary = summarize_rows(rows)
        alpha = krippendorff_alpha_nominal(rows)
        fleiss = fleiss_kappa_binary(rows)
        if alpha is not None:
            alpha_values.append(alpha)
        if fleiss is not None:
            fleiss_values.append(fleiss)

        lines.append(class_code)
        lines.append(f"  Colunas: {', '.join(columns)}")
        lines.append(f"  Unidades com alguma anotacao: {summary['with_any']} / {summary['units']}")
        lines.append(f"  Unidades com pelo menos 2 anotacoes: {summary['with_two_or_more']} / {summary['units']}")
        lines.append(f"  Unidades completas: {summary['complete']} / {summary['units']}")
        lines.append(f"  Anotacoes validas: {summary['valid_assignments']}")
        lines.append(f"  Alfa de Krippendorff nominal: {format_metric(alpha)}")
        lines.append(f"  Kappa de Fleiss: {format_metric(fleiss)}")
        if summary["with_two_or_more"] == 0:
            lines.append("  Status: pendente de preenchimento por pelo menos dois anotadores.")
        elif fleiss is None:
            lines.append("  Observacao: Fleiss exige o mesmo numero de anotacoes validas em todas as unidades.")
        lines.append("")

    pooled_alpha = krippendorff_alpha_nominal(pooled_rows)
    pooled_fleiss = fleiss_kappa_binary(pooled_rows)

    lines.extend(
        [
            "Resumo agregado",
            f"Macro alfa de Krippendorff: {format_metric(sum(alpha_values) / len(alpha_values) if alpha_values else None)}",
            f"Macro Kappa de Fleiss: {format_metric(sum(fleiss_values) / len(fleiss_values) if fleiss_values else None)}",
            f"Alfa de Krippendorff empilhado classe-comentario: {format_metric(pooled_alpha)}",
            f"Kappa de Fleiss empilhado classe-comentario: {format_metric(pooled_fleiss)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    report = build_report(frame, args.input)
    args.output.write_text(report, encoding="utf-8")
    print(f"Relatorio salvo em: {args.output}")


if __name__ == "__main__":
    main()
