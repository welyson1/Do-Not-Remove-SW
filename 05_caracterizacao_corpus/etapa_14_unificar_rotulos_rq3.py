from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    CLASS_AGREEMENT_SAMPLE_CSV,
    CLASS_FINAL_LABELS_CSV,
    CLASS_LABEL_MERGE_REPORT_TXT,
    CLASS_LABELING_FULL_SAMPLE_CSV,
    CLASS_STRATIFIED_SAMPLE_CSV,
    CLASS_TEST_LABELS_CONSOLIDATED_CSV,
)

CLASS_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6"]
EXPECTED_ANNOTATORS = 3
SAMPLE_COLUMNS = [
    "sample_id",
    "sample_class",
    "sample_class_label",
    "sample_year",
    "comment_id",
    "video_id",
    "published_at",
    "is_reply",
    "like_count",
    "comment_length",
    "label_combo",
    "n_labels",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "comment",
]
LABEL_COLUMNS = ["comment_id", "video_id", "comment"] + CLASS_COLUMNS
TRUE_VALUES = {"1", "true", "t", "yes", "y", "sim", "s"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "nao", "não", "nÃ£o", "nÃƒÂ£o", "nÃƒÆ’Ã‚Â£o"}


class ValidationFailure(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unifica rotulos exportados por 3 avaliadores fixos e calcula concordancia da RQ3."
    )
    parser.add_argument("--sample-pilot", type=Path, default=CLASS_AGREEMENT_SAMPLE_CSV)
    parser.add_argument("--sample-full", type=Path, default=CLASS_LABELING_FULL_SAMPLE_CSV)
    parser.add_argument("--legacy-sample", type=Path, default=CLASS_STRATIFIED_SAMPLE_CSV)
    parser.add_argument("--pilot-labels", nargs="*", type=Path, default=[])
    parser.add_argument("--full-labels", nargs="*", type=Path, default=[])
    parser.add_argument("--output-pilot", type=Path, default=CLASS_TEST_LABELS_CONSOLIDATED_CSV)
    parser.add_argument("--output-final", type=Path, default=CLASS_FINAL_LABELS_CSV)
    parser.add_argument("--report", type=Path, default=CLASS_LABEL_MERGE_REPORT_TXT)
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


def is_blank_label(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def format_metric(value: float | None) -> str:
    if value is None:
        return "nao calculado"
    return f"{value:.4f}"


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


def read_sample(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in SAMPLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationFailure(f"{name}: colunas ausentes no arquivo de amostra: {', '.join(missing)}")
    duplicate_count = int(frame["comment_id"].astype(str).duplicated().sum())
    if duplicate_count:
        raise ValidationFailure(f"{name}: comment_id duplicados na amostra = {duplicate_count}.")
    return frame[SAMPLE_COLUMNS].copy()


def read_legacy_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["comment_id"], low_memory=False)
    return set(frame["comment_id"].astype(str))


def label_file_display_name(path: Path, index: int) -> str:
    stem = path.stem
    if stem.lower().startswith("rotulos-"):
        stem = stem[len("rotulos-") :]
    return stem or f"avaliador{index}"


def require_three_files(paths: list[Path], label: str) -> None:
    if paths and len(paths) != EXPECTED_ANNOTATORS:
        raise ValidationFailure(f"{label}: informe exatamente {EXPECTED_ANNOTATORS} arquivos de rotulos.")


def read_label_file(
    path: Path,
    expected_sample: pd.DataFrame,
    expected_name: str,
    legacy_ids: set[str],
) -> pd.DataFrame:
    labels = pd.read_csv(path, low_memory=False)
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValidationFailure(f"{path}: colunas obrigatorias ausentes: {', '.join(missing)}")

    labels = labels[LABEL_COLUMNS].copy()
    duplicate_count = int(labels["comment_id"].astype(str).duplicated().sum())
    if duplicate_count:
        raise ValidationFailure(f"{path}: comment_id duplicados no arquivo de rotulos = {duplicate_count}.")

    expected_ids = set(expected_sample["comment_id"].astype(str))
    label_ids = set(labels["comment_id"].astype(str))
    outside = label_ids - expected_ids
    missing_ids = expected_ids - label_ids

    if outside or missing_ids or len(labels) != len(expected_sample):
        detail = [
            f"{path}: arquivo nao corresponde a {expected_name}.",
            f"  Linhas no arquivo: {len(labels)}; esperado: {len(expected_sample)}.",
            f"  IDs fora da amostra esperada: {len(outside)}.",
            f"  IDs esperados ausentes: {len(missing_ids)}.",
            f"  Intersecao com a amostra esperada: {len(label_ids & expected_ids)}.",
        ]
        if legacy_ids and label_ids.issubset(legacy_ids):
            detail.append("  Diagnostico: estes rotulos parecem pertencer a amostra antiga de 4000 comentarios.")
        raise ValidationFailure("\n".join(detail))

    labels["_comment_id_key"] = labels["comment_id"].astype(str)
    ordered = expected_sample[["comment_id"]].copy()
    ordered["_comment_id_key"] = ordered["comment_id"].astype(str)
    ordered = ordered[["_comment_id_key"]].merge(labels, on="_comment_id_key", how="left", validate="one_to_one")
    ordered = ordered.drop(columns=["_comment_id_key"])

    invalid_messages: list[str] = []
    for class_code in CLASS_COLUMNS:
        parsed = ordered[class_code].apply(parse_binary_label)
        blank_mask = ordered[class_code].apply(is_blank_label)
        invalid_count = int((parsed.isna() & ~blank_mask).sum())
        if invalid_count:
            invalid_messages.append(f"{class_code}: {invalid_count} valores invalidos")
        ordered[class_code] = parsed
    if invalid_messages:
        raise ValidationFailure(f"{path}: valores de rotulo invalidos: " + "; ".join(invalid_messages))

    return ordered


def majority_vote(values: list[int | None]) -> tuple[int | None, bool]:
    valid = [value for value in values if value in (0, 1)]
    if len(valid) < 2:
        return None, True
    ones = valid.count(1)
    zeros = valid.count(0)
    if ones > zeros:
        return 1, False
    if zeros > ones:
        return 0, False
    return None, True


def build_consolidated_labels(sample: pd.DataFrame, label_frames: list[pd.DataFrame]) -> pd.DataFrame:
    consolidated = sample.copy()
    for index, labels in enumerate(label_frames, start=1):
        for class_code in CLASS_COLUMNS:
            consolidated[f"ann{index}_{class_code}"] = labels[class_code].apply(parse_binary_label).astype("Int64")

    review_classes_by_row: list[str] = []
    for row_index, row in consolidated.iterrows():
        review_classes: list[str] = []
        for class_code in CLASS_COLUMNS:
            values = [row[f"ann{index}_{class_code}"] for index in range(1, len(label_frames) + 1)]
            final_value, needs_review = majority_vote([None if pd.isna(value) else int(value) for value in values])
            consolidated.at[row_index, f"final_{class_code}"] = final_value
            if needs_review:
                review_classes.append(class_code)
        review_classes_by_row.append(";".join(review_classes))

    consolidated["needs_review"] = [bool(value) for value in review_classes_by_row]
    consolidated["review_classes"] = review_classes_by_row
    return consolidated


def metric_rows(consolidated: pd.DataFrame, class_code: str, annotator_count: int) -> list[list[int | None]]:
    rows: list[list[int | None]] = []
    for _, row in consolidated.iterrows():
        values: list[int | None] = []
        for index in range(1, annotator_count + 1):
            value = row[f"ann{index}_{class_code}"]
            values.append(None if pd.isna(value) else int(value))
        rows.append(values)
    return rows


def append_concordance_report(
    lines: list[str],
    consolidated: pd.DataFrame,
    annotator_count: int,
    section_title: str,
) -> None:
    alpha_values: list[float] = []
    fleiss_values: list[float] = []
    pooled_rows: list[list[int | None]] = []

    lines.extend(["", section_title])
    for class_code in CLASS_COLUMNS:
        rows = metric_rows(consolidated, class_code, annotator_count)
        pooled_rows.extend(rows)
        alpha = krippendorff_alpha_nominal(rows)
        fleiss = fleiss_kappa_binary(rows)
        if alpha is not None:
            alpha_values.append(alpha)
        if fleiss is not None:
            fleiss_values.append(fleiss)

        valid_counts = [sum(value in (0, 1) for value in row) for row in rows]
        lines.append(f"{class_code}")
        lines.append(f"  Unidades com pelo menos 2 anotacoes: {sum(count >= 2 for count in valid_counts)} / {len(rows)}")
        lines.append(f"  Unidades completas: {sum(count == annotator_count for count in valid_counts)} / {len(rows)}")
        lines.append(f"  Alfa de Krippendorff nominal: {format_metric(alpha)}")
        lines.append(f"  Kappa de Fleiss: {format_metric(fleiss)}")

    pooled_alpha = krippendorff_alpha_nominal(pooled_rows)
    pooled_fleiss = fleiss_kappa_binary(pooled_rows)
    lines.extend(
        [
            "",
            f"Resumo - {section_title}",
            f"Macro alfa de Krippendorff: {format_metric(sum(alpha_values) / len(alpha_values) if alpha_values else None)}",
            f"Macro Kappa de Fleiss: {format_metric(sum(fleiss_values) / len(fleiss_values) if fleiss_values else None)}",
            f"Alfa de Krippendorff empilhado classe-comentario: {format_metric(pooled_alpha)}",
            f"Kappa de Fleiss empilhado classe-comentario: {format_metric(pooled_fleiss)}",
        ]
    )


def build_final_labels(sample_full: pd.DataFrame, consolidated_full: pd.DataFrame) -> pd.DataFrame:
    output = sample_full.copy()
    for class_code in CLASS_COLUMNS:
        output[f"dict_{class_code}"] = output[class_code]
        output[class_code] = consolidated_full[f"final_{class_code}"].astype("Int64")
    output["label_source"] = "maioria_3_rotuladores"
    output["needs_review"] = consolidated_full["needs_review"]
    output["review_classes"] = consolidated_full["review_classes"]
    return output.sort_values("sample_id").reset_index(drop=True)


def load_label_frames(
    paths: list[Path],
    sample: pd.DataFrame,
    sample_name: str,
    legacy_ids: set[str],
    report_lines: list[str],
    prefix: str,
) -> list[pd.DataFrame]:
    label_frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        labels = read_label_file(path, sample, sample_name, legacy_ids)
        label_frames.append(labels)
        report_lines.append(f"{prefix}{index}: {path} ({label_file_display_name(path, index)})")
    return label_frames


def main() -> None:
    args = parse_args()
    report_lines = [
        "Relatorio de unificacao de rotulos e concordancia da RQ3",
        "=" * 72,
        f"Amostra completa: {args.sample_full}",
        f"Amostra piloto: {args.sample_pilot}",
    ]

    try:
        require_three_files(args.pilot_labels, "--pilot-labels")
        require_three_files(args.full_labels, "--full-labels")

        sample_full = read_sample(args.sample_full, "amostra completa")
        sample_pilot = read_sample(args.sample_pilot, "amostra piloto")
        legacy_ids = read_legacy_ids(args.legacy_sample)

        full_ids = set(sample_full["comment_id"].astype(str))
        pilot_ids = set(sample_pilot["comment_id"].astype(str))
        if not pilot_ids.issubset(full_ids):
            raise ValidationFailure("A amostra piloto nao e subconjunto da amostra completa.")

        report_lines.extend(
            [
                "",
                "Validacao das amostras",
                f"Completa: {len(sample_full)} linhas.",
                f"Piloto: {len(sample_pilot)} linhas.",
                "OK: piloto e subconjunto da amostra completa.",
            ]
        )

        if args.pilot_labels:
            report_lines.extend(["", "Arquivos do piloto"])
            pilot_frames = load_label_frames(
                args.pilot_labels,
                sample_pilot,
                "amostra piloto de 300 comentarios",
                legacy_ids,
                report_lines,
                prefix="piloto_ann",
            )
            pilot_consolidated = build_consolidated_labels(sample_pilot, pilot_frames)
            pilot_consolidated.to_csv(args.output_pilot, index=False, encoding="utf-8")
            report_lines.append(f"Consolidado do piloto salvo em: {args.output_pilot}")
            report_lines.append(f"Linhas do piloto que precisam de revisao por falta de maioria: {int(pilot_consolidated['needs_review'].sum())}")
            append_concordance_report(
                report_lines,
                pilot_consolidated,
                annotator_count=len(pilot_frames),
                section_title="Concordancia preliminar do piloto",
            )
        else:
            report_lines.extend(
                [
                    "",
                    "Piloto",
                    "Pendente: informe 3 arquivos em --pilot-labels para calcular a concordancia preliminar do piloto.",
                ]
            )

        if args.full_labels:
            report_lines.extend(["", "Arquivos da amostra completa"])
            full_frames = load_label_frames(
                args.full_labels,
                sample_full,
                "amostra completa de 4800 comentarios",
                legacy_ids,
                report_lines,
                prefix="final_ann",
            )
            full_consolidated = build_consolidated_labels(sample_full, full_frames)
            final_output = build_final_labels(sample_full, full_consolidated)
            duplicate_count = int(final_output["comment_id"].astype(str).duplicated().sum())
            if len(final_output) != len(sample_full) or duplicate_count:
                raise ValidationFailure(
                    f"Arquivo final invalido: linhas={len(final_output)}, duplicatas={duplicate_count}."
                )
            final_output.to_csv(args.output_final, index=False, encoding="utf-8")
            report_lines.append(f"CSV final salvo em: {args.output_final}")
            report_lines.append(f"Linhas finais: {len(final_output)}")
            report_lines.append(f"Linhas finais que precisam de revisao por falta de maioria: {int(final_output['needs_review'].sum())}")
            append_concordance_report(
                report_lines,
                full_consolidated,
                annotator_count=len(full_frames),
                section_title="Concordancia final da amostra completa",
            )
        else:
            report_lines.extend(
                [
                    "",
                    "Arquivo final",
                    "Pendente: informe 3 arquivos em --full-labels para gerar o CSV final com os 4800 comentarios.",
                ]
            )

        args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"Relatorio salvo em: {args.report}")
    except ValidationFailure as exc:
        report_lines.extend(["", "Falha de validacao", str(exc)])
        args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
