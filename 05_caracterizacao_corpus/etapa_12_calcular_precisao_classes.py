from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import CLASS_AUDIT_TEMPLATE_CSV, CLASS_PRECISION_REPORT_TXT

TRUE_VALUES = {"1", "true", "t", "yes", "y", "sim", "s"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "nao", "não"}


def parse_manual_flag(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    phat = successes / total
    denominator = 1 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denominator
    margin = (z * math.sqrt((phat * (1 - phat) / total) + ((z * z) / (4 * total * total)))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    audit = pd.read_csv(CLASS_AUDIT_TEMPLATE_CSV, low_memory=False)
    if audit.empty:
        lines = [
            "Relatorio de precisao das classes da RQ3",
            "=" * 72,
            "O template de auditoria esta vazio. Gere a amostra antes de calcular a precisao.",
        ]
        CLASS_PRECISION_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Relatorio salvo em: {CLASS_PRECISION_REPORT_TXT}")
        return

    audit["manual_flag"] = audit["manual_is_true"].apply(parse_manual_flag)

    lines = [
        "Relatorio de precisao das classes da RQ3",
        "=" * 72,
        f"Arquivo auditado: {CLASS_AUDIT_TEMPLATE_CSV.name}",
        f"Linhas no template: {len(audit)}",
        "",
        "Resultados por classe",
    ]

    precision_values: list[float] = []
    audited_classes = 0

    for class_code, group in audit.groupby("predicted_class", sort=True):
        sampled_total = len(group)
        audited = group.loc[group["manual_flag"].notna()].copy()
        audited_total = len(audited)
        pending_total = sampled_total - audited_total
        true_positives = int((audited["manual_flag"] == True).sum())
        false_positives = int((audited["manual_flag"] == False).sum())

        lines.append(f"{class_code}")
        lines.append(f"  Amostrados: {sampled_total}")
        lines.append(f"  Auditados: {audited_total}")
        lines.append(f"  Pendentes: {pending_total}")
        lines.append(f"  Confirmados como corretos: {true_positives}")
        lines.append(f"  Considerados incorretos: {false_positives}")

        if audited_total > 0:
            precision = true_positives / audited_total
            interval = wilson_interval(true_positives, audited_total)
            if interval is None:
                lines.append(f"  Precisao: {precision:.2%}")
            else:
                lower, upper = interval
                lines.append(f"  Precisao: {precision:.2%} (IC95%: {lower:.2%} a {upper:.2%})")
            precision_values.append(precision)
            audited_classes += 1
        else:
            lines.append("  Precisao: nao calculada (nenhuma linha auditada)")

        false_examples = audited.loc[audited["manual_flag"] == False, "comment"].head(3).tolist()
        if false_examples:
            lines.append("  Exemplos de falso positivo:")
            for example in false_examples:
                lines.append(f"  - {example}")

        lines.append("")

    if audited_classes > 0:
        macro_precision = sum(precision_values) / audited_classes
        lines.extend(
            [
                "Resumo agregado",
                f"Macro-precisao nas classes auditadas: {macro_precision:.2%}",
            ]
        )
    else:
        lines.extend(
            [
                "Resumo agregado",
                "Macro-precisao nao calculada porque a coluna manual_is_true ainda nao foi preenchida.",
            ]
        )

    CLASS_PRECISION_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatorio salvo em: {CLASS_PRECISION_REPORT_TXT}")


if __name__ == "__main__":
    main()
