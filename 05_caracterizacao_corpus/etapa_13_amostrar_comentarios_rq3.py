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
    CLASS_AGREEMENT_SAMPLE_REPORT_TXT,
    CLASS_LABELING_FULL_SAMPLE_CSV,
    CLASS_LABELING_FULL_SAMPLE_REPORT_TXT,
    ENGLISH_COMMENTS_PERIOD_CSV,
    LANGUAGE_FILTER_COLUMNS,
    ensure_directories,
    load_language_comment_file,
    parse_utc_datetime,
)
from etapa_10_testar_sanidade_classes import (
    CLASS_COLUMNS,
    CLASS_DEFINITIONS,
    classify_comments,
)

DEFAULT_RANDOM_SEED = 42
FULL_TARGET_PER_CLASS = 800
PILOT_TARGET_PER_CLASS = 50
MIN_COMMENT_LENGTH = 25
YEARS = list(range(2018, 2026))

OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera amostra completa e piloto balanceadas para rotulacao multirrotulo da RQ3."
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--full-per-class", type=int, default=FULL_TARGET_PER_CLASS)
    parser.add_argument("--pilot-per-class", type=int, default=PILOT_TARGET_PER_CLASS)
    return parser.parse_args()


def prepare_eligible_comments(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    classified, dedup_stats = classify_comments(comments)
    classified["published_at"] = parse_utc_datetime(classified["published_at"])
    classified["sample_year"] = classified["published_at"].dt.year
    classified["comment_length"] = classified["comment"].fillna("").astype(str).str.strip().str.len()

    period_mask = classified["sample_year"].isin(YEARS)
    length_mask = classified["comment_length"] > MIN_COMMENT_LENGTH
    labeled_mask = classified["n_labels"] > 0
    eligible_mask = period_mask & length_mask & labeled_mask

    stats = {
        **dedup_stats,
        "period_count": int(period_mask.sum()),
        "length_count": int(length_mask.sum()),
        "labeled_count": int(labeled_mask.sum()),
        "eligible_count": int(eligible_mask.sum()),
        "eligible_multilabel_count": int(((classified["n_labels"] > 1) & eligible_mask).sum()),
        "eligible_unlabeled_count": int(((classified["n_labels"] == 0) & period_mask & length_mask).sum()),
    }
    return classified.loc[eligible_mask].copy(), stats


def allocate_largest_remainders(available_by_year: pd.Series, target_total: int, class_code: str) -> dict[int, int]:
    available = available_by_year.reindex(YEARS, fill_value=0).astype(int)
    total_available = int(available.sum())
    if target_total <= 0:
        raise ValueError("A cota alvo precisa ser positiva.")
    if total_available < target_total:
        raise ValueError(
            f"Amostra insuficiente para {class_code}: disponivel={total_available}, quota={target_total}."
        )

    raw = {year: (int(available[year]) * target_total) / total_available for year in YEARS}
    quotas = {year: math.floor(raw[year]) for year in YEARS}
    remaining = target_total - sum(quotas.values())
    candidates = sorted(
        YEARS,
        key=lambda year: (raw[year] - quotas[year], int(available[year]), -year),
        reverse=True,
    )

    while remaining > 0:
        changed = False
        for year in candidates:
            if remaining == 0:
                break
            if quotas[year] < int(available[year]):
                quotas[year] += 1
                remaining -= 1
                changed = True
        if not changed:
            break

    if remaining:
        raise ValueError(f"Nao foi possivel distribuir {remaining} cotas anuais de {class_code}.")
    return quotas


def build_equal_class_quotas(target_per_class: int) -> dict[str, int]:
    return {class_code: target_per_class for class_code in CLASS_COLUMNS}


def build_year_quotas_from_labels(source: pd.DataFrame, target_per_class: int) -> dict[str, dict[int, int]]:
    quotas: dict[str, dict[int, int]] = {}
    for class_code in CLASS_COLUMNS:
        available_by_year = (
            source.loc[source[class_code]]
            .groupby("sample_year")
            .size()
            .reindex(YEARS, fill_value=0)
        )
        quotas[class_code] = allocate_largest_remainders(available_by_year, target_per_class, class_code)
    return quotas


def available_by_class(source: pd.DataFrame, use_sample_class: bool) -> dict[str, int]:
    if use_sample_class:
        counts = source["sample_class"].value_counts().reindex(CLASS_COLUMNS, fill_value=0).astype(int)
        return counts.to_dict()
    return {class_code: int(source[class_code].sum()) for class_code in CLASS_COLUMNS}


def stable_random_state(random_seed: int, class_code: str, year: int, offset: int = 0) -> int:
    class_index = CLASS_COLUMNS.index(class_code) + 1
    return random_seed + offset + (class_index * 10000) + year


def sample_without_replacement(
    source: pd.DataFrame,
    year_quotas: dict[str, dict[int, int]],
    random_seed: int,
    use_sample_class: bool,
    random_offset: int = 0,
    insert_sample_id: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    used_comment_texts: set[str] = set()
    class_availability = available_by_class(source, use_sample_class=use_sample_class)
    sample_order = sorted(CLASS_COLUMNS, key=lambda code: (class_availability[code], code))

    for class_code in sample_order:
        for year in YEARS:
            quota = year_quotas[class_code][year]
            if quota == 0:
                continue

            class_mask = source["sample_class"].eq(class_code) if use_sample_class else source[class_code]
            subset = source.loc[
                class_mask
                & source["sample_year"].eq(year)
                & ~source["comment_normalized"].isin(used_comment_texts)
            ].copy()
            if len(subset) < quota:
                raise ValueError(
                    f"Amostra insuficiente para {class_code}/{year}: disponivel={len(subset)}, quota={quota}."
                )

            sampled = subset.sample(
                n=quota,
                random_state=stable_random_state(random_seed, class_code, year, offset=random_offset),
            ).copy()
            sampled["sample_class"] = class_code
            sampled["sample_class_label"] = CLASS_DEFINITIONS[class_code]["label"]
            used_comment_texts.update(sampled["comment_normalized"].tolist())
            frames.append(sampled)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    class_order = {class_code: index for index, class_code in enumerate(CLASS_COLUMNS)}
    output["_sample_class_order"] = output["sample_class"].map(class_order)
    output = output.sort_values(["_sample_class_order", "sample_year", "comment_id"]).reset_index(drop=True)
    output = output.drop(columns=["_sample_class_order"])
    if insert_sample_id:
        output.insert(0, "sample_id", range(1, len(output) + 1))
    return output


def build_class_year_counts(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["sample_class", "sample_year"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CLASS_COLUMNS, columns=YEARS, fill_value=0)
        .astype(int)
    )


def validate_sample(
    sample: pd.DataFrame,
    class_quotas: dict[str, int],
    year_quotas: dict[str, dict[int, int]],
    expected_size: int,
) -> list[str]:
    messages: list[str] = []

    if len(sample) != expected_size:
        messages.append(f"Falha: linhas geradas = {len(sample)}, esperado = {expected_size}.")
    else:
        messages.append(f"OK: linhas geradas = {expected_size}.")

    duplicate_count = int(sample["comment_normalized"].duplicated().sum())
    if duplicate_count:
        messages.append(f"Falha: comentarios normalizados duplicados = {duplicate_count}.")
    else:
        messages.append("OK: nenhum comment_normalized repetido.")

    short_count = int((sample["comment_length"] <= MIN_COMMENT_LENGTH).sum())
    if short_count:
        messages.append(f"Falha: comentarios com ate {MIN_COMMENT_LENGTH} caracteres = {short_count}.")
    else:
        messages.append(f"OK: todos os comentarios tem mais de {MIN_COMMENT_LENGTH} caracteres.")

    sem_classe_count = int(sample["sample_class"].eq("sem_classe").sum())
    if sem_classe_count:
        messages.append(f"Falha: linhas com sample_class=sem_classe = {sem_classe_count}.")
    else:
        messages.append("OK: nenhuma linha com sample_class=sem_classe.")

    class_counts = sample["sample_class"].value_counts().reindex(CLASS_COLUMNS, fill_value=0).astype(int).to_dict()
    if class_counts != class_quotas:
        messages.append(f"Falha: cotas por classe divergentes = {class_counts}.")
    else:
        messages.append("OK: cotas por sample_class conferem.")

    class_year_counts = build_class_year_counts(sample)
    class_year_ok = True
    for class_code in CLASS_COLUMNS:
        observed = class_year_counts.loc[class_code].to_dict()
        if observed != year_quotas[class_code]:
            messages.append(f"Falha: cotas anuais de {class_code} divergentes = {observed}.")
            class_year_ok = False
    if class_year_ok:
        messages.append("OK: cotas por sample_class e sample_year conferem.")

    failures = [message for message in messages if message.startswith("Falha:")]
    if failures:
        raise AssertionError("\n".join(messages))
    return messages


def validate_pilot_subset(pilot_sample: pd.DataFrame, full_sample: pd.DataFrame) -> list[str]:
    full_ids = set(full_sample["comment_id"].astype(str))
    pilot_ids = set(pilot_sample["comment_id"].astype(str))
    missing_count = len(pilot_ids - full_ids)
    if missing_count:
        raise AssertionError(f"Falha: comentarios do piloto fora da amostra completa = {missing_count}.")
    return ["OK: amostra piloto e subconjunto da amostra completa."]


def write_sample(sample: pd.DataFrame, path: Path) -> None:
    sample[OUTPUT_COLUMNS].to_csv(path, index=False, encoding="utf-8")


def build_report(
    title: str,
    output_path: Path,
    sample: pd.DataFrame,
    eligible: pd.DataFrame,
    stats: dict[str, int],
    validation_messages: list[str],
    class_quotas: dict[str, int],
    year_quotas: dict[str, dict[int, int]],
    random_seed: int,
    repaired_in_memory: bool,
    target_per_class: int,
    extra_lines: list[str] | None = None,
) -> str:
    eligible_class_counts = {class_code: int(eligible[class_code].sum()) for class_code in CLASS_COLUMNS}
    eligible_memberships = sum(eligible_class_counts.values())
    class_counts = sample["sample_class"].value_counts().reindex(CLASS_COLUMNS, fill_value=0).astype(int)
    class_year_counts = build_class_year_counts(sample)

    lines = [
        title,
        "=" * 72,
        f"Arquivo de entrada: {ENGLISH_COMMENTS_PERIOD_CSV}",
        f"Arquivo de saida: {output_path}",
        f"Seed aleatoria: {random_seed}",
        f"CSV de entrada reparado apenas em memoria: {repaired_in_memory}",
    ]
    if extra_lines:
        lines.extend(extra_lines)

    lines.extend(
        [
            "",
            "Premissas",
            "- Comentarios unicos apos deduplicacao por texto normalizado.",
            f"- Comentarios com mais de {MIN_COMMENT_LENGTH} caracteres apos strip().",
            "- Periodo: 2018 a 2025.",
            "- Classes elegiveis: C1-C6; sem_classe excluido.",
            "- sample_class indica o estrato de amostragem, nao um rotulo exclusivo.",
            "- Cabecalho mantido compativel com o CSV antigo de entrada do software de rotulacao.",
            "- Comentarios e replies sao tratados como comentarios para a rotulacao.",
            "- A amostra balanceada nao estima prevalencia real das classes no corpus.",
            "",
            "Totais de filtragem",
            f"Comentarios antes da deduplicacao: {stats['total_before_dedup']}",
            f"Comentarios duplicados removidos: {stats['duplicates_removed']}",
            f"Comentarios apos deduplicacao: {stats['total_after_dedup']}",
            f"Comentarios no periodo 2018-2025: {stats['period_count']}",
            f"Comentarios com mais de {MIN_COMMENT_LENGTH} caracteres: {stats['length_count']}",
            f"Comentarios com pelo menos uma classe: {stats['labeled_count']}",
            f"Comentarios elegiveis finais: {stats['eligible_count']}",
            f"Comentarios elegiveis multirrotulo: {stats['eligible_multilabel_count']}",
            f"Comentarios elegiveis sem classe excluidos: {stats['eligible_unlabeled_count']}",
            "",
            "Contagens elegiveis por classe (>25 chars)",
        ]
    )

    for class_code in CLASS_COLUMNS:
        count = eligible_class_counts[class_code]
        share = count / eligible_memberships if eligible_memberships else 0
        lines.append(f"{class_code}: {count} ({share:.2%} das associacoes de classe elegiveis)")

    lines.extend(["", f"Cotas finais por classe ({target_per_class} por classe)"])
    for class_code in CLASS_COLUMNS:
        lines.append(f"{class_code}: {int(class_counts[class_code])} / {class_quotas[class_code]}")

    lines.extend(["", "Cotas finais por classe e ano"])
    lines.append("Classe\t" + "\t".join(str(year) for year in YEARS) + "\tTotal")
    for class_code in CLASS_COLUMNS:
        counts = [int(class_year_counts.loc[class_code, year]) for year in YEARS]
        expected_counts = [year_quotas[class_code][year] for year in YEARS]
        lines.append(f"{class_code}\t" + "\t".join(str(value) for value in counts) + f"\t{sum(counts)}")
        if counts != expected_counts:
            lines.append(f"Esperado {class_code}\t" + "\t".join(str(value) for value in expected_counts))

    lines.extend(["", "Validacoes"])
    lines.extend(validation_messages)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_directories()

    if args.full_per_class <= args.pilot_per_class:
        raise ValueError("--full-per-class precisa ser maior que --pilot-per-class.")

    full_class_quotas = build_equal_class_quotas(args.full_per_class)
    pilot_class_quotas = build_equal_class_quotas(args.pilot_per_class)

    comments, repaired = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    eligible, stats = prepare_eligible_comments(comments)

    full_year_quotas = build_year_quotas_from_labels(eligible, args.full_per_class)
    pilot_year_quotas = build_year_quotas_from_labels(eligible, args.pilot_per_class)

    full_sample = sample_without_replacement(
        source=eligible,
        year_quotas=full_year_quotas,
        random_seed=args.random_seed,
        use_sample_class=False,
    )
    pilot_sample = sample_without_replacement(
        source=full_sample,
        year_quotas=pilot_year_quotas,
        random_seed=args.random_seed,
        use_sample_class=True,
        random_offset=500000,
        insert_sample_id=False,
    )

    full_validation = validate_sample(
        full_sample,
        class_quotas=full_class_quotas,
        year_quotas=full_year_quotas,
        expected_size=args.full_per_class * len(CLASS_COLUMNS),
    )
    pilot_validation = validate_sample(
        pilot_sample,
        class_quotas=pilot_class_quotas,
        year_quotas=pilot_year_quotas,
        expected_size=args.pilot_per_class * len(CLASS_COLUMNS),
    )
    pilot_validation.extend(validate_pilot_subset(pilot_sample, full_sample))

    write_sample(full_sample, CLASS_LABELING_FULL_SAMPLE_CSV)
    write_sample(pilot_sample, CLASS_AGREEMENT_SAMPLE_CSV)

    CLASS_LABELING_FULL_SAMPLE_REPORT_TXT.write_text(
        build_report(
            title="Relatorio da amostra balanceada de rotulacao completa da RQ3",
            output_path=CLASS_LABELING_FULL_SAMPLE_CSV,
            sample=full_sample,
            eligible=eligible,
            stats=stats,
            validation_messages=full_validation,
            class_quotas=full_class_quotas,
            year_quotas=full_year_quotas,
            random_seed=args.random_seed,
            repaired_in_memory=repaired,
            target_per_class=args.full_per_class,
            extra_lines=["Desenho de anotacao: os mesmos 4800 comentarios serao rotulados por 3 rotuladores fixos."],
        ),
        encoding="utf-8",
    )
    CLASS_AGREEMENT_SAMPLE_REPORT_TXT.write_text(
        build_report(
            title="Relatorio da amostra piloto de concordancia da RQ3",
            output_path=CLASS_AGREEMENT_SAMPLE_CSV,
            sample=pilot_sample,
            eligible=eligible,
            stats=stats,
            validation_messages=pilot_validation,
            class_quotas=pilot_class_quotas,
            year_quotas=pilot_year_quotas,
            random_seed=args.random_seed,
            repaired_in_memory=repaired,
            target_per_class=args.pilot_per_class,
            extra_lines=[
                f"Subconjunto de: {CLASS_LABELING_FULL_SAMPLE_CSV}",
                "Uso: piloto de calibracao do protocolo; nao substitui a concordancia final dos 4800 comentarios.",
            ],
        ),
        encoding="utf-8",
    )

    print(f"Amostra completa salva em: {CLASS_LABELING_FULL_SAMPLE_CSV}")
    print(f"Amostra piloto salva em: {CLASS_AGREEMENT_SAMPLE_CSV}")
    print(f"Linhas da amostra completa: {len(full_sample)}")
    print(f"Linhas da amostra piloto: {len(pilot_sample)}")


if __name__ == "__main__":
    main()
