from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    FILTERED_THEME_COMMENTS_CSV,
    LANGUAGE_COLUMNS,
    LANGUAGE_COMPARISON_CSV,
    LANGUAGE_COMPARISON_REPORT_TXT,
    LANGUAGE_MODEL_BIN,
    ensure_directories,
    normalize_text,
    read_csv_with_fallback,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import fasttext

    fasttext.FastText.eprint = lambda _text: None
    FASTTEXT_MODEL = fasttext.load_model(str(LANGUAGE_MODEL_BIN))
    FASTTEXT_AVAILABLE = True
except Exception:
    FASTTEXT_MODEL = None
    FASTTEXT_AVAILABLE = False

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid

    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


def detect_fasttext(text: object) -> str:
    if not FASTTEXT_AVAILABLE:
        return "unavailable"
    cleaned = normalize_text(text)
    if not cleaned:
        return "empty"
    try:
        prediction = FASTTEXT_MODEL.predict(cleaned, k=1)
        return prediction[0][0].replace("__label__", "")
    except Exception:
        return "error"


def detect_langdetect(text: object) -> str:
    if not LANGDETECT_AVAILABLE:
        return "unavailable"
    cleaned = normalize_text(text)
    if not cleaned:
        return "empty"
    try:
        return detect(cleaned)
    except Exception:
        return "error"


def detect_langid(text: object) -> str:
    if not LANGID_AVAILABLE:
        return "unavailable"
    cleaned = normalize_text(text)
    if not cleaned:
        return "empty"
    try:
        language, _score = langid.classify(cleaned)
        return language
    except Exception:
        return "error"


def build_report(dataframe) -> str:
    total_comments = len(dataframe)
    lines = [
        "Relatorio de comparacao de idioma",
        "=" * 72,
        f"Total de comentarios analisados: {total_comments}",
        "",
    ]

    for tool_name, column_name in [
        ("fastText", "lang_fasttext"),
        ("langdetect", "lang_langdetect"),
        ("langid", "lang_langid"),
    ]:
        counts = dataframe[column_name].value_counts()
        en_count = int(counts.get("en", 0))
        pt_count = int(counts.get("pt", 0))
        lines.extend(
            [
                f"{tool_name}",
                "-" * 32,
                f"Ingles (en): {en_count} ({(en_count / total_comments) * 100:.2f}%)",
                f"Portugues (pt): {pt_count} ({(pt_count / total_comments) * 100:.2f}%)",
                "Top 10 idiomas:",
            ]
        )
        for language, count in counts.head(10).items():
            lines.append(f"  {language}: {count} ({(count / total_comments) * 100:.2f}%)")
        lines.append("")

    dataframe = dataframe.copy()
    dataframe["all_agree"] = (
        (dataframe["lang_fasttext"] == dataframe["lang_langdetect"])
        & (dataframe["lang_langdetect"] == dataframe["lang_langid"])
    )
    dataframe["majority_en"] = (
        (dataframe["lang_fasttext"] == "en").astype(int)
        + (dataframe["lang_langdetect"] == "en").astype(int)
        + (dataframe["lang_langid"] == "en").astype(int)
    ) >= 2
    dataframe["majority_pt"] = (
        (dataframe["lang_fasttext"] == "pt").astype(int)
        + (dataframe["lang_langdetect"] == "pt").astype(int)
        + (dataframe["lang_langid"] == "pt").astype(int)
    ) >= 2
    majority_language = dataframe[LANGUAGE_COLUMNS].mode(axis=1)[0]
    agreement_count = sum((dataframe[column] == majority_language).astype(int) for column in LANGUAGE_COLUMNS)

    lines.extend(
        [
            "Concordancia entre ferramentas",
            "-" * 32,
            f"Total com concordancia completa: {int(dataframe['all_agree'].sum())} ({(dataframe['all_agree'].mean()) * 100:.2f}%)",
            f"Maioria em ingles (2 de 3): {int(dataframe['majority_en'].sum())} ({(dataframe['majority_en'].mean()) * 100:.2f}%)",
            f"Maioria em portugues (2 de 3): {int(dataframe['majority_pt'].sum())} ({(dataframe['majority_pt'].mean()) * 100:.2f}%)",
            "",
            "Distribuicao do nivel de concordancia:",
        ]
    )
    for total_agreement in [3, 2, 1]:
        count = int((agreement_count == total_agreement).sum())
        lines.append(f"  {total_agreement} ferramentas: {count} ({(count / total_comments) * 100:.2f}%)")

    lines.extend(
        [
            "",
            "Casos de discordancia relevantes:",
            f"  fastText=pt e outras duas=en: {int(((dataframe['lang_fasttext'] == 'pt') & (dataframe['lang_langdetect'] == 'en') & (dataframe['lang_langid'] == 'en')).sum())}",
            f"  fastText=en e outras duas=pt: {int(((dataframe['lang_fasttext'] == 'en') & (dataframe['lang_langdetect'] == 'pt') & (dataframe['lang_langid'] == 'pt')).sum())}",
            f"  langdetect discorda das outras duas: {int(((dataframe['lang_fasttext'] == dataframe['lang_langid']) & (dataframe['lang_langdetect'] != dataframe['lang_fasttext'])).sum())}",
            f"  langid discorda das outras duas: {int(((dataframe['lang_fasttext'] == dataframe['lang_langdetect']) & (dataframe['lang_langid'] != dataframe['lang_fasttext'])).sum())}",
            "",
            "Regra operacional sugerida: usar maioria simples de 2 entre 3 ferramentas.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_directories()

    if not LANGUAGE_MODEL_BIN.exists():
        raise SystemExit(f"Modelo fastText nao encontrado em: {LANGUAGE_MODEL_BIN}")

    comments = read_csv_with_fallback(FILTERED_THEME_COMMENTS_CSV, low_memory=False, on_bad_lines="skip")
    if "comment" not in comments.columns:
        raise SystemExit("A coluna 'comment' nao foi encontrada no arquivo de comentarios filtrados.")

    comments = comments.copy()
    comments["comment"] = comments["comment"].fillna("")
    total_comments = len(comments)
    print(f"Processando {total_comments} comentarios...")

    comments["lang_fasttext"] = [detect_fasttext(comment) for comment in tqdm(comments["comment"], total=total_comments, desc="fastText")]
    comments["lang_langdetect"] = [detect_langdetect(comment) for comment in tqdm(comments["comment"], total=total_comments, desc="langdetect")]
    comments["lang_langid"] = [detect_langid(comment) for comment in tqdm(comments["comment"], total=total_comments, desc="langid")]

    comments.to_csv(LANGUAGE_COMPARISON_CSV, index=False)
    LANGUAGE_COMPARISON_REPORT_TXT.write_text(build_report(comments), encoding="utf-8")

    print(f"Arquivo salvo em: {LANGUAGE_COMPARISON_CSV}")
    print(f"Relatorio salvo em: {LANGUAGE_COMPARISON_REPORT_TXT}")


if __name__ == "__main__":
    main()
