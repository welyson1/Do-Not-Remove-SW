from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

import pandas as pd

FILES_DIR = Path(__file__).resolve().parent

STAGE_01_DIR = FILES_DIR / "01_entrada_bruta"
STAGE_02_DIR = FILES_DIR / "02_filtragem_tematica"
STAGE_03_DIR = FILES_DIR / "03_idioma"
STAGE_04_DIR = FILES_DIR / "04_topicos"
STAGE_04_RESULTS_DIR = STAGE_04_DIR / "05_resultados_topicos"
STAGE_05_DIR = FILES_DIR / "05_caracterizacao_corpus"

RAW_VIDEOS_CSV = STAGE_01_DIR / "01_videos_youtube_brutos.csv"
RAW_COMMENTS_CSV = STAGE_01_DIR / "02_comentarios_youtube_brutos.csv"

FILTERED_THEME_VIDEOS_CSV = STAGE_02_DIR / "01_videos_filtrados_tematica.csv"
FILTERED_THEME_COMMENTS_CSV = STAGE_02_DIR / "02_comentarios_filtrados_tematica.csv"
FILTERED_THEME_REPORT_TXT = STAGE_02_DIR / "03_relatorio_filtragem_tematica.txt"

LANGUAGE_MODEL_BIN = STAGE_03_DIR / "01_modelo_fasttext_lid176.bin"
LANGUAGE_COMPARISON_CSV = STAGE_03_DIR / "02_comentarios_comparacao_idioma.csv"
LANGUAGE_COMPARISON_REPORT_TXT = STAGE_03_DIR / "03_relatorio_comparacao_idioma.txt"
LANGUAGE_THRESHOLD_REPORT_TXT = STAGE_03_DIR / "04_relatorio_sensibilidade_limiar.txt"
LANGUAGE_THRESHOLD_PLOT_PNG = STAGE_03_DIR / "05_grafico_sensibilidade_limiar.png"
ENGLISH_VIDEOS_CSV = STAGE_03_DIR / "06_videos_corpus_ingles.csv"
ENGLISH_COMMENTS_CSV = STAGE_03_DIR / "07_comentarios_corpus_ingles.csv"
ENGLISH_COMMENTS_PERIOD_CSV = STAGE_03_DIR / "08_comentarios_corpus_ingles_periodo.csv"
ENGLISH_FILTER_REPORT_TXT = STAGE_03_DIR / "09_relatorio_filtragem_idioma.txt"

LEGACY_TOPIC_COMMENTS_CSV = STAGE_04_DIR / "00_comentarios_com_topicos_legado.csv"
LEGACY_TOPIC_DESCRIPTIONS_CSV = STAGE_04_DIR / "00_descricoes_topicos_legado.csv"
TOPIC_BASE_COMMENTS_CSV = STAGE_04_DIR / "01_comentarios_base_topicos.csv"
TOPIC_ALIGNED_COMMENTS_CSV = STAGE_04_DIR / "02_comentarios_com_topicos.csv"
TOPIC_DESCRIPTIONS_CSV = STAGE_04_DIR / "03_descricoes_topicos.csv"
TOPIC_COVERAGE_REPORT_TXT = STAGE_04_DIR / "04_relatorio_cobertura_topicos.txt"
TOPIC_ANALYSIS_REPORT_TXT = STAGE_04_RESULTS_DIR / "01_relatorio_analise_topicos.txt"
TOPIC_TEMPORAL_PDF = STAGE_04_RESULTS_DIR / "03_figura_evolucao_temporal_topicos.pdf"
TOPIC_PLATFORM_HEATMAP_PDF = STAGE_04_RESULTS_DIR / "04_figura_heatmap_plataformas_topicos.pdf"
MACROTHEME_MAPPING_CSV = STAGE_04_DIR / "04_mapeamento_macrotemas.csv"
MACROTHEME_TEMPORAL_PNG = STAGE_04_RESULTS_DIR / "05_figura_evolucao_macrotemas.png"
MACROTHEME_TEMPORAL_PDF = STAGE_04_RESULTS_DIR / "06_figura_evolucao_macrotemas.pdf"
MACROTHEME_REPORT_TXT = STAGE_04_RESULTS_DIR / "08_relatorio_macrotemas.txt"
MACROTHEME_LATEX_TABLE = STAGE_04_RESULTS_DIR / "09_tabela_mapeamento_macrotemas.tex"

CHARACTERIZATION_METRICS_PDF = STAGE_05_DIR / "01_figura_metricas_corpus.pdf"
CHARACTERIZATION_PLATFORM_HEATMAP_PDF = STAGE_05_DIR / "02_figura_heatmap_top30_plataformas.pdf"
CHARACTERIZATION_REPORT_TXT = STAGE_05_DIR / "03_relatorio_caracterizacao_corpus.txt"
CLASS_SANITY_REPORT_TXT = STAGE_05_DIR / "04_relatorio_teste_sanidade_classes.txt"
CLASS_SANITY_SAMPLE_CSV = STAGE_05_DIR / "05_amostra_teste_sanidade_classes.csv"
CLASS_AUDIT_TEMPLATE_CSV = STAGE_05_DIR / "06_auditoria_precisao_classes.csv"
CLASS_PRECISION_REPORT_TXT = STAGE_05_DIR / "07_relatorio_precisao_classes.txt"
CLASS_LABELING_FULL_SAMPLE_CSV = STAGE_05_DIR / "10_amostra_rotulacao_completa_4800_comentarios.csv"
CLASS_LABELING_FULL_SAMPLE_REPORT_TXT = STAGE_05_DIR / "10_relatorio_amostra_rotulacao_completa_4800.txt"
CLASS_AGREEMENT_SAMPLE_CSV = STAGE_05_DIR / "11_amostra_piloto_concordancia_300_comentarios.csv"
CLASS_AGREEMENT_SAMPLE_REPORT_TXT = STAGE_05_DIR / "11_relatorio_amostra_piloto_concordancia_300.txt"
CLASS_AGREEMENT_METRICS_REPORT_TXT = STAGE_05_DIR / "12_relatorio_concordancia_interanotador_rq3.txt"
CLASS_TEST_LABELS_CONSOLIDATED_CSV = STAGE_05_DIR / "13_rotulos_piloto_concordancia_consolidado.csv"
CLASS_FINAL_LABELS_CSV = STAGE_05_DIR / "14_rotulos_rq3_final_4800.csv"
CLASS_LABEL_MERGE_REPORT_TXT = STAGE_05_DIR / "13_relatorio_unificacao_rotulos_rq3.txt"
CLASS_STRATIFIED_SAMPLE_CSV = STAGE_05_DIR / "10_amostra_estratificada_4000_comentarios.csv"
CLASS_STRATIFIED_SAMPLE_REPORT_TXT = STAGE_05_DIR / "10_relatorio_amostra_estratificada_4000.txt"

STUDY_START = pd.Timestamp("2018-01-01")
STUDY_END = pd.Timestamp("2025-09-30")
STUDY_END_EXCLUSIVE = pd.Timestamp("2025-10-01")

LANGUAGE_VIDEO_THRESHOLD = 50.0
MIN_TOPIC_COMMENT_LENGTH = 15

COMMENT_BASE_COLUMNS = [
    "video_id",
    "comment_id",
    "author",
    "author_profile_image_url",
    "author_channel_url",
    "author_channel_id",
    "comment",
    "published_at",
    "updated_at",
    "like_count",
    "viewer_rating",
    "can_rate",
    "is_reply",
    "parent_id",
    "channel_id",
]

LANGUAGE_COLUMNS = ["lang_fasttext", "lang_langdetect", "lang_langid"]
LANGUAGE_FILTER_COLUMNS = LANGUAGE_COLUMNS + ["en_agreement_count", "is_majority_en"]

LOW_CODE_TERMS = [
    "OutSystems",
    "Appian Platform",
    "Power Apps",
    "Mendix Platform",
    "Salesforce Platform",
    "Quickbase",
    "APEX Application Development",
    "Zoho Creator",
    "App Engine",
    "Kissflow",
    "Pega Platform",
    "Retool",
    "Google App Maker",
    "GeneXus",
    "Studio Creatio",
    "BRYTER",
    "Jitterbit App Builder",
    "Kintone",
    "Joget DX",
    "Quixy",
    "Neptune DXP",
    "AgilePoint",
    "NewgenONE Platform",
    "Astro Zero",
    "TrueContext",
    "WaveMaker",
    "WEBCON BPS",
    "BettyBlocks Platform",
    "Bubble.io",
    "HCL Volt MX",
    "Caspio",
    "AuraQuantic",
    "SAP Build",
    "MobileFrame",
    "WEM No-Code Platform",
    "Definesys DeCod",
    "Bizagi",
    "eLegere",
    "TrackVia",
    "Unqork",
    "Zvolv",
    "UI Bakery",
    "DronaHQ",
    "FileMaker",
    "Boomi",
    "ClickPaaS Platform",
    "Mingdao",
    "3forge",
    "RunMyProcess",
    "Axpert",
    "Liberty Create",
    "Simplicite Software",
    "Superblocks",
    "OpenText AppWorks",
    "WebRatio Platform",
    "Comidor Platform",
    "Vantiq",
    "CITSmart",
    "OnBase",
    "Thinkwise Platform",
    "YiDA",
    "AppMaster.io",
    "CALS Platform",
    "Ninox Low",
    "Reasy",
    "sterlo",
    "The m-Power Development Platform",
    "ToolJet",
    "Xpoda",
    "Zuilder",
    "Aapli",
    "ACTICO",
    "ActiveBatch",
    "Airtool",
    "AMODIT",
    "AmperAXP",
    "Appenate",
    "Appsynergy",
    "AppWay",
    "Aqtra Platform",
    "Archman NAVIGATOR",
    "AWS Amplify Studio",
    "Baserow",
    "BIC Platform",
    "Bimser Synergy",
    "BMC Helix Platform",
    "Canonic",
    "CGaaS.ai",
    "ClaySys AppForms",
    "CMW Platform",
    "Codease",
    "Codespell.ai",
    "CODIUM",
    "Convertigo Platform",
    "Dynamics 365 Business Central",
    "EIQ Platform",
    "FAB Builder",
    "Ferryt",
    "Flutterflow",
    "GenCodex",
    "Genero Enterprise",
    "Graphite Studio",
    "HokuApps Platform",
    "HumongouS.io",
    "Ideagen Smartforms",
    "Infodator HAP",
    "JET Workflow",
    "Katonic Generative AI Platform",
    "KnowledgeKube aPaaS",
    "Kuika",
    "Maestro Blocks",
    "Mekari Officeless",
    "Microsoft Power Platform",
    "N-AOS Low(No)-Code Development Platform",
    "Netigma",
    "Nocoly HAP",
    "Olympe Platform",
    "Onventis Workflow Automation",
    "Oro Platform",
    "Perfeqta",
    "Pipefy",
    "Rintagi",
    "Servoy Enterprise",
    "Simplifier",
    "Slingr",
    "Structr",
    "TAAP Accelerate",
    "TerraGo Magic",
    "Tonkean Process Experience Platform",
    "Tyler Application Platform",
    "UCBOS",
    "UnityBaseNext",
    "USoft",
    "Vahana Cloud",
    "Verj.io",
    "VisionX",
    "XOne Platform",
    "n8n",
    "AppSheet",
    "budibase",
    "uibakery",
    "appsmith",
]

EXCLUSION_TERMS = [
    "music",
    "dance",
    "song",
    "lyrics",
    "official video",
    "choreography",
    "vlog",
    "gaming",
    "makeup",
    "official music video",
    "remix",
    "cover",
    "zumba",
    "aliexpress",
    "funny",
    "noodles",
    "eating",
    "truk",
    "cooking",
    "food",
    "konsert 2017",
    "eat",
    "seafood",
    "urbanism",
    "shaun johnson",
    "gta 5",
    "guitar",
]

TOPICS_TO_REMOVE = [0, 2, 13, 20, 21, 22, 29, 35, 36, 68, 92, 94, 102, 106, 107]

MACROTHEME_ORDER = [
    "Plataformas e ecossistemas",
    "Desenvolvimento de aplicações, UI e lógica",
    "Funcionalidades e casos de uso",
    "Dados, arquivos e artefatos",
    "Integração, backend e infraestrutura",
    "Automação e workflows",
    "IA, agentes e conversação",
    "Aprendizagem, tutoriais e capacitação",
    "Comunidade, engajamento e interação",
    "Suporte, erros e limitações",
    "Governança, custos, licenças e disponibilidade",
]

PLATFORM_TERMS = sorted(set(LOW_CODE_TERMS), key=lambda value: (-len(value), value.lower()))


def ensure_directories() -> None:
    for directory in [
        STAGE_01_DIR,
        STAGE_02_DIR,
        STAGE_03_DIR,
        STAGE_04_DIR,
        STAGE_04_RESULTS_DIR,
        STAGE_05_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def study_period_mask(series: pd.Series) -> pd.Series:
    timestamps = series
    if not pd.api.types.is_datetime64_any_dtype(series):
        timestamps = parse_utc_datetime(series)
    return (timestamps >= STUDY_START) & (timestamps < STUDY_END_EXCLUSIVE)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def build_terms_regex(terms: list[str]) -> re.Pattern[str]:
    unique_terms = sorted(set(terms), key=lambda value: (-len(value), value.lower()))
    parts = [rf"(?<!\w){re.escape(term)}(?!\w)" for term in unique_terms]
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def build_term_patterns(terms: list[str]) -> dict[str, re.Pattern[str]]:
    return {term: re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE) for term in terms}


PLATFORM_PATTERNS = build_term_patterns(PLATFORM_TERMS)


def extract_matching_terms(text: object, patterns: dict[str, re.Pattern[str]] | None = None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    active_patterns = patterns or PLATFORM_PATTERNS
    found_terms = [term for term, pattern in active_patterns.items() if pattern.search(normalized)]
    return sorted(set(found_terms), key=lambda value: (value.lower(), value))


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("low_memory", None)
        fallback_kwargs["engine"] = "python"
        fallback_kwargs.setdefault("on_bad_lines", "skip")
        return pd.read_csv(path, **fallback_kwargs)


def _combine_comment_fragments(fragments: list[list[str]]) -> list[str]:
    flattened = list(itertools.chain.from_iterable(fragments))
    if len(flattened) < len(COMMENT_BASE_COLUMNS):
        raise ValueError(f"Registro incompleto durante o reparo do CSV: {fragments[:2]}")
    prefix = flattened[:6]
    suffix = flattened[-8:]
    comment_chunks = flattened[6:-8]
    comment_text = "\n".join(comment_chunks) if comment_chunks else ""
    rebuilt_row = prefix + [comment_text] + suffix
    if len(rebuilt_row) != len(COMMENT_BASE_COLUMNS):
        raise ValueError(f"Registro reconstruido com tamanho invalido: {len(rebuilt_row)}")
    return rebuilt_row


def repair_split_comment_tail_csv(path: Path, tail_columns: list[str]) -> pd.DataFrame:
    header_full = COMMENT_BASE_COLUMNS + tail_columns
    header_split_tail = [""] + tail_columns
    rows: list[list[str]] = []
    pending_fragments: list[list[str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return pd.DataFrame(columns=header_full)

        second_row = next(reader, None)
        buffered_rows: list[list[str]] = []

        if first_row == header_full:
            if second_row is not None:
                buffered_rows.append(second_row)
        elif first_row == COMMENT_BASE_COLUMNS and second_row == header_split_tail:
            pass
        else:
            raise ValueError(f"Cabecalho inesperado em {path}")

        for row in itertools.chain(buffered_rows, reader):
            if not row:
                continue
            if len(row) == len(header_full):
                rows.append(row)
                pending_fragments = []
                continue
            if row[0] == "" and len(row) == len(tail_columns) + 1:
                if not pending_fragments:
                    continue
                base_row = _combine_comment_fragments(pending_fragments)
                rows.append(base_row + row[1:])
                pending_fragments = []
                continue
            pending_fragments.append(row)

    if pending_fragments:
        raise ValueError(f"Registro pendente sem fechamento em {path}")

    return pd.DataFrame(rows, columns=header_full)


def load_language_comment_file(path: Path, tail_columns: list[str]) -> tuple[pd.DataFrame, bool]:
    full_columns = COMMENT_BASE_COLUMNS + tail_columns
    try:
        dataframe = read_csv_with_fallback(path, low_memory=False)
        if all(column in dataframe.columns for column in full_columns):
            return dataframe[full_columns].copy(), False
    except Exception:
        pass
    repaired_dataframe = repair_split_comment_tail_csv(path, tail_columns)
    return repaired_dataframe, True
