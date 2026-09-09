# Pipeline Overview

This document describes each stage of the research pipeline, including inputs, outputs, and how to run each step.

---

## Architecture

```
Raw YouTube Data (API)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 01 — Raw Input (01_entrada_bruta/)                                  │
│  • 01_videos_youtube_brutos.csv   — all collected videos                   │
│  • 02_comentarios_youtube_brutos.csv — all collected comments              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 02 — Thematic Filtering (02_filtragem_tematica/)                    │
│  Script: etapa_01_filtrar_tematica.py                                      │
│  Keeps only videos whose title/description matches a curated list of       │
│  low-code and no-code platform names. Removes off-topic content.           │
│  Output: 29,669 videos · 612,760 comments                                 │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 03 — Language Detection (03_idioma/)                                │
│  Scripts: etapa_02_detectar_idioma.py                                      │
│           etapa_03_analisar_limiar.py                                      │
│           etapa_04_filtrar_corpus_ingles.py                                │
│                                                                             │
│  Runs three language classifiers in ensemble:                              │
│    · fastText (lid.176.bin) — Meta's neural language identifier            │
│    · langdetect — Google Compact Language Detector port                    │
│    · langid — Bayesian language identifier                                 │
│                                                                             │
│  A video is included if >50% of its comments are classified as English     │
│  by majority vote (2-of-3 tools).                                          │
│  Output: 16,019 videos · 265,301 comments (analytical period: 260,791)    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 04 — Topic Modelling (04_topicos/)                                  │
│  Scripts: etapa_05_preparar_base_topicos.py                                │
│           etapa_06_alinhar_topicos_legado.py                               │
│           etapa_07_analisar_topicos.py                                     │
│           etapa_08_analisar_macrotemas.py                                  │
│           gerar_tabelas_latex_topicos.py                                   │
│                                                                             │
│  BERTopic was run externally and produced a topic assignment for each      │
│  comment (stored in 00_comentarios_com_topicos_legado.csv).                │
│  These scripts:                                                             │
│    1. Prepare a clean text base for topic matching (min 15 chars,          │
│       deduplicated by normalised text).                                    │
│    2. Align comments to BERTopic topic IDs by exact normalised text match. │
│    3. Analyse temporal trends and platform co-occurrence (z-score          │
│       heatmaps).                                                            │
│    4. Curate 101 substantive topics into 11 macro-themes.                  │
│  Output: 227,811 comments with topics · 11 macro-themes · 6 figures       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 05 — Corpus Characterisation & RQ3 Labelling (05_caracterizacao/)   │
│  Scripts: etapa_09_caracterizar_corpus.py  ← corpus-level metrics         │
│           etapa_10_testar_sanidade_classes.py                              │
│           etapa_11_preparar_auditoria_precisao_classes.py                  │
│           etapa_12_calcular_precisao_classes.py                            │
│           etapa_13_amostrar_comentarios_rq3.py                             │
│           etapa_14_calcular_concordancia_rq3.py                            │
│           etapa_14_unificar_rotulos_rq3.py                                 │
│           app_auditoria_rq3.py  ← Streamlit auditing app                  │
│                                                                             │
│  Produces corpus statistics (videos, channels, views, comments over time)  │
│  and orchestrates the multi-label human annotation round for RQ3 with 3   │
│  annotators, computing Krippendorff's α and Fleiss's κ.                   │
│  Macro-precision of the dictionary classifier: 92.33% (300 audited items). │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Running the Pipeline

### Prerequisites

1. **Python ≥ 3.10** installed.
2. All dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **FastText model** downloaded:
   ```bash
   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
        -O 03_idioma/01_modelo_fasttext_lid176.bin
   ```
   On Windows (PowerShell):
   ```powershell
   Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" `
                     -OutFile "03_idioma\01_modelo_fasttext_lid176.bin"
   ```
4. **Raw data** placed in `01_entrada_bruta/`:
   - `01_videos_youtube_brutos.csv`
   - `02_comentarios_youtube_brutos.csv`
   
   See [Data Availability](../README.md#data-availability) in the main README.

5. **BERTopic legacy files** (Stage 04 prerequisite):
   - `04_topicos/00_comentarios_com_topicos_legado.csv`
   - `04_topicos/00_descricoes_topicos_legado.csv`
   
   See [Reproducing BERTopic](#reproducing-bertopic) below.

---

### Running All Stages at Once

```bash
# From the repository root
python pipeline_01_executar.py
```

### Running a Specific Range of Stages

```bash
python pipeline_01_executar.py --from-stage 01 --to-stage 04
```

### Running a Single Stage Directly

```bash
python 02_filtragem_tematica/etapa_01_filtrar_tematica.py
python 03_idioma/etapa_02_detectar_idioma.py
python 03_idioma/etapa_03_analisar_limiar.py
python 03_idioma/etapa_04_filtrar_corpus_ingles.py
python 04_topicos/etapa_05_preparar_base_topicos.py
python 04_topicos/etapa_06_alinhar_topicos_legado.py
python 04_topicos/etapa_07_analisar_topicos.py
python 04_topicos/etapa_08_analisar_macrotemas.py
python 05_caracterizacao_corpus/etapa_09_caracterizar_corpus.py
```

---

## Reproducing BERTopic

The BERTopic step (topic modelling) was run as a standalone notebook/script not included in this automated pipeline. The model assigns topic IDs to each comment.

To reproduce:
1. Use the English-filtered comment base: `04_topicos/01_comentarios_base_topicos.csv`
2. Run BERTopic with `language="english"`, `min_topic_size=50`, and an embedding model of your choice (e.g., `all-MiniLM-L6-v2`).
3. Export the resulting topic assignments to `04_topicos/00_comentarios_com_topicos_legado.csv` with columns: `comment_id`, `video_id`, `comment`, `topic_id`, `Name`, `Representation`.
4. Export topic descriptions to `04_topicos/00_descricoes_topicos_legado.csv` with columns: `Topic`, `Count`, `Name`, `Representation`, `Representative_Docs`.
5. Run `python 04_topicos/etapa_06_alinhar_topicos_legado.py` to align.

See also the [BERTopic documentation](https://maartengr.github.io/BERTopic/).

---

## Stage Inputs & Outputs Summary

| Stage | Script | Key Input | Key Output |
|-------|--------|-----------|------------|
| 01 | *(raw collection)* | YouTube Data API | `01_entrada_bruta/` CSVs |
| 02 | `etapa_01_filtrar_tematica.py` | Raw CSVs | Filtered CSVs + report |
| 03a | `etapa_02_detectar_idioma.py` | Filtered comments | Language labels CSV |
| 03b | `etapa_03_analisar_limiar.py` | Language labels | Threshold report + figure |
| 03c | `etapa_04_filtrar_corpus_ingles.py` | Language labels | English corpus CSVs |
| 04a | `etapa_05_preparar_base_topicos.py` | English period CSV | Topic base CSV |
| 04b | `etapa_06_alinhar_topicos_legado.py` | Topic base + BERTopic | Aligned CSV + descriptions |
| 04c | `etapa_07_analisar_topicos.py` | Aligned comments | Heatmap figures + report |
| 04d | `etapa_08_analisar_macrotemas.py` | Aligned + macro-mapping | Macro-theme figures + report |
| 05a | `etapa_09_caracterizar_corpus.py` | English corpus | Metrics figures + report |
| 05b | `etapa_10_testar_sanidade_classes.py` | English period CSV | Sanity test report |
| 05c | `etapa_11_preparar_auditoria_precisao_classes.py` | English corpus | Audit template CSV |
| 05d | `etapa_12_calcular_precisao_classes.py` | Audited CSV | Precision report |
| 05e | `etapa_13_amostrar_comentarios_rq3.py` | English period CSV | Labelling sample CSVs |
| 05f | `etapa_14_unificar_rotulos_rq3.py` | Annotator label files | Unified labels + agreement metrics |

---

## Configuration

All paths, constants, and shared utilities live in `pipeline_00_config.py`.

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STUDY_START` | 2018-01-01 | Beginning of analytical period |
| `STUDY_END` | 2025-09-30 | End of analytical period |
| `LANGUAGE_VIDEO_THRESHOLD` | 50.0 | Min % English comments to keep a video |
| `MIN_TOPIC_COMMENT_LENGTH` | 15 | Min normalised characters for topic base |
| `TOPICS_TO_REMOVE` | [0, 2, 13, …] | Noise topic IDs excluded from analysis |

---

## Figures

All publication-ready figures are in `figures/`. See the table in the main [README.md](../README.md#figures).
