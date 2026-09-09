# Research Questions

This document describes the three research questions (RQs) that guide this study, the data used to answer each one, and the methodological decisions made.

---

## Context

Low-code and no-code (LCNC) development platforms have grown rapidly since 2018, democratising software creation for non-programmers. Despite their commercial growth, the discourse community around these tools — how users learn, what they struggle with, what they celebrate — remains understudied.

This research mines **YouTube comments** as a large-scale naturalistic corpus of user experience, capturing authentic reactions over a period of seven years (2018–2025).

---

## RQ1 — What are the main discussion topics in the LCNC YouTube community?

**Goal:** Identify the latent thematic structure of user comments using unsupervised topic modelling.

**Method:**
- Corpus: 227,811 English comments on 16,019 LCNC videos.
- Model: BERTopic with transformer embeddings (`all-MiniLM-L6-v2` or equivalent).
- Post-processing: 101 substantive topics curated and grouped into **11 macro-themes** by manual inspection of c-TF-IDF representative terms.
- Noise topics excluded: IDs -1, 0, 2, 13, 20, 21, 22, 29, 35, 36, 68, 92, 94, 102, 106, 107.

**Key macro-themes identified:**
1. Plataformas e ecossistemas (14.47%)
2. Desenvolvimento de aplicações, UI e lógica (16.96%)
3. Funcionalidades e casos de uso (8.80%)
4. Dados, arquivos e artefatos (6.96%)
5. Integração, backend e infraestrutura (7.98%)
6. Automação e workflows (3.73%)
7. IA, agentes e conversação (3.30%)
8. Aprendizagem, tutoriais e capacitação (9.41%)
9. Comunidade, engajamento e interação (18.42%)
10. Suporte, erros e limitações (6.51%)
11. Governança, custos, licenças e disponibilidade (3.46%)

**Temporal analysis:** Z-score heatmaps identify months where each topic surges significantly above its baseline, revealing adoption waves and event-driven spikes.

**Output artifacts:**
- `figures/fig04_evolucao_temporal_topicos.pdf` — temporal z-score heatmap
- `figures/fig05_heatmap_plataformas_topicos.pdf` — platform–topic co-occurrence heatmap
- `figures/fig06_evolucao_macrotemas.pdf` — macro-theme proportion over time
- `reports/06_analise_topicos.txt`
- `reports/07_macrotemas.txt`

---

## RQ2 — How does the LCNC discourse community evolve over time and across platforms?

**Goal:** Characterise the corpus temporally and by platform to understand growth patterns and platform-specific communities.

**Method:**
- Metrics computed monthly: videos, channels, comments, views, likes, unique users.
- Platform co-occurrence: the top-30 LCNC platforms are identified by name in video titles/descriptions, and their quarterly mention counts are normalised via z-score to reveal relative prominence over time.
- Period: January 2018 – September 2025.

**Key findings:**
- 16,019 videos across 4,852 channels.
- 260,791 comments in the analytical period.
- 196,604,871 total views.
- 118,744 unique commenters.
- Top platforms by video count: Power Apps, FileMaker, FlutterFlow, Bubble.io, AppSheet, n8n.

**Output artifacts:**
- `figures/fig02_metricas_corpus.pdf` — 6-panel corpus metrics time series
- `figures/fig03_heatmap_top30_plataformas.pdf` — platform prominence heatmap
- `reports/08_caracterizacao_corpus.txt`

---

## RQ3 — What types of user intent are expressed in LCNC YouTube comments?

**Goal:** Classify comments into six functional discourse categories using a lexical rule-based classifier validated by human annotation.

**Six classes:**

| ID | Label | Definition |
|----|-------|-----------|
| C1 | **Pergunta / pedido de ajuda** | Question, request for explanation, guidance, example, or tutorial |
| C2 | **Problema operacional** | Operational problem, error, failure, blocking issue, or concrete difficulty using a tool |
| C3 | **Elogio / testemunho positivo** | Praise, gratitude, positive testimonial, or recognition of utility |
| C4 | **Crítica / avaliação negativa** | Critique, frustration, objection, perceived cost problem, limitation, or negative evaluation |
| C5 | **Recomendação / divulgação** | Recommendation, referral to resources/alternatives, including links, channels, courses, communities |
| C6 | **Ruído / humor / informalidade** | Humour, evident noise, dominant informality, isolated links, or low-analytical comments |

**Annotation design:**
- **Pilot round:** 300 comments (50 per class), stratified by class, used for inter-rater agreement calibration.
- **Full round:** 4,800 comments (800 per class), 3 annotators.
- The pilot is a strict subset of the full sample.
- Final label: majority vote (2-of-3 annotators).
- Agreement metrics: Krippendorff's α and Fleiss's κ per class.

**Classifier precision audit:**
- 300 comments audited (50 per class) by a single expert reviewer using the Streamlit app (`app_auditoria_rq3.py`).
- **Macro-precision: 92.33%** (95% CI: overall high, see `reports/10_precisao_classes.txt`).
- Per-class: C1=96%, C2=92%, C3=96%, C4=88%, C5=92%, C6=90%.

**Running the annotation app:**
```bash
streamlit run 05_caracterizacao_corpus/app_auditoria_rq3.py
```

**See also:**
- `05_caracterizacao_corpus/MANUAL_ROTULACAO_RQ3.md` — annotator instructions
- `05_caracterizacao_corpus/PROTOCOLO_AUDITORIA_RQ3.md` — audit protocol

---

## Methodological Notes

### Language Detection Ensemble
Three tools were used in a majority-vote scheme to reduce false positives in English detection:
- **fastText** (Meta's `lid.176.bin`) — neural, fast, handles short texts well.
- **langdetect** (Google CLD port) — probabilistic, sensitive to text length.
- **langid** — Bayesian, robust to domain shifts.

A comment is labelled English if at least 2-of-3 tools agree. A video is included if >50% of its comments are majority-English. Sensitivity analysis of this threshold is documented in `reports/03_sensibilidade_limiar.txt` and `figures/fig01_sensibilidade_limiar_idioma.png`.

### Topic Noise Removal
Topics labelled -1 by BERTopic (outlier) and topics with IDs {0, 2, 13, 20, 21, 22, 29, 35, 36, 68, 92, 94, 102, 106, 107} were manually reviewed and excluded as noise or out-of-scope content before analysis.

### No Generative AI in Curation
The manual curation of topics into macro-themes was done entirely by inspection of c-TF-IDF representative terms and exemplar comments — no chat LLM or generative model was used in this decision.

---

## Target Publication

This research was prepared for submission to a peer-reviewed venue in software engineering / information systems. Manuscript prepared following **Elsevier's Information and Software Technology (IST)** double-anonymised peer review guidelines.
