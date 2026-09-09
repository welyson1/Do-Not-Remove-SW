# Research Questions

This document describes the three research questions (RQs) that guide this study, the data used to answer each one, and the methodological decisions made.

---

## Context

Low-code and no-code (LCNC) development platforms have grown rapidly since 2018, democratising software creation for non-programmers. Despite their commercial growth, the discourse community around these tools — how users learn, what they struggle with, what they celebrate — remains understudied.

This research mines **YouTube comments** as a large-scale naturalistic corpus of user experience, capturing authentic reactions over a period of seven years (2018–2025).

---

## RQ1 — What are the scale, temporal distribution, public interaction, and platform coverage of the retrieved video-centered corpus in low-code ecosystems?

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

## RQ2 — What software engineering and learning concerns are represented in the video-centered corpus and how are they distributed across development activities?

**Goal:** Identify the latent thematic structure of user comments using unsupervised topic modelling and map them to development activities.

**Method:**
- Corpus: 227,811 English comments on 16,019 LCNC videos.
- Model: BERTopic with transformer embeddings (`all-MiniLM-L6-v2` or `all-mpnet-base-v2` equivalent).
- Post-processing: 101 substantive topics curated and grouped into **11 macro-themes** and mapped to **6 development activities**.
- Noise topics excluded manually based on c-TF-IDF terms.

**Key development activities identified (Share of technical texts):**
1. Construction and implementation (39.10%)
2. Platform and capability understanding (28.53%)
3. Knowledge acquisition and onboarding (11.53%)
4. Orchestration and emerging capability (8.61%)
5. Maintenance and troubleshooting (7.98%)
6. Governance, licensing and cost (4.24%)

*(Note: Non-technical interaction, such as praise and social engagement, accounted for 18.42% of all texts and was separated from the technical share.)*

**Output artifacts:**
- `reports/06_analise_topicos.txt`
- `reports/07_macrotemas.txt`

---

## RQ3 — How does the composition of these concerns change over the observation window and what does this trajectory indicate about the boundary of low-code ecosystems in public discussion?

**Goal:** Understand how the identified topics and macro-themes evolve over the 7-year period, revealing shifts in community focus (e.g., the rise of AI agents and workflow automation).

**Method:**
- Temporal analysis: Monthly composition of the 11 macrothemes among distinct texts.
- Z-score heatmaps identify months where each topic surges significantly above its baseline, revealing adoption waves and event-driven spikes.
- Platform–topic co-occurrence matrices map specific concerns to particular platforms.

**Output artifacts:**
- `figures/fig04_evolucao_temporal_topicos.pdf` — temporal z-score heatmap
- `figures/fig05_heatmap_plataformas_topicos.pdf` — platform–topic co-occurrence heatmap
- `figures/fig06_evolucao_macrotemas.pdf` — macro-theme proportion over time

---

## Methodological Notes

### Language Detection Ensemble
Three tools were used in a majority-vote scheme to reduce false positives in English detection:
- **fastText** (Meta's `lid.176.bin`) — neural, fast, handles short texts well.
- **langdetect** (Google CLD port) — probabilistic, sensitive to text length.
- **langid** — Bayesian, robust to domain shifts.

A comment is labelled English if at least 2-of-3 tools agree. A video is included if >50% of its comments are majority-English. Sensitivity analysis of this threshold is documented in `reports/03_sensibilidade_limiar.txt` and `figures/fig01_sensibilidade_limiar_idioma.png`.

### Topic Noise Removal
Residual, incoherent, generic, and out-of-scope clusters were excluded. The manual curation of topics into macro-themes was done entirely by inspection of c-TF-IDF representative terms and exemplar comments — no chat LLM or generative model was used in this decision.

---

## Target Publication

This research was prepared for submission to a peer-reviewed venue in software engineering / information systems. Manuscript prepared following **Elsevier's Information and Software Technology (IST)** double-anonymised peer review guidelines.
