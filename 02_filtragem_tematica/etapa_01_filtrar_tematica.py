from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    EXCLUSION_TERMS,
    FILTERED_THEME_COMMENTS_CSV,
    FILTERED_THEME_REPORT_TXT,
    FILTERED_THEME_VIDEOS_CSV,
    LOW_CODE_TERMS,
    RAW_COMMENTS_CSV,
    RAW_VIDEOS_CSV,
    build_terms_regex,
    ensure_directories,
    parse_utc_datetime,
    read_csv_with_fallback,
    study_period_mask,
)


def main() -> None:
    ensure_directories()

    print("Carregando videos brutos...")
    videos = read_csv_with_fallback(RAW_VIDEOS_CSV, low_memory=False, on_bad_lines="skip")
    print("Carregando comentarios brutos...")
    comments = read_csv_with_fallback(RAW_COMMENTS_CSV, low_memory=False, on_bad_lines="skip")

    stats: dict[str, int] = {}
    stats["videos_total_bruto"] = len(videos)
    videos = videos.drop_duplicates(subset=["video_id"]).copy()
    stats["videos_apos_deduplicacao"] = len(videos)
    stats["videos_removidos_duplicatas"] = stats["videos_total_bruto"] - stats["videos_apos_deduplicacao"]

    videos["published_at"] = parse_utc_datetime(videos["published_at"])
    videos = videos.dropna(subset=["published_at"]).copy()
    stats["videos_apos_limpeza_data"] = len(videos)
    stats["videos_removidos_data_invalida"] = stats["videos_apos_deduplicacao"] - stats["videos_apos_limpeza_data"]

    videos = videos.loc[study_period_mask(videos["published_at"])].copy()
    stats["videos_apos_filtro_data"] = len(videos)
    stats["videos_removidos_fora_data"] = stats["videos_apos_limpeza_data"] - stats["videos_apos_filtro_data"]

    videos["texto_busca"] = videos["title"].fillna("") + " " + videos["description"].fillna("")
    positive_pattern = build_terms_regex(LOW_CODE_TERMS)
    negative_pattern = build_terms_regex(EXCLUSION_TERMS)

    positive_mask = videos["texto_busca"].str.contains(positive_pattern, na=False)
    negative_mask = videos["texto_busca"].str.contains(negative_pattern, na=False)

    videos_positive = videos.loc[positive_mask].copy()
    stats["videos_apos_filtro_positivo"] = len(videos_positive)
    stats["videos_removidos_sem_termos_positivos"] = stats["videos_apos_filtro_data"] - stats["videos_apos_filtro_positivo"]

    videos_final = videos_positive.loc[~negative_mask.loc[videos_positive.index]].copy()
    stats["videos_final_selecionados"] = len(videos_final)
    stats["videos_removidos_por_exclusao"] = stats["videos_apos_filtro_positivo"] - stats["videos_final_selecionados"]

    videos_final = videos_final.drop(columns=["texto_busca"])
    videos_final.to_csv(FILTERED_THEME_VIDEOS_CSV, index=False)

    stats["comments_total_bruto"] = len(comments)
    comments = comments.drop_duplicates(subset=["comment_id"]).copy()
    stats["comments_apos_deduplicacao"] = len(comments)
    stats["comments_removidos_duplicatas"] = stats["comments_total_bruto"] - stats["comments_apos_deduplicacao"]

    selected_video_ids = set(videos_final["video_id"])
    comments_final = comments.loc[comments["video_id"].isin(selected_video_ids)].copy()
    stats["comments_final_filtrados"] = len(comments_final)
    stats["comments_removidos_video_nao_selecionado"] = stats["comments_apos_deduplicacao"] - stats["comments_final_filtrados"]
    comments_final.to_csv(FILTERED_THEME_COMMENTS_CSV, index=False)

    lines = [
        f"Relatorio de filtragem tematica - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "Secao 1. Videos",
        f"1. Total bruto: {stats['videos_total_bruto']}",
        f"2. Duplicatas removidas: {stats['videos_removidos_duplicatas']}",
        f"3. Datas invalidas removidas: {stats['videos_removidos_data_invalida']}",
        f"4. Fora do periodo de estudo removidos: {stats['videos_removidos_fora_data']}",
        f"5. Removidos pelo filtro positivo: {stats['videos_removidos_sem_termos_positivos']}",
        f"6. Removidos pelo filtro negativo: {stats['videos_removidos_por_exclusao']}",
        f"7. Videos finais: {stats['videos_final_selecionados']}",
        "",
        "Secao 2. Comentarios",
        f"1. Total bruto: {stats['comments_total_bruto']}",
        f"2. Duplicatas removidas: {stats['comments_removidos_duplicatas']}",
        f"3. Removidos por nao pertencerem aos videos finais: {stats['comments_removidos_video_nao_selecionado']}",
        f"4. Comentarios finais: {stats['comments_final_filtrados']}",
        "",
        "Secao 3. Parametros",
        "Filtro positivo: termos de low-code e no-code no titulo ou descricao.",
        "Filtro negativo: termos nao relacionados ao dominio.",
        f"Total de termos positivos: {len(LOW_CODE_TERMS)}",
        f"Total de termos negativos: {len(EXCLUSION_TERMS)}",
        "",
        f"Arquivo de videos: {FILTERED_THEME_VIDEOS_CSV.name}",
        f"Arquivo de comentarios: {FILTERED_THEME_COMMENTS_CSV.name}",
    ]
    FILTERED_THEME_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Videos finais: {len(videos_final)}")
    print(f"Comentarios finais: {len(comments_final)}")
    print(f"Relatorio salvo em: {FILTERED_THEME_REPORT_TXT}")


if __name__ == "__main__":
    main()
