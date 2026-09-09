from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_00_config import STAGE_02_DIR, STAGE_03_DIR, STAGE_04_DIR, STAGE_05_DIR

STAGE_SCRIPTS = {
    "01": STAGE_02_DIR / "etapa_01_filtrar_tematica.py",
    "02": STAGE_03_DIR / "etapa_02_detectar_idioma.py",
    "03": STAGE_03_DIR / "etapa_03_analisar_limiar.py",
    "04": STAGE_03_DIR / "etapa_04_filtrar_corpus_ingles.py",
    "05": STAGE_04_DIR / "etapa_05_preparar_base_topicos.py",
    "06": STAGE_04_DIR / "etapa_06_alinhar_topicos_legado.py",
    "07": STAGE_04_DIR / "etapa_07_analisar_topicos.py",
    "08": STAGE_04_DIR / "etapa_08_analisar_macrotemas.py",
    "09": STAGE_05_DIR / "etapa_09_caracterizar_corpus.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo ou um intervalo de etapas.")
    parser.add_argument("--from-stage", default="01", choices=sorted(STAGE_SCRIPTS), dest="from_stage")
    parser.add_argument("--to-stage", default="09", choices=sorted(STAGE_SCRIPTS), dest="to_stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_ids = sorted(STAGE_SCRIPTS)
    start_index = stage_ids.index(args.from_stage)
    end_index = stage_ids.index(args.to_stage)
    if start_index > end_index:
        raise SystemExit("--from-stage precisa vir antes de --to-stage.")

    for stage_id in stage_ids[start_index : end_index + 1]:
        script_path = STAGE_SCRIPTS[stage_id]
        print(f"\n=== Executando etapa {stage_id}: {script_path.name} ===")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nPipeline concluido com sucesso.")


if __name__ == "__main__":
    main()
