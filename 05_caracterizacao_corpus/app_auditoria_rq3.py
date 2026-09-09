from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pipeline_00_config import CLASS_AUDIT_TEMPLATE_CSV, CLASS_PRECISION_REPORT_TXT
from etapa_12_calcular_precisao_classes import parse_manual_flag


AUDIT_COLUMNS = [
    "manual_is_true",
    "manual_primary_label",
    "manual_should_be_other_class",
    "manual_notes",
    "auditor",
    "audit_date",
]
CLASS_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]
ALTERNATIVE_OPTIONS = CLASS_ORDER + ["sem_classe", "multirrotulo_ambiguo"]
MANUAL_CHOICES = ["Pendente", "Correto", "Incorreto"]
MANUAL_TO_VALUE = {"Pendente": "", "Correto": "1", "Incorreto": "0"}
MANUAL_DISPLAY = {
    "Pendente": "Ainda nao avaliei",
    "Correto": "Sim, pertence a classe prevista",
    "Incorreto": "Nao, nao pertence a classe prevista",
}
VALUE_TO_MANUAL = {None: "Pendente", True: "Correto", False: "Incorreto"}
STATUS_OPTIONS = ["Todos", "Pendentes", "Auditados", "Corretos", "Incorretos"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_audit(path: Path = CLASS_AUDIT_TEMPLATE_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False, keep_default_na=False)
    missing = [column for column in AUDIT_COLUMNS if column not in frame.columns]
    for column in missing:
        frame[column] = ""
    for column in AUDIT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def save_audit(frame: pd.DataFrame, path: Path = CLASS_AUDIT_TEMPLATE_CSV) -> Path | None:
    backup_path = None
    if not st.session_state.get("backup_created", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        st.session_state.backup_created = True
        st.session_state.backup_path = str(backup_path)

    output = frame.copy()
    for column in AUDIT_COLUMNS:
        output[column] = output[column].fillna("").astype(str)
    output.to_csv(path, index=False, encoding="utf-8")
    return backup_path


def load_state() -> None:
    if "audit_df" not in st.session_state:
        st.session_state.audit_df = load_audit()
    if "selected_position" not in st.session_state:
        st.session_state.selected_position = 0
    if "target_row_index" not in st.session_state:
        st.session_state.target_row_index = None


def class_label_map(frame: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for class_code, group in frame.groupby("predicted_class", sort=True):
        label = clean_text(group["predicted_class_label"].iloc[0])
        labels[class_code] = label
    return labels


def class_format(class_code: str, labels: dict[str, str]) -> str:
    label = labels.get(class_code, "")
    return f"{class_code} - {label}" if label else class_code


def manual_status(value: object) -> str:
    return VALUE_TO_MANUAL[parse_manual_flag(value)]


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_code in sorted(frame["predicted_class"].unique()):
        subset = frame.loc[frame["predicted_class"] == class_code]
        flags = subset["manual_is_true"].apply(parse_manual_flag)
        total = len(subset)
        correct = int((flags == True).sum())
        incorrect = int((flags == False).sum())
        audited = correct + incorrect
        pending = total - audited
        precision = "" if audited == 0 else f"{correct / audited:.1%}"
        rows.append(
            {
                "classe": class_code,
                "total": total,
                "auditados": audited,
                "pendentes": pending,
                "corretos": correct,
                "incorretos": incorrect,
                "precisao": precision,
            }
        )
    return pd.DataFrame(rows)


def filtered_indices(
    frame: pd.DataFrame,
    selected_classes: list[str],
    selected_status: str,
    search_text: str,
) -> list[int]:
    mask = frame["predicted_class"].isin(selected_classes)
    flags = frame["manual_is_true"].apply(parse_manual_flag)

    if selected_status == "Pendentes":
        mask &= flags.isna()
    elif selected_status == "Auditados":
        mask &= flags.notna()
    elif selected_status == "Corretos":
        mask &= flags == True
    elif selected_status == "Incorretos":
        mask &= flags == False

    normalized_search = search_text.strip().lower()
    if normalized_search:
        haystack = (
            frame["comment"].astype(str)
            + " "
            + frame["comment_id"].astype(str)
            + " "
            + frame["video_id"].astype(str)
            + " "
            + frame["predicted_class_matches"].astype(str)
        ).str.lower()
        mask &= haystack.str.contains(normalized_search, regex=False, na=False)

    return frame.index[mask].tolist()


def item_label(frame: pd.DataFrame, row_index: int) -> str:
    row = frame.loc[row_index]
    status = manual_status(row["manual_is_true"])
    comment = clean_text(row["comment"]).replace("\n", " ")
    if len(comment) > 80:
        comment = comment[:77] + "..."
    rank = clean_text(row["sample_rank_within_class"])
    return f"{row_index + 1:03d} | {row['predicted_class']} #{rank} | {status} | {comment}"


def split_labels(value: object) -> list[str]:
    text = clean_text(value).strip()
    if not text:
        return []
    labels = [part.strip() for part in text.replace("|", "+").replace(",", "+").split("+")]
    return [label for label in labels if label in ALTERNATIVE_OPTIONS]


def join_labels(labels: list[str]) -> str:
    return "+".join(label for label in labels if label)


def update_row(
    frame: pd.DataFrame,
    row_index: int,
    manual_choice: str,
    manual_should_be_other_classes: list[str],
    manual_notes: str,
    auditor: str,
) -> None:
    row = frame.loc[row_index]
    predicted_class = clean_text(row["predicted_class"])
    manual_value = MANUAL_TO_VALUE[manual_choice]
    manual_primary_label = ""
    manual_should_be_other_class = join_labels(manual_should_be_other_classes)
    if manual_choice == "Correto":
        manual_primary_label = predicted_class
        manual_should_be_other_class = ""
    elif manual_choice == "Incorreto":
        manual_primary_label = manual_should_be_other_classes[0] if manual_should_be_other_classes else ""

    frame.at[row_index, "manual_is_true"] = manual_value
    frame.at[row_index, "manual_primary_label"] = manual_primary_label
    frame.at[row_index, "manual_should_be_other_class"] = manual_should_be_other_class
    frame.at[row_index, "manual_notes"] = manual_notes.strip()
    frame.at[row_index, "auditor"] = auditor.strip()
    frame.at[row_index, "audit_date"] = datetime.now().strftime("%Y-%m-%d") if manual_value else ""


def go_to_next_pending(indices: list[int], current_row_index: int, frame: pd.DataFrame) -> None:
    if not indices:
        return

    current_position = indices.index(current_row_index) if current_row_index in indices else 0
    ordered = indices[current_position + 1 :] + indices[: current_position + 1]
    for candidate in ordered:
        if parse_manual_flag(frame.at[candidate, "manual_is_true"]) is None:
            st.session_state.selected_position = indices.index(candidate)
            st.session_state.target_row_index = candidate
            return

    next_position = min(current_position + 1, len(indices) - 1)
    st.session_state.selected_position = next_position
    st.session_state.target_row_index = indices[next_position]


def go_to_next_item(indices: list[int], current_row_index: int) -> None:
    if not indices:
        return
    current_position = indices.index(current_row_index) if current_row_index in indices else 0
    st.session_state.selected_position = min(current_position + 1, len(indices) - 1)
    st.session_state.target_row_index = indices[st.session_state.selected_position]


def run_precision_report() -> subprocess.CompletedProcess[str]:
    script_path = CURRENT_DIR / "etapa_12_calcular_precisao_classes.py"
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def render_header(frame: pd.DataFrame) -> None:
    flags = frame["manual_is_true"].apply(parse_manual_flag)
    total = len(frame)
    correct = int((flags == True).sum())
    incorrect = int((flags == False).sum())
    audited = correct + incorrect
    pending = total - audited

    st.title("Auditoria RQ3")
    st.caption("Valide se cada comentario corresponde a classe prevista pelo dicionario lexical.")

    col_total, col_audited, col_pending, col_precision = st.columns(4)
    col_total.metric("Total", total)
    col_audited.metric("Auditados", audited)
    col_pending.metric("Pendentes", pending)
    col_precision.metric("Precisao parcial", "-" if audited == 0 else f"{correct / audited:.1%}")
    st.progress(0 if total == 0 else audited / total)


def render_sidebar(frame: pd.DataFrame) -> tuple[list[str], str, str, str]:
    labels = class_label_map(frame)
    classes = [class_code for class_code in CLASS_ORDER if class_code in labels]

    st.sidebar.header("Filtros")
    auditor_default = st.sidebar.text_input("Auditor", value=os.environ.get("USERNAME", ""))
    with st.sidebar.expander("Protocolo rapido", expanded=True):
        st.markdown(
            """
            Julgue apenas a classe prevista na tela.

            Marque **Sim** quando o comentario tiver essa funcao discursiva, mesmo que tambem tenha outras.

            Marque **Nao** quando a classe prevista nao aparecer; nesse caso, escolha uma ou mais classes alternativas.

            Use observacoes so para casos ambiguos.
            """
        )
    selected_classes = st.sidebar.multiselect(
        "Classes",
        options=classes,
        default=classes,
        format_func=lambda value: class_format(value, labels),
    )
    selected_status = st.sidebar.selectbox("Status", options=STATUS_OPTIONS, index=0)
    search_text = st.sidebar.text_input("Buscar no comentario")

    st.sidebar.divider()
    if st.sidebar.button("Recarregar CSV"):
        st.session_state.audit_df = load_audit()
        st.session_state.selected_position = 0
        st.rerun()

    if st.sidebar.button("Atualizar relatorio de precisao"):
        result = run_precision_report()
        if result.returncode == 0:
            st.sidebar.success("Relatorio atualizado.")
        else:
            st.sidebar.error("Falha ao atualizar o relatorio.")
            st.sidebar.code(result.stderr or result.stdout)

    if st.session_state.get("backup_path"):
        st.sidebar.caption(f"Backup: {st.session_state.backup_path}")

    return selected_classes, selected_status, search_text, auditor_default


def render_summary(frame: pd.DataFrame) -> None:
    with st.expander("Resumo por classe", expanded=False):
        st.dataframe(build_summary(frame), use_container_width=True, hide_index=True)


def render_navigation(frame: pd.DataFrame, indices: list[int]) -> int | None:
    if not indices:
        st.warning("Nenhum comentario encontrado com os filtros atuais.")
        return None

    if st.session_state.selected_position >= len(indices):
        st.session_state.selected_position = 0
    if st.session_state.target_row_index in indices:
        st.session_state.selected_position = indices.index(st.session_state.target_row_index)
        st.session_state.target_row_index = None

    col_prev, col_select, col_next = st.columns([1, 6, 1])
    with col_prev:
        if st.button("Anterior", use_container_width=True):
            st.session_state.selected_position = max(0, st.session_state.selected_position - 1)
            st.rerun()
    with col_next:
        if st.button("Proximo", use_container_width=True):
            st.session_state.selected_position = min(len(indices) - 1, st.session_state.selected_position + 1)
            st.rerun()
    with col_select:
        st.session_state.selected_position = st.selectbox(
            "Comentario filtrado",
            options=list(range(len(indices))),
            index=st.session_state.selected_position,
            format_func=lambda position: item_label(frame, indices[position]),
            label_visibility="collapsed",
        )

    return indices[st.session_state.selected_position]


def render_comment(frame: pd.DataFrame, row_index: int, indices: list[int], auditor_default: str) -> None:
    row = frame.loc[row_index]
    labels = class_label_map(frame)
    status = manual_status(row["manual_is_true"])

    st.subheader(f"Classe avaliada: {clean_text(row['predicted_class'])}")
    st.caption(clean_text(row["predicted_class_label"]))
    if status == "Correto":
        st.success("Status: correto")
    elif status == "Incorreto":
        st.error("Status: incorreto")
    else:
        st.warning("Status: pendente")

    st.write(f"Amostra: {clean_text(row['sample_rank_within_class'])}")
    st.write(f"Regras acionadas: {clean_text(row['predicted_class_matches']) or '-'}")
    st.write(f"Rotulos automaticos: {clean_text(row['all_predicted_labels']) or '-'}")

    meta_col_1, meta_col_2, meta_col_3, meta_col_4 = st.columns(4)
    meta_col_1.caption(f"Data: {clean_text(row['published_at']) or '-'}")
    meta_col_2.caption(f"Tipo: {'reply' if str(row['is_reply']).lower() == 'true' else 'top-level'}")
    meta_col_3.caption(f"Likes: {clean_text(row['like_count']) or '0'}")
    meta_col_4.caption(f"Video: {clean_text(row['video_id']) or '-'}")
    st.caption(f"comment_id: {clean_text(row['comment_id'])}")

    st.divider()
    st.write("Comentario")
    st.write(clean_text(row["comment"]))
    st.divider()

    with st.form(key=f"audit_form_{row_index}"):
        current_choice = manual_status(row["manual_is_true"])
        manual_choice = st.radio(
            "Decisao",
            options=MANUAL_CHOICES,
            index=MANUAL_CHOICES.index(current_choice),
            horizontal=True,
            format_func=lambda value: MANUAL_DISPLAY[value],
        )

        other_classes = split_labels(row["manual_should_be_other_class"])
        if manual_choice == "Incorreto":
            other_classes = st.multiselect(
                "Classes alternativas",
                options=ALTERNATIVE_OPTIONS,
                default=other_classes,
                format_func=lambda value: class_format(value, labels) if value in labels else (value or "-"),
                help="Selecione todas as classes que descrevem melhor o comentario. Use sem_classe quando nenhuma classe se aplica.",
            )
        elif manual_choice == "Correto":
            st.info("Esta linha sera salva como acerto da classe prevista.")
            other_classes = []

        notes = st.text_area("Observacoes opcionais", value=clean_text(row["manual_notes"]), height=80)

        col_save, col_save_next = st.columns(2)
        save_current = col_save.form_submit_button("Salvar", use_container_width=True)
        save_next = col_save_next.form_submit_button("Salvar e continuar", use_container_width=True)

    if save_current or save_next:
        if manual_choice == "Incorreto" and not other_classes:
            st.warning("Escolha pelo menos uma classe alternativa para a marcacao incorreta.")
            return

        update_row(
            frame,
            row_index=row_index,
            manual_choice=manual_choice,
            manual_should_be_other_classes=other_classes,
            manual_notes=notes,
            auditor=clean_text(row["auditor"]) or auditor_default,
        )
        backup_path = save_audit(frame)
        if backup_path:
            st.info(f"Backup criado em: {backup_path}")
        st.success("Auditoria salva no CSV.")

        if save_next:
            if parse_manual_flag(frame.at[row_index, "manual_is_true"]) is not None:
                go_to_next_pending(indices, row_index, frame)
            else:
                go_to_next_item(indices, row_index)
            st.rerun()


def render_report_preview() -> None:
    if not CLASS_PRECISION_REPORT_TXT.exists():
        return
    with st.expander("Previa do relatorio de precisao", expanded=False):
        st.text(CLASS_PRECISION_REPORT_TXT.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Auditoria RQ3", layout="wide")
    load_state()

    frame = st.session_state.audit_df
    render_header(frame)
    selected_classes, selected_status, search_text, auditor_default = render_sidebar(frame)
    render_summary(frame)

    indices = filtered_indices(frame, selected_classes, selected_status, search_text)
    row_index = render_navigation(frame, indices)
    if row_index is not None:
        render_comment(frame, row_index, indices, auditor_default)

    render_report_preview()


if __name__ == "__main__":
    main()
