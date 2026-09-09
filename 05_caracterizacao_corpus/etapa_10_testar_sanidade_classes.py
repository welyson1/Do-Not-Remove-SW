from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_00_config import (
    CLASS_SANITY_REPORT_TXT,
    CLASS_SANITY_SAMPLE_CSV,
    ENGLISH_COMMENTS_PERIOD_CSV,
    LANGUAGE_FILTER_COLUMNS,
    ensure_directories,
    load_language_comment_file,
    normalize_series,
)

ANALYSIS_DESCRIPTION = (
    "aplicar dicionarios lexicais no nivel do comentario apos deduplicacao "
    "para verificar se a taxonomia produz sinais plausiveis."
)


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: re.Pattern[str]


def compile_rules(patterns: list[tuple[str, str]]) -> list[RegexRule]:
    return [RegexRule(name, re.compile(pattern, flags=re.IGNORECASE)) for name, pattern in patterns]


CLASS_DEFINITIONS = {
    "C1": {
        "label": "Pergunta, pedido de ajuda ou pedido de tutorial",
        "rules": compile_rules(
            [
                ("question_mark", r"\?"),
                ("how_can", r"\bhow\s+can\b"),
                ("can_you", r"\bcan\s+you\b"),
                ("can_i", r"\bcan\s+i\b|\bcan\s+we\b"),
                ("could_you", r"\bcould\s+you\b"),
                ("would_you", r"\bwould\s+you\b|\bwould\s+it\s+be\s+possible\b"),
                ("please_request", r"\bplease\b.{0,40}\b(help|explain|show|share|advise|make|do|guide|clarify)\b"),
                ("please_explain_show", r"\bplease\s+(explain|show|clarify|guide)\b"),
                ("any_idea", r"\bany\s+idea\b"),
                ("any_suggestions", r"\bany\s+suggestions?\b|\bwhat\s+are\s+your\s+suggestions?\b"),
                ("anyone_know", r"\banyone\s+know\b"),
                ("help_me", r"\bhelp\s+me\b|\bneed\s+help\b|\bi\s+need\s+help\b"),
                ("advise", r"\badvise\s+me\b|\bhope\s+you\s+can\s+advise\b"),
                ("show_us", r"\bshow\s+us\b"),
                ("how_do_i", r"\bhow\s+do\s+i\b|\bhow\s+do\s+we\b"),
                ("how_to_request", r"(^|[\r\n])\s*how\s+to\b|\bhow\s+to\b.{0,80}\?"),
                ("tutorial_request", r"\b(tutorial\s+on|tutorial\s+for|step\s+by\s+step)\b|\b(make|do|create|record)\b.{0,35}\b(tutorial|video)\b"),
                ("what_do_i_need", r"\bwhat\s+do\s+i\s+need\b"),
                ("where_can_i", r"\bwhere\s+can\s+i\b"),
                ("is_there_a_way", r"\bis\s+there\s+(any\s+)?way\b|\bis\s+it\s+possible\b"),
                ("do_you_have", r"\bdo\s+you\s+have\b"),
                ("please_make_video", r"\b(make|do|create|record)\b.{0,40}\b(video|tutorial)\b"),
                ("part_two_request", r"\b(part|episode)\s*2\b|\bsecond\s+(part|episode|video)\b|\bmore\s+videos?\b"),
            ]
        ),
    },
    "C2": {
        "label": "Relato de problema, erro, falha ou limitacao operacional",
        "rules": compile_rules(
            [
                ("same_issue", r"\bsame\s+(issue|problem|error)\b"),
                ("issue_context", r"\b(i\s+have|i'm\s+having|im\s+having|having|facing|encountering|running\s+into|ran\s+into|got|getting|there\s+is|there'?s)\b.{0,25}\b(issue|problem)\b|\b(issue|problem)\b.{0,25}\b(with|when|here)\b"),
                ("bug", r"\bbug\b"),
                ("error", r"\berrors?\b"),
                ("exception", r"\bexceptions?\b|\btraceback\b"),
                ("crash", r"\bcrash(?:es|ed|ing)?\b"),
                ("timeout", r"\btime\s*out\b|\btimeout\b|\btimed\s+out\b"),
                ("failed", r"\bfailed\b|\bfailure\b"),
                ("not_working", r"\bnot\s+working\b"),
                ("keeps_failing", r"\bkeeps?\s+(failing|crashing)\b"),
                ("doesnt_work", r"\bdoesn'?t\s+work\b|\bdidn'?t\s+work\b"),
                ("cant_operational", r"\b(can't|cannot|cant)\b.{0,25}\b(access|open|install|save|load|connect|create|import|upload|login|log\s*in|get|see|run|find|make|use|duplicate|trigger|show|work)\b"),
                ("unable_to", r"\bunable\s+to\b|\bnot\s+able\s+to\b"),
                ("stuck", r"\bstuck\b"),
                ("broken_state", r"\bis\s+broken\b|\bgot\s+broken\b|\bbroken\s+again\b"),
                ("went_wrong", r"\bsomething\s+went\s+wrong\b"),
                ("connection_refused", r"\bconnection\s+refused\b|\bconnection\s+denied\b"),
                ("permission_denied", r"\bpermission\s+denied\b|\baccess\s+denied\b|\bunauthori[sz]ed\b"),
                ("not_found", r"\bnot\s+found\b|\bpage\s+not\s+found\b|\b404\b"),
                ("blank_or_missing", r"\bblank\b|\bmissing\b|\bgreyed\s+out\b|\bgrayed\s+out\b"),
                ("invalid", r"\binvalid\b"),
                ("fix_or_resolve", r"\b(please\s+)?(fix|resolve|solve)\b"),
                ("not_showing", r"\bnot\s+(showing|loading|available|saving)\b|\bis\s+not\s+showing\b|\bwon'?t\s+load\b"),
                ("manifest", r"\bmanifest\s+file\b"),
            ]
        ),
    },
    "C3": {
        "label": "Elogio ou testemunho positivo",
        "rules": compile_rules(
            [
                ("thanks", r"\bthanks?\b|\bthank\s+you\b"),
                ("great", r"\bgreat\b"),
                ("amazing", r"\bamazing\b"),
                ("awesome", r"\bawesome\b"),
                ("clear", r"\bclear\b|\bconcise\b"),
                ("excellent", r"\bexcellent\b"),
                ("fantastic", r"\bfantastic\b"),
                ("helpful", r"\bhelpful\b|\bsuper\s+helpful\b"),
                ("useful", r"\buseful\b|\bvaluable\b"),
                ("informative", r"\binformative\b|\bvery\s+informative\b"),
                ("clear_explanation", r"\b(clear|good|great)\s+explanation\b|\bexplained\s+(it\s+)?(well|clearly)\b|\bwell\s+explained\b"),
                ("great_content", r"\bgreat\s+content\b|\bquality\s+content\b"),
                ("best_tutorial", r"\bbest\s+(tutorial|video|explanation)\b|\bbest\s+resource\b"),
                ("learned_saved", r"\blearned\s+a\s+lot\b|\bsaved\s+me\b|\bsaved\s+my\b"),
                ("works_perfectly", r"\bworks?\s+(perfectly|great|well)\b"),
                ("love", r"\blove\s+this\b|\bloved\b"),
                ("well_done", r"\bwell\s+done\b|\bnice\s+job\b"),
                ("congrats", r"\bcongrats?\b|\bcongratulations\b"),
                ("appreciate", r"\bappreciate\b|\bappreciated\b"),
                ("brilliant", r"\bbrilliant\b|\binsightful\b"),
                ("exactly_needed", r"\bexactly\s+what\s+i\s+needed\b|\bperfect\b"),
            ]
        ),
    },
    "C4": {
        "label": "Critica, frustracao ou objecao",
        "rules": compile_rules(
            [
                ("expensive", r"\btoo\s+expensive\b|\bexpensive\b|\boverpriced\b"),
                ("not_worth", r"\bnot\s+worth\b|\bwaste\s+of\s+money\b"),
                ("not_free", r"\bnot\s+free\b"),
                ("clunky", r"\bclunky\b|\bwonky\b"),
                ("pain", r"\bpain\b|\bfrustrating\b|\bfrustration\b|\bannoying\b"),
                ("repetitive", r"\brepetitive\b"),
                ("limitations", r"\blimitations?\b|\btoo\s+limited\b|\bvery\s+limited\b"),
                ("not_there_yet", r"\bnot\s+there\s+yet\b"),
                ("hard_difficult", r"\bhard\s+to\b|\bdifficult\b|\bconfusing\b|\btoo\s+complicated\b"),
                ("slow_laggy", r"\btoo\s+slow\b|\bslow\b|\blaggy\b|\blags\b"),
                ("disappointing", r"\bdisappointing\b|\bdisappointed\b"),
                ("missing_feature", r"\bmissing\s+feature\b|\black\s+of\b|\blacks?\b.{0,20}\b(feature|support|option)\b"),
                ("terrible", r"\bterrible\b|\bworst\b|\bbad\s+quality\b|\bunusable\b|\bwaste\s+of\s+time\b"),
                ("bad_ui_quality", r"\bblurry\b|\bpoor\s+quality\b"),
                ("not_best", r"\bnot\s+the\s+best\b|\bless\s+user\s+friendly\b|\bdoesn'?t\s+make\s+sense\b"),
                ("too_late", r"\btoo\s+late\b|\byou'?re\s+hooked\b"),
            ]
        ),
    },
    "C5": {
        "label": "Recomendacao, divulgacao ou encaminhamento a recursos/alternativas",
        "rules": compile_rules(
            [
                ("subscribe", r"\bplease\s+subscribe\b|\bsubscribe\s+to\s+(the\s+)?channel\b|\blike\s+and\s+subscribe\b"),
                ("join_membership", r"\b(join|consider\s+joining)\b.{0,25}\b(my|our|the)\b.{0,15}\b(channel|membership|course|community|whatsapp)\b|\bmembers?\b.{0,20}\b(access|join)\b"),
                ("watch_full", r"\bwatch\b.{0,20}\b(full|here)\b"),
                ("my_channel", r"\bon\s+my\s+channel\b|\bcheck\s+out\s+my\b|\bcheck\s+out\b"),
                ("playlist", r"\bplaylist\b"),
                ("linktree_or_bio", r"\blinktree\b|\blink\s+in\s+bio\b|\blink\s+in\s+(the\s+)?description\b"),
                ("github_repo", r"\bgithub\s+(repo|repository|link)\b|\bgithub\.com\b"),
                ("docs_forum_community", r"\b(documentation|docs|forum|forums|discord)\b|\b(join|check|see|post|ask|visit)\b.{0,25}\bcommunity\b"),
                ("template_resource", r"\btemplate(s)?\b|\bresource(s)?\b|\bsample\s+(app|file|code)\b"),
                ("contact_me", r"\bcontact\s+me\b|\breach\s+out\s+to\s+me\b|\bdm\s+me\b|\bemail\s+me\b"),
                ("offer_service", r"\bi\s+(design|offer|provide)\b"),
                ("free_demo", r"\bfree\s+demo\b"),
                ("register_now", r"\bregister\s+now\b|\bbuy\s+now\b|\bdiscount\b|\baffiliate\s+link\b"),
                ("private_classes", r"\bprivate\s+classes\b|\b1-on-1\b|\bcourse(s)?\b"),
                ("recommend_alt", r"\b(recommend|recommended|recommendation)\b.{0,40}\b(tool|tools|course|channel|platform|alternative|plan|tier|service|approach|solution)\b"),
                ("you_should_use", r"\byou\s+should\s+(use|try|check)\b"),
                ("use_instead", r"\b(use|try|go\s+with)\b.{0,40}\binstead\b|\balternative(s)?\b"),
                ("cheaper_better", r"\bcheaper\b.{0,20}\bbetter\b|\bbetter\b.{0,20}\bcheaper\b"),
            ]
        ),
    },
    "C6": {
        "label": "Outros, humor ou ruido evidente",
        "rules": compile_rules(
            [
                ("lol_haha", r"\b(lol|haha|hahaha|lmao|rofl)\b"),
                ("bro_only", r"^\s*(bro|bruh)\b"),
                ("bruh", r"\bbruh\b"),
                ("laugh_emoji", r"[😂🤣]{1,}"),
                ("real_laugh_emoji", "[\U0001F602\U0001F923\U0001F605\U0001F606]"),
                ("wtf", r"\bwtf\b"),
                ("meme", r"\bmeme\b"),
                ("only_link", r"^\s*https?://\S+\s*$"),
                ("minimal_noise", r"^\s*(ok|okay|yes|no|wow|nice|cool|first|done|hi|hello|thanks?)\s*[.!?]*\s*$"),
                ("sales_spam", r"\beye-?catching\s+thumbnails?\b|\bvery\s+affordable\s+price\b"),
            ]
        ),
    },
}

CLASS_COLUMNS = list(CLASS_DEFINITIONS)


def find_rule_matches(text: str, rules: list[RegexRule]) -> list[str]:
    return [rule.name for rule in rules if rule.pattern.search(text)]


def label_combo_from_flags(flag_row: list[bool]) -> str:
    active = [CLASS_COLUMNS[index] for index, value in enumerate(flag_row) if value]
    return "+".join(active) if active else "sem_classe"


def summarize_rule_hits(series: pd.Series) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for matches in series:
        counter.update(matches)
    return counter.most_common(8)


def prepare_comments_for_classification(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    output = comments.copy()
    output["comment_normalized"] = normalize_series(output["comment"])
    total_before_dedup = len(output)
    output = output.drop_duplicates(subset=["comment_normalized"], keep="first").copy()
    duplicates_removed = total_before_dedup - len(output)
    output["is_reply"] = output["is_reply"].astype(str).str.lower().eq("true")
    output["like_count"] = pd.to_numeric(output["like_count"], errors="coerce").fillna(0)
    stats = {
        "total_before_dedup": total_before_dedup,
        "total_after_dedup": len(output),
        "duplicates_removed": duplicates_removed,
    }
    return output, stats


def classify_comments(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    output, stats = prepare_comments_for_classification(comments)
    for class_code, definition in CLASS_DEFINITIONS.items():
        rules = definition["rules"]
        match_series = output["comment_normalized"].apply(lambda text: find_rule_matches(text, rules))
        output[f"{class_code}_matches"] = match_series
        output[class_code] = match_series.str.len().gt(0)

    flag_matrix = output[CLASS_COLUMNS].to_numpy(dtype=bool)
    output["label_combo"] = [label_combo_from_flags(list(row)) for row in flag_matrix]
    output["n_labels"] = output[CLASS_COLUMNS].sum(axis=1)
    return output, stats


def sample_comments(frame: pd.DataFrame, label_column: str, sample_size: int = 20) -> pd.DataFrame:
    if label_column == "sem_classe":
        subset = frame.loc[frame["label_combo"] == "sem_classe"].copy()
    else:
        subset = frame.loc[frame[label_column]].copy()

    if subset.empty:
        return subset

    n = min(sample_size, len(subset))
    sample = subset.sample(n=n, random_state=42).copy() if len(subset) > n else subset.copy()
    sample["sample_group"] = label_column
    return sample


def main() -> None:
    ensure_directories()

    comments, repaired = load_language_comment_file(ENGLISH_COMMENTS_PERIOD_CSV, LANGUAGE_FILTER_COLUMNS)
    if repaired:
        comments.to_csv(ENGLISH_COMMENTS_PERIOD_CSV, index=False)

    comments, dedup_stats = classify_comments(comments)

    total_comments = len(comments)
    total_replies = int(comments["is_reply"].sum())
    total_top_level = total_comments - total_replies
    any_label_count = int((comments["n_labels"] > 0).sum())
    multi_label_count = int((comments["n_labels"] > 1).sum())
    unlabeled_count = int((comments["label_combo"] == "sem_classe").sum())

    report_lines = [
        "Relatorio de teste de sanidade das classes da RQ3",
        "=" * 72,
        f"Objetivo: {ANALYSIS_DESCRIPTION}",
        f"Comentarios antes da deduplicacao: {dedup_stats['total_before_dedup']}",
        f"Comentarios duplicados removidos: {dedup_stats['duplicates_removed']} ({dedup_stats['duplicates_removed'] / dedup_stats['total_before_dedup']:.2%})",
        f"Comentarios analisados: {total_comments}",
        f"Comentarios de topo: {total_top_level}",
        f"Replies: {total_replies}",
        f"Comentarios com pelo menos uma classe: {any_label_count} ({any_label_count / total_comments:.2%})",
        f"Comentarios multirrotulo: {multi_label_count} ({multi_label_count / total_comments:.2%})",
        f"Comentarios sem classe: {unlabeled_count} ({unlabeled_count / total_comments:.2%})",
        "",
        "Contagens por classe",
    ]

    for class_code, definition in CLASS_DEFINITIONS.items():
        mask = comments[class_code]
        count = int(mask.sum())
        replies = int(comments.loc[mask, "is_reply"].sum())
        top_level = count - replies
        report_lines.append(
            f"{class_code} | {definition['label']} | {count} ({count / total_comments:.2%}) | topo={top_level} | replies={replies}"
        )

        top_rules = summarize_rule_hits(comments.loc[mask, f"{class_code}_matches"])
        if top_rules:
            formatted_rules = ", ".join(f"{rule}={hits}" for rule, hits in top_rules)
            report_lines.append(f"  Regras mais frequentes: {formatted_rules}")

    report_lines.extend(["", "Combinacoes mais frequentes"])
    combo_counts = comments["label_combo"].value_counts().head(15)
    for combo, count in combo_counts.items():
        report_lines.append(f"{combo}: {count} ({count / total_comments:.2%})")

    report_lines.extend(["", "Exemplos por classe"])
    for class_code, definition in CLASS_DEFINITIONS.items():
        examples = comments.loc[comments[class_code], "comment_normalized"].head(3).tolist()
        report_lines.append(f"{class_code} | {definition['label']}")
        if not examples:
            report_lines.append("  Nenhum exemplo encontrado.")
            continue
        for example in examples:
            report_lines.append(f"  - {example}")

    CLASS_SANITY_REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sample_frames = []
    sample_columns = [
        "sample_group",
        "comment_id",
        "video_id",
        "published_at",
        "is_reply",
        "like_count",
        "label_combo",
        "comment",
    ]
    for class_code in CLASS_COLUMNS + ["sem_classe"]:
        sampled = sample_comments(comments, class_code, sample_size=20)
        if sampled.empty:
            continue
        for active_class in CLASS_COLUMNS:
            sampled[f"{active_class}_matches"] = sampled[f"{active_class}_matches"].apply(lambda value: " | ".join(value))
        sample_frames.append(
            sampled[
                sample_columns
                + CLASS_COLUMNS
                + [f"{active_class}_matches" for active_class in CLASS_COLUMNS]
            ]
        )

    if sample_frames:
        sample_output = pd.concat(sample_frames, ignore_index=True)
        sample_output.to_csv(CLASS_SANITY_SAMPLE_CSV, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=sample_columns).to_csv(CLASS_SANITY_SAMPLE_CSV, index=False, encoding="utf-8")

    print(f"Relatorio salvo em: {CLASS_SANITY_REPORT_TXT}")
    print(f"Amostra salva em: {CLASS_SANITY_SAMPLE_CSV}")


if __name__ == "__main__":
    main()
