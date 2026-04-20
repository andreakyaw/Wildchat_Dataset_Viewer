import json
import re
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


FIELD_NAME = "Psychology"
DEFAULT_MATCH_LIMIT = 1_000
BASE_DIR = Path(__file__).resolve().parents[1]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUTPUT_PATH = BASE_DIR / "data" / "psychology_posts.jsonl"

FIELD_PATTERN = re.compile(
    r"\b(psychology|mental health|therapy|therapist|counseling|psychiatry|psychiatrist|"
    r"anxiety|depression|stress|trauma|ptsd|adhd|autism|ocd|bipolar|"
    r"personality disorder|narcissism|psychosis|hallucination|emotion|"
    r"behavior|cognitive|cbt|dbt|mindfulness|attachment theory|subconscious|"
    r"fear|phobia|panic attack|addiction|self.?esteem|motivation|burnout|"
    r"loneliness|grief|sadness|anger|mood|coping|therapy session|"
    r"mental illness|well.?being|emotional regulation|trauma response|"
    r"childhood trauma|behavioral patterns|intrusive thoughts|overthinking|"
    r"self.?worth|identity crisis|imposter syndrome|psychotherapy|psychoanalysis|"
    r"counselor|neuropsychology|schizophrenia|psychologist)\b",
    re.I,
)

CLINICAL_PATTERNS = {
    "active_care_terms": re.compile(
        r"\b(treat(?:ment|ing)?|therapy session|care plan|treatment plan|diagnos(?:is|ed)|"
        r"assessment|evaluate|screening|intervention|counseling session|"
        r"clinical notes?|intake|case formulation|follow[- ]?up|progress notes?)\b",
        re.I,
    ),
    "provider_roles": re.compile(
        r"\b(psychologist|psychiatrist|therapist|counselor|social worker|clinician|"
        r"patient|client)\b",
        re.I,
    ),
    "symptom_discussion": re.compile(
        r"\b(symptoms?|panic attacks?|suicidal|self[- ]harm|depression|anxiety|"
        r"hallucinations?|obsessions?|compulsions?|trauma response|flashbacks?)\b",
        re.I,
    ),
    "documentation_language": re.compile(
        r"\b(chief complaint|mental status exam|risk assessment|soap note|progress note|"
        r"history of present illness|treatment goals?)\b",
        re.I,
    ),
}


def parse_conversation(raw_conversation):
    if isinstance(raw_conversation, list):
        return raw_conversation
    if isinstance(raw_conversation, str):
        try:
            parsed = json.loads(raw_conversation)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def get_user_text(conversation):
    return "\n\n".join(
        message.get("content", "")
        for message in conversation
        if isinstance(message, dict) and message.get("role") == "user"
    )


def detect_clinical_work(text):
    matches = [label for label, pattern in CLINICAL_PATTERNS.items() if pattern.search(text)]
    return {
        "is_clinical_work": len(matches) >= 2,
        "clinical_signals": matches,
        "clinical_summary": (
            "Likely clinical-style psychology work based on care/process language, participant roles, or symptom documentation."
            if len(matches) >= 2
            else "No strong clinical-style psychology signal detected."
        ),
    }


def iter_source_rows():
    parquet_files = sorted(PARQUET_DIR.glob("chunk_*.parquet"))
    if parquet_files:
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            for row in df.to_dict(orient="records"):
                yield row
        return

    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    yield from dataset


def collect_matches(match_limit=DEFAULT_MATCH_LIMIT):
    matched_rows = []

    for row in iter_source_rows():
        if row.get("language") != "English":
            continue

        conversation = parse_conversation(row.get("conversation", []))
        user_text = get_user_text(conversation)
        if not user_text or not FIELD_PATTERN.search(user_text):
            continue

        clinical_info = detect_clinical_work(user_text)
        matched_rows.append(
            {
                "field": FIELD_NAME,
                "model": row.get("model", "unknown"),
                "toxic": row.get("toxic", False),
                "user_text": user_text,
                "conversation": conversation,
                **clinical_info,
            }
        )

        if len(matched_rows) >= match_limit:
            break

    return pd.DataFrame(matched_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Export psychology-related WildChat posts.")
    parser.add_argument("--limit", type=int, default=DEFAULT_MATCH_LIMIT, help="Number of matched posts to export")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = collect_matches(match_limit=args.limit)
    df.to_json(OUTPUT_PATH, orient="records", lines=True, force_ascii=True)

    flagged = int(df["is_clinical_work"].sum()) if not df.empty else 0
    print(f"Saved {len(df)} {FIELD_NAME.lower()} posts to {OUTPUT_PATH}")
    print(f"Flagged {flagged} posts as likely clinical work")
    print(f"Source: {'local parquet' if any(PARQUET_DIR.glob('chunk_*.parquet')) else 'Hugging Face streaming'}")
    print("Clinical work here usually looks like therapy/care process language, symptom assessment, and provider-patient framing.")


if __name__ == "__main__":
    main()