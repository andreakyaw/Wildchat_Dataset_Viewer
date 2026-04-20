import json
import re
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


FIELD_NAME = "Medicine"
DEFAULT_MATCH_LIMIT = 1_000
BASE_DIR = Path(__file__).resolve().parents[1]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUTPUT_PATH = BASE_DIR / "data" / "medicine_posts.jsonl"

FIELD_PATTERN = re.compile(
    r"\b(medicine|medical|doctor|physician|hospital|clinic|diagnosis|prognosis|"
    r"treatment|symptom|disease|illness|infection|virus|bacteria|vaccine|"
    r"surgery|medication|drug|dosage|prescription|pain|fever|"
    r"cardiology|neurology|oncology|dermatology|pediatrics|pharmacology|"
    r"diabetes|cancer|covid|flu|blood pressure|heart rate|"
    r"inflammation|chronic|acute|emergency|healthcare|patient|"
    r"injury|fracture|allergy|immune system|antibiotics|side effects|"
    r"clinical|diagnostic|x.?ray|mri|ct.?scan|ultrasound|lab test|blood test|"
    r"anatomy|physiology|pathology|biopsy|chemotherapy|cardiovascular|"
    r"radiology|hiv|aids|insomnia|fatigue|sleep disorder|"
    r"medical condition|health condition|treatment plan)\b",
    re.I,
)

CLINICAL_PATTERNS = {
    "active_care_terms": re.compile(
        r"\b(diagnos(?:is|ed)|treat(?:ment|ing)?|prescri(?:be|ption)|dosage|"
        r"care plan|treatment plan|follow[- ]?up|clinical decision|management plan)\b",
        re.I,
    ),
    "care_setting": re.compile(
        r"\b(hospital|clinic|er|urgent care|icu|outpatient|inpatient|ward|provider)\b",
        re.I,
    ),
    "patient_and_symptoms": re.compile(
        r"\b(patient|symptoms?|pain|fever|shortness of breath|chest pain|rash|"
        r"lab results?|blood pressure|heart rate|history of present illness)\b",
        re.I,
    ),
    "documentation_language": re.compile(
        r"\b(differential diagnosis|assessment and plan|soap note|chief complaint|"
        r"physical exam|clinical findings|test results?)\b",
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
            "Likely clinical medical work based on care setting, patient symptoms, and diagnostic or treatment language."
            if len(matches) >= 2
            else "No strong clinical medical signal detected."
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
    parser = argparse.ArgumentParser(description="Export medicine-related WildChat posts.")
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
    print("Clinical work here usually looks like patient-specific symptoms, diagnostic framing, and treatment planning.")


if __name__ == "__main__":
    main()