import json
import re
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


FIELD_NAME = "Law"
DEFAULT_MATCH_LIMIT = 1_000
BASE_DIR = Path(__file__).resolve().parents[1]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUTPUT_PATH = BASE_DIR / "data" / "law_posts.jsonl"

FIELD_PATTERN = re.compile(
    r"\b(law|legal|lawyer|attorney|litigation|lawsuit|defendant|plaintiff|"
    r"jurisdiction|contract|statute|constitutional|criminal justice|civil law|"
    r"notary|affidavit|subpoena|courtroom|testimony|legislation|"
    r"copyright|trademark|intellectual property|tort|liability|negligence|"
    r"prosecutor|arbitration|appeal|felony|misdemeanor|paralegal|judiciary|"
    r"malpractice|compliance|patent|court|judge|jury|arrest|evidence|trial|"
    r"nda|settlement|damages|fraud|theft|assault|harassment|defamation|"
    r"terms of service|privacy policy|data protection|regulation|ordinance|"
    r"legal advice|legal rights|criminal|police)\b",
    re.I,
)

CLINICAL_PATTERNS = {
    "health_law_context": re.compile(
        r"\b(patient|hospital|doctor|medical record|medical malpractice|healthcare|"
        r"clinic|consent form|health insurance|hipaa|injury claim|disability claim)\b",
        re.I,
    ),
    "casework_language": re.compile(
        r"\b(case|claim|complaint|evidence|damages|liability|negligence|settlement|"
        r"representation|legal strategy|deposition|court filing)\b",
        re.I,
    ),
    "advice_request": re.compile(
        r"\b(can i sue|do i have a case|should i get a lawyer|legal options|"
        r"what are my rights|what should i file|can they be liable)\b",
        re.I,
    ),
    "regulated_care_terms": re.compile(
        r"\b(clinical trial|medical negligence|standard of care|duty of care|"
        r"informed consent|patient safety|licensing board)\b",
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
            "Likely healthcare-adjacent legal work based on patient-care regulation, malpractice, or casework language."
            if len(matches) >= 2
            else "No strong healthcare-adjacent legal signal detected."
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
    parser = argparse.ArgumentParser(description="Export law-related WildChat posts.")
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
    print("Clinical-style law work here usually looks like malpractice, patient rights, or healthcare-regulation case discussion.")


if __name__ == "__main__":
    main()