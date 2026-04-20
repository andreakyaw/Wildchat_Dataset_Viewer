"""
extract_fields.py
-----------------
Streams English conversations from WildChat-1M and exports ~1,000 matched
posts for each of Psychology, Medicine, and Law into separate JSONL files.

Clinical-work detection is run per-field so you can immediately see which
posts look like practitioner use vs. general-public queries.

Usage
-----
    python extract_fields.py               # 1 000 posts per field (default)
    python extract_fields.py --limit 500   # 500 per field
"""

import json
import re
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

#paths
BASE_DIR   = Path(__file__).resolve().parent
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUTPUT_DIR  = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LIMIT = 1_000

#field keyword patterns
FIELD_PATTERNS = {
    "Psychology": re.compile(
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
    ),
    "Medicine": re.compile(
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
    ),
    "Law": re.compile(
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
    ),
}

#clinical-work signal patterns (per field)
CLINICAL_PATTERNS = {
    "Psychology": {
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
    },
    "Medicine": {
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
    },
    "Law": {
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
    },
}

VERIFICATION_PATTERN = re.compile(
    r"\b(verify|validation|confirm|check|accurate|correct|citation|source|evidence|proof|"
    r"are you sure|fact.?check|double.?check|can you confirm|please confirm|references?|"
    r"based on what|how do you know|where did you get|is that true|truth|credible|"
    r"is this right|is that right|are these correct|are you certain|double check that)\b",
    re.I,
)

#helpers

def parse_conversation(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def get_user_text(conv):
    return "\n\n".join(
        m.get("content", "")
        for m in conv
        if isinstance(m, dict) and m.get("role") == "user"
    )


def detect_clinical(text, field):
    patterns = CLINICAL_PATTERNS[field]
    matched = [label for label, pat in patterns.items() if pat.search(text)]
    is_clinical = len(matched) >= 2
    summaries = {
        "Psychology": "Likely clinical: care/process language, provider-patient roles, or symptom documentation.",
        "Medicine":   "Likely clinical: patient symptoms, diagnostic framing, and treatment planning.",
        "Law":        "Likely clinical-adjacent: healthcare regulation, malpractice, or casework language.",
    }
    return {
        "is_clinical_work": is_clinical,
        "clinical_signals": matched,
        "clinical_summary": summaries[field] if is_clinical else "No strong clinical signal detected.",
    }


def count_turns(conv):
    return len([m for m in conv if isinstance(m, dict)])


def iter_source_rows():
    parquet_files = sorted(PARQUET_DIR.glob("chunk_*.parquet"))
    if parquet_files:
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for row in df.to_dict(orient="records"):
                yield row
        return
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    yield from dataset


#main collection function

def collect_all(limit: int = DEFAULT_LIMIT) -> dict[str, pd.DataFrame]:
    """Stream once, distribute matching rows into per-field buckets."""
    buckets: dict[str, list] = {f: [] for f in FIELD_PATTERNS}
    done = set()

    for row in iter_source_rows():
        if row.get("language") != "English":
            continue

        conv      = parse_conversation(row.get("conversation", []))
        user_text = get_user_text(conv)
        if not user_text:
            continue

        for field, pat in FIELD_PATTERNS.items():
            if field in done:
                continue
            if not pat.search(user_text):
                continue

            clinical = detect_clinical(user_text, field)
            buckets[field].append({
                "field":           field,
                "model":           row.get("model", "unknown"),
                "toxic":           row.get("toxic", False),
                "turn_count":      count_turns(conv),
                "has_verification": bool(VERIFICATION_PATTERN.search(user_text)),
                "user_text":       user_text,
                "conversation":    conv,
                **clinical,
            })

            if len(buckets[field]) >= limit:
                done.add(field)

        if len(done) == len(FIELD_PATTERNS):
            break  # all fields saturated

    return {field: pd.DataFrame(rows) for field, rows in buckets.items()}


#entry point 

def parse_args():
    p = argparse.ArgumentParser(description="Extract field-specific WildChat posts.")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Posts to collect per field (default: 1 000)")
    return p.parse_args()


def main():
    args  = parse_args()
    dfs   = collect_all(limit=args.limit)
    source = "local parquet" if any(PARQUET_DIR.glob("chunk_*.parquet")) else "HuggingFace streaming"

    for field, df in dfs.items():
        out_path = OUTPUT_DIR / f"{field.lower()}_posts.jsonl"
        df.to_json(out_path, orient="records", lines=True, force_ascii=True)
        clinical_n = int(df["is_clinical_work"].sum()) if not df.empty else 0
        print(f"[{field}]  saved {len(df):,} posts → {out_path}")
        print(f"          clinical-work flagged: {clinical_n:,} ({clinical_n/max(len(df),1)*100:.1f}%)")

    print(f"\nSource: {source}")
    print("Done.")


if __name__ == "__main__":
    main()
