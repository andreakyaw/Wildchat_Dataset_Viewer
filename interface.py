import streamlit as st
import pandas as pd
import json
import re
from datasets import load_dataset

st.set_page_config(layout="wide", page_title="WildChat Explorer")
st.title("WildChat Dataset Explorer")
st.caption("Browsing English-only conversations from the WildChat-1M dataset")


#Keywords for detecting fields
VERIFICATION_PATTERN = re.compile(
    r"\b(verify|validation|confirm|check|accurate|correct|citation|source|evidence|proof|"
    r"are you sure|fact.?check|double.?check|can you confirm|please confirm|references?|"
    r"based on what|how do you know|where did you get|is that true|truth|credible|"
    r"is this right|is that right|are these correct|are you certain|double check that)\b",
    re.I,
)

TOPIC_PATTERNS = {
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

#all eng

@st.cache_data(show_spinner=False)
def load_data(sample_size: int = 50_000) -> pd.DataFrame:
    collected = []
    dataset = load_dataset(
        "allenai/WildChat-1M",
        split="train",
         #fetch rows lazily instead of downloading the whole dataset
        streaming=True,
    )

    for row in dataset:
         # skip non-English conversations
        if row.get("language", "") != "English":
            continue
        #conversation is usually a list, but fall back to JSON parsing if not
        conv = row.get("conversation", [])
        if not isinstance(conv, list):
            try:
                conv = json.loads(conv)
            except Exception:
                #treat unparseable conversations as empty
                conv = []

        collected.append(
            {
                "model": row.get("model", "unknown"),
                "conversation": conv,
                "toxic": row.get("toxic", False),
            }
        )
        #stop once we've gathered enough rows
        if len(collected) >= sample_size:
            break

    return pd.DataFrame(collected)


#helpers 
#return list of topic labels (Psychology, Medicine, Law)

def get_user_text(conv: list) -> str:
    return "\n\n".join(
        msg.get("content", "")
        for msg in conv
        if isinstance(msg, dict) and msg.get("role") == "user"
    )

#return true if text contains any verification keywords
def detect_topics(text: str) -> list:
    return [
        topic
        for topic, pat in TOPIC_PATTERNS.items()
        if isinstance(text, str) and pat.search(text)
    ]

#return true if the text contains any verification language
def has_verification(text: str) -> bool:
    return bool(VERIFICATION_PATTERN.search(text)) if isinstance(text, str) else False

#highlight keyword in text
def highlight(text: str, keyword: str) -> str:
    if not keyword or not isinstance(text, str):
        return text
    return re.sub(
        f"({re.escape(keyword)})",
        r"<mark style='background:#fde68a;color:#111;border-radius:3px;padding:0 2px'>\1</mark>",
        text,
        flags=re.IGNORECASE,
    )


#process the data to the different info and highlight keywords
#each block runs only once
@st.cache_data(show_spinner=False)
def process(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    df["user_text"] = df["conversation"].apply(get_user_text)
    df["topics_list"] = df["user_text"].apply(detect_topics)
    df["has_verification"] = df["user_text"].apply(has_verification)
    return df
with st.spinner("Streaming English conversations from WildChat-1M..."):
    raw_df = load_data(sample_size=50_000)
with st.spinner("Running keyword detection..."):
    df = process(raw_df)

#build model list dynamically from whatever models appear in the loaded data
st.sidebar.header("Filters")
model_opts = ["All"] + sorted(df["model"].dropna().unique().tolist())
sel_model = st.sidebar.selectbox("Model", model_opts)
sel_keyword = st.sidebar.text_input("Keyword search (user messages only)")
sel_topic = st.sidebar.selectbox(
    "Field",
    ["All", "Psychology", "Medicine", "Law", "Any Field", "General (no field)"],
)

show_full = st.sidebar.checkbox("Show AI responses", value=False)
show_only_verified = st.sidebar.checkbox("Only show verification language", value=False)

#
@st.cache_data(show_spinner=False)
def apply_filters(
    _df: pd.DataFrame,
    model: str,
    keyword: str,
    topic: str,
    only_verified: bool,
) -> pd.DataFrame:
    out = _df.copy()

    if model != "All":
        out = out[out["model"] == model]
        #case-insensitive substring match against the pre-built user_text column
    if keyword:
        out = out[out["user_text"].str.contains(keyword, case=False, na=False)]
        #keep only rows with no detected topic
    if topic == "General (no field)":
        out = out[out["topics_list"].apply(len) == 0]
    #keep rows that matched at least one topic
    elif topic == "Any Field":
        out = out[out["topics_list"].apply(len) > 0]
    elif topic != "All":
        #keep rows where the specific topic is present in the list
        out = out[out["topics_list"].apply(lambda x: topic in x)]

    if only_verified:
        out = out[out["has_verification"]]

    return out


filtered_df = apply_filters(df, sel_model, sel_keyword, sel_topic, show_only_verified)

#render a conversation in chat bubbles
def render_conversation(conv: list, keyword: str = "", user_only: bool = True):
    st.markdown(
        '<div style="max-height:520px;overflow-y:auto;padding-right:6px;">',
        unsafe_allow_html=True,
    )

    rendered_any = False
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        #guard against None content
        content = msg.get("content", "") or ""

        if user_only and role != "user":
            continue

        rendered_any = True
        content_display = highlight(content, keyword)

        if role == "user":
            bubble_style = (
                "background:#dbeafe;color:#1e3a5f;"
                "padding:14px 16px;border-radius:16px 16px 16px 4px;"
                "margin-bottom:10px;max-width:820px;font-size:14.5px;line-height:1.65;"
            )
            label = "<b>User</b>"
        else:
            bubble_style = (
                "background:#f3f4f6;color:#1f2937;"
                "padding:14px 16px;border-radius:16px 16px 4px 16px;"
                "margin-bottom:10px;margin-left:auto;max-width:820px;"
                "font-size:14.5px;line-height:1.65;"
            )
            label = "<b>Assistant</b>"

        st.markdown(
            f'<div style="{bubble_style}">{label}<br><br>{content_display}</div>',
            unsafe_allow_html=True,
        )

    if not rendered_any:
        st.info("No messages to display for this filter.")

    st.markdown("</div>", unsafe_allow_html=True)

mode_label = "User messages only" if not show_full else "Full thread (user + AI)"
st.subheader(f"Conversations — {mode_label}")
#number of conversations shown per page
PAGE_SIZE = 50
total_results = len(filtered_df)
#always at least 1 page
max_page = max(1, (total_results - 1) // PAGE_SIZE + 1)
#result count on the left page selector on the right
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.write(f"**{total_results:,}** conversations match current filters")
with top_col2:
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)

start = (page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, total_results)
subset = filtered_df.iloc[start:end]

st.caption(f"Showing {start + 1}–{end} of {total_results:,}")

for i, row in subset.iterrows():
    topics_str = ", ".join(row["topics_list"]) if row["topics_list"] else "General"
    verif_badge = " [verified]" if row["has_verification"] else ""
    label = f"#{i}  |  {topics_str}{verif_badge}"

    with st.expander(label):
        info_cols = st.columns(3)
        info_cols[0].markdown(f"**Field(s):** {topics_str}")
        info_cols[1].markdown(f"**Model:** `{row.get('model', 'N/A')}`")
        info_cols[2].markdown(
            f"**Verification lang:** {'Yes' if row['has_verification'] else 'No'}"
        )

        render_conversation(
            row["conversation"],
            keyword=sel_keyword,
            #mirror the sidebar checkbox
            user_only=not show_full,
        )