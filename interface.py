import streamlit as st
import pandas as pd
import json
import re

st.set_page_config(layout="wide")
st.title("WildChat Dataset Visualization Interface")

# ------------------------
# LOAD DATA
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/parquet/chunk_0.parquet")
    return df.head(10000)  # LIMIT SIZE FOR SPEED

df = load_data()

# ------------------------
# PATTERNS
# ------------------------
VERIFICATION_PATTERN = re.compile(
    r"\b(verify|validation|confirm|check|accurate|correct|citation|source|evidence|proof|are you sure)\b",
    flags=re.IGNORECASE
)

TOPIC_PATTERNS = {
    "Psychology": re.compile(r"\b(psychology|therapy|mental health|anxiety|depression|stress|emotion)\b", re.I),
    "Medicine": re.compile(r"\b(doctor|patient|diagnosis|symptom|treatment|medical|medicine|medication|disease)\b", re.I),
    "Law": re.compile(r"\b(legal|lawyer|court|judge|contract|lawsuit|crime|police)\b", re.I),
}

# ------------------------
# HELPERS
# ------------------------
def normalize_conversation(conv):
    try:
        if hasattr(conv, "to_pylist"):
            conv = conv.to_pylist()
        elif hasattr(conv, "tolist") and not isinstance(conv, (list, dict, str)):
            conv = conv.tolist()

        if isinstance(conv, str):
            conv = json.loads(conv)

        if isinstance(conv, list):
            return [msg for msg in conv if isinstance(msg, dict)]
    except:
        pass
    return []

def get_user_turn_text(conv):
    return "\n\n".join(
        msg.get("content", "")
        for msg in conv if msg.get("role") == "user"
    )

def has_verification_language(text):
    return bool(VERIFICATION_PATTERN.search(text)) if isinstance(text, str) else False

def detect_topic_multi(text):
    matches = []
    for topic, pattern in TOPIC_PATTERNS.items():
        if isinstance(text, str) and pattern.search(text):
            matches.append(topic)
    return matches if matches else ["Psychology"]

def highlight_keyword(text, keyword):
    if not keyword or not isinstance(text, str):
        return text
    return re.sub(
        f"({re.escape(keyword)})",
        r"<span style='background-color:yellow; color:black;'>\1</span>",
        text,
        flags=re.IGNORECASE
    )

# ------------------------
# PROCESS DATA (CACHED)
# ------------------------
@st.cache_data
def process_data(df):
    df = df.copy()
    df["raw_conversation"] = df["conversation"].apply(normalize_conversation)
    df["user_text"] = df["raw_conversation"].apply(get_user_turn_text)
    df["has_verification_language"] = df["user_text"].apply(has_verification_language)
    df["topics_list"] = df["user_text"].apply(detect_topic_multi)

    exploded_df = df.explode("topics_list")
    exploded_df["topic"] = exploded_df["topics_list"]

    return df, exploded_df

df, exploded_df = process_data(df)

# ------------------------
# FILTERING (CACHED)
# ------------------------
@st.cache_data
def filter_data(df, language, model, keyword, topic_filter):
    filtered = df

    if language != "All":
        filtered = filtered[filtered["language"] == language]

    if model != "All":
        filtered = filtered[filtered["model"] == model]

    if keyword:
        filtered = filtered[
            filtered["user_text"].str.contains(keyword, case=False, na=False)
        ]

    if topic_filter != "All":
        filtered = filtered[
            filtered["topics_list"].apply(lambda x: topic_filter in x)
        ]

    return filtered

# ------------------------
# SIDEBAR FILTERS
# ------------------------
st.sidebar.header("Filters")

language = st.sidebar.selectbox("Language", ["All"] + sorted(df["language"].dropna().unique()))
model = st.sidebar.selectbox("Model", ["All"] + sorted(df["model"].dropna().unique()))
keyword = st.sidebar.text_input("Search keyword")

topic_filter = st.sidebar.selectbox(
    "Field",
    ["All", "Psychology", "Medicine", "Law"]
)

filtered_df = filter_data(df, language, model, keyword, topic_filter)

# ------------------------
# CHARTS (CACHED)
# ------------------------
@st.cache_data
def compute_field_counts(exploded_df):
    return exploded_df["topic"].value_counts()

@st.cache_data
def compute_validation(exploded_df):
    return (
        exploded_df.groupby("topic")["has_verification_language"]
        .mean()
        .mul(100)
    )

st.subheader("Field Distribution")
st.bar_chart(compute_field_counts(exploded_df))

st.subheader("Verification Behavior by Field")
st.bar_chart(compute_validation(exploded_df))

# ------------------------
# CHAT RENDERER
# ------------------------
def render_conversation(conv, keyword=None):
    conv = normalize_conversation(conv)

    st.markdown('<div style="max-height:500px; overflow-y:auto;">', unsafe_allow_html=True)

    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if keyword:
            content = highlight_keyword(content, keyword)

        if role == "user":
            st.markdown(f"""
                <div style="
                    background-color:#dbeafe;
                    color:#111827;
                    padding:14px;
                    border-radius:14px;
                    margin-bottom:12px;
                    max-width:700px;
                    font-size:15px;
                ">
                👤 <b>User</b><br><br>{content}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="
                    background-color:#e5e7eb;
                    color:#111827;
                    padding:14px;
                    border-radius:14px;
                    margin-bottom:12px;
                    margin-left:auto;
                    max-width:700px;
                    font-size:15px;
                ">
                🤖 <b>Assistant</b><br><br>{content}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# PAGINATION (BIG SPEED BOOST)
# ------------------------
st.subheader("Conversations")

page_size = 5
page = st.number_input("Page", min_value=1, value=1)

start = (page - 1) * page_size
end = start + page_size

subset = filtered_df.iloc[start:end]

st.write(f"Showing {start} - {end} of {len(filtered_df)} results")

for i, row in subset.iterrows():
    topics = ", ".join(row["topics_list"])

    with st.expander(f"Conversation {i} | {topics}"):
        st.markdown(f"**Field(s):** {topics}")
        render_conversation(row["raw_conversation"], keyword)