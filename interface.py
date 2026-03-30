import streamlit as st
import pandas as pd
import json
import re

st.set_page_config(layout="wide")
st.title("WildChat Dataset Visualization Interface")

#load data
@st.cache_data
def load_data():
    return pd.read_parquet("data/parquet/chunk_0.parquet")

df = load_data()

# Format Conversation Column
def format_conv(conv):
    try:
        if hasattr(conv, "to_pylist"):
            conv = conv.to_pylist()

        if isinstance(conv, str):
            conv = json.loads(conv)

        return "\n\n".join(
            f"{msg.get('role','')}: {msg.get('content','')}"
            for msg in conv if isinstance(msg, dict)
        )
    except:
        return str(conv)

#format columns
def format_json(obj):
    try:
        if hasattr(obj, "to_pylist"):
            obj = obj.to_pylist()
        if hasattr(obj, "as_py"):
            obj = obj.as_py()

        if isinstance(obj, list) and len(obj) > 0:
            obj = obj[0]

        if isinstance(obj, dict):
            return ", ".join(f"{k}:{v}" for k, v in obj.items())

        return str(obj)
    except:
        return str(obj)

# Apply formatting
df["conversation"] = df["conversation"].apply(format_conv)
df["openai_moderation"] = df["openai_moderation"].apply(format_json)
df["detoxify_moderation"] = df["detoxify_moderation"].apply(format_json)

#filter
st.sidebar.header("Filters")

language = st.sidebar.selectbox(
    "Language",
    options=df["language"].dropna().unique()
)

model = st.sidebar.selectbox(
    "Model",
    options=df["model"].dropna().unique()
)
#Keyword search input
keyword = st.sidebar.text_input("Search keyword (in conversation)")
use_regex = st.sidebar.checkbox("Use regex")

filtered_df = df[
    (df["language"] == language) &
    (df["model"] == model)
].copy()

if keyword:
    filtered_df = filtered_df[
        filtered_df["conversation"].str.contains(
            keyword,
            case=False,
            na=False,
            regex=use_regex
        )
    ]

#Keyword
def highlight_keyword(text, keyword):
    if not keyword or not isinstance(text, str):
        return text
    return re.sub(
        f"({re.escape(keyword)})",
        r"**\1**",
        text,
        flags=re.IGNORECASE
    )

if keyword:
    filtered_df["conversation"] = filtered_df["conversation"].apply(
        lambda x: highlight_keyword(x, keyword)
    )

#charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Language Distribution")
    st.bar_chart(df["language"].value_counts())

with col2:
    st.subheader("Model Usage")
    st.bar_chart(df["model"].value_counts())

#convo len
filtered_df.loc[:, "conversation_length"] = filtered_df["conversation"].apply(
    lambda x: x.count("\n") if isinstance(x, str) else 0
)

st.subheader("Conversation Length Distribution")
st.bar_chart(filtered_df["conversation_length"].value_counts().sort_index())

#count how many results 
st.write(f"Results found: {len(filtered_df)}")

#data
st.subheader("Filtered Conversations")

st.dataframe(
    filtered_df[
        [
            "conversation",
            "language",
            "model",
            "toxic",
            "openai_moderation",
            "detoxify_moderation"
        ]
    ].head(50)
)