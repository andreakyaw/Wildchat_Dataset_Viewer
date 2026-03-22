import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("WildChat Dataset Visualization Interface")

#load in the data
@st.cache_data
def load_data():
    return pd.read_parquet("data/parquet/chunk_0.parquet")
df = load_data()

#convos column
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

#fix columns mod
def format_json(obj):
    try:
        if hasattr(obj, "to_pylist"):
            obj = obj.to_pylist()
        if hasattr(obj, "as_py"):
            obj = obj.as_py()
        #if its a list then take first eleme
        if isinstance(obj, list) and len(obj) > 0:
            obj = obj[0]
        #if its a dict then flatten
        if isinstance(obj, dict):
            return ", ".join(f"{k}:{v}" for k, v in obj.items())
        return str(obj)
    except:
        return str(obj)
df["conversation"] = df["conversation"].apply(format_conv)
df["openai_moderation"] = df["openai_moderation"].apply(format_json)
df["detoxify_moderation"] = df["detoxify_moderation"].apply(format_json)

#filter it
st.sidebar.header("Filters")
language = st.sidebar.selectbox(
    "Language",
    options=df["language"].dropna().unique()
)
model = st.sidebar.selectbox(
    "Model",
    options=df["model"].dropna().unique()
)
filtered_df = df[
    (df["language"] == language) &
    (df["model"] == model)
].copy()

#layout of tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("Language Distribution")
    st.bar_chart(df["language"].value_counts())

with col2:
    st.subheader("Model Usage")
    st.bar_chart(df["model"].value_counts())

#convos len
filtered_df.loc[:, "conversation_length"] = filtered_df["conversation"].apply(
    lambda x: x.count("\n") if isinstance(x, str) else 0
)
st.subheader("Conversation Length Distribution")
st.bar_chart(filtered_df["conversation_length"].value_counts().sort_index())

#data table obj fix
st.subheader("Filtered Conversations")
st.dataframe(
    filtered_df[
        ["conversation", "language", "model", "toxic", "openai_moderation", "detoxify_moderation"]
    ].head(50)
)