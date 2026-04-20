"""
field_dashboard.py
------------------
Visual analytics dashboard for the Psychology / Medicine / Law exports
produced by extract_fields.py.

Run:
    streamlit run field_dashboard.py
"""

import json
import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

#page config
st.set_page_config(
    layout="wide",
    page_title="Field Analytics — WildChat",
    page_icon="📊",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "scripts" / "data"

FIELD_FILES = {
    "Psychology": DATA_DIR / "psychology_posts.jsonl",
    "Medicine":   DATA_DIR / "medicine_posts.jsonl",
    "Law":        DATA_DIR / "law_posts.jsonl",
}

FIELD_COLORS = {
    "Psychology": "#7F77DD",  
    "Medicine":   "#1D9E75",  
    "Law":        "#D85A30",   
}

CLINICAL_SIGNAL_LABELS = {
    #Psychology
    "active_care_terms":       "Active care terms",
    "provider_roles":          "Provider / patient roles",
    "symptom_discussion":      "Symptom discussion",
    "documentation_language":  "Documentation language",
    #Medicine
    "care_setting":            "Care setting",
    "patient_and_symptoms":    "Patient & symptoms",
    #Law
    "health_law_context":      "Health-law context",
    "casework_language":       "Casework language",
    "advice_request":          "Advice request",
    "regulated_care_terms":    "Regulated care terms",
}

#loading in data

@st.cache_data(show_spinner=False)
def load_field(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    #normalise list-typed columns that may have been stringified
    if "clinical_signals" in df.columns:
        df["clinical_signals"] = df["clinical_signals"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    if "turn_count" not in df.columns:
        df["turn_count"] = df.get("conversation", pd.Series(dtype=object)).apply(
            lambda c: len(c) if isinstance(c, list) else 0
        )
    if "has_verification" not in df.columns:
        df["has_verification"] = False
    return df


@st.cache_data(show_spinner=False)
def load_all() -> dict[str, pd.DataFrame]:
    return {field: load_field(path) for field, path in FIELD_FILES.items()}

def pct(n, total):
    if total == 0:
        return "0%"
    return f"{n / total * 100:.1f}%"


def render_text_box(text: str, background: str, max_chars: int, max_height: int) -> None:
    snippet = (text or "")[:max_chars]
    safe_text = escape(snippet).replace("\n", "<br>")
    ellipsis = "…" if len(text or "") > max_chars else ""
    st.markdown(
        f'<div style="background:{background};color:#0f172a;padding:12px 16px;'
        f'border:1px solid #cbd5e1;border-radius:8px;font-size:14px;line-height:1.65;'
        f'white-space:normal;max-height:{max_height}px;overflow-y:auto;">'
        f'{safe_text}{ellipsis}</div>',
        unsafe_allow_html=True,
    )


def signal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for signals in df["clinical_signals"]:
        for s in signals:
            counts[s] = counts.get(s, 0) + 1
    rows = [
        {"signal": CLINICAL_SIGNAL_LABELS.get(k, k), "count": v}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def model_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["model"]
        .value_counts()
        .reset_index()
        .rename(columns={"model": "model", "count": "count"})
        .head(10)
    )

#sidebar controls and data loading
all_dfs = load_all()
missing = [f for f, df in all_dfs.items() if df.empty]

st.sidebar.header("WildChat field explorer")
st.sidebar.caption("Select a field to analyse, or view the combined overview.")

view = st.sidebar.radio(
    "View",
    ["Overview (all fields)", "Psychology", "Medicine", "Law"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters (single-field views)**")
only_clinical = st.sidebar.checkbox("Clinical work only", value=False)
only_verified = st.sidebar.checkbox("Verification language only", value=False)

if missing:
    st.sidebar.warning(
        f"Missing data files:\n" + "\n".join(f"• {f}" for f in missing)
        + "\n\nRun `python extract_fields.py` to generate them."
    )

st.title("Field analytics — WildChat")
st.caption("Psychology · Medicine · Law — 1 000 posts per field from WildChat-1M")


if view == "Overview (all fields)":

    combined_rows = []
    for field, df in all_dfs.items():
        if df.empty:
            continue
        n = len(df)
        clinical_n = int(df["is_clinical_work"].sum())
        verif_n    = int(df["has_verification"].sum()) if "has_verification" in df.columns else 0
        combined_rows.append({
            "Field":    field,
            "Posts":    n,
            "Clinical": clinical_n,
            "Verif.":   verif_n,
            "Clinical %": round(clinical_n / n * 100, 1) if n else 0,
        })

    if not combined_rows:
        st.info("No data loaded yet. Run `python extract_fields.py` first.")
        st.stop()

    summary = pd.DataFrame(combined_rows)

    col1, col2, col3 = st.columns(3)
    for col, row in zip([col1, col2, col3], combined_rows):
        with col:
            st.metric(row["Field"], f"{row['Posts']:,} posts")
            st.caption(
                f"Clinical: **{row['Clinical']:,}** ({row['Clinical %']}%)  "
                f"· Verif: **{row['Verif.']:,}**"
            )

    st.markdown("---")

    # side-by-side bar: clinical %
    st.subheader("Clinical-work rate by field")
    c_chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Field:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Clinical %:Q", scale=alt.Scale(domain=[0, 20]),
                    title="% flagged as clinical work"),
            color=alt.Color(
                "Field:N",
                scale=alt.Scale(
                    domain=list(FIELD_COLORS.keys()),
                    range=list(FIELD_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=["Field", "Posts", "Clinical", "Clinical %"],
        )
        .properties(height=280)
    )
    st.altair_chart(c_chart, use_container_width=True)

    
    # verification language presence
    st.subheader("Verification language presence")
    verif_rows = []
    for row in combined_rows:
        verif_rows.append({"Field": row["Field"], "Category": "With verification", "n": row["Verif."]})
        verif_rows.append({"Field": row["Field"], "Category": "Without", "n": row["Posts"] - row["Verif."]})

    verif_df = pd.DataFrame(verif_rows)
    verif_chart = (
        alt.Chart(verif_df)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="posts"),
            y=alt.Y("Field:N"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=["With verification", "Without"], range=["#D85A30", "#D3D1C7"]),
            ),
            tooltip=["Field", "Category", "n"],
        )
        .properties(height=200)
    )
    st.altair_chart(verif_chart, use_container_width=True)


#single field view
else:
    field = view
    df_raw = all_dfs.get(field, pd.DataFrame())

    if df_raw.empty:
        st.warning(
            f"No data found for **{field}**. "
            f"Run `python extract_fields.py` to generate `{field.lower()}_posts.jsonl`."
        )
        st.stop()

    # apply sidebar filters
    df = df_raw.copy()
    if only_clinical and "is_clinical_work" in df.columns:
        df = df[df["is_clinical_work"]]
    if only_verified and "has_verification" in df.columns:
        df = df[df["has_verification"]]

    n          = len(df)
    clinical_n = int(df["is_clinical_work"].sum()) if "is_clinical_work" in df.columns else 0
    verif_n    = int(df["has_verification"].sum())  if "has_verification" in df.columns else 0

    color = FIELD_COLORS[field]

    st.subheader(f"{field} — {n:,} posts")

    #metrics cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Total posts",        f"{n:,}")
    m2.metric("Clinical work",      f"{clinical_n:,}",  delta=pct(clinical_n, n))
    m3.metric("Verification lang.", f"{verif_n:,}",     delta=pct(verif_n, n))

    st.markdown("---")

    left, right = st.columns([1, 1])

    #clinical vs general
    with left:
        st.markdown("**Clinical vs. general**")
        donut_df = pd.DataFrame({
            "Category": ["Clinical", "General"],
            "Count":    [clinical_n, n - clinical_n],
        })
        donut = (
            alt.Chart(donut_df)
            .mark_arc(innerRadius=55, outerRadius=100)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=["Clinical", "General"], range=[color, "#B4B2A9"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["Category", "Count"],
            )
            .properties(height=240)
        )
        st.altair_chart(donut, use_container_width=True)

    #clinical signal for comments flagged as clinical work
    with right:
        st.markdown("**What clinical work looks like signal breakdown**")
        sig_df = signal_breakdown(df[df["is_clinical_work"]] if "is_clinical_work" in df.columns else df)
        if sig_df.empty:
            st.caption("No clinical-work posts found.")
        else:
            sig_chart = (
                alt.Chart(sig_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("signal:N", sort="-x", title=""),
                    x=alt.X("count:Q", title="posts with signal"),
                    color=alt.value(color),
                    tooltip=["signal", "count"],
                )
                .properties(height=240)
            )
            st.altair_chart(sig_chart, use_container_width=True)


    st.markdown("---")
    left2, right2 = st.columns([1, 1])

    # model breakdown
    with left2:
        st.markdown("**Top models used**")
        mod_df = model_breakdown(df)
        if not mod_df.empty:
            mod_chart = (
                alt.Chart(mod_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("model:N", sort="-x", title=""),
                    x=alt.X("count:Q", title="conversations"),
                    color=alt.value(color),
                    tooltip=["model", "count"],
                )
                .properties(height=280)
            )
            st.altair_chart(mod_chart, use_container_width=True)

    #conversation length distribution
    with right2:
        st.markdown("**Conversation turn count**")
        if "turn_count" in df.columns and df["turn_count"].sum() > 0:
            turn_df = df["turn_count"].clip(upper=20).value_counts().reset_index()
            turn_df.columns = ["turns", "posts"]
            turn_df = turn_df.sort_values("turns")
            turn_chart = (
                alt.Chart(turn_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("turns:O", title="turns (capped at 20)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("posts:Q", title="posts"),
                    color=alt.value(color),
                    tooltip=["turns", "posts"],
                )
                .properties(height=280)
            )
            st.altair_chart(turn_chart, use_container_width=True)
        else:
            st.caption("Turn-count data not available (run extract_fields.py again).")

    st.markdown("---")

    #clinical examples
    st.subheader("Sample clinical-work posts")
    clinical_sample = df[df["is_clinical_work"]].head(5) if "is_clinical_work" in df.columns else pd.DataFrame()

    if clinical_sample.empty:
        st.info("No clinical-work posts in the current filtered view.")
    else:
        for _, row in clinical_sample.iterrows():
            signals = ", ".join(
                CLINICAL_SIGNAL_LABELS.get(s, s) for s in row.get("clinical_signals", [])
            )
            with st.expander(f"Signals: {signals}  |  Model: {row.get('model', 'N/A')}"):
                st.caption(row.get("clinical_summary", ""))
                user_text = row.get("user_text", "")
                render_text_box(user_text, background="#f8fafc", max_chars=2000, max_height=320)

    st.markdown("---")

    #full browse table
    st.subheader("Browse all posts")
    PAGE = 25
    total = len(df)
    max_p = max(1, (total - 1) // PAGE + 1)
    col_pg1, col_pg2 = st.columns([4, 1])
    with col_pg1:
        st.caption(f"{total:,} posts match current filters")
    with col_pg2:
        page = st.number_input("Page", 1, max_p, 1, step=1)

    start = (page - 1) * PAGE
    subset = df.iloc[start : start + PAGE]

    for _, row in subset.iterrows():
        signals = ", ".join(
            CLINICAL_SIGNAL_LABELS.get(s, s) for s in row.get("clinical_signals", [])
        ) or "none"
        clin_badge  = f"{field} · clinical" if row.get("is_clinical_work") else field
        verif_badge = " · ✓ verif"  if row.get("has_verification") else ""
        label = f"{clin_badge}{verif_badge}  |  {row.get('model','?')}  |  signals: {signals}"
        with st.expander(label):
            user_text = row.get("user_text", "")
            render_text_box(user_text, background="#eff6ff", max_chars=1500, max_height=280)
