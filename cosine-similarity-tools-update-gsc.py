import streamlit as st
import pandas as pd
import numpy as np
import math
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Streamlit page config ---
st.set_page_config(
    page_title="GSC Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- Helper Functions ---
def get_openai_client():
    """Initialize OpenAI client from Streamlit secrets."""
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.warning("OpenAI API key not found. GPT labeling will be disabled.")
        return None
    return OpenAI(api_key=api_key)

def clean_metric(series: pd.Series) -> pd.Series:
    """Strip % and commas, coerce to numeric."""
    s = series.astype(str).str.replace("%", "", regex=False)\
                         .str.replace(r"[<>,]", "", regex=True)\
                         .str.strip()\
                         .replace({"": np.nan, "N/A": np.nan, "--": np.nan})
    return pd.to_numeric(s, errors="coerce")

def calculate_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted average, ignoring zero/NaN weights."""
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return values.mean() if values.notna().any() else np.nan
    try:
        return np.average(values[valid], weights=weights[valid])
    except ZeroDivisionError:
        return values[valid].mean()

def calculate_yoy_pct_change(delta: float, before: float) -> float:
    """YOY % change, handling zero and NaN safely."""
    if pd.isna(delta) or pd.isna(before):
        return np.nan
    if before == 0:
        return np.inf if delta != 0 else 0.0
    return delta / before * 100

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_gpt_cluster_label(client: OpenAI, queries: list, cluster_id: int) -> str:
    """Ask GPT for a short cluster label."""
    prompt = (
        "The following queries belong to one topic cluster: "
        + ", ".join(queries[:15])
        + ". Provide a concise 3–5 word Title Case label."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are an SEO expert."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        label = resp.choices[0].message.content.strip()
        return label or f"Cluster {cluster_id}"
    except Exception:
        return f"Cluster {cluster_id}"

# --- Main GSC Analyzer Page ---
def google_search_console_analysis_page():
    st.title("🔍 Google Search Console Analyzer")
    st.markdown("""
    Upload two CSVs of your GSC query data — one for the **Before** period and one for the **After** period — 
    then review YOY changes by query cluster.
    """)

    before_file = st.file_uploader("Upload GSC CSV: BEFORE period", type="csv", key="before")
    after_file  = st.file_uploader("Upload GSC CSV: AFTER period",  type="csv", key="after")

    if not before_file or not after_file:
        st.info("Please upload both CSV files to begin analysis.")
        return

    client = get_openai_client()
    progress = st.progress(0)
    status = st.empty()

    # --- 1) Read & Clean ---
    status.text("Reading and cleaning data…")
    df_b = pd.read_csv(before_file)
    df_a = pd.read_csv(after_file)

    def find_col(df, names):
        for name in names:
            for col in df.columns:
                if col.strip().lower() == name.lower():
                    return col
        return None

    # Identify columns
    qc_b = find_col(df_b, ["Top queries", "Query"])
    qc_a = find_col(df_a, ["Top queries", "Query"])
    pos_b = find_col(df_b, ["Average position", "Position"])
    pos_a = find_col(df_a, ["Average position", "Position"])
    clk_b = find_col(df_b, ["Clicks"])
    clk_a = find_col(df_a, ["Clicks"])
    imp_b = find_col(df_b, ["Impressions", "Impr."])
    imp_a = find_col(df_a, ["Impressions", "Impr."])
    ctr_b = find_col(df_b, ["CTR"])
    ctr_a = find_col(df_a, ["CTR"])

    # Rename to standard
    rename_b = {qc_b:"Query", pos_b:"AvgPos"}
    rename_a = {qc_a:"Query", pos_a:"AvgPos"}
    if clk_b: rename_b[clk_b] = "Clicks"
    if imp_b: rename_b[imp_b] = "Impressions"
    if ctr_b: rename_b[ctr_b] = "CTR"
    if clk_a: rename_a[clk_a] = "Clicks"
    if imp_a: rename_a[imp_a] = "Impressions"
    if ctr_a: rename_a[ctr_a] = "CTR"

    df_b = df_b.rename(columns=rename_b)
    df_a = df_a.rename(columns=rename_a)

    # Clean metrics
    metrics = ["AvgPos","Clicks","Impressions","CTR"]
    for df in (df_b, df_a):
        for m in metrics:
            if m in df:
                df[m] = clean_metric(df[m])

    progress.progress(10)

    # --- 2) Dashboard Summary ---
    status.text("Computing overall changes…")
    col1, col2, col3, col4 = st.columns(4)

    # Clicks change
    if "Clicks" in df_b and "Clicks" in df_a:
        cb, ca = df_b["Clicks"].sum(), df_a["Clicks"].sum()
        d  = ca - cb
        dp = calculate_yoy_pct_change(d, cb)
        col1.metric("Clicks Δ", f"{d:,.0f}", f"{dp:.1f}%")
    else:
        col1.metric("Clicks Δ","N/A")

    # Impressions change
    if "Impressions" in df_b and "Impressions" in df_a:
        ib, ia = df_b["Impressions"].sum(), df_a["Impressions"].sum()
        d  = ia - ib
        dp = calculate_yoy_pct_change(d, ib)
        col2.metric("Impr. Δ", f"{d:,.0f}", f"{dp:.1f}%")
    else:
        col2.metric("Impr. Δ","N/A")

    # AvgPos change (weighted by Impressions)
    if "AvgPos" in df_b and "AvgPos" in df_a and "Impressions" in df_b and "Impressions" in df_a:
        apb = calculate_weighted_average(df_b["AvgPos"], df_b["Impressions"])
        apa = calculate_weighted_average(df_a["AvgPos"], df_a["Impressions"])
        d   = apb - apa
        dp  = calculate_yoy_pct_change(d, apb)
        col3.metric("AvgPos Δ", f"{d:.1f}", f"{dp:.1f}%", delta_color="inverse")
    else:
        col3.metric("AvgPos Δ","N/A")

    # CTR change
    if "CTR" in df_b and "CTR" in df_a and "Impressions" in df_b and "Impressions" in df_a:
        cb = calculate_weighted_average(df_b["CTR"], df_b["Impressions"])
        ca = calculate_weighted_average(df_a["CTR"], df_a["Impressions"])
        d  = ca - cb
        dp = calculate_yoy_pct_change(d, cb)
        col4.metric("CTR Δ", f"{d:.2f}% pts", f"{dp:.1f}%")
    else:
        col4.metric("CTR Δ","N/A")

    progress.progress(20)

    # --- 3) Merge & Compute YOY by Query ---
    status.text("Merging data and computing YOY…")
    to_keep = ["Query"] + [m for m in metrics if m in df_b and m in df_a]
    merged = pd.merge(
        df_b[to_keep], df_a[to_keep],
        on="Query", suffixes=("_b","_a"),
        how="outer"
    )

    if merged.empty:
        st.error("No overlapping queries to merge.")
        return

    # Absolute changes
    if "Clicks_b" in merged and "Clicks_a" in merged:
        merged["Δ_Clicks"] = merged["Clicks_a"] - merged["Clicks_b"]
    if "Impressions_b" in merged and "Impressions_a" in merged:
        merged["Δ_Impr"] = merged["Impressions_a"] - merged["Impressions_b"]
    if "CTR_b" in merged and "CTR_a" in merged:
        merged["Δ_CTR"] = merged["CTR_a"] - merged["CTR_b"]
    if "AvgPos_b" in merged and "AvgPos_a" in merged:
        merged["Δ_Pos"] = merged["AvgPos_b"] - merged["AvgPos_a"]

    # % changes
    for col, base in [("Δ_Clicks","Clicks_b"), ("Δ_Impr","Impressions_b"),
                      ("Δ_CTR","CTR_b"),    ("Δ_Pos","AvgPos_b")]:
        pct = col.replace("Δ_","Δ%_")
        if col in merged and base in merged:
            merged[pct] = merged.apply(lambda r: calculate_yoy_pct_change(r[col], r[base]), axis=1)

    progress.progress(30)

    # --- 4) Embeddings & Clustering ---
    status.text("Embedding queries and clustering…")
    model = load_sentence_transformer()
    qs = merged["Query"].fillna("").unique().tolist()
    embs = model.encode(qs, show_progress_bar=False)

    # Let user choose K, default to 10
    max_k = min(30, len(qs)-1) if len(qs)>1 else 1
    k_default = 10 if max_k >= 10 else max_k
    k = st.slider("Number of clusters (K)", min_value=1, max_value=max_k, value=k_default, step=1)

    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(embs)
    merged["Cluster"] = merged["Query"].map(dict(zip(qs, labels)))

    progress.progress(50)

    # --- 5) GPT Labeling ---
    status.text("Generating topic labels…")
    topics = {}
    if client:
        prog2 = st.progress(0)
        clu_ids = sorted(merged["Cluster"].unique())
        for i, cid in enumerate(clu_ids):
            qs_in = merged[merged["Cluster"]==cid]["Query"].tolist()
            topics[cid] = get_gpt_cluster_label(client, qs_in, cid)
            prog2.progress((i+1)/len(clu_ids))
        prog2.empty()
    else:
        for cid in merged["Cluster"].unique():
            topics[cid] = f"Cluster {cid}"

    merged["Topic"] = merged["Cluster"].map(topics)
    progress.progress(70)

    # --- 6) Display & Aggregate by Topic ---
    st.subheader("Merged Data with Topics")
    st.dataframe(merged, use_container_width=True)

    st.subheader("Aggregated by Topic")
    agg_funcs = {
        "Clicks_b":"sum","Clicks_a":"sum","Δ_Clicks":"sum","Δ%_Clicks":"mean",
        "Impressions_b":"sum","Impressions_a":"sum","Δ_Impr":"sum","Δ%_Impr":"mean",
        "CTR_b":"mean","CTR_a":"mean","Δ_CTR":"mean","Δ%_CTR":"mean",
        "AvgPos_b":"mean","AvgPos_a":"mean","Δ_Pos":"mean","Δ%_Pos":"mean"
    }
    valid = {c:agg_funcs[c] for c in agg_funcs if c in merged.columns}
    agg = merged.groupby("Topic").agg(valid).reset_index()
    st.dataframe(agg, use_container_width=True)

    progress.progress(100)
    status.text("Done! 🎉")

# --- Run the app ---
if __name__ == "__main__":
    google_search_console_analysis_page()
