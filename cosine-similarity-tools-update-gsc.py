import streamlit as st
import pandas as pd
import numpy as np
import math
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import plotly.express as px

# --- Helper to initialize OpenAI client ---
def get_openai_client():
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]
    if not api_key:
        st.warning("OpenAI API key not found in Streamlit Secrets. GPT labeling will be disabled.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

# --- Cache SBERT model ---
@st.cache_resource
def initialize_sentence_transformer():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- Main GSC Analyzer Tool ---
def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        Compare GSC query data from two periods to identify performance changes.
        This tool uses:
        1. **KMeans clustering** on query embeddings (SentenceTransformer)
        2. **GPT-based labeling** (optional) to group queries into topics.
        Upload CSV files for the 'Before' and 'After' periods, then:
        - Compute overall dashboard metrics
        - Merge data (outer join)
        - Calculate YOY absolute & % changes
        - Cluster queries and label topics
        - Aggregate metrics by topic
        - Visualize YOY % change by topic for each metric
        **Note:** Requires an OpenAI API key in Streamlit Secrets for topic labeling.
        """
    )

    # --- Upload files ---
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after  = st.file_uploader("Upload GSC CSV for 'After' period",  type=["csv"], key="gsc_after")

    if not uploaded_file_before or not uploaded_file_after:
        st.info("Please upload both GSC CSV files to begin analysis.")
        return

    openai_client = get_openai_client()
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Steps 1–4: Read, Rename, Clean, Dashboard
        status_text.text("Reading, cleaning, and computing dashboard metrics...")
        df_before = pd.read_csv(uploaded_file_before)
        df_after  = pd.read_csv(uploaded_file_after)

        def find_col_name(df, names):
            for name in names:
                for col in df.columns:
                    if col.strip().lower() == name.strip().lower():
                        return col
            return None

        # Identify columns
        query_col_before = find_col_name(df_before, ["Top queries", "Query"])
        pos_col_before   = find_col_name(df_before, ["Average position", "Position"])
        query_col_after  = find_col_name(df_after,  ["Top queries", "Query"])
        pos_col_after    = find_col_name(df_after,  ["Average position", "Position"])
        clicks_before    = find_col_name(df_before, ["Clicks"])
        clicks_after     = find_col_name(df_after,  ["Clicks"])
        impr_before      = find_col_name(df_before, ["Impressions", "Impr."])
        impr_after       = find_col_name(df_after,  ["Impressions", "Impr."])
        ctr_before       = find_col_name(df_before, ["CTR"])
        ctr_after        = find_col_name(df_after,  ["CTR"])

        if not query_col_before or not pos_col_before or not query_col_after or not pos_col_after:
            st.error("Could not detect Query/Position columns in one or both files.")
            return

        # Rename maps
        rename_map_before = {query_col_before: "Query", pos_col_before: "Average Position"}
        rename_map_after  = {query_col_after:  "Query", pos_col_after:  "Average Position"}
        for col, std in [(clicks_before,"Clicks"), (impr_before,"Impressions"), (ctr_before,"CTR")]:
            if col: rename_map_before[col] = std
        for col, std in [(clicks_after,"Clicks"), (impr_after,"Impressions"), (ctr_after,"CTR")]:
            if col: rename_map_after[col] = std

        df_before = df_before.rename(columns=rename_map_before)
        df_after  = df_after.rename(columns=rename_map_after)

        # Cleaning helper
        def clean_metric(series):
            if pd.api.types.is_numeric_dtype(series):
                return series
            s = series.astype(str)\
                      .str.replace('%','',regex=False)\
                      .str.replace('<|>|,','',regex=True)\
                      .str.strip()\
                      .replace({'':np.nan,'N/A':np.nan,'--':np.nan})
            return pd.to_numeric(s, errors='coerce')

        for df in (df_before, df_after):
            for m in ["Average Position","Clicks","Impressions","CTR"]:
                if m in df.columns:
                    df[m] = clean_metric(df[m])

        progress_bar.progress(10)

        # Dashboard summary
        def weigh_avg(vals, wts):
            valid = vals.notna() & wts.notna() & (wts>0)
            if not valid.any():
                return vals.mean() if vals.notna().any() else np.nan
            try:
                return np.average(vals[valid], weights=wts[valid])
            except ZeroDivisionError:
                return vals[valid].mean()

        def calc_yoy_pct(delta, before):
            if pd.isna(delta) or pd.isna(before):
                return np.nan
            if before == 0:
                return np.inf if delta!=0 else 0.0
            return (delta/before)*100

        cols = st.columns(4)
        # Clicks
        if "Clicks" in df_before and "Clicks" in df_after:
            cb, ca = df_before["Clicks"].sum(), df_after["Clicks"].sum()
            d = ca - cb
            dp = calc_yoy_pct(d, cb)
            cols[0].metric("Clicks Change", f"{d:,.0f}", f"{dp:.1f}%")
        else:
            cols[0].metric("Clicks Change","N/A")
        # Impressions
        if "Impressions" in df_before and "Impressions" in df_after:
            ib, ia = df_before["Impressions"].sum(), df_after["Impressions"].sum()
            d = ia - ib
            dp = calc_yoy_pct(d, ib)
            cols[1].metric("Impressions Change", f"{d:,.0f}", f"{dp:.1f}%")
        else:
            cols[1].metric("Impressions Change","N/A")
        # Average Position
        if all(k in df_before for k in ["Average Position","Impressions"]) and all(k in df_after for k in ["Average Position","Impressions"]):
            apb = weigh_avg(df_before["Average Position"], df_before["Impressions"])
            apa = weigh_avg(df_after["Average Position"],  df_after["Impressions"])
            d   = apb - apa
            dp  = calc_yoy_pct(d, apb)
            cols[2].metric("Avg. Position Change", f"{d:.1f}", f"{dp:.1f}%", delta_color="inverse")
        else:
            cols[2].metric("Avg. Position Change","N/A")
        # CTR
        if all(k in df_before for k in ["CTR","Impressions"]) and all(k in df_after for k in ["CTR","Impressions"]):
            cb = weigh_avg(df_before["CTR"], df_before["Impressions"])
            ca = weigh_avg(df_after["CTR"],  df_after["Impressions"])
            d  = ca - cb
            dp = calc_yoy_pct(d, cb)
            cols[3].metric("Avg. CTR Change", f"{d:.2f}% pts", f"{dp:.1f}%")
        else:
            cols[3].metric("Avg. CTR Change","N/A")

        progress_bar.progress(20)

        # Step 5: Merge and YOY per query
        status_text.text("Merging data and calculating YOY per query…")
        metrics = ["Average Position","Clicks","Impressions","CTR"]
        cols_before = ["Query"] + [m for m in metrics if m+"_b" not in locals() and m in df_before]
        cols_after  = ["Query"] + [m for m in metrics if m in df_after]
        merged_df = pd.merge(
            df_before[["Query"] + [m for m in metrics if m in df_before]].rename(columns={m:m+"_before" for m in metrics if m in df_before}),
            df_after[ ["Query"] + [m for m in metrics if m in df_after ]].rename(columns={m:m+"_after" for m in metrics if m in df_after}),
            on="Query", how="outer"
        )

        # Absolute YOY
        if "Clicks_before" in merged_df and "Clicks_after" in merged_df:
            merged_df["Clicks_YOY"] = merged_df["Clicks_after"] - merged_df["Clicks_before"]
        if "Impressions_before" in merged_df and "Impressions_after" in merged_df:
            merged_df["Impressions_YOY"] = merged_df["Impressions_after"] - merged_df["Impressions_before"]
        if "CTR_before" in merged_df and "CTR_after" in merged_df:
            merged_df["CTR_YOY"] = merged_df["CTR_after"] - merged_df["CTR_before"]
        if "Average Position_before" in merged_df and "Average Position_after" in merged_df:
            merged_df["Position_YOY"] = merged_df["Average Position_before"] - merged_df["Average Position_after"]

        # % YOY
        for abs_col, base_col in [
            ("Clicks_YOY","Clicks_before"),
            ("Impressions_YOY","Impressions_before"),
            ("CTR_YOY","CTR_before"),
            ("Position_YOY","Average Position_before"),
        ]:
            pct_col = abs_col.replace("_YOY","_YOY_pct")
            if abs_col in merged_df and base_col in merged_df:
                merged_df[pct_col] = merged_df.apply(
                    lambda r: calc_yoy_pct(r[abs_col], r[base_col]), axis=1
                )

        progress_bar.progress(35)

        # Step 7: Embeddings
        status_text.text("Embedding queries and clustering…")
        model = initialize_sentence_transformer()
        queries = merged_df["Query"].fillna("").astype(str).unique().tolist()
        embeddings = model.encode(queries, show_progress_bar=True)

        # Silhouette-based K suggestion (3 to min(30,n−1))
        n = len(queries)
        max_k = min(30, n-1) if n>1 else 1
        min_k = 3
        optimal_k = min(max(min_k, max_k//2), max_k) if max_k>=min_k else max_k
        if max_k>=min_k:
            scores = {}
            for k in range(min_k, max_k+1):
                lab = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(embeddings)
                scores[k] = silhouette_score(embeddings, lab) if k>1 else -1
            best = max(scores, key=scores.get)
            optimal_k = best

        slider_min = max(1, min_k)
        slider_max = max(1, max_k)
        slider_default = optimal_k
        n_clusters_selected = st.slider(
            "Select number of query clusters (K):",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
            key="kmeans_clusters_gsc"
        )

        kmeans = KMeans(n_clusters=n_clusters_selected, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        merged_df["Cluster_ID"] = pd.Series(labels, index=merged_df.loc[merged_df["Query"].isin(queries)].index).astype('Int64')

        progress_bar.progress(55)

        # Step 9: GPT labeling
        status_text.text("Generating topic labels with GPT…")
        cluster_topics = {}
        valid_clusters = merged_df["Cluster_ID"].dropna().unique().tolist()
        if openai_client:
            bar2 = st.progress(0)
            for i, cid in enumerate(sorted(valid_clusters)):
                qlist = merged_df[merged_df["Cluster_ID"]==cid]["Query"].unique().tolist()
                prompt = (
                    "The following search queries belong to one topic cluster: "
                    + ", ".join(qlist[:15])
                    + ". Give a short 3–5 word topic label in Title Case."
                )
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role":"system","content":"You are an expert SEO assistant."},
                            {"role":"user","content":prompt}
                        ],
                        temperature=0.4,
                        max_tokens=20,
                        timeout=20.0
                    )
                    topic = resp.choices[0].message.content.strip().replace('"','')
                    topic = topic or f"Cluster {cid}"
                except Exception:
                    topic = f"Cluster {cid}"
                cluster_topics[cid] = topic
                bar2.progress((i+1)/len(valid_clusters))
            bar2.empty()
        else:
            for cid in valid_clusters:
                cluster_topics[cid] = f"Cluster {cid}"
        cluster_topics[pd.NA] = "Unclustered"
        merged_df["Query_Topic"] = merged_df["Cluster_ID"].map(cluster_topics).fillna("Unclustered")

        progress_bar.progress(70)

        # Step 10: Display merged table
        st.markdown("### Combined Data with Topic Labels")
        display_cols = ["Query","Cluster_ID","Query_Topic"]
        for metric in ["Average Position","Clicks","Impressions","CTR"]:
            for suffix in ["_before","_after","_YOY","_YOY_pct"]:
                col = (metric if suffix in ["_before","_after"] else
                       "Position" if metric=="Average Position" and suffix=="_YOY" else
                       "Position_YOY_pct" if metric=="Average Position" and suffix=="_YOY_pct" else
                       f"{metric}{suffix}")
                if col in merged_df.columns:
                    display_cols.append(col)
        st.dataframe(merged_df[display_cols], use_container_width=True)

        # Step 11: Aggregate & visualize YOY % by topic
        st.markdown("### Aggregated Metrics by Topic")
        agg_dict = {}
        if "Average Position_before" in merged_df and "Impressions_before" in merged_df:
            agg_dict["Average Position_before"] = lambda x: weigh_avg(x, merged_df.loc[x.index,"Impressions_before"])
        elif "Average Position_before" in merged_df:
            agg_dict["Average Position_before"] = "mean"
        if "Average Position_after" in merged_df and "Impressions_after" in merged_df:
            agg_dict["Average Position_after"] = lambda x: weigh_avg(x, merged_df.loc[x.index,"Impressions_after"])
        elif "Average Position_after" in merged_df:
            agg_dict["Average Position_after"] = "mean"
        for m in ["Clicks","Impressions"]:
            if f"{m}_before" in merged_df: agg_dict[f"{m}_before"] = "sum"
            if f"{m}_after"  in merged_df: agg_dict[f"{m}_after"]  = "sum"
        if "CTR_before" in merged_df and "Impressions_before" in merged_df:
            agg_dict["CTR_before"] = lambda x: weigh_avg(x, merged_df.loc[x.index,"Impressions_before"])
        elif "CTR_before" in merged_df:
            agg_dict["CTR_before"] = "mean"
        if "CTR_after" in merged_df and "Impressions_after" in merged_df:
            agg_dict["CTR_after"] = lambda x: weigh_avg(x, merged_df.loc[x.index,"Impressions_after"])
        elif "CTR_after" in merged_df:
            agg_dict["CTR_after"] = "mean"

        aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index()
        aggregated.rename(columns={"Query_Topic":"Topic"}, inplace=True)

        # Recalculate aggregated YOY
        if "Average Position_before" in aggregated and "Average Position_after" in aggregated:
            aggregated["Position_YOY"] = aggregated["Average Position_before"] - aggregated["Average Position_after"]
            aggregated["Position_YOY_pct"] = aggregated.apply(
                lambda r: calc_yoy_pct(r["Position_YOY"], r["Average Position_before"]), axis=1
            )
        if "Clicks_before" in aggregated and "Clicks_after" in aggregated:
            aggregated["Clicks_YOY"] = aggregated["Clicks_after"] - aggregated["Clicks_before"]
            aggregated["Clicks_YOY_pct"] = aggregated.apply(
                lambda r: calc_yoy_pct(r["Clicks_YOY"], r["Clicks_before"]), axis=1
            )
        if "Impressions_before" in aggregated and "Impressions_after" in aggregated:
            aggregated["Impressions_YOY"] = aggregated["Impressions_after"] - aggregated["Impressions_before"]
            aggregated["Impressions_YOY_pct"] = aggregated.apply(
                lambda r: calc_yoy_pct(r["Impressions_YOY"], r["Impressions_before"]), axis=1
            )
        if "CTR_before" in aggregated and "CTR_after" in aggregated:
            aggregated["CTR_YOY"] = aggregated["CTR_after"] - aggregated["CTR_before"]
            aggregated["CTR_YOY_pct"] = aggregated.apply(
                lambda r: calc_yoy_pct(r["CTR_YOY"], r["CTR_before"]), axis=1
            )

        st.dataframe(aggregated, use_container_width=True)

        # YOY visual chart
        st.markdown("### YOY % Change by Topic for Each Metric")
        vis_data = []
        order = ["Clicks","Impressions","Average Position","CTR"]
        for _, row in aggregated.iterrows():
            topic = row["Topic"]
            for m in order:
                pct = f"{m}_YOY_pct" if m!="Average Position" else "Position_YOY_pct"
                if pct in row and pd.notna(row[pct]) and np.isfinite(row[pct]):
                    vis_data.append({
                        "Topic": topic,
                        "Metric": m,
                        "YOY % Change": row[pct]
                    })
        if vis_data:
            vis_df = pd.DataFrame(vis_data)
            fig = px.bar(
                vis_df, x="Topic", y="YOY % Change", color="Metric",
                category_orders={"Metric": order},
                barmode="group",
                title="YOY % Change by Topic for Each Metric",
            )
            st.plotly_chart(fig, use_container_width=True)

        progress_bar.progress(100)
        status_text.text("Analysis Complete!")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Run only this tool ---
if __name__ == "__main__":
    google_search_console_analysis_page()

