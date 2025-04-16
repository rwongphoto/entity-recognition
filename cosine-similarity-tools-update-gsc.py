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
        This tool now uses **KMeans clustering** on query embeddings (SentenceTransformer) and **GPT-based labeling** to group queries into topics.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Calculate overall performance changes (based on full input data).
        - Merge data using an outer join to preserve all queries.
        - Compute embeddings for each query.
        - Cluster queries using KMeans (optimal K suggested via Silhouette Score).
        - Generate descriptive topic labels for each cluster using OpenAI's GPT.
        - Display the original merged data table with GPT topic labels (includes queries unique to one period).
        - Aggregate metrics by topic.
        - Visualize the YOY % change by topic for each metric.
        **Note:** Requires an OpenAI API key set in Streamlit Secrets for topic labeling.
        """
    )

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        openai_client = get_openai_client()
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            # --- Steps 1-4: Read, Validate, Clean, Dashboard ---
            status_text.text("Reading, Cleaning, Initial Metrics...")
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            # --- Find Columns & Rename ---
            required_query_col = "Top queries"
            required_pos_col = "Position"
            def find_col_name(df, potential_names):
                for name in potential_names:
                    for col in df.columns:
                        if col.strip().lower() == name.strip().lower():
                            return col
                return None
            query_col_before = find_col_name(df_before, [required_query_col, "Query"])
            pos_col_before = find_col_name(df_before, [required_pos_col, "Average position", "Position"])
            query_col_after = find_col_name(df_after, [required_query_col, "Query"])
            pos_col_after = find_col_name(df_after, [required_pos_col, "Average position", "Position"])
            clicks_col_before = find_col_name(df_before, ["Clicks"])
            impressions_col_before = find_col_name(df_before, ["Impressions", "Impr."])
            ctr_col_before = find_col_name(df_before, ["CTR"])
            clicks_col_after = find_col_name(df_after, ["Clicks"])
            impressions_col_after = find_col_name(df_after, ["Impressions", "Impr."])
            ctr_col_after = find_col_name(df_after, ["CTR"])
            if not query_col_before or not pos_col_before or not query_col_after or not pos_col_after:
                st.error("Query/Position column missing in one of the files.")
                return
            rename_map_before = {query_col_before: "Query", pos_col_before: "Average Position"}
            rename_map_after = {query_col_after: "Query", pos_col_after: "Average Position"}
            if clicks_col_before:
                rename_map_before[clicks_col_before] = "Clicks"
            if impressions_col_before:
                rename_map_before[impressions_col_before] = "Impressions"
            if ctr_col_before:
                rename_map_before[ctr_col_before] = "CTR"
            if clicks_col_after:
                rename_map_after[clicks_col_after] = "Clicks"
            if impressions_col_after:
                rename_map_after[impressions_col_after] = "Impressions"
            if ctr_col_after:
                rename_map_after[ctr_col_after] = "CTR"
            df_before = df_before.rename(columns=rename_map_before)
            df_after = df_after.rename(columns=rename_map_after)
            # --- Clean numeric values ---
            def clean_metric(series):
                if pd.api.types.is_numeric_dtype(series):
                    return series
                series_str = series.astype(str)
                cleaned = series_str.str.replace('%', '', regex=False).str.replace('<|>|,', '', regex=True).str.strip()
                cleaned = cleaned.replace('', np.nan).replace('N/A', np.nan).replace('--', np.nan)
                return pd.to_numeric(cleaned, errors='coerce')
            potential_metrics = ["Average Position", "Clicks", "Impressions", "CTR"]
            df_before_cleaned = df_before.copy()
            df_after_cleaned = df_after.copy()
            for df in [df_before_cleaned, df_after_cleaned]:
                for col in potential_metrics:
                    if col in df.columns:
                        df[col] = clean_metric(df[col])
            # --- Dashboard Summary ---
            st.markdown("## Dashboard Summary")
            cols = st.columns(4)
            def calculate_weighted_average(values, weights):
                if values is None or weights is None:
                    return np.nan
                valid_indices = values.notna() & weights.notna() & (weights > 0)
                if not valid_indices.any():
                    return values.mean() if values.notna().any() else np.nan
                try:
                    return np.average(values[valid_indices], weights=weights[valid_indices])
                except ZeroDivisionError:
                    return values.mean() if values.notna().any() else np.nan
            # Clicks
            if "Clicks" in df_before_cleaned.columns and "Clicks" in df_after_cleaned.columns:
                total_clicks_before = df_before_cleaned["Clicks"].sum()
                total_clicks_after = df_after_cleaned["Clicks"].sum()
                overall_clicks_change = total_clicks_after - total_clicks_before
                overall_clicks_change_pct = (overall_clicks_change / total_clicks_before * 100) if pd.notna(total_clicks_before) and total_clicks_before != 0 else 0
                cols[0].metric(label="Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
            else:
                cols[0].metric(label="Clicks Change", value="N/A")
            # Impressions
            if "Impressions" in df_before_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                total_impressions_before = df_before_cleaned["Impressions"].sum()
                total_impressions_after = df_after_cleaned["Impressions"].sum()
                overall_impressions_change = total_impressions_after - total_impressions_before
                overall_impressions_change_pct = (overall_impressions_change / total_impressions_before * 100) if pd.notna(total_impressions_before) and total_impressions_before != 0 else 0
                cols[1].metric(label="Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
            else:
                cols[1].metric(label="Impressions Change", value="N/A")
            # Position (updated for Average Position)
            overall_avg_position_before = np.nan
            if "Average Position" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                overall_avg_position_before = calculate_weighted_average(df_before_cleaned["Average Position"], df_before_cleaned["Impressions"])
            overall_avg_position_after = np.nan
            if "Average Position" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                overall_avg_position_after = calculate_weighted_average(df_after_cleaned["Average Position"], df_after_cleaned["Impressions"])
            if pd.notna(overall_avg_position_before) and pd.notna(overall_avg_position_after):
                # Calculate change as (Before - After) so that a drop (improvement) gives a positive value
                overall_position_change = overall_avg_position_before - overall_avg_position_after
                overall_position_change_pct = (overall_position_change / overall_avg_position_before * 100) if overall_avg_position_before != 0 else 0
                cols[2].metric(label="Avg. Position Change", value=f"{overall_position_change:.1f}", delta=f"{overall_position_change_pct:.1f}%", delta_color="inverse")
            else:
                cols[2].metric(label="Avg. Position Change", value="N/A")
            # CTR
            overall_ctr_before = np.nan
            if "CTR" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                overall_ctr_before = calculate_weighted_average(df_before_cleaned["CTR"], df_before_cleaned["Impressions"])
            overall_ctr_after = np.nan
            if "CTR" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                overall_ctr_after = calculate_weighted_average(df_after_cleaned["CTR"], df_after_cleaned["Impressions"])
            if pd.notna(overall_ctr_before) and pd.notna(overall_ctr_after):
                overall_ctr_change = overall_ctr_after - overall_ctr_before
                overall_ctr_change_pct = (overall_ctr_change / overall_ctr_before * 100) if pd.notna(overall_ctr_before) and overall_ctr_before != 0 else 0
                cols[3].metric(label="Avg. CTR Change", value=f"{overall_ctr_change:.2f}% pts", delta=f"{overall_ctr_change_pct:.1f}%")
            else:
                cols[3].metric(label="Avg. CTR Change", value="N/A")
            progress_bar.progress(10)

            # --- Step 5: Merge Data using OUTER JOIN ---
            status_text.text("Merging data (Outer Join)...")
            cols_to_keep_before = ["Query"] + [col for col in potential_metrics if col in df_before_cleaned.columns]
            cols_to_keep_after = ["Query"] + [col for col in potential_metrics if col in df_after_cleaned.columns]
            merged_df = pd.merge(
                df_before_cleaned[cols_to_keep_before],
                df_after_cleaned[cols_to_keep_after],
                on="Query", suffixes=("_before", "_after"), how='outer'
            )
            if merged_df.empty:
                st.error("Merge failed. No matching queries found.")
                return
            progress_bar.progress(15)

            # --- Step 6: Calculate YOY changes ---
            status_text.text("Calculating YOY changes...")
            def calculate_yoy_pct_change(yoy_abs, before):
                if pd.isna(yoy_abs) or pd.isna(before):
                    return np.nan
                if before == 0:
                    return np.inf if yoy_abs != 0 else 0.0
                return (yoy_abs / before) * 100

            if "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns:
                merged_df["Clicks_YOY"] = merged_df.apply(lambda row: row["Clicks_after"] - row["Clicks_before"], axis=1)
            if "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns:
                merged_df["Impressions_YOY"] = merged_df.apply(lambda row: row["Impressions_after"] - row["Impressions_before"], axis=1)
            if "CTR_before" in merged_df.columns and "CTR_after" in merged_df.columns:
                merged_df["CTR_YOY"] = merged_df.apply(lambda row: row["CTR_after"] - row["CTR_before"], axis=1)
            if "Average Position_before" in merged_df.columns and "Average Position_after" in merged_df.columns:
                merged_df["Position_YOY"] = merged_df.apply(lambda row: row["Average Position_before"] - row["Average Position_after"], axis=1)

            if "Clicks_YOY" in merged_df.columns:
                merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Clicks_YOY"], row["Clicks_before"]), axis=1)
            if "Impressions_YOY" in merged_df.columns:
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Impressions_YOY"], row["Impressions_before"]), axis=1)
            if "CTR_YOY" in merged_df.columns:
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["CTR_YOY"], row["CTR_before"]), axis=1)
            if "Position_YOY" in merged_df.columns:
                merged_df["Position_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Position_YOY"], row["Average Position_before"]), axis=1)

            progress_bar.progress(20)

            # --- Step 7: Embeddings ---
            status_text.text("Computing query embeddings...")
            model = initialize_sentence_transformer()
            if model is None:
                st.error("Sentence Transformer model failed to load.")
                return
            queries = merged_df["Query"].astype(str).unique().tolist()
            if not queries:
                st.error("No queries found in the merged data.")
                return
            with st.spinner(f"Generating embeddings for {len(queries)} unique queries..."):
                try:
                    query_embeddings_unique = model.encode(queries, show_progress_bar=True)
                except Exception as encode_err:
                    st.error(f"Embedding generation failed: {encode_err}")
                    return
            if query_embeddings_unique is None or len(query_embeddings_unique) != len(queries):
                st.error("Mismatch between queries and generated embeddings.")
                return
            query_to_embedding = {query: emb for query, emb in zip(queries, query_embeddings_unique)}
            merged_df['query_embedding'] = merged_df['Query'].map(query_to_embedding)
            valid_embedding_mask = merged_df['query_embedding'].notna()
            if not valid_embedding_mask.any():
                st.error("No valid query embeddings were generated.")
                return
            embeddings_matrix = np.vstack(merged_df.loc[valid_embedding_mask, 'query_embedding'].values)
            progress_bar.progress(35)

            # --- Step 8: Clustering ---
            status_text.text("Performing KMeans clustering...")
            num_to_cluster = embeddings_matrix.shape[0]
            max_k = min(30, num_to_cluster - 1) if num_to_cluster > 1 else 1
            min_k = 3
            optimal_k = min(max(min_k, max_k // 2), max_k) if max_k >= min_k else max_k
            if max_k < min_k:
                optimal_k = max_k if max_k > 0 else 1
            else:
                silhouette_scores = {}
                for k in range(min_k, max_k + 1):
                    try:
                        from sklearn.cluster import KMeans
                        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        labels_temp = kmeans_temp.fit_predict(embeddings_matrix)
                        score = silhouette_score(embeddings_matrix, labels_temp)
                        silhouette_scores[k] = score
                    except Exception:
                        continue
                if silhouette_scores and max(silhouette_scores.values()) > -1:
                    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
                else:
                    optimal_k = max(min_k, math.ceil(num_to_cluster / 50))
                    optimal_k = min(optimal_k, max_k)
            optimal_k = max(1, optimal_k)
            slider_min = max(1, min_k if num_to_cluster >= min_k else 1)
            slider_max = max(1, max_k)
            slider_default = max(slider_min, min(int(optimal_k), slider_max))
            n_clusters_selected = st.slider("Select number of query clusters (K):", min_value=slider_min, max_value=slider_max, value=slider_default, key="kmeans_clusters_gsc")
            kmeans = KMeans(n_clusters=n_clusters_selected, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            merged_df["Cluster_ID"] = np.nan
            merged_df.loc[valid_embedding_mask, "Cluster_ID"] = cluster_labels
            merged_df["Cluster_ID"] = merged_df["Cluster_ID"].astype('Int64')
            progress_bar.progress(55)

            # --- Step 9: GPT Labeling ---
            status_text.text("Generating topic labels with GPT...")
            cluster_topics = {}
            valid_cluster_ids = merged_df["Cluster_ID"].dropna().unique()
            if openai_client:
                gpt_prog_bar = st.progress(0)
                status_text.text(f"Requesting labels from OpenAI for {len(valid_cluster_ids)} clusters...")
                for i, cluster_id in enumerate(sorted(valid_cluster_ids)):
                    queries_in_cluster = merged_df[merged_df["Cluster_ID"] == cluster_id]["Query"].unique().tolist()
                    if queries_in_cluster:
                        cluster_topics[cluster_id] = get_gpt_cluster_label(openai_client, queries_in_cluster, cluster_id)
                    else:
                        cluster_topics[cluster_id] = f"Cluster {cluster_id + 1} (Empty)"
                    time.sleep(0.1)
                    gpt_prog_bar.progress((i + 1) / len(valid_cluster_ids))
                gpt_prog_bar.empty()
            else:
                st.warning("OpenAI client not initialized. Using default labels.")
                for cluster_id in sorted(valid_cluster_ids):
                    cluster_topics[cluster_id] = f"Cluster {cluster_id + 1}"
            cluster_topics[pd.NA] = "Unclustered / No Embedding"
            merged_df["Query_Topic"] = merged_df["Cluster_ID"].map(cluster_topics).fillna("Unclustered")
            progress_bar.progress(70)

            # --- Display Merged Data Table ---
            st.markdown("### Combined Data with Topic Labels")
            st.markdown("Merged data (outer join) with cluster ID and GPT-generated topic labels.")
            display_order = ["Query", "Cluster_ID", "Query_Topic"]
            metrics_ordered = ["Average Position", "Clicks", "Impressions", "CTR"]
            for metric in metrics_ordered:
                for suffix in ["_before", "_after", "_YOY", "_YOY_pct"]:
                    if metric == "Average Position":
                        col = "Average Position_before" if suffix == "_before" else (
                            "Average Position_after" if suffix == "_after" else
                            "Position_YOY" if suffix == "_YOY" else
                            "Position_YOY_pct"
                        )
                    else:
                        col = f"{metric}{suffix}"
                    if col in merged_df.columns:
                        display_order.append(col)
            merged_df_display = merged_df[[col for col in display_order if col in merged_df.columns]]
            format_dict_merged = {}
            def add_format(col_name, fmt_str):
                if col_name in merged_df_display.columns:
                    format_dict_merged[col_name] = fmt_str
            add_format("Cluster_ID", "{:.0f}")
            add_format("Average Position_before", "{:.1f}")
            add_format("Average Position_after", "{:.1f}")
            add_format("Position_YOY", "{:+.1f}")
            add_format("Position_YOY_pct", "{:+.1f}%")
            add_format("Clicks_before", "{:,.0f}")
            add_format("Clicks_after", "{:,.0f}")
            add_format("Clicks_YOY", "{:+,.0f}")
            add_format("Clicks_YOY_pct", "{:+.1f}%")
            add_format("Impressions_before", "{:,.0f}")
            add_format("Impressions_after", "{:,.0f}")
            add_format("Impressions_YOY", "{:+,.0f}")
            add_format("Impressions_YOY_pct", "{:+.1f}%")
            add_format("CTR_before", "{:.2f}%")
            add_format("CTR_after", "{:.2f}%")
            add_format("CTR_YOY", "{:+.2f}%")
            add_format("CTR_YOY_pct", "{:+.1f}%")
            st.dataframe(merged_df_display.style.format(format_dict_merged, na_rep="N/A"))

            # --- Step 10: Aggregated Metrics by Topic ---
            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")
            agg_dict = {}
            if "Average Position_before" in merged_df.columns and "Impressions_before" in merged_df.columns:
                agg_dict["Average Position_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "Average Position_before" in merged_df.columns:
                agg_dict["Average Position_before"] = "mean"
            if "Average Position_after" in merged_df.columns and "Impressions_after" in merged_df.columns:
                agg_dict["Average Position_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "Average Position_after" in merged_df.columns:
                agg_dict["Average Position_after"] = "mean"
            if "Clicks_before" in merged_df.columns:
                agg_dict["Clicks_before"] = "sum"
            if "Clicks_after" in merged_df.columns:
                agg_dict["Clicks_after"] = "sum"
            if "Impressions_before" in merged_df.columns:
                agg_dict["Impressions_before"] = "sum"
            if "Impressions_after" in merged_df.columns:
                agg_dict["Impressions_after"] = "sum"
            if "CTR_before" in merged_df.columns and "Impressions_before" in merged_df.columns:
                agg_dict["CTR_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "CTR_before" in merged_df.columns:
                agg_dict["CTR_before"] = "mean"
            if "CTR_after" in merged_df.columns and "Impressions_after" in merged_df.columns:
                agg_dict["CTR_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "CTR_after" in merged_df.columns:
                agg_dict["CTR_after"] = "mean"

            aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index()
            aggregated.rename(columns={"Query_Topic": "Topic"}, inplace=True)

            # Recalculate Aggregated YOY changes AFTER aggregation
            if "Average Position_before" in aggregated.columns and "Average Position_after" in aggregated.columns:
                aggregated["Position_YOY"] = aggregated["Average Position_before"] - aggregated["Average Position_after"]
            if "Clicks_before" in aggregated.columns and "Clicks_after" in aggregated.columns:
                aggregated["Clicks_YOY"] = aggregated["Clicks_after"] - aggregated["Clicks_before"]
            if "Impressions_before" in aggregated.columns and "Impressions_after" in aggregated.columns:
                aggregated["Impressions_YOY"] = aggregated["Impressions_after"] - aggregated["Impressions_before"]
            if "CTR_before" in aggregated.columns and "CTR_after" in aggregated.columns:
                aggregated["CTR_YOY"] = aggregated["CTR_after"] - aggregated["CTR_before"]

            if "Position_YOY" in aggregated.columns:
                aggregated["Position_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Position_YOY"], row["Average Position_before"]), axis=1)
            if "Clicks_YOY" in aggregated.columns:
                aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Clicks_YOY"], row["Clicks_before"]), axis=1)
            if "Impressions_YOY" in aggregated.columns:
                aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Impressions_YOY"], row["Impressions_before"]), axis=1)
            if "CTR_YOY" in aggregated.columns:
                aggregated["CTR_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["CTR_YOY"], row["CTR_before"]), axis=1)

            progress_bar.progress(85)

            # Reorder columns for the aggregated table
            new_order_agg = ["Topic"]
            agg_yoy_cols_ordered = []
            metrics_ordered = ["Average Position", "Clicks", "Impressions", "CTR"]
            for metric in metrics_ordered:
                if metric == "Average Position":
                    before_col, after_col, yoy_col, yoy_pct_col = "Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"
                else:
                    before_col, after_col, yoy_col, yoy_pct_col = f"{metric}_before", f"{metric}_after", f"{metric}_YOY", f"{metric}_YOY_pct"
                if before_col in aggregated.columns:
                    new_order_agg.append(before_col)
                if after_col in aggregated.columns:
                    new_order_agg.append(after_col)
                if yoy_col in aggregated.columns:
                    new_order_agg.append(yoy_col)
                if yoy_pct_col in aggregated.columns:
                    new_order_agg.append(yoy_pct_col)
                    agg_yoy_cols_ordered.append((yoy_pct_col, metric))
            aggregated = aggregated[[col for col in new_order_agg if col in aggregated.columns]]
            format_dict_agg = {}
            def add_agg_format(col_name, fmt_str):
                if col_name in aggregated.columns:
                    format_dict_agg[col_name] = fmt_str
            add_agg_format("Average Position_before", "{:.1f}")
            add_agg_format("Average Position_after", "{:.1f}")
            add_agg_format("Position_YOY", "{:+.1f}")
            add_agg_format("Position_YOY_pct", "{:+.1f}%")
            add_agg_format("Clicks_before", "{:,.0f}")
            add_agg_format("Clicks_after", "{:,.0f}")
            add_agg_format("Clicks_YOY", "{:+,.0f}")
            add_agg_format("Clicks_YOY_pct", "{:+.1f}%")
            add_agg_format("Impressions_before", "{:,.0f}")
            add_agg_format("Impressions_after", "{:,.0f}")
            add_agg_format("Impressions_YOY", "{:+,.0f}")
            add_agg_format("Impressions_YOY_pct", "{:+.1f}%")
            add_agg_format("CTR_before", "{:.2f}%")
            add_agg_format("CTR_after", "{:.2f}%")
            add_agg_format("CTR_YOY", "{:+.2f}%")
            add_agg_format("CTR_YOY_pct", "{:+.1f}%")
            display_count = st.number_input("Number of aggregated topics to display:", min_value=1, value=min(aggregated.shape[0], 50), max_value=aggregated.shape[0])
            sort_metric_agg = "Impressions_after" if "Impressions_after" in aggregated.columns else "Topic"
            aggregated_sorted = aggregated.sort_values(by=sort_metric_agg, ascending=False, na_position='last')
            st.dataframe(aggregated_sorted.head(display_count).style.format(format_dict_agg, na_rep="N/A"))
            progress_bar.progress(90)

            # --- Step 11: Visualization ---
            status_text.text("Generating visualizations...")
            st.markdown("### YOY % Change by Topic for Each Metric")
            default_topics = [t for t in aggregated["Topic"].unique() if t != "Unclustered / No Embedding"]
            available_topics = aggregated["Topic"].unique().tolist()
            actual_default_topics = [t for t in default_topics if t in available_topics]
            if not actual_default_topics and available_topics:
                actual_default_topics = available_topics
            selected_topics = st.multiselect("Select topics to display on the chart:", options=available_topics, default=actual_default_topics)
            vis_data = []
            found_metrics_for_plot = set()
            if not agg_yoy_cols_ordered:
                st.warning("Could not determine which YOY % columns to plot.")
            else:
                for idx, row in aggregated_sorted.iterrows():
                    topic = row["Topic"]
                    if topic not in selected_topics:
                        continue
                    for yoy_pct_col, metric_name in agg_yoy_cols_ordered:
                        if yoy_pct_col in row:
                            yoy_value = row[yoy_pct_col]
                            if pd.notna(yoy_value) and np.isfinite(yoy_value):
                                vis_data.append({"Topic": topic, "Metric": metric_name, "YOY % Change": yoy_value})
                                found_metrics_for_plot.add(metric_name)
            if vis_data:
                vis_df = pd.DataFrame(vis_data)
                # Reorder the metrics using a categorical order:
                vis_df['Metric'] = pd.Categorical(vis_df['Metric'],
                                                  categories=["Clicks", "Impressions", "Average Position", "CTR"],
                                                  ordered=True)
                topic_order_plot = sorted([t for t in aggregated_sorted['Topic'] if t in selected_topics])
                color_discrete_map = {
                    "Clicks": px.colors.qualitative.Plotly[1],
                    "Impressions": px.colors.qualitative.Plotly[2],
                    "Average Position": px.colors.qualitative.Plotly[0],
                    "CTR": px.colors.qualitative.Plotly[3],
                }
                fig = px.bar(vis_df, x="Topic", y="YOY % Change", color="Metric",
                             barmode="group", title="YOY % Change by Topic for Each Metric",
                             labels={"YOY % Change": "YOY Change (%)", "Topic": "GPT-Generated Topic"},
                             category_orders={"Topic": topic_order_plot, "Metric": ["Clicks", "Impressions", "Average Position", "CTR"]},
                             color_discrete_map=color_discrete_map)
                fig.update_layout(height=600, yaxis_title="YOY Change (%)", legend_title_text="Metric")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid YOY % change data available to plot for the selected topics.")

            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Error: CSV file not found.")
        except pd.errors.EmptyDataError:
            st.error("Error: CSV file is empty.")
        except KeyError as e:
            st.error(f"Error: Column mismatch: {e}.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    else:
        st.info("Please upload both GSC CSV files.")


# --- Run only this tool ---
if __name__ == "__main__":
    google_search_console_analysis_page()

