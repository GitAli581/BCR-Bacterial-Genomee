import streamlit as st
import pandas as pd

st.set_page_config(page_title="Categorizing GUI", layout="wide")

st.title("Categorizing GUI")
st.markdown("Put the CSV down below and you will be happy.")

uploaded_file = st.file_uploader("Upload the input.csv file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).fillna("")
    df_lower = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # The filters for the sidebars
    st.sidebar.header("🔍 Filter Categories")

    selected_filters = {}
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() > 1:
            options = sorted(df[col].dropna().unique())
            selected = st.sidebar.multiselect(f"{col}", options)
            if selected:
                selected_filters[col] = selected

    # Using the filters
    if selected_filters:
        mask = pd.Series([True] * len(df))
        for col, vals in selected_filters.items():
            mask &= df[col].isin(vals)
        filtered_df = df[mask]

        st.subheader("✅ Results are Filtered")
        st.write(f"{len(filtered_df)} rows match your filters")
        st.dataframe(filtered_df, use_container_width=True)

        # Download of the filters
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download the CSV", csv, "filtered_results.csv", "text/csv")
    else:
        st.info("Look at the sidebars for any filters")
else:
    st.warning("Upload the CSV file when trying to start.")
