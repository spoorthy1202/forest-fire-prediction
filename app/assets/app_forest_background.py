"""Streamlit UI for the Extended Algerian Forest Fire XGBoost classifier."""
import warnings
import base64
import sys
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Robust path handling whether run from repo root, app/, or app/assets/
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent if CURRENT_FILE.parent.name == "app" else CURRENT_FILE.parent.parent
ASSET_DIR = APP_DIR / "assets"
CSS_PATH = APP_DIR / "styles.css"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import core

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Forest Fire Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the external stylesheet if it exists.
if CSS_PATH.exists():
    css_text = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

# Use assets/forest_hero.jpg as the page background if it exists.
HERO_BACKGROUND = ASSET_DIR / "forest_hero.jpg"
background_css = ""
if HERO_BACKGROUND.exists():
    image_bytes = HERO_BACKGROUND.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_mime = "image/jpeg"
    background_css = f"""
        [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(
                    rgba(244, 243, 236, 0.90),
                    rgba(244, 243, 236, 0.90)
                ),
                url("data:{image_mime};base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        [data-testid="stMain"] {{
            background: transparent !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            background: transparent !important;
        }}
    """

st.markdown(
    f"""
    <style>
    {background_css}
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {{
        color: #F5F5ED !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #183D32 !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: #657168 !important;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: #183D32;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return core.load_model()


@st.cache_data
def get_data():
    return core.load_data()


@st.cache_data
def get_metadata():
    return core.load_metadata()


def safe_default(feature, ranges, fallback):
    low, high = ranges[feature]
    return float(min(max(fallback, low), high))


# Load resources
try:
    metadata = get_metadata()
    df = get_data()
    model = get_model()
except Exception as exc:
    st.error("The application could not load its model or dataset.")
    st.exception(exc)
    st.stop()

ranges = core.data_ranges(df)
df_view = df.copy()
df_view["Fire label"] = df_view["Classes"].map(core.TARGET_LABELS)
df_view["Region name"] = df_view["Region"].map(core.REGION_LABELS).fillna(
    df_view["Region"].astype(str)
)

# --- SIDEBAR (No emojis) ---
st.sidebar.title("Forest Fire AI")
st.sidebar.caption("Extended Algerian Forest Fires · XGBoost")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Exploratory Data Analysis", "Model Performance", "Fire Prediction"],
)
st.sidebar.divider()
st.sidebar.caption("Minor Specialization Project · MIT Manipal")

# ------------------------------------------------------------------- OVERVIEW
if page == "Project Overview":
    st.title("Forest Fire Risk Assessment System")
    st.markdown("### Extended Algerian Forest Fires Dataset")
    st.write(
        "This system predicts forest fire occurrence using a tuned XGBoost classifier trained on "
        "1,215 daily observations across 10 geographical region-year groups (2012–2021). "
        "The model utilizes 11 meteorological and fire-weather features to provide real-time risk assessment."
    )
    st.divider()

    st.markdown("### Platform at a glance")
    total_observations = len(df)
    fire_days = int(df["Classes"].sum())
    no_fire_days = total_observations - fire_days
    regions = df["Region"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total observations", f"{total_observations:,}")
    col2.metric("Fire observations", f"{fire_days:,}")
    col3.metric("Non-fire observations", f"{no_fire_days:,}")
    col4.metric("Regions covered", regions)
    st.divider()

    st.markdown("### Understanding the dataset")
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        st.markdown("#### Fire occurrence distribution")
        class_counts = df_view["Fire label"].value_counts().rename_axis("Class").reset_index(name="Observations")
        fig = px.pie(class_counts, names="Class", values="Observations", hole=0.5, color="Class",
                     color_discrete_map={"Fire": "#E47B42", "Not Fire": "#66866B"})
        fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_right:
        st.markdown("#### Regional fire distribution")
        region_data = df_view.groupby(["Region name", "Fire label"]).size().reset_index(name="Observations")
        fig2 = px.bar(region_data, x="Region name", y="Observations", color="Fire label", barmode="group",
                      color_discrete_map={"Fire": "#E47B42", "Not Fire": "#66866B"})
        fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.divider()

# ------------------------------------------------------------------------- EDA
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Class balance", "Feature distributions", "Correlation", "Time & region"]
    )

    with tab1:
        counts = df_view["Fire label"].value_counts().rename_axis("Class").reset_index(name="Days")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(counts, names="Class", values="Days", hole=0.4, title="Fire vs not-fire"), use_container_width=True)
        with col2:
            by_region = df_view.groupby(["Region name", "Fire label"]).size().reset_index(name="Days")
            st.plotly_chart(px.bar(by_region, x="Region name", y="Days", color="Fire label", barmode="group", title="Class by region"), use_container_width=True)

    with tab2:
        available = [f for f in core.DEFAULT_FEATURES if f != "Region"]
        feature = st.selectbox("Select feature", available, format_func=lambda f: core.FEATURE_LABELS.get(f, f))
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.histogram(df_view, x=feature, color="Fire label", nbins=25, barmode="overlay", opacity=0.7, title=f"{feature} distribution"), use_container_width=True)
        with right:
            st.plotly_chart(px.box(df_view, x="Fire label", y=feature, color="Fire label", title=f"{feature} by class"), use_container_width=True)

    with tab3:
        numeric = df.select_dtypes(include="number").drop(columns=["Classes"], errors="ignore")
        corr = numeric.corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Feature correlation"), use_container_width=True)

    with tab4:
        if "month" in df_view.columns:
            monthly = df_view.groupby(["month", "Fire label"]).size().reset_index(name="Days")
            st.plotly_chart(px.bar(monthly, x="month", y="Days", color="Fire label", barmode="group", title="Monthly fire incidence"), use_container_width=True)
        st.plotly_chart(px.scatter(df_view, x="Temperature", y="RH", color="Fire label", hover_data=["Ws", "Rain", "Region name"], title="Temp vs RH"), use_container_width=True)

# --------------------------------------------------------------- PERFORMANCE
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Metrics below are from 5-Fold TimeSeriesSplit validation on the 1,215-row Extended Dataset.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "86.73%")
    c2.metric("Precision (fire)", "79.67%")
    c3.metric("Recall (fire)", "97.61%")
    c4.metric("ROC-AUC", "0.9318")

    st.info(
        "We used TimeSeriesSplit to prevent temporal leakage. This realistic accuracy (86.73%) proves the model generalizes to unseen future data, unlike a random split which yields a fake 100%."
    )

    st.subheader("ROC Curve & Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/roc_curve_extended.png", use_container_width=True)
    with col2:
        st.image("outputs/confusion_matrix_extended.png", use_container_width=True)

# ------------------------------------------------------------------ PREDICTION
elif page == "Fire Prediction":
    st.title("Fire Prediction")
    st.write("Enter the month, region and weather/FWI readings. The application will return a fire/no-fire classification and the model's fire probability.")

    means = df[core.DEFAULT_FEATURES].mean(numeric_only=True)
    with st.form("fire_prediction_form"):
        st.subheader("Time and location")
        loc_cols = st.columns(2)
        month = loc_cols[0].number_input("Month", min_value=6, max_value=9, value=int(round(means["month"])), step=1, help="Summer months: June (6) to September (9)")
        
        region_options = sorted(list(core.REGION_LABELS.keys()))
        selected_region = loc_cols[1].selectbox("Region", region_options, format_func=lambda r: core.REGION_LABELS.get(r, str(r)), help="Select one of the 10 region-year groups")
        region_value = int(selected_region)

        st.subheader("Weather conditions")
        weather_cols = st.columns(4)
        weather_features = ["Temperature", "Ws", "Rain", "RH"]
        values = {"month": int(month), "Region": int(region_value)}
        weather_limits = {"Temperature": (-10.0, 60.0), "Ws": (0.0, 150.0), "Rain": (0.0, 300.0), "RH": (0.0, 100.0)}
        
        for col, feature in zip(weather_cols, weather_features):
            low, high = weather_limits[feature]
            default = safe_default(feature, ranges, means[feature])
            values[feature] = col.number_input(core.FEATURE_LABELS[feature], min_value=float(low), max_value=float(high), value=float(default), step=0.1 if feature == "Rain" else 1.0)

        st.subheader("Fire Weather Index indicators")
        index_cols = st.columns(5)
        index_features = ["FFMC", "DMC", "DC", "ISI", "BUI"]
        index_limits = {"FFMC": (0.0, 101.0), "DMC": (0.0, 500.0), "DC": (0.0, 1000.0), "ISI": (0.0, 100.0), "BUI": (0.0, 500.0)}
        
        for col, feature in zip(index_cols, index_features):
            low, high = index_limits[feature]
            default = safe_default(feature, ranges, means[feature])
            values[feature] = col.number_input(core.FEATURE_LABELS[feature], min_value=float(low), max_value=float(high), value=float(default), step=0.1)

        submitted = st.form_submit_button("Run XGBoost Inference", type="primary", use_container_width=True)

    if submitted:
        errors = core.validate_inputs(values)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                label, probability = core.predict(values, model)
                left, right = st.columns(2)
                with left:
                    st.subheader("Prediction")
                    if label == 1:
                        st.error("Fire predicted")
                    else:
                        st.success("No fire predicted")
                with right:
                    st.subheader("Estimated fire probability")
                    st.metric("Fire probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                st.caption("This is a machine-learning estimate. It is not an official warning system.")
                with st.expander("Inputs sent to the model"):
                    st.json({feature: values[feature] for feature in core.DEFAULT_FEATURES})
            except Exception as exc:
                st.error("Prediction failed. Check the model file.")
                st.exception(exc)