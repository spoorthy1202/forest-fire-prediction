"""Streamlit UI for the Random Forest Algerian Forest Fire classifier."""
import warnings
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import core

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pathlib import Path

APP_DIR = Path(__file__).parent
ASSET_DIR = APP_DIR / "assets"
CSS_PATH = APP_DIR / "styles.css"

# Load the external stylesheet.
if CSS_PATH.exists():
    css_text = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
else:
    st.warning("styles.css was not found. The application is using Streamlit's default styling.")

# Use assets/forest_hero.jpg as the page background.
# Embedding the image as a data URI avoids browser path/URL issues on Windows.
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
                    rgba(244, 243, 236, 0.84),
                    rgba(244, 243, 236, 0.84)
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
else:
    st.warning("assets/forest_hero.jpg was not found. The page will use the default background.")

# Targeted overrides: keep sidebar text and Plotly labels readable.
st.markdown(
    """
    <style>
    {background_css}
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {
        color: #F5F5ED !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #C5D4CA !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label,
    section[data-testid="stSidebar"] [role="radiogroup"] label p,
    section[data-testid="stSidebar"] [role="radiogroup"] label span {
        color: #F5F5ED !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.10) !important;
    }

    [data-testid="stMetricValue"] {
        color: #183D32 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #657168 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #183D32;
    }
    [data-testid="stImage"] img {
        border-radius: 14px;
        object-fit: cover;
    }
    header[data-testid="stHeader"] {
        background: rgba(244,243,236,0.92);
    }
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


# Load resources once per Streamlit session/cache.
try:
    metadata = get_metadata()
    df = get_data()
    model = get_model()
except Exception as exc:
    st.error("The application could not load its model or dataset.")
    st.exception(exc)
    st.info(
        "Check that models/fire_classifier.joblib, models/metadata.json and "
        "data/Algerian_forest_fires_dataset_CLEANED.csv are in the expected "
        "locations. See the integration instructions."
    )
    st.stop()

ranges = core.data_ranges(df)
df_view = df.copy()
df_view["Fire label"] = df_view["Classes"].map(core.TARGET_LABELS)
df_view["Region name"] = df_view["Region"].map(core.REGION_LABELS).fillna(
    df_view["Region"].astype(str)
)

st.sidebar.title("🔥 Forest Fire AI")
st.sidebar.caption("Algerian Forest Fires · Random Forest")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Project Overview", "📊 Exploratory Data Analysis",
     "📈 Model Performance", "🔥 Fire Prediction"],
)
st.sidebar.divider()
st.sidebar.caption("Minor Specialization Project · MIT Manipal")

# ------------------------------------------------------------------- OVERVIEW

# ------------------------------------------------------------
# PROJECT OVERVIEW - REDESIGNED HOMEPAGE
# ------------------------------------------------------------

if page.startswith("🏠"):

    st.markdown(
        """
        <div style="margin-bottom:8px;">
            <span style="
                background:#E5ECE4;
                color:#285744;
                padding:7px 13px;
                border-radius:20px;
                font-size:12px;
                font-weight:600;
                letter-spacing:1px;">
                ENVIRONMENTAL INTELLIGENCE PLATFORM
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # HERO SECTION
    hero_left, hero_right = st.columns([1.15, 0.85], gap="large")

    with hero_left:
        st.markdown(
            """
            <div style="padding-top:25px;">
                <h1 style="
                    font-size:43px;
                    line-height:1.15;
                    color:#183D32;
                    margin-bottom:15px;">
                    Forest Fire<br>
                    Intelligence.
                </h1>
                <p style="
                    font-size:17px;
                    line-height:1.7;
                    color:#657168;">
                    Understanding fire patterns through weather
                    intelligence and machine learning.
                    Predict potential forest fire occurrence
                    using environmental conditions and
                    Fire Weather Index indicators.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="margin-top:20px;">
                <span style="
                    background:#183D32;
                    color:white;
                    padding:10px 17px;
                    border-radius:8px;
                    font-size:13px;">
                    RANDOM FOREST MODEL
                </span>
                <span style="
                    background:#E9E8DE;
                    color:#285744;
                    padding:10px 17px;
                    border-radius:8px;
                    font-size:13px;
                    margin-left:7px;">
                    13 INPUT FEATURES
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with hero_right:
        # The forest_hero.jpg image is now used as the full-page background,
        # so a second copy of the same image is intentionally not displayed here.
        st.markdown(
            """
            <div style="
                min-height:300px;
                border-radius:18px;
                background:rgba(24,61,50,0.10);
                border:1px solid rgba(24,61,50,0.10);
                display:flex;
                align-items:flex-end;
                padding:22px;">
                <div style="
                    background:rgba(24,61,50,0.88);
                    color:#FFFFFF;
                    padding:13px 17px;
                    border-radius:10px;
                    font-size:13px;
                    letter-spacing:0.5px;">
                    ENVIRONMENTAL MONITORING · FIRE RISK INTELLIGENCE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # KEY STATISTICS
    st.markdown("### Platform at a glance")

    total_observations = len(df)
    fire_days = int(df["Classes"].sum())
    no_fire_days = total_observations - fire_days
    regions = df["Region"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total observations",
        f"{total_observations:,}",
        help="Total observations in the cleaned dataset"
    )

    col2.metric(
        "Fire observations",
        f"{fire_days:,}",
        help="Observations classified as fire"
    )

    col3.metric(
        "Non-fire observations",
        f"{no_fire_days:,}"
    )

    col4.metric(
        "Regions covered",
        regions
    )

    st.markdown("")
    st.divider()

    # DATA INSIGHTS
    st.markdown("### Understanding the dataset")

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:

        st.markdown("#### Fire occurrence distribution")

        class_counts = (
            df_view["Fire label"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Observations")
        )

        fig = px.pie(
            class_counts,
            names="Class",
            values="Observations",
            hole=0.62,
            color="Class",
            color_discrete_map={
                "Fire": "#E47B42",
                "Not Fire": "#66866B"
            }
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=20, l=10, r=10),
            legend=dict(
                orientation="h",
                y=-0.10,
                font=dict(color="#303B34", size=12)
            ),
            font=dict(
                family="Arial, sans-serif",
                size=13,
                color="#303B34"
            )
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hole=0.65,
            textfont=dict(color="#FFFFFF", size=12),
            marker=dict(line=dict(color="#F4F3EC", width=2))
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with chart_right:

        st.markdown("#### Regional fire distribution")

        region_data = (
            df_view.groupby(["Region name", "Fire label"])
            .size()
            .reset_index(name="Observations")
        )

        fig2 = px.bar(
            region_data,
            x="Region name",
            y="Observations",
            color="Fire label",
            barmode="group",
            color_discrete_map={
                "Fire": "#E47B42",
                "Not Fire": "#66866B"
            }
        )

        fig2.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=25, b=35, l=10, r=10),
            xaxis_title="",
            yaxis_title="Observations",
            legend_title="Classification",
            font=dict(
                family="Arial, sans-serif",
                size=13,
                color="#303B34"
            ),
            xaxis=dict(
                tickfont=dict(color="#303B34", size=12),
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title_font=dict(color="#657168"),
                tickfont=dict(color="#657168", size=11),
                gridcolor="#E2E5DC",
                zeroline=False
            ),
            legend=dict(
                font=dict(color="#303B34", size=12),
                bgcolor="rgba(255,255,255,0)"
            ),
            bargap=0.25
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.divider()

    # HOW THE SYSTEM WORKS
    st.markdown("### How the prediction system works")

    step1, step2, step3 = st.columns(3, gap="medium")

    with step1:
        st.markdown("#### 01 · Environmental data")
        weather_image = ASSET_DIR / "weather_monitoring.jpg"

        if weather_image.exists():
            st.image(str(weather_image), use_container_width=True)

        st.write(
            "Collect temperature, humidity, wind speed, "
            "rainfall and Fire Weather Index indicators."
        )

    with step2:
        st.markdown("#### 02 · Machine learning")
        satellite_image = ASSET_DIR / "wildfire_satellite.jpg"

        if satellite_image.exists():
            st.image(str(satellite_image), use_container_width=True)

        st.write(
            "Process the 13 input features through the "
            "trained Random Forest classification pipeline."
        )

    with step3:
        st.markdown("#### 03 · Fire prediction")

        st.markdown(
            """
            <div style="
                background:#183D32;
                border-radius:12px;
                padding:30px;
                min-height:145px;
                display:flex;
                align-items:center;
                justify-content:center;">
                <div style="text-align:center;color:white;">
                    <div style="font-size:42px;">🔥</div>
                    <div style="font-size:17px;font-weight:600;">
                        Fire Intelligence
                    </div>
                    <div style="font-size:12px;color:#C8D9CB;">
                        Classification & Probability
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Generate a fire or no-fire prediction "
            "along with the estimated fire probability."
        )

    st.divider()

    st.caption(
        "Forest Fire Intelligence | Algerian Forest Fires Dataset | "
        "Random Forest Classification"
    )


# ------------------------------------------------------------------------- EDA
elif page.startswith("📊"):
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Class balance", "Feature distributions", "Correlation", "Time & region"]
    )

    with tab1:
        counts = df_view["Fire label"].value_counts().rename_axis("Class").reset_index(name="Days")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.pie(counts, names="Class", values="Days", hole=0.4,
                       title="Fire vs not-fire observations"),
                use_container_width=True,
            )
        with col2:
            by_region = (
                df_view.groupby(["Region name", "Fire label"])
                .size().reset_index(name="Days")
            )
            st.plotly_chart(
                px.bar(by_region, x="Region name", y="Days", color="Fire label",
                       barmode="group", title="Class distribution by region"),
                use_container_width=True,
            )

    with tab2:
        available = [f for f in core.DEFAULT_FEATURES if f != "Region"] + ["FWI"]
        feature = st.selectbox(
            "Select feature",
            available,
            format_func=lambda f: core.FEATURE_LABELS.get(f, f),
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(df_view, x=feature, color="Fire label", nbins=25,
                             barmode="overlay", opacity=0.7,
                             title=f"{feature} distribution"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.box(df_view, x="Fire label", y=feature, color="Fire label",
                       title=f"{feature} by class"),
                use_container_width=True,
            )

    with tab3:
        numeric = df.select_dtypes(include="number").drop(
            columns=["Classes"], errors="ignore"
        )
        corr = numeric.corr().round(2)
        st.plotly_chart(
            px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, aspect="auto", title="Feature correlation"),
            use_container_width=True,
        )

    with tab4:
        if "month" in df_view.columns:
            month_order = [6, 7, 8, 9]
            monthly = (
                df_view.groupby(["month", "Fire label"]).size()
                .reset_index(name="Days")
            )
            st.plotly_chart(
                px.bar(monthly, x="month", y="Days", color="Fire label",
                       barmode="group", category_orders={"month": month_order},
                       title="Monthly fire incidence"),
                use_container_width=True,
            )
        st.plotly_chart(
            px.scatter(df_view, x="Temperature", y="RH", color="Fire label",
                       hover_data=["Ws", "Rain", "Region name"],
                       title="Temperature vs relative humidity"),
            use_container_width=True,
        )

# --------------------------------------------------------------- PERFORMANCE
elif page.startswith("📈"):
    st.title("📈 Model Performance")
    st.caption(
        "Metrics below are recomputed on the same stratified 80/20 holdout split "
        "used in src/train_model.py (test_size=0.2, random_state=42)."
    )
    try:
        y_true, y_pred, y_probability = core.holdout_predictions(df, model)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_true, y_probability)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Precision (fire)", f"{precision:.2%}")
        c3.metric("Recall (fire)", f"{recall:.2%}")
        c4.metric("ROC-AUC", f"{auc:.3f}")

        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            st.plotly_chart(
                px.imshow(
                    cm, text_auto=True, x=["Predicted: Not fire", "Predicted: Fire"],
                    y=["Actual: Not fire", "Actual: Fire"],
                    color_continuous_scale="Blues", title="Confusion matrix",
                ),
                use_container_width=True,
            )
        with right:
            fpr, tpr, _ = roc_curve(y_true, y_probability)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc:.3f})"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash"), name="Random baseline"))
            roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                                  yaxis_title="True positive rate")
            st.plotly_chart(roc_fig, use_container_width=True)

        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Not Fire", "Fire"], output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        classifier = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
        if classifier is not None and hasattr(classifier, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": core.DEFAULT_FEATURES,
                "Importance": classifier.feature_importances_,
            }).sort_values("Importance")
            st.subheader("Feature importance")
            st.plotly_chart(
                px.bar(importance, x="Importance", y="Feature", orientation="h"),
                use_container_width=True,
            )
    except Exception as exc:
        st.error("Could not calculate holdout performance.")
        st.exception(exc)

# ------------------------------------------------------------------ PREDICTION
else:
    st.title("🔥 Fire Prediction")
    st.write(
        "Enter the date, region and weather/FWI readings. The application will "
        "return a fire/no-fire classification and the model's fire probability."
    )

    means = df[core.DEFAULT_FEATURES].mean(numeric_only=True)
    with st.form("fire_prediction_form"):
        st.subheader("Date and location")
        date_cols = st.columns(4)
        day = date_cols[0].number_input("Day", min_value=1, max_value=31,
                                        value=int(round(means["day"])), step=1)
        month = date_cols[1].number_input("Month", min_value=1, max_value=12,
                                          value=int(round(means["month"])), step=1)
        year = date_cols[2].number_input("Year", min_value=1900, max_value=2100,
                                         value=int(round(means["year"])), step=1)
        region_name = date_cols[3].selectbox("Region", ["Bejaia", "Sidi Bel-Abbes"])
        region_value = 1 if region_name == "Bejaia" else 2

        st.subheader("Weather conditions")
        weather_cols = st.columns(4)
        weather_features = ["Temperature", "RH", "Ws", "Rain"]
        values = {
            "day": int(day), "month": int(month), "year": int(year),
            "Region": int(region_value),
        }
        weather_limits = {
            "Temperature": (-10.0, 60.0),
            "RH": (0.0, 100.0),
            "Ws": (0.0, 150.0),
            "Rain": (0.0, 300.0),
        }
        for col, feature in zip(weather_cols, weather_features):
            low, high = weather_limits[feature]
            default = safe_default(feature, ranges, means[feature])
            values[feature] = col.number_input(
                core.FEATURE_LABELS[feature], min_value=float(low), max_value=float(high),
                value=float(default), step=0.1 if feature == "Rain" else 1.0,
                help=f"Observed training range: {ranges[feature][0]:.2f}–{ranges[feature][1]:.2f}",
            )

        st.subheader("Fire Weather Index indicators")
        index_cols = st.columns(5)
        index_features = ["FFMC", "DMC", "DC", "ISI", "BUI"]
        index_limits = {
            "FFMC": (0.0, 101.0), "DMC": (0.0, 500.0),
            "DC": (0.0, 1000.0), "ISI": (0.0, 100.0), "BUI": (0.0, 500.0),
        }
        for col, feature in zip(index_cols, index_features):
            low, high = index_limits[feature]
            default = safe_default(feature, ranges, means[feature])
            values[feature] = col.number_input(
                core.FEATURE_LABELS[feature], min_value=float(low), max_value=float(high),
                value=float(default), step=0.1,
                help=f"Observed training range: {ranges[feature][0]:.2f}–{ranges[feature][1]:.2f}",
            )

        submitted = st.form_submit_button("Predict fire risk", type="primary",
                                          use_container_width=True)

    if submitted:
        errors = core.validate_inputs(values)
        if errors:
            for error in errors:
                st.error(error)
        else:
            outside = core.out_of_training_range(values, ranges)
            if outside:
                labels = [core.FEATURE_LABELS[f] for f in outside]
                st.warning(
                    "These values are outside the model's training-data range: "
                    + ", ".join(labels)
                    + ". The prediction may be less reliable."
                )
            try:
                label, probability = core.predict(values, model)
                left, right = st.columns(2)
                with left:
                    st.subheader("Prediction")
                    if label == 1:
                        st.error("🔥 Fire predicted")
                    else:
                        st.success("🌲 No fire predicted")
                with right:
                    st.subheader("Estimated fire probability")
                    st.metric("Fire probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                st.caption(
                    "This is a machine-learning estimate based on the Algerian "
                    "Forest Fires dataset. It is not an official warning system."
                )
                with st.expander("Inputs sent to the model"):
                    st.json({feature: values[feature] for feature in core.DEFAULT_FEATURES})
            except Exception as exc:
                st.error("Prediction failed. Check the model file and input feature order.")
                st.exception(exc)
