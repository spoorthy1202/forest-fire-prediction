"""Forest Fire Prediction – Streamlit dashboard (Algerian Forest Fires dataset).

Models and preprocessing come from the original project by Aravind Selvam
(https://github.com/aravind-selvam/forest-fire-prediction); this app re-packages them as a
Streamlit dashboard.
"""
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             precision_recall_fscore_support, r2_score)

import core

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥", layout="wide")


# ---------- cached loaders ----------
@st.cache_resource
def get_models():
    return core.load_models()


@st.cache_data
def get_data():
    return core.load_data()


@st.cache_resource
def get_scaler():
    return core.full_data_scaler(get_data())


df = get_data()
clf, reg = get_models()
ranges = core.data_ranges(df)
df_view = df.assign(Fire=df["Classes"].map({0: "Not fire", 1: "Fire"}),
                    Region_name=df["Region"].map({1: "Bejaia", 2: "Sidi Bel-Abbes"}))

# Metrics reported in the training notebook ("Forest Fire Part-2 Model.ipynb")
REG_METRICS = pd.DataFrame({
    "Model": ["Random Forest", "Linear Regression", "Ridge", "K-Neighbors", "Lasso",
              "Support Vector Regressor", "Random Forest (tuned)", "Ridge (tuned)"],
    "R²": [0.9818, 0.9700, 0.9690, 0.9422, 0.9377, 0.9340, 0.9803, 0.9695],
    "MAE": [0.5735, 0.6453, 0.6648, 0.9816, 1.1209, 0.8676, 0.5739, 0.6537],
})
CLF_METRICS = pd.DataFrame({
    "Model": ["XGBoost (tuned)", "Logistic Regression", "Decision Tree", "Random Forest*"],
    "Test accuracy": [0.9726, 0.9589, 0.9726, 0.9589],
    "10-fold stratified CV accuracy": [0.9752, 0.9752, 0.9752, 0.9710],
})

page = st.sidebar.radio("Navigate", ["🏠 Project Overview", "📊 EDA", "📈 Model Performance",
                                     "🔥 Fire Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("Minor Specialization Project · MIT Manipal\n\nOriginal models & notebook: "
                   "Aravind Selvam")

# ====================================================================== OVERVIEW
if page.startswith("🏠"):
    st.title("🔥 Algerian Forest Fire Prediction")
    st.write("A machine-learning dashboard that predicts **whether a forest fire is likely** "
             "(classification) and the **Fire Weather Index, FWI** (regression) from weather "
             "and fuel-moisture readings.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Fire days", int(df["Classes"].sum()))
    c3.metric("Regions", "2")
    c4.metric("Model inputs", len(core.FEATURES))

    st.subheader("Dataset")
    st.write("UCI **Algerian Forest Fires** dataset: daily observations from two regions of "
             "Algeria – **Bejaia** and **Sidi Bel-Abbes** – from June to September 2012.")
    st.subheader("Pipeline")
    st.markdown(
        "1. Clean the data and encode `Classes` (fire = 1, not fire = 0)\n"
        "2. Remove highly correlated features (|r| > 0.75: BUI, DC)\n"
        "3. Standardise with `StandardScaler`\n"
        "4. Compare models, tune the best with `RandomizedSearchCV`\n"
        "5. Keep the **5 most useful inputs** for the app: Temperature, Wind speed, FFMC, DMC, ISI\n"
        "6. Serve the final models: **XGBoost** (fire / no fire) and **Random Forest** (FWI)")
    st.subheader("Input features")
    st.table(pd.DataFrame({
        "Feature": [core.FEATURE_LABELS[f] for f in core.FEATURES],
        "Training min": [round(ranges[f][0], 1) for f in core.FEATURES],
        "Training max": [round(ranges[f][1], 1) for f in core.FEATURES],
    }))
    with st.expander("Dataset preview"):
        st.dataframe(df_view.drop(columns=["Classes", "Region"]).head(20), width="stretch")

# ====================================================================== EDA
elif page.startswith("📊"):
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Class balance", "Distributions", "Correlation",
                                      "Time & region"])
    with tab1:
        counts = df_view["Fire"].value_counts().reset_index()
        counts.columns = ["Class", "Days"]
        st.plotly_chart(px.pie(counts, names="Class", values="Days", hole=0.4,
                               color="Class", color_discrete_map={"Fire": "#d9534f",
                                                                  "Not fire": "#5cb85c"}),
                        width="stretch")
        by_region = df_view.groupby(["Region_name", "Fire"]).size().reset_index(name="Days")
        st.plotly_chart(px.bar(by_region, x="Region_name", y="Days", color="Fire", barmode="group",
                               color_discrete_map={"Fire": "#d9534f", "Not fire": "#5cb85c"}),
                        width="stretch")
    with tab2:
        feat = st.selectbox("Feature", core.FEATURES + ["FWI"],
                            format_func=lambda f: core.FEATURE_LABELS.get(f, "Fire Weather Index (FWI)"))
        st.plotly_chart(px.histogram(df_view, x=feat, color="Fire", nbins=30, barmode="overlay",
                                     opacity=0.7,
                                     color_discrete_map={"Fire": "#d9534f", "Not fire": "#5cb85c"}),
                        width="stretch")
        st.plotly_chart(px.box(df_view, x="Fire", y=feat, color="Fire",
                               color_discrete_map={"Fire": "#d9534f", "Not fire": "#5cb85c"}),
                        width="stretch")
    with tab3:
        cols = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
        corr = df[cols].corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), width="stretch")
        st.caption("BUI and DC are highly correlated with other variables, which is why the "
                   "original notebook drops them before modelling.")
    with tab4:
        month_names = {6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
        m = df_view.assign(Month=df_view["month"].map(month_names)).groupby(
            ["Month", "Fire"]).size().reset_index(name="Days")
        st.plotly_chart(px.bar(m, x="Month", y="Days", color="Fire", barmode="group",
                               category_orders={"Month": ["Jun", "Jul", "Aug", "Sep"]},
                               color_discrete_map={"Fire": "#d9534f", "Not fire": "#5cb85c"}),
                        width="stretch")
        st.plotly_chart(px.scatter(df_view, x="FFMC", y="FWI", color="Fire",
                                   color_discrete_map={"Fire": "#d9534f", "Not fire": "#5cb85c"}),
                        width="stretch")

# ====================================================================== PERFORMANCE
elif page.startswith("📈"):
    st.title("📈 Model Performance")
    st.caption("Tables show the results recorded in the training notebook. The confusion matrix "
               "and predicted-vs-actual plot are recomputed here by loading the saved models on "
               "the notebook's exact test split.")
    st.subheader("Classification – Fire / Not fire (deployed: tuned XGBoost)")
    st.dataframe(CLF_METRICS, hide_index=True, width="stretch")
    st.caption("*In the notebook, the cell labelled 'KNN' actually trains a Random Forest, so it "
               "is shown here as Random Forest.")

    y_c, p_c = core.classification_holdout(df, clf)
    cm = confusion_matrix(y_c, p_c)
    prec, rec, f1, _ = precision_recall_fscore_support(y_c, p_c, zero_division=0)
    a, b, c = st.columns(3)
    a.metric("Deployed model – test accuracy", f"{accuracy_score(y_c, p_c):.2%}")
    b.metric("Precision (fire)", f"{prec[1]:.2f}")
    c.metric("Recall (fire)", f"{rec[1]:.2f}")
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              x=["Pred: Not fire", "Pred: Fire"], y=["Actual: Not fire", "Actual: Fire"],
                              title="Confusion matrix (73 test rows)"), width="stretch")

    st.subheader("Regression – Fire Weather Index (deployed: tuned Random Forest)")
    st.dataframe(REG_METRICS, hide_index=True, width="stretch")
    y_r, p_r = core.regression_holdout(df, reg)
    a, b = st.columns(2)
    a.metric("Deployed model – R²", f"{r2_score(y_r, p_r):.4f}")
    b.metric("Deployed model – MAE", f"{mean_absolute_error(y_r, p_r):.4f}")
    fig = px.scatter(x=y_r, y=p_r, labels={"x": "Actual FWI", "y": "Predicted FWI"},
                     title="Predicted vs actual FWI (61 test rows)")
    mx = float(max(y_r.max(), p_r.max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect fit",
                             line=dict(dash="dash")))
    st.plotly_chart(fig, width="stretch")
    imp = pd.DataFrame({"Feature": core.FEATURES, "Importance": reg.feature_importances_})
    st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature",
                           orientation="h", title="Random Forest feature importance (FWI model)"),
                    width="stretch")

# ====================================================================== PREDICTION
else:
    st.title("🔥 Fire Prediction")
    st.write("Enter weather and fuel-moisture readings. Defaults are the dataset averages.")
    mean = df[core.FEATURES].mean()
    with st.form("predict_form"):
        cols = st.columns(5)
        values = {}
        steps = {"Temperature": 1.0, "Ws": 1.0, "FFMC": 0.1, "DMC": 0.1, "ISI": 0.1}
        for col, f in zip(cols, core.FEATURES):
            lo, hi = core.HARD_LIMITS[f]
            values[f] = col.number_input(core.FEATURE_LABELS[f], min_value=lo, max_value=hi,
                                         value=float(round(mean[f], 1)), step=steps[f],
                                         help=f"Training data range: {ranges[f][0]:.1f} – "
                                              f"{ranges[f][1]:.1f}")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        errors = core.validate_inputs(values)
        if errors:
            for e in errors:
                st.error(e)
        else:
            outside = core.out_of_training_range(values, ranges)
            if outside:
                st.warning("Outside the range the models were trained on for: "
                           + ", ".join(core.FEATURE_LABELS[f] for f in outside)
                           + ". Treat the result with caution.")
            label, p_fire, fwi = core.predict(values, clf, reg, get_scaler())
            left, right = st.columns(2)
            with left:
                st.subheader("Classification")
                if label == 1:
                    st.error("🔥 **Forest in danger** – fire predicted")
                else:
                    st.success("🌲 **Forest is safe** – no fire predicted")
                st.progress(min(max(p_fire, 0.0), 1.0), text=f"Estimated fire probability: {p_fire:.1%}")
            with right:
                st.subheader("Fire Weather Index (FWI)")
                st.metric("Predicted FWI", f"{fwi:.2f}")
                if fwi > core.HIGH_HAZARD_FWI:
                    st.error("High hazard rating (FWI > 15)")
                else:
                    st.success("Low hazard rating (FWI ≤ 15)")
            st.caption("The two models are trained separately, so they can disagree near the "
                       "boundary (the classifier is driven mainly by FFMC, with a switch point "
                       "around FFMC ≈ 80).")
            with st.expander("Inputs used"):
                st.json(values)
