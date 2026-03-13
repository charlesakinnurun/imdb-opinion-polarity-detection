"""
IMDB Sentiment Analysis - Streamlit Web Application
A comprehensive data science dashboard with EDA, ML training, and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import io
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background and font */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
        border-right: 1px solid #2d2f45;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3a3d55;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h2 { color: #7c83fd; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #9399b2; margin: 5px 0 0 0; font-size: 0.9rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #7c83fd20, transparent);
        border-left: 4px solid #7c83fd;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 15px 0;
    }
    .section-header h3 { color: #7c83fd; margin: 0; }

    /* Success / warning / info banners */
    .banner-success {
        background: #0d2818;
        border: 1px solid #1a7a40;
        border-radius: 8px;
        padding: 12px 16px;
        color: #4ade80;
        margin: 10px 0;
    }
    .banner-info {
        background: #0d1f3c;
        border: 1px solid #1a4d8a;
        border-radius: 8px;
        padding: 12px 16px;
        color: #60a5fa;
        margin: 10px 0;
    }
    .banner-warning {
        background: #2a1f0d;
        border: 1px solid #8a6a1a;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fbbf24;
        margin: 10px 0;
    }

    /* Prediction result */
    .pred-positive {
        background: linear-gradient(135deg, #0d2818, #0f3520);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-negative {
        background: linear-gradient(135deg, #2a0d0d, #3a0f0f);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-label { font-size: 2rem; font-weight: bold; margin: 0; }
    .pred-conf  { color: #9399b2; margin-top: 5px; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #9399b2; }
    .stTabs [aria-selected="true"] { background: #7c83fd30; color: #7c83fd; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #5c63d8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6b72f0, #4b52c7);
        transform: translateY(-1px);
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "trained_models": {},
        "le": None,
        "vectorizer": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "X_train_vec": None, "X_test_vec": None,
        "results": {},
        "tuned_model": None,
        "pipeline_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helper Functions ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def plotly_theme():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )


def section(title: str, icon: str = ""):
    st.markdown(
        f'<div class="section-header"><h3>{icon} {title}</h3></div>',
        unsafe_allow_html=True,
    )


def metric_card(col, label: str, value, suffix: str = ""):
    col.markdown(
        f'<div class="metric-card"><h2>{value}{suffix}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 Sentiment Analysis")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📂 Upload & Preview", "🧹 Data Cleaning", "📊 EDA",
         "🤖 Model Training", "📈 Evaluation", "🔮 Predict"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    test_size    = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    max_features = st.select_slider("TF-IDF Features", [1000, 2000, 3000, 5000, 8000], 5000)
    random_state = st.number_input("Random State", 0, 999, 42)

    st.markdown("---")
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        st.markdown(f"**Dataset:** `{len(df):,}` rows")
        if "sentiment" in df.columns:
            vc = df["sentiment"].value_counts()
            for s, c in vc.items():
                st.markdown(f"- **{s.capitalize()}:** {c:,}")

# ─── Page: Upload & Preview ──────────────────────────────────────────────────

if page == "📂 Upload & Preview":
    st.title("📂 Upload & Data Preview")

    uploaded = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Upload a CSV with at minimum a 'review' and 'sentiment' column.",
    )

    if uploaded:
        with st.spinner("Loading dataset…"):
            df = pd.read_csv(uploaded)
            st.session_state.df_raw = df.copy()
            st.session_state.df_clean = None  # reset downstream
        st.markdown('<div class="banner-success">✅ File loaded successfully!</div>', unsafe_allow_html=True)

    df = st.session_state.df_raw

    if df is None:
        st.markdown('<div class="banner-info">👆 Upload a CSV file to get started.</div>', unsafe_allow_html=True)

        # Demo data option
        if st.button("🎲 Load Sample Data (Demo)"):
            np.random.seed(42)
            positive = [
                "This movie was absolutely fantastic! Best film I've seen.",
                "Incredible performances and stunning cinematography.",
                "A masterpiece of modern cinema. Highly recommended!",
                "Loved every minute of it. The story was captivating.",
                "Outstanding direction and a brilliant cast.",
            ]
            negative = [
                "Terrible waste of time. Boring and predictable.",
                "The worst movie I have ever seen. Avoid at all costs.",
                "Poor acting, weak plot, and dreadful pacing.",
                "Fell asleep halfway through. Not recommended.",
                "Disappointing sequel that ruins the original.",
            ]
            reviews  = (positive * 200) + (negative * 200)
            sentiments = (["positive"] * 1000) + (["negative"] * 1000)
            idx = np.random.permutation(len(reviews))
            demo_df = pd.DataFrame({"review": np.array(reviews)[idx],
                                    "sentiment": np.array(sentiments)[idx]})
            st.session_state.df_raw = demo_df
            st.rerun()
    else:
        # Overview metrics
        section("Dataset Overview", "📋")
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Total Rows",    f"{len(df):,}")
        metric_card(c2, "Columns",       len(df.columns))
        metric_card(c3, "Missing Values", df.isnull().sum().sum())
        metric_card(c4, "Duplicates",    df.duplicated().sum())

        # Data preview
        section("Data Preview", "👀")
        n_rows = st.slider("Rows to display", 5, 100, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

        # Column info
        col1, col2 = st.columns(2)
        with col1:
            section("Column Info", "🗂️")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-Null": df.notnull().sum().values,
                "Null %": (df.isnull().mean() * 100).round(2).values,
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True)

        with col2:
            section("Sample Reviews", "📝")
            if "review" in df.columns and "sentiment" in df.columns:
                for sent in df["sentiment"].unique()[:2]:
                    sample = df[df["sentiment"] == sent]["review"].iloc[0]
                    label_color = "#22c55e" if "pos" in str(sent).lower() else "#ef4444"
                    st.markdown(
                        f'<div style="border-left:3px solid {label_color};padding:8px 12px;'
                        f'background:#1a1d2e;border-radius:0 8px 8px 0;margin-bottom:8px;">'
                        f'<b style="color:{label_color}">{sent.upper()}</b><br>'
                        f'<small style="color:#9399b2">{sample[:200]}…</small></div>',
                        unsafe_allow_html=True,
                    )

        # Download raw
        section("Download", "⬇️")
        csv_raw = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Raw Data", csv_raw, "raw_data.csv", "text/csv")

# ─── Page: Data Cleaning ─────────────────────────────────────────────────────

elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")

    if st.session_state.df_raw is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_raw.copy()

    # Options
    section("Cleaning Options", "⚙️")
    col1, col2 = st.columns(2)
    with col1:
        drop_dupes   = st.checkbox("Remove duplicate rows", True)
        drop_nulls   = st.checkbox("Drop rows with missing values", True)
        clean_reviews = st.checkbox("Clean review text (lowercase, strip HTML/special chars)", True)
    with col2:
        min_length   = st.slider("Minimum review length (chars)", 0, 200, 10)
        max_length   = st.slider("Maximum review length (chars)", 100, 5000, 3000)

    if st.button("🧹 Apply Cleaning"):
        with st.spinner("Cleaning data…"):
            before = len(df)

            if drop_dupes:
                df.drop_duplicates(inplace=True)
            if drop_nulls:
                df.dropna(inplace=True)
            if "review" in df.columns:
                if clean_reviews:
                    df["review_clean"] = df["review"].apply(clean_text)
                df["review_length"] = df["review"].str.len()
                df = df[(df["review_length"] >= min_length) & (df["review_length"] <= max_length)]
            df.reset_index(drop=True, inplace=True)

            st.session_state.df_clean = df

        after = len(df)
        st.markdown(
            f'<div class="banner-success">✅ Cleaning complete! '
            f'Removed <b>{before - after:,}</b> rows. '
            f'<b>{after:,}</b> rows remain.</div>',
            unsafe_allow_html=True,
        )

    df_clean = st.session_state.df_clean if st.session_state.df_clean is not None else df

    section("Before / After Comparison", "🔍")
    c1, c2, c3, c4 = st.columns(4)
    raw = st.session_state.df_raw
    metric_card(c1, "Rows (Before)", f"{len(raw):,}")
    metric_card(c2, "Rows (After)",  f"{len(df_clean):,}")
    metric_card(c3, "Removed",       f"{len(raw) - len(df_clean):,}")
    metric_card(c4, "Reduction",     f"{(1 - len(df_clean)/len(raw))*100:.1f}", "%")

    section("Cleaned Data Preview", "📋")
    st.dataframe(df_clean.head(20), use_container_width=True)

    if "review" in df_clean.columns and "review_clean" in df_clean.columns:
        section("Text Transformation Example", "✨")
        idx = st.number_input("Row index", 0, len(df_clean)-1, 0)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original:**")
            st.info(df_clean["review"].iloc[idx][:400])
        with col2:
            st.markdown("**Cleaned:**")
            st.success(df_clean["review_clean"].iloc[idx][:400])

    # Download cleaned
    csv_clean = df_clean.to_csv(index=False).encode()
    st.download_button("⬇️ Download Cleaned Data", csv_clean, "cleaned_data.csv", "text/csv")

# ─── Page: EDA ───────────────────────────────────────────────────────────────

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df is None:
        st.warning("Please upload and (optionally) clean a dataset first.")
        st.stop()

    if "review" not in df.columns or "sentiment" not in df.columns:
        st.error("Dataset must have 'review' and 'sentiment' columns.")
        st.stop()

    theme = plotly_theme()

    # ── Sentiment Distribution ──
    section("Sentiment Distribution", "🎯")
    vc = df["sentiment"].value_counts().reset_index()
    vc.columns = ["Sentiment", "Count"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(vc, x="Sentiment", y="Count", color="Sentiment",
                     color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                     title="Sentiment Counts", **theme)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(vc, names="Sentiment", values="Count",
                      color="Sentiment",
                      color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                      title="Sentiment Proportion", **theme)
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Review Length Analysis ──
    section("Review Length Analysis", "📏")
    df["_len"] = df["review"].str.len()
    df["_wc"]  = df["review"].str.split().str.len()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="_len", color="sentiment", nbins=60,
                           barmode="overlay", opacity=0.7,
                           color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                           title="Character Length Distribution",
                           labels={"_len": "Characters"}, **theme)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="_wc", color="sentiment", nbins=60,
                           barmode="overlay", opacity=0.7,
                           color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                           title="Word Count Distribution",
                           labels={"_wc": "Words"}, **theme)
        st.plotly_chart(fig, use_container_width=True)

    # Box plots
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="sentiment", y="_len", color="sentiment",
                     color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                     title="Char Length by Sentiment", **theme)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="sentiment", y="_wc", color="sentiment",
                     color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                     title="Word Count by Sentiment", **theme)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Stats Table ──
    section("Descriptive Statistics", "📐")
    stats = df.groupby("sentiment")[["_len", "_wc"]].agg(["mean", "median", "std", "min", "max"])
    stats.columns = ["Avg Chars", "Median Chars", "Std Chars", "Min Chars", "Max Chars",
                     "Avg Words", "Median Words", "Std Words", "Min Words", "Max Words"]
    st.dataframe(stats.round(1), use_container_width=True)

    # Cleanup temp cols
    df.drop(columns=["_len", "_wc"], inplace=True, errors="ignore")

    # ── Missing Values Heatmap ──
    section("Missing Values", "❓")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.markdown('<div class="banner-success">✅ No missing values found!</div>', unsafe_allow_html=True)
    else:
        fig = px.bar(x=missing.index, y=missing.values,
                     labels={"x": "Column", "y": "Missing Count"},
                     title="Missing Values per Column", **theme)
        st.plotly_chart(fig, use_container_width=True)

# ─── Page: Model Training ────────────────────────────────────────────────────

elif page == "🤖 Model Training":
    st.title("🤖 Model Training")

    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    text_col = "review_clean" if "review_clean" in df.columns else "review"

    section("Model Selection", "🎛️")
    col1, col2 = st.columns(2)
    with col1:
        use_lr = st.checkbox("Logistic Regression", True)
        use_nb = st.checkbox("Multinomial Naive Bayes", True)
        use_dt = st.checkbox("Decision Tree", True)
    with col2:
        tune_lr     = st.checkbox("Tune Logistic Regression (GridSearchCV)", False)
        stop_words  = st.selectbox("TF-IDF Stop Words", ["english", "None"])
        min_df      = st.slider("TF-IDF Min Doc Freq", 1, 20, 5)
        max_df_val  = st.slider("TF-IDF Max Doc Freq (%)", 50, 100, 80) / 100

    if st.button("🚀 Train Models"):
        if not (use_lr or use_nb or use_dt):
            st.error("Select at least one model.")
            st.stop()

        df_model = df[[text_col, "sentiment"]].dropna().copy()
        X_raw = df_model[text_col]
        y_raw = df_model["sentiment"]

        # Encode
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y_enc, test_size=test_size,
            random_state=random_state, stratify=y_enc
        )

        # Vectorize
        sw = "english" if stop_words == "english" else None
        vec = TfidfVectorizer(max_features=max_features, min_df=min_df,
                              max_df=max_df_val, stop_words=sw)
        X_tr_v = vec.fit_transform(X_tr)
        X_te_v = vec.transform(X_te)

        st.session_state.update({
            "le": le, "vectorizer": vec,
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "X_train_vec": X_tr_v, "X_test_vec": X_te_v,
            "trained_models": {}, "results": {},
            "tuned_model": None, "pipeline_ready": True,
        })

        models_to_train = {}
        if use_lr:
            models_to_train["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=random_state)
        if use_nb:
            models_to_train["Naive Bayes"] = MultinomialNB()
        if use_dt:
            models_to_train["Decision Tree"] = DecisionTreeClassifier(random_state=random_state)

        progress = st.progress(0)
        status   = st.empty()
        trained  = {}
        results  = {}

        for i, (name, mdl) in enumerate(models_to_train.items()):
            status.markdown(f"Training **{name}**…")
            t0 = time.time()
            mdl.fit(X_tr_v, y_tr)
            elapsed = time.time() - t0

            y_pred = mdl.predict(X_te_v)
            acc    = accuracy_score(y_te, y_pred)
            cm     = confusion_matrix(y_te, y_pred)
            report = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)

            trained[name] = mdl
            results[name] = {
                "accuracy": acc, "cm": cm,
                "report": report, "time": elapsed,
                "y_pred": y_pred,
            }
            progress.progress((i + 1) / len(models_to_train))

        # Optional tuning
        if tune_lr and use_lr:
            status.markdown("Running GridSearch for Logistic Regression…")
            param_grid = {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 1, 10],
                "clf__solver": ["liblinear"],
            }
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=max_features, min_df=min_df,
                                          max_df=max_df_val, stop_words=sw)),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ])
            gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
            gs.fit(X_tr, y_tr)
            tuned = gs.best_estimator_
            y_pred_t = tuned.predict(X_te)
            acc_t    = accuracy_score(y_te, y_pred_t)
            cm_t     = confusion_matrix(y_te, y_pred_t)
            report_t = classification_report(y_te, y_pred_t, target_names=le.classes_, output_dict=True)

            st.session_state.tuned_model = tuned
            results["Tuned LR (GridSearch)"] = {
                "accuracy": acc_t, "cm": cm_t,
                "report": report_t, "time": 0,
                "y_pred": y_pred_t,
                "best_params": gs.best_params_,
            }

        st.session_state.trained_models = trained
        st.session_state.results = results
        status.empty()
        progress.empty()

        st.markdown('<div class="banner-success">✅ Training complete! Go to Evaluation.</div>',
                    unsafe_allow_html=True)

        # Quick summary
        section("Training Summary", "📋")
        rows = []
        for name, r in results.items():
            rows.append({"Model": name, "Accuracy": f"{r['accuracy']:.4f}",
                         "Time (s)": f"{r.get('time', 0):.2f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─── Page: Evaluation ────────────────────────────────────────────────────────

elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation")

    if not st.session_state.results:
        st.warning("Train models first.")
        st.stop()

    results = st.session_state.results
    le      = st.session_state.le
    theme   = plotly_theme()

    # ── Accuracy comparison ──
    section("Accuracy Comparison", "🏆")
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    colors = ["#22c55e" if a == max(accs) else "#7c83fd" for a in accs]

    fig = go.Figure(go.Bar(x=names, y=accs, marker_color=colors,
                           text=[f"{a:.4f}" for a in accs], textposition="outside"))
    fig.update_layout(title="Model Accuracy", yaxis=dict(range=[0, 1.05]),
                      xaxis_tickangle=-30, **theme)
    st.plotly_chart(fig, use_container_width=True)

    # Metric cards
    c1, c2, c3 = st.columns(3)
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    metric_card(c1, "Best Model",    best_name)
    metric_card(c2, "Best Accuracy", f"{results[best_name]['accuracy']:.4f}")
    metric_card(c3, "Models Tested", len(results))

    # ── Per-model details ──
    section("Detailed Results", "🔬")
    model_sel = st.selectbox("Select model", names)
    r = results[model_sel]

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve"])

    with tab1:
        cm_df = pd.DataFrame(r["cm"],
                             index=[f"Actual {c}" for c in le.classes_],
                             columns=[f"Pred {c}" for c in le.classes_])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Viridis",
                        title=f"Confusion Matrix — {model_sel}", **theme)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rep = r["report"]
        rows = []
        for cls in le.classes_:
            if cls in rep:
                m = rep[cls]
                rows.append({"Class": cls,
                             "Precision": round(m["precision"], 4),
                             "Recall": round(m["recall"], 4),
                             "F1-Score": round(m["f1-score"], 4),
                             "Support": int(m["support"])})
        for avg in ["macro avg", "weighted avg"]:
            m = rep[avg]
            rows.append({"Class": avg.title(),
                         "Precision": round(m["precision"], 4),
                         "Recall": round(m["recall"], 4),
                         "F1-Score": round(m["f1-score"], 4),
                         "Support": int(m["support"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if "best_params" in r:
            st.markdown("**Best Hyperparameters:**")
            st.json(r["best_params"])

    with tab3:
        # ROC curve (binary)
        y_te    = st.session_state.y_test
        mdl_key = model_sel.replace("Tuned LR (GridSearch)", "")

        try:
            if model_sel == "Tuned LR (GridSearch)" and st.session_state.tuned_model:
                X_eval = st.session_state.X_test
                proba  = st.session_state.tuned_model.predict_proba(X_eval)
            else:
                X_eval = st.session_state.X_test_vec
                mdl    = st.session_state.trained_models[model_sel]
                proba  = mdl.predict_proba(X_eval)

            pos_idx = list(le.classes_).index("positive") if "positive" in le.classes_ else 1
            fpr, tpr, _ = roc_curve(y_te, proba[:, pos_idx], pos_label=pos_idx)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color="#7c83fd", width=2),
                                     name=f"AUC = {roc_auc:.4f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="#555", dash="dash"), name="Random"))
            fig.update_layout(title=f"ROC Curve — {model_sel}",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", **theme)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"ROC curve unavailable: {e}")

    # ── Multi-metric comparison ──
    section("Multi-Metric Comparison", "📊")
    rows2 = []
    for name, res in results.items():
        rep = res["report"]
        rows2.append({
            "Model": name,
            "Accuracy":       round(res["accuracy"], 4),
            "Precision (macro)": round(rep["macro avg"]["precision"], 4),
            "Recall (macro)":    round(rep["macro avg"]["recall"], 4),
            "F1 (macro)":        round(rep["macro avg"]["f1-score"], 4),
        })
    comp_df = pd.DataFrame(rows2).sort_values("Accuracy", ascending=False)
    st.dataframe(comp_df, use_container_width=True)

    fig = px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                 x="Model", y="Score", color="Metric", barmode="group",
                 title="All Metrics by Model", **theme)
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # ── Download results ──
    section("Download Results", "⬇️")
    col1, col2 = st.columns(2)
    with col1:
        csv_comp = comp_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Comparison CSV", csv_comp,
                           "model_comparison.csv", "text/csv")
    with col2:
        # Full predictions
        y_te_labels = le.inverse_transform(st.session_state.y_test)
        pred_rows = []
        for name, res in results.items():
            for actual, pred in zip(y_te_labels, le.inverse_transform(res["y_pred"])):
                pred_rows.append({"model": name, "actual": actual, "predicted": pred})
        preds_df   = pd.DataFrame(pred_rows)
        csv_preds  = preds_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download All Predictions", csv_preds,
                           "all_predictions.csv", "text/csv")

# ─── Page: Predict ───────────────────────────────────────────────────────────

elif page == "🔮 Predict":
    st.title("🔮 Predict Sentiment")

    if not st.session_state.pipeline_ready:
        st.warning("Train models first.")
        st.stop()

    le      = st.session_state.le
    results = st.session_state.results
    names   = list(results.keys())
    theme   = plotly_theme()

    tab1, tab2 = st.tabs(["✏️ Single Review", "📋 Batch Upload"])

    # ── Single ──
    with tab1:
        section("Single Review Prediction", "🎬")
        model_sel = st.selectbox("Model", names, key="pred_model")
        review_in = st.text_area("Enter your movie review:", height=150,
                                 placeholder="Type a movie review here…")

        if st.button("🔮 Predict Sentiment"):
            if not review_in.strip():
                st.error("Please enter a review.")
            else:
                cleaned = clean_text(review_in)

                if model_sel == "Tuned LR (GridSearch)" and st.session_state.tuned_model:
                    pipe   = st.session_state.tuned_model
                    proba  = pipe.predict_proba([review_in])[0]
                    pred_i = pipe.predict([review_in])[0]
                else:
                    mdl    = st.session_state.trained_models[model_sel]
                    vec    = st.session_state.vectorizer
                    X_in   = vec.transform([cleaned])
                    proba  = mdl.predict_proba(X_in)[0]
                    pred_i = mdl.predict(X_in)[0]

                sentiment  = le.inverse_transform([pred_i])[0]
                confidence = proba[pred_i]
                is_pos     = "pos" in sentiment.lower()
                emoji      = "😊" if is_pos else "😞"
                css_class  = "pred-positive" if is_pos else "pred-negative"
                color      = "#22c55e" if is_pos else "#ef4444"

                st.markdown(
                    f'<div class="{css_class}">'
                    f'<p class="pred-label" style="color:{color}">'
                    f'{emoji} {sentiment.upper()}</p>'
                    f'<p class="pred-conf">Confidence: {confidence:.4f} '
                    f'({confidence*100:.1f}%)</p></div>',
                    unsafe_allow_html=True,
                )

                st.markdown("")
                labels = le.classes_
                fig = go.Figure(go.Bar(x=labels, y=proba,
                                       marker_color=["#22c55e" if "pos" in l else "#ef4444" for l in labels],
                                       text=[f"{p:.3f}" for p in proba], textposition="outside"))
                fig.update_layout(title="Prediction Probabilities",
                                  yaxis=dict(range=[0, 1.1]), **theme)
                st.plotly_chart(fig, use_container_width=True)

    # ── Batch ──
    with tab2:
        section("Batch Prediction", "📋")
        model_sel2   = st.selectbox("Model", names, key="batch_model")
        batch_upload = st.file_uploader("Upload CSV with a 'review' column", type=["csv"], key="batch_up")

        if batch_upload:
            batch_df = pd.read_csv(batch_upload)
            st.dataframe(batch_df.head(), use_container_width=True)

            if "review" not in batch_df.columns:
                st.error("CSV must have a 'review' column.")
            elif st.button("🚀 Run Batch Prediction"):
                cleaned = batch_df["review"].apply(clean_text)

                if model_sel2 == "Tuned LR (GridSearch)" and st.session_state.tuned_model:
                    pipe   = st.session_state.tuned_model
                    preds  = pipe.predict(batch_df["review"])
                    probas = pipe.predict_proba(batch_df["review"])
                else:
                    mdl    = st.session_state.trained_models[model_sel2]
                    vec    = st.session_state.vectorizer
                    X_b    = vec.transform(cleaned)
                    preds  = mdl.predict(X_b)
                    probas = mdl.predict_proba(X_b)

                batch_df["predicted_sentiment"] = le.inverse_transform(preds)
                batch_df["confidence"]          = np.max(probas, axis=1).round(4)

                section("Results", "✅")
                st.dataframe(batch_df, use_container_width=True)

                vc = batch_df["predicted_sentiment"].value_counts().reset_index()
                vc.columns = ["Sentiment", "Count"]
                fig = px.pie(vc, names="Sentiment", values="Count",
                             color="Sentiment",
                             color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                             title="Predicted Sentiment Distribution", **theme)
                st.plotly_chart(fig, use_container_width=True)

                csv_out = batch_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions", csv_out,
                                   "batch_predictions.csv", "text/csv")

        # Manual multi-review input
        section("Or Enter Reviews Manually", "✍️")
        manual_text = st.text_area("One review per line:", height=150)
        model_sel3  = st.selectbox("Model", names, key="manual_model")

        if st.button("🔮 Predict All"):
            lines = [l.strip() for l in manual_text.split("\n") if l.strip()]
            if not lines:
                st.error("Enter at least one review.")
            else:
                cleaned = [clean_text(l) for l in lines]

                if model_sel3 == "Tuned LR (GridSearch)" and st.session_state.tuned_model:
                    pipe   = st.session_state.tuned_model
                    preds  = pipe.predict(lines)
                    probas = pipe.predict_proba(lines)
                else:
                    mdl    = st.session_state.trained_models[model_sel3]
                    vec    = st.session_state.vectorizer
                    X_m    = vec.transform(cleaned)
                    preds  = mdl.predict(X_m)
                    probas = mdl.predict_proba(X_m)

                labels_out = le.inverse_transform(preds)
                confs      = np.max(probas, axis=1)

                out_df = pd.DataFrame({
                    "Review": [l[:80] + "…" if len(l) > 80 else l for l in lines],
                    "Sentiment": labels_out,
                    "Confidence": confs.round(4),
                })
                st.dataframe(out_df, use_container_width=True)

                csv_manual = out_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download", csv_manual, "manual_predictions.csv", "text/csv")
