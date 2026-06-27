"""
Streamlit Dashboard — Unsupervised Credit Card Fraud Detection
==============================================================
Run:
    pip install streamlit
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 600; color: #1a1a2e; margin: 0; }
    .metric-label { font-size: 13px; color: #6c757d; margin: 0; }
    .fraud    { color: #D85A30; }
    .normal   { color: #1D9E75; }
    .section-header {
        font-size: 16px; font-weight: 600;
        border-left: 4px solid #7F77DD;
        padding-left: 10px; margin: 1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTOENCODER
# ─────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 14),        nn.ReLU(),
            nn.Linear(14, 8),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8,  14), nn.ReLU(),
            nn.Linear(14, 20), nn.ReLU(),
            nn.Linear(20, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────
# CACHED FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def preprocess(_df):
    df = _df.drop(columns=["Time"])
    df["Amount"] = np.log1p(df["Amount"])
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train  = X_scaled[y == 0]
    return X_scaled, y, X_train

@st.cache_resource
def run_models(X_train_key, X_scaled_key, contamination, n_estimators, ae_epochs):
    # re-load from session state (hashing workaround)
    X_scaled = st.session_state["X_scaled"]
    X_train  = st.session_state["X_train"]

    # Isolation Forest
    iso = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_train)
    if_scores = -iso.decision_function(X_scaled)

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                             novelty=True, n_jobs=-1)
    lof.fit(X_train)
    lof_scores = -lof.score_samples(X_scaled)

    # Autoencoder
    device = torch.device("cpu")
    model  = Autoencoder(input_dim=X_train.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=256, shuffle=True)
    ae_losses = []
    for _ in range(ae_epochs):
        model.train()
        total = 0
        for (b,) in loader:
            r = model(b); loss = crit(r, b)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        ae_losses.append(total / len(X_train))

    model.eval()
    with torch.no_grad():
        recon = model(torch.FloatTensor(X_scaled)).numpy()
    ae_scores = np.mean((X_scaled - recon) ** 2, axis=1)

    return if_scores, lof_scores, ae_scores, ae_losses


def get_metrics(scores, y):
    prec, rec, thr = precision_recall_curve(y, scores)
    f1s   = 2 * prec * rec / (prec + rec + 1e-8)
    best  = np.argmax(f1s[:-1])
    thresh = thr[best]
    preds  = (scores >= thresh).astype(int)
    cm     = confusion_matrix(y, preds)
    return {
        "roc_auc": roc_auc_score(y, scores),
        "ap":      average_precision_score(y, scores),
        "f1":      f1s[best],
        "thresh":  thresh,
        "preds":   preds,
        "prec":    prec,
        "rec":     rec,
        "cm":      cm,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fraud.png", width=60)
    st.title("Fraud Detection")
    st.caption("Unsupervised ML Dashboard")
    st.divider()

    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])

    st.markdown("#### Model settings")
    contamination  = st.slider("Contamination rate", 0.001, 0.01, 0.0017, 0.0001,
                                help="Estimated fraud rate in dataset")
    n_estimators   = st.slider("IF — n_estimators", 50, 300, 200, 50)
    ae_epochs      = st.slider("Autoencoder epochs", 5, 50, 20, 5)

    run_btn = st.button("Run models", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Models**")
    show_if  = st.checkbox("Isolation Forest", value=True)
    show_lof = st.checkbox("Local Outlier Factor", value=True)
    show_ae  = st.checkbox("Autoencoder", value=True)

    st.divider()
    st.markdown("**About**")
    st.caption("Built by an AI/ML Engineering student.\nModels: IF · LOF · PyTorch Autoencoder")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

st.title("🔍 Credit Card Fraud Detection")
st.caption("Unsupervised anomaly detection — no labels used during training")

if uploaded is None:
    st.info("👈 Upload `creditcard.csv` from Kaggle to get started.", icon="📂")
    st.markdown("""
    **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    - 284,807 transactions
    - 492 fraudulent (0.17%)
    - Features V1–V28 are PCA-anonymised
    """)
    st.stop()

# ── Load & preprocess ──────────────────────────────────────────────────────
df = load_data(uploaded)
X_scaled, y, X_train = preprocess(df)

st.session_state["X_scaled"] = X_scaled
st.session_state["X_train"]  = X_train

fraud_count  = int(y.sum())
normal_count = int((y == 0).sum())
fraud_pct    = fraud_count / len(y) * 100

# ── Dataset overview ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">Dataset overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total transactions", f"{len(df):,}")
with c2:
    st.metric("Normal", f"{normal_count:,}", delta=None)
with c3:
    st.metric("Fraud", f"{fraud_count:,}", delta=None)
with c4:
    st.metric("Fraud rate", f"{fraud_pct:.4f}%")

# ── EDA ────────────────────────────────────────────────────────────────────
with st.expander("Exploratory Data Analysis", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Normal", "Fraud"], [normal_count, fraud_count],
               color=["#1D9E75", "#D85A30"])
        ax.set_title("Class distribution"); ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[df["Class"]==0]["Amount"].hist(bins=60, ax=ax, alpha=0.6,
            color="#1D9E75", label="Normal", density=True)
        df[df["Class"]==1]["Amount"].hist(bins=60, ax=ax, alpha=0.8,
            color="#D85A30", label="Fraud",  density=True)
        ax.set_xlabel("Amount (USD)"); ax.set_title("Amount distribution")
        ax.legend(); st.pyplot(fig); plt.close()

    st.dataframe(df.describe().round(3), use_container_width=True)


# ── Run models ─────────────────────────────────────────────────────────────
if run_btn or "if_scores" in st.session_state:

    if run_btn:
        with st.spinner("Training models... this takes ~30 seconds"):
            cache_key = f"{contamination}_{n_estimators}_{ae_epochs}"
            if_scores, lof_scores, ae_scores, ae_losses = run_models(
                cache_key, cache_key, contamination, n_estimators, ae_epochs
            )
            st.session_state.update({
                "if_scores": if_scores, "lof_scores": lof_scores,
                "ae_scores": ae_scores, "ae_losses":  ae_losses,
            })
        st.success("Models trained!")

    if "if_scores" not in st.session_state:
        st.warning("Click 'Run models' to train."); st.stop()

    if_scores  = st.session_state["if_scores"]
    lof_scores = st.session_state["lof_scores"]
    ae_scores  = st.session_state["ae_scores"]
    ae_losses  = st.session_state["ae_losses"]

    models = {}
    if show_if:  models["Isolation Forest"]    = if_scores
    if show_lof: models["Local Outlier Factor"] = lof_scores
    if show_ae:  models["Autoencoder"]          = ae_scores

    if not models:
        st.warning("Select at least one model in the sidebar."); st.stop()

    metrics   = {name: get_metrics(s, y) for name, s in models.items()}
    clr_map   = {"Isolation Forest": "#7F77DD",
                 "Local Outlier Factor": "#1D9E75",
                 "Autoencoder": "#D85A30"}

    # ── Model comparison metrics ───────────────────────────────────────────
    st.markdown('<div class="section-header">Model comparison</div>', unsafe_allow_html=True)
    cols = st.columns(len(models))
    for col, (name, m) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            st.metric("Avg Precision", f"{m['ap']:.4f}")
            st.metric("Best F1", f"{m['f1']:.4f}")
            cm = m["cm"]
            st.caption(f"TP: {cm[1,1]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TN: {cm[0,0]:,}")

    # ── PR curves ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Precision-Recall curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, m in metrics.items():
        ax.plot(m["rec"], m["prec"], label=f"{name}  AP={m['ap']:.3f}",
                color=clr_map[name], lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    # ── Score distributions ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Anomaly score distributions</div>', unsafe_allow_html=True)
    dist_cols = st.columns(len(models))
    for col, (name, scores) in zip(dist_cols, models.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(scores[y==0], bins=80, alpha=0.6, color="#1D9E75",
                    label="Normal", density=True)
            ax.hist(scores[y==1], bins=80, alpha=0.85, color="#D85A30",
                    label="Fraud",  density=True)
            ax.axvline(metrics[name]["thresh"], color="black",
                       linestyle="--", lw=1.2, label="Threshold")
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("Score"); ax.legend(fontsize=8)
            st.pyplot(fig); plt.close()

    # ── PCA visualization ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">PCA 2D projection</div>', unsafe_allow_html=True)
    pca_model = st.selectbox("Colour by model score",
                              list(models.keys()), index=list(models.keys()).index(
                                  "Autoencoder" if "Autoencoder" in models else list(models.keys())[0]
                              ))
    pca    = PCA(n_components=2, random_state=SEED)
    X2     = pca.fit_transform(X_scaled)
    scores = models[pca_model]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(X2[y==0, 0], X2[y==0, 1],
                    c=scores[y==0], cmap="YlOrRd",
                    s=2, alpha=0.25)
    ax.scatter(X2[y==1, 0], X2[y==1, 1],
               c="blue", s=20, alpha=0.7, label="Fraud (true)", marker="x")
    plt.colorbar(sc, ax=ax, label="Anomaly score")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA projection — coloured by {pca_model} score")
    ax.legend(markerscale=4); ax.grid(alpha=0.2)
    st.pyplot(fig); plt.close()

    # ── Autoencoder training loss ──────────────────────────────────────────
    if show_ae:
        st.markdown('<div class="section-header">Autoencoder training loss</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ae_losses, color="#D85A30", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    # ── Business cost tradeoff ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Business cost tradeoff</div>', unsafe_allow_html=True)
    st.caption("Tune the threshold based on your business priority — catching more fraud vs reducing customer friction.")

    cost_model = st.selectbox("Model for threshold analysis",
                               list(models.keys()), key="cost_model")
    c_scores   = models[cost_model]
    thresholds = np.percentile(c_scores, np.linspace(50, 99, 50))
    fn_list, fp_list = [], []
    for t in thresholds:
        p  = (c_scores >= t).astype(int)
        cm = confusion_matrix(y, p)
        fn_list.append(cm[1, 0])
        fp_list.append(cm[0, 1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, fn_list, color="#D85A30", lw=2,
            label="Missed fraud (FN) — financial loss")
    ax.plot(thresholds, fp_list, color="#7F77DD", lw=2,
            label="Blocked legit txn (FP) — customer friction")
    ax.axvline(metrics[cost_model]["thresh"], color="black",
               linestyle="--", lw=1.2, label="Best F1 threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    # ── Classification report ──────────────────────────────────────────────
    with st.expander("Full classification reports"):
        for name, m in metrics.items():
            st.markdown(f"**{name}**")
            report = classification_report(y, m["preds"],
                                           target_names=["Normal", "Fraud"])
            st.code(report)

    # ── Confusion matrices ─────────────────────────────────────────────────
    with st.expander("Confusion matrices"):
        cm_cols = st.columns(len(models))
        for col, (name, m) in zip(cm_cols, metrics.items()):
            with col:
                st.markdown(f"**{name}**")
                cm = m["cm"]
                fig, ax = plt.subplots(figsize=(3.5, 3))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Normal","Fraud"])
                ax.set_yticklabels(["Normal","Fraud"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{cm[i,j]:,}", ha="center",
                                va="center", color="white" if cm[i,j] > cm.max()/2 else "black",
                                fontsize=12, fontweight="bold")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()

else:
    st.info("Configure settings in the sidebar and click **Run models** to start.", icon="⚙️")
