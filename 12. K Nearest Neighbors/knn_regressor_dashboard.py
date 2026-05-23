import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KNN Regressor Explorer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d14;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13131f 0%, #0d0d14 100%);
    border-right: 1px solid #2a2a40;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

/* Main area */
.main { background-color: #0d0d14; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #16162a 0%, #1e1e35 100%);
    border: 1px solid #2e2e50;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7070a0;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-unit {
    font-size: 0.75rem;
    color: #5050780;
    margin-top: 2px;
    color: #606090;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7070a0;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #2a2a40;
    padding-bottom: 6px;
}

/* K badge */
.k-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 4px 16px;
    border-radius: 30px;
    margin: 4px;
    box-shadow: 0 2px 10px rgba(99,102,241,0.4);
}

/* Title */
.dashboard-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.dashboard-sub {
    color: #606090;
    font-size: 0.85rem;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

/* Streamlit overrides */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 8px; }
div[data-testid="stMetricValue"] { color: #a78bfa; }

/* Plot container */
.plot-container {
    background: #12121e;
    border: 1px solid #2a2a40;
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

hr { border-color: #2a2a40; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (used in matplotlib) ──────────────────────────────────────
BG       = "#0d0d14"
PANEL    = "#12121e"
GRID     = "#1e1e30"
ACCENT1  = "#a78bfa"   # violet  – main prediction line
ACCENT2  = "#60a5fa"   # blue    – scatter points
ACCENT3  = "#34d399"   # green   – residuals / feature 2
ACCENT4  = "#f97316"   # orange  – highlight
TEXT     = "#e8e8f0"
SUBTEXT  = "#606090"

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="dashboard-title">KNN Regressor</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-sub">Interactive Explorer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">🎛 Model Controls</div>', unsafe_allow_html=True)

    k_single = st.slider("**Single K value**", min_value=1, max_value=30, value=5, step=1,
                         help="Select K for the main plot. Lower K = complex model, Higher K = smoother model.")

    st.markdown('<div class="section-header">⚖️ Compare Multiple K</div>', unsafe_allow_html=True)
    compare_k = st.multiselect(
        "**Select K values to compare**",
        options=list(range(1, 31)),
        default=[1, 5, 15, 25],
        help="Pick multiple K values to overlay their prediction lines."
    )

    st.markdown('<div class="section-header">🔬 Dataset Settings</div>', unsafe_allow_html=True)
    n_samples  = st.slider("**Number of samples**",    100, 800, 300, step=50)
    noise_lvl  = st.slider("**Noise level (σ)**",      0.1, 5.0, 1.5, step=0.1)
    random_seed = st.slider("**Random seed**",          0,   99,  42, step=1)

    st.markdown('<div class="section-header">🔢 Distance Metric</div>', unsafe_allow_html=True)
    metric = st.selectbox("**Metric**", ["euclidean", "manhattan", "chebyshev"], index=0)

    st.markdown('<div class="section-header">⚙️ Weighting</div>', unsafe_allow_html=True)
    weights = st.selectbox("**Weights**", ["uniform", "distance"], index=0,
                           help="'distance' weights closer neighbours more heavily.")

    st.markdown("---")
    st.markdown(
        '<span style="font-family:Space Mono;font-size:0.65rem;color:#40406a;">'
        'BIAS ↔ VARIANCE TRADEOFF<br>'
        '▲ K → underfitting (high bias)<br>'
        '▼ K → overfitting (high variance)'
        '</span>', unsafe_allow_html=True
    )

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n, noise, seed):
    rng = np.random.RandomState(seed)
    X1  = rng.uniform(-3, 3, n)
    X2  = rng.uniform(-3, 3, n)
    y   = (np.sin(X1) * np.cos(X2)
           + 0.5 * X1
           - 0.3 * X2 ** 2
           + noise * rng.randn(n))
    return np.column_stack([X1, X2]), y

X, y = generate_data(n_samples, noise_lvl, random_seed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sort by Feature 1 for clean 1-D fit line
sort_idx = np.argsort(X[:, 0])
X_sorted = X_scaled[sort_idx]

# 1-D grid over feature 1 (fix feature 2 at median)
grid_f1   = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
grid_f2   = np.full(300, np.median(X[:, 1]))
grid_raw  = np.column_stack([grid_f1, grid_f2])
grid_sc   = scaler.transform(grid_raw)

# ── Fit models & collect predictions ─────────────────────────────────────────
def fit_predict(k):
    m = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights)
    m.fit(X_scaled, y)
    y_pred_full = m.predict(X_scaled)
    y_pred_line = m.predict(grid_sc)
    r2  = r2_score(y, y_pred_full)
    mse = mean_squared_error(y, y_pred_full)
    return y_pred_full, y_pred_line, r2, mse

y_pred_full, y_pred_line, r2_val, mse_val = fit_predict(k_single)
rmse_val = np.sqrt(mse_val)

# ── Dashboard Header ──────────────────────────────────────────────────────────
col_title, col_k = st.columns([3, 1])
with col_title:
    st.markdown('<div class="dashboard-title">KNN Regressor Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dashboard-sub">Synthetic data · {n_samples} samples · '
        f'2 features · noise σ={noise_lvl} · {metric} · {weights} weights</div>',
        unsafe_allow_html=True
    )
with col_k:
    st.markdown(
        f'<div style="text-align:right;margin-top:8px;">'
        f'<div class="metric-label">Active K</div>'
        f'<div class="k-badge">K = {k_single}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Metric cards ──────────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
for col, label, value, unit in [
    (mc1, "K Neighbours",    f"{k_single}",        "neighbours"),
    (mc2, "R² Score",        f"{r2_val:.4f}",      "goodness of fit"),
    (mc3, "RMSE",            f"{rmse_val:.4f}",    "root mean sq. error"),
    (mc4, "MSE",             f"{mse_val:.4f}",     "mean sq. error"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("")

# ── Figure layout: 2 rows × 2 cols ───────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.titlesize":    10,
    "axes.labelsize":    8.5,
})

fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                        left=0.06, right=0.97, top=0.94, bottom=0.07)

# ── PLOT 1 – Main fit line (feature 1 vs y, feature 2 held at median) ────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X[:, 0], y, color=ACCENT2, s=18, alpha=0.45, zorder=2, label="Data points")
ax1.plot(grid_f1, y_pred_line, color=ACCENT1, lw=2.5, zorder=3,
         label=f"KNN fit (K={k_single})")
ax1.axvline(np.median(X[:, 1]), color=SUBTEXT, lw=0.5, linestyle=":")
ax1.set_title(f"Fit Line — Feature 1 vs Target  |  K = {k_single}", pad=10)
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Target y")
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=7.5, framealpha=0.2)

# ── PLOT 2 – Compare multiple K values ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(X[:, 0], y, color=ACCENT2, s=14, alpha=0.30, zorder=2, label="Data")

cmap_k = LinearSegmentedColormap.from_list("kmap", ["#f97316", "#a78bfa", "#60a5fa", "#34d399"])
ks_to_compare = sorted(compare_k) if compare_k else [k_single]
for i, k_c in enumerate(ks_to_compare):
    clr = cmap_k(i / max(len(ks_to_compare) - 1, 1))
    _, pred_line_c, r2_c, _ = fit_predict(k_c)
    ax2.plot(grid_f1, pred_line_c, color=clr, lw=2.2, alpha=0.9,
             label=f"K={k_c}  (R²={r2_c:.3f})")

ax2.set_title("Multi-K Comparison — Feature 1 vs Target", pad=10)
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Target y")
ax2.grid(True, alpha=0.4)
ax2.legend(fontsize=7, framealpha=0.2, ncol=2)

# ── PLOT 3 – Residuals plot ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
residuals = y - y_pred_full
ax3.axhline(0, color=ACCENT4, lw=1.2, linestyle="--", alpha=0.7)
ax3.scatter(y_pred_full, residuals, color=ACCENT3, s=18, alpha=0.55, zorder=2)
ax3.set_title(f"Residuals (Actual − Predicted)  |  K = {k_single}", pad=10)
ax3.set_xlabel("Predicted ŷ")
ax3.set_ylabel("Residual")
ax3.grid(True, alpha=0.4)

# shade ±1σ band
sigma = np.std(residuals)
ax3.axhspan(-sigma, sigma, color=ACCENT3, alpha=0.06, label=f"±1σ = {sigma:.2f}")
ax3.legend(fontsize=7.5, framealpha=0.2)

# ── PLOT 4 – R² vs K curve ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
k_range = np.arange(1, 31)
r2_train = []
for k_iter in k_range:
    m_tmp = KNeighborsRegressor(n_neighbors=k_iter, metric=metric, weights=weights)
    m_tmp.fit(X_scaled, y)
    r2_train.append(r2_score(y, m_tmp.predict(X_scaled)))

ax4.plot(k_range, r2_train, color=ACCENT1, lw=2.2, marker="o",
         markersize=4, alpha=0.85, label="Train R²")
ax4.axvline(k_single, color=ACCENT4, lw=1.5, linestyle="--",
            label=f"Selected K={k_single}")
ax4.scatter([k_single], [r2_val], color=ACCENT4, s=80, zorder=5)
ax4.set_title("R² Score vs K  (Training Set)", pad=10)
ax4.set_xlabel("K  (number of neighbours)")
ax4.set_ylabel("R² Score")
ax4.set_xticks(range(1, 31, 2))
ax4.grid(True, alpha=0.4)
ax4.legend(fontsize=7.5, framealpha=0.2)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    f"KNN Regressor Dashboard   ·   K = {k_single}   ·   "
    f"R² = {r2_val:.4f}   ·   RMSE = {rmse_val:.4f}",
    fontsize=11, fontweight="bold", color=TEXT, y=0.975
)

st.markdown('<div class="plot-container">', unsafe_allow_html=True)
st.pyplot(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

plt.close(fig)

# ── Raw data table (collapsible) ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Dataset (first 50 rows)"):
    df_show = pd.DataFrame(
        np.column_stack([X, y, y_pred_full, y - y_pred_full]),
        columns=["Feature 1", "Feature 2", "Actual y", f"Predicted ŷ (K={k_single})", "Residual"]
    ).round(4).head(50)
    st.dataframe(
        df_show.style.background_gradient(subset=["Residual"], cmap="RdYlGn_r"),
        use_container_width=True
    )

# ── Intuition block ───────────────────────────────────────────────────────────
with st.expander("📖 How KNN Regression works (intuition)"):
    st.markdown(f"""
**K-Nearest Neighbours Regression** predicts a target value for a new point by:
1. Finding the **K closest training points** (using {metric} distance after scaling).
2. Averaging their target values → that average is the prediction.

#### Bias–Variance tradeoff with K
| K | Model behaviour | Bias | Variance |
|---|---|---|---|
| 1 | Memorises every point (jagged fit) | Low | Very High |
| ~5-10 | Balanced | Medium | Medium |
| 30 | Extremely smooth (near flat) | High | Low |

> **Current K = {k_single}** → {"Overfitting risk (high variance)" if k_single <= 3 else "Underfitting risk (high bias)" if k_single >= 20 else "Balanced region"}

#### Why we scale features
KNN is distance-based. If Feature 1 is in [0, 1000] and Feature 2 in [0, 1], 
the algorithm ignores Feature 2 entirely. `StandardScaler` brings both to μ=0, σ=1.

#### Synthetic data formula used
`y = sin(X₁) · cos(X₂) + 0.5·X₁ − 0.3·X₂² + noise`
""")
