import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnScope — Customer Retention Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0D0F14;
    color: #E2E8F0;
}

/* ---------- Header ---------- */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ---------- KPI Cards ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: #13161E;
    border: 1px solid #1e2432;
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #6366f1; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.danger::before  { background: linear-gradient(90deg, #ef4444, #f97316); }
.kpi-card.warning::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.success::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.info::before    { background: linear-gradient(90deg, #6366f1, #818cf8); }

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #475569;
}

/* ---------- Section headers ---------- */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2432;
    margin-left: 0.5rem;
}

/* ---------- Upload zone ---------- */
.upload-zone {
    background: #13161E;
    border: 2px dashed #1e2432;
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #6366f1; }

/* ---------- Risk badges ---------- */
.risk-high    { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.risk-medium  { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.risk-low     { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }

/* ---------- Insight box ---------- */
.insight-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.insight-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* ---------- Dataframe overrides ---------- */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    background: #13161E;
    border: 2px dashed #1e2432;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    background: #13161E !important;
    border: 1px solid #1e2432 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0D0F14 !important;
    border-right: 1px solid #1e2432;
}

/* ---------- Progress bar color ---------- */
.stProgress > div > div { background: #6366f1 !important; }

/* ---------- Info / warning boxes ---------- */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: preprocessing pipeline (mirrors Phase 2 notebook) ─────────────────
BINARY_COLS = ['Gender', 'Senior Citizen', 'Partner', 'Dependents',
               'Phone Service', 'Paperless Billing']
MULTI_COLS  = ['Multiple Lines', 'Internet Service',
               'Online Security', 'Online Backup', 'Device Protection',
               'Tech Support', 'Streaming TV', 'Streaming Movies',
               'Contract', 'Payment Method']

# City frequency map derived from training data — loaded from model artifact if available
CITY_FREQ_MAP = {}  # filled lazily

def load_model():
    model_path = "churn_model_final.pkl"
    if not os.path.exists(model_path):
        st.error("❌ `churn_model_final.pkl` not found. Place it in the same folder as `app.py`.")
        st.stop()
    return joblib.load(model_path)


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Fix Total Charges
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(0, inplace=True)

    # Binary encode
    binary_map = {
        'Gender':            {'Male': 1, 'Female': 0},
        'Senior Citizen':    {'Yes': 1, 'No': 0},
        'Partner':           {'Yes': 1, 'No': 0},
        'Dependents':        {'Yes': 1, 'No': 0},
        'Phone Service':     {'Yes': 1, 'No': 0},
        'Paperless Billing': {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=MULTI_COLS)

    # City frequency encode
    if 'City' in df.columns:
        if CITY_FREQ_MAP:
            df['City'] = df['City'].map(CITY_FREQ_MAP).fillna(0)
        else:
            df.drop(columns=['City'], inplace=True)

    return df


def align_features(df_processed: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Add missing dummy columns and drop extras to match training feature set."""
    for col in model_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    return df_processed[model_features]


def get_risk_tier(prob: float) -> tuple:
    if prob >= 0.65:
        return "High", "🔴", "risk-high"
    elif prob >= 0.35:
        return "Medium", "🟡", "risk-medium"
    else:
        return "Low", "🟢", "risk-low"


def revenue_at_risk(df_results: pd.DataFrame) -> float:
    """Sum monthly charges of high-risk customers."""
    if 'Monthly Charges' in df_results.columns:
        high = df_results[df_results['Risk Tier'] == 'High']['Monthly Charges'].sum()
        return high
    return 0.0


def build_feature_importance_chart(model, feature_names: list):
    """Return a matplotlib figure of top-15 feature importances."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(model.coef_[0])
        except Exception:
            return None

    fi = pd.Series(importances, index=feature_names).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#13161E')
    ax.set_facecolor('#13161E')

    colors = ['#6366f1' if v > fi.mean() else '#4338ca' for v in fi.values]
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor='none', height=0.65)

    # Clean labels
    clean_labels = [l.replace('_', ' ').replace('Internet Service ', 'Internet: ')
                     .replace('Contract ', 'Contract: ').replace('Payment Method ', 'Pay: ')
                     .replace('Multiple Lines ', 'Lines: ')
                    for l in fi.index]
    ax.set_yticklabels(clean_labels, color='#94a3b8', fontsize=9)
    ax.tick_params(axis='x', colors='#475569', labelsize=8)
    ax.set_xlabel('Importance Score', color='#64748b', fontsize=9)
    ax.set_title('Top Churn Drivers (Model Feature Importance)', 
                 color='#e2e8f0', fontsize=11, fontweight='600', pad=12)
    for spine in ax.spines.values():
        spine.set_color('#1e2432')
    ax.grid(axis='x', color='#1e2432', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    return fig


def build_risk_distribution_chart(probs: np.ndarray):
    """Histogram of churn probabilities."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#13161E')
    ax.set_facecolor('#13161E')

    # Color by risk tier
    n, bins, patches = ax.hist(probs, bins=25, edgecolor='none')
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.65:
            patch.set_facecolor('#ef4444')
        elif left >= 0.35:
            patch.set_facecolor('#f59e0b')
        else:
            patch.set_facecolor('#10b981')

    ax.axvline(0.35, color='#f59e0b', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(0.65, color='#ef4444', linewidth=1.2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Churn Probability', color='#64748b', fontsize=9)
    ax.set_ylabel('Customer Count', color='#64748b', fontsize=9)
    ax.set_title('Risk Distribution', color='#e2e8f0', fontsize=11, fontweight='600', pad=10)
    ax.tick_params(colors='#475569', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e2432')
    ax.grid(axis='y', color='#1e2432', linewidth=0.5, alpha=0.7)

    low_patch   = mpatches.Patch(color='#10b981', label='Low risk')
    med_patch   = mpatches.Patch(color='#f59e0b', label='Medium risk')
    high_patch  = mpatches.Patch(color='#ef4444', label='High risk')
    ax.legend(handles=[low_patch, med_patch, high_patch],
              facecolor='#0D0F14', edgecolor='#1e2432', labelcolor='#94a3b8', fontsize=8)

    plt.tight_layout()
    return fig


def churn_by_contract_chart(df_results: pd.DataFrame):
    """Bar chart of avg churn probability by contract type."""
    if 'Contract' not in df_results.columns:
        return None
    avg = df_results.groupby('Contract')['Churn Probability'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#13161E')
    ax.set_facecolor('#13161E')
    ax.bar(avg.index, avg.values, color='#6366f1', edgecolor='none', width=0.5)
    ax.set_ylabel('Avg Churn Probability', color='#64748b', fontsize=9)
    ax.set_title('Churn Risk by Contract Type', color='#e2e8f0', fontsize=11, fontweight='600', pad=10)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1e2432')
    ax.grid(axis='y', color='#1e2432', linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # HERO
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">🔮 AI-Powered Analytics</div>
        <h1 class="hero-title">ChurnScope</h1>
        <p class="hero-subtitle">Upload your customer data and get instant churn risk predictions — ranked, explained, and ready to act on.</p>
    </div>
    """, unsafe_allow_html=True)

    # MODEL LOAD
    model = load_model()

    # Feature names from model
    try:
        model_features = model.feature_names_in_.tolist()
    except AttributeError:
        model_features = None

    # ── UPLOAD SECTION ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📂 Upload Customer Data</div>', unsafe_allow_html=True)

    col_up, col_hint = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop a CSV file here",
            type=["csv"],
            label_visibility="collapsed",
            help="CSV with the same columns as the IBM Telco dataset (excluding leakage columns)."
        )
    with col_hint:
        st.markdown("""
        <div style="background:#13161E; border:1px solid #1e2432; border-radius:10px; padding:1rem; font-size:0.82rem; color:#64748b;">
        <b style="color:#94a3b8;">Required columns include:</b><br>
        Tenure Months · Monthly Charges · Contract · Internet Service · Payment Method · and all standard Telco feature columns.<br><br>
        <b style="color:#94a3b8;">Leakage columns are ignored:</b><br>
        Churn Score · CLTV · Churn Reason
        </div>
        """, unsafe_allow_html=True)

    if uploaded is None:
        # Show a minimal demo prompt
        st.markdown("""
        <div style="background:#13161E; border:1px dashed #312e81; border-radius:12px; padding:3rem; text-align:center; color:#475569; margin-top:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
            <div style="font-size:1rem; color:#64748b;">Upload a CSV file above to begin analysis</div>
            <div style="font-size:0.82rem; margin-top:0.5rem;">Predictions will appear here for all customers at once</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # READ
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    # Columns to drop from MODEL INPUT (leakage/target) — keep CustomerID for display
    model_drop = ['Churn Score', 'CLTV', 'Churn Reason', 'Churn Label', 'Churn Value',
                  'Count', 'Country', 'State', 'Zip Code',
                  'Lat Long', 'Latitude', 'Longitude']
    df_input = df_raw.drop(columns=[c for c in model_drop if c in df_raw.columns])

    # Separate CustomerID for display; exclude from features fed to model
    customer_id_col = None
    if 'CustomerID' in df_input.columns:
        customer_id_col = df_input['CustomerID'].reset_index(drop=True)
        df_features = df_input.drop(columns=['CustomerID'])
    else:
        df_features = df_input.copy()

    # ── PREDICT ───────────────────────────────────────────────────────────────
    with st.spinner("Running predictions…"):
        df_proc = preprocess(df_features)
        if model_features:
            df_aligned = align_features(df_proc, model_features)
        else:
            df_aligned = df_proc

        probs = model.predict_proba(df_aligned)[:, 1]

    # Build results table — include CustomerID as first column if present
    df_results = df_features.copy()
    if customer_id_col is not None:
        df_results.insert(0, 'CustomerID', customer_id_col)
    df_results.insert(0 if customer_id_col is None else 1, 'Churn Probability', np.round(probs, 4))
    df_results.insert(1 if customer_id_col is None else 2, 'Risk Tier', [get_risk_tier(p)[0] for p in probs])
    df_results = df_results.sort_values('Churn Probability', ascending=False).reset_index(drop=True)

    high  = (df_results['Risk Tier'] == 'High').sum()
    med   = (df_results['Risk Tier'] == 'Medium').sum()
    low   = (df_results['Risk Tier'] == 'Low').sum()
    avg_p = probs.mean()
    rev   = revenue_at_risk(df_results)

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card info">
            <div class="kpi-label">Customers Analysed</div>
            <div class="kpi-value">{len(df_results):,}</div>
            <div class="kpi-sub">from uploaded file</div>
        </div>
        <div class="kpi-card danger">
            <div class="kpi-label">High-Risk Customers</div>
            <div class="kpi-value">{high:,}</div>
            <div class="kpi-sub">≥ 65% churn probability</div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-label">Avg Churn Probability</div>
            <div class="kpi-value">{avg_p:.1%}</div>
            <div class="kpi-sub">across all customers</div>
        </div>
        <div class="kpi-card success">
            <div class="kpi-label">Revenue at Risk / mo</div>
            <div class="kpi-value">${rev:,.0f}</div>
            <div class="kpi-sub">high-risk monthly charges</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHARTS ROW ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Insights</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig_dist = build_risk_distribution_chart(probs)
        st.pyplot(fig_dist, use_container_width=True)
        plt.close()

    with c2:
        fi_fig = build_feature_importance_chart(model, df_aligned.columns.tolist())
        if fi_fig:
            st.pyplot(fi_fig, use_container_width=True)
            plt.close()

    with c3:
        fig_contract = churn_by_contract_chart(df_results)
        if fig_contract:
            st.pyplot(fig_contract, use_container_width=True)
            plt.close()

    # ── WHY DO CUSTOMERS CHURN — EXPLANATION PANEL ────────────────────────────
    st.markdown('<div class="section-title">💡 Why Customers Churn — Model Explanation</div>', unsafe_allow_html=True)

    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else None

    if importances is not None:
        fi_series = pd.Series(importances, index=df_aligned.columns).nlargest(8)
        
        # Map features to human-readable business explanations
        explanations = {
            'Tenure Months':                      ("📅 Tenure", "Customers who have been with you less than 12 months are significantly more likely to churn. Early-tenure customers haven't yet built loyalty or switching inertia."),
            'Monthly Charges':                    ("💸 Monthly Charges", "Higher monthly bills correlate strongly with churn. Customers paying premium rates without perceiving equivalent value are most at risk."),
            'Total Charges':                      ("🧾 Total Charges", "Lifetime spend is a proxy for tenure — low total charges flag new customers who haven't yet committed."),
            'Contract_Month-to-month':            ("📋 Month-to-Month Contract", "Month-to-month customers have zero financial penalty for leaving. This is the single strongest behavioural risk indicator in the model."),
            'Contract_Two year':                  ("📋 Two-Year Contract", "Two-year contracts dramatically reduce churn — the switching cost keeps customers committed."),
            'Internet Service_Fiber optic':       ("🌐 Fiber Optic Internet", "Fiber customers churn more often, likely because this segment has higher expectations and more competitive alternatives available."),
            'Internet Service_No':                ("🌐 No Internet Service", "Customers without internet are less engaged with the full product suite and therefore lower-risk overall."),
            'Payment Method_Electronic check':    ("💳 Electronic Check", "Electronic check users churn at higher rates — this payment method often correlates with less automated, more transactional relationships."),
            'Online Security_No':                 ("🔒 No Online Security", "Customers without add-on security features churn more, possibly due to lower overall product engagement."),
            'Tech Support_No':                    ("🛠 No Tech Support", "Customers who don't use Tech Support are less embedded in the service ecosystem."),
            'Paperless Billing_Yes':              ("📧 Paperless Billing", "Paperless billing customers slightly over-index on churn — they may be more digitally mobile and comparison-shopping."),
        }

        insight_html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">'
        for feat, importance in fi_series.items():
            label, desc = explanations.get(feat, (feat.replace('_', ' '), "This feature contributes meaningfully to the model's churn predictions."))
            pct = int(importance / importances.sum() * 100)
            insight_html += f"""
            <div class="insight-box">
                <div class="insight-title">{label} — {pct}% of model weight</div>
                <div style="background:#1e2432; border-radius:4px; height:4px; margin-bottom:0.75rem;">
                    <div style="background:#6366f1; width:{min(pct*5, 100)}%; height:4px; border-radius:4px;"></div>
                </div>
                <p style="font-size:0.88rem; color:#94a3b8; margin:0; line-height:1.6;">{desc}</p>
            </div>
            """
        insight_html += '</div>'
        st.markdown(insight_html, unsafe_allow_html=True)

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗂 Full Prediction Results</div>', unsafe_allow_html=True)

    # Filter controls
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            label_visibility="visible"
        )
    with f2:
        min_prob = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
    with f3:
        search = st.text_input("Search by Customer ID", placeholder="Type a Customer ID…")

    df_display = df_results[
        (df_results['Risk Tier'].isin(tier_filter)) &
        (df_results['Churn Probability'] >= min_prob)
    ]
    if search and 'CustomerID' in df_display.columns:
        df_display = df_display[df_display['CustomerID'].astype(str).str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(df_display):,} of {len(df_results):,} customers")

    # Show top columns only for readability — check df_display, not df_raw
    show_cols = ['Churn Probability', 'Risk Tier']
    nice_cols = ['Tenure Months', 'Monthly Charges', 'Contract', 'Internet Service', 'Payment Method']
    if 'CustomerID' in df_display.columns:
        show_cols = ['CustomerID'] + show_cols
    for c in nice_cols:
        if c in df_display.columns:
            show_cols.append(c)
    # Final safety: only keep columns that actually exist
    show_cols = [c for c in show_cols if c in df_display.columns]

    st.dataframe(
        df_display[show_cols].style
            .background_gradient(subset=['Churn Probability'], cmap='RdYlGn_r', vmin=0, vmax=1)
            .format({'Churn Probability': '{:.1%}'}),
        use_container_width=True,
        height=420
    )

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⬇️ Export</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        csv_full = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Results (CSV)",
            data=csv_full,
            file_name="churn_predictions_all.csv",
            mime="text/csv",
        )
    with d2:
        high_risk = df_results[df_results['Risk Tier'] == 'High']
        csv_high = high_risk.to_csv(index=False).encode('utf-8')
        st.download_button(
            "🔴 Download High-Risk Only (CSV)",
            data=csv_high,
            file_name="churn_predictions_high_risk.csv",
            mime="text/csv",
        )

    # ── ACTION GUIDE ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Retention Playbook</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;">
        <div style="background:#13161E; border:1px solid #1e2432; border-top:2px solid #ef4444; border-radius:10px; padding:1.25rem;">
            <div style="color:#fca5a5; font-weight:600; margin-bottom:0.5rem;">🔴 High Risk (≥65%)</div>
            <ul style="color:#94a3b8; font-size:0.85rem; padding-left:1.2rem; margin:0; line-height:2;">
                <li>Trigger immediate personal outreach</li>
                <li>Offer contract upgrade incentive</li>
                <li>Flag for customer success team</li>
                <li>Consider loyalty discount</li>
            </ul>
        </div>
        <div style="background:#13161E; border:1px solid #1e2432; border-top:2px solid #f59e0b; border-radius:10px; padding:1.25rem;">
            <div style="color:#fcd34d; font-weight:600; margin-bottom:0.5rem;">🟡 Medium Risk (35–65%)</div>
            <ul style="color:#94a3b8; font-size:0.85rem; padding-left:1.2rem; margin:0; line-height:2;">
                <li>Enrol in automated nurture sequence</li>
                <li>Highlight unused features (security, support)</li>
                <li>Satisfaction survey at next touch</li>
                <li>Monitor for escalation signals</li>
            </ul>
        </div>
        <div style="background:#13161E; border:1px solid #1e2432; border-top:2px solid #10b981; border-radius:10px; padding:1.25rem;">
            <div style="color:#6ee7b7; font-weight:600; margin-bottom:0.5rem;">🟢 Low Risk (&lt;35%)</div>
            <ul style="color:#94a3b8; font-size:0.85rem; padding-left:1.2rem; margin:0; line-height:2;">
                <li>Maintain standard engagement cadence</li>
                <li>Upsell opportunities (bundling)</li>
                <li>Referral / advocacy programmes</li>
                <li>Re-score monthly</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#334155; font-size:0.78rem; margin-top:2rem;">
        ChurnScope · Built with Streamlit · Random Forest · IBM Telco Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
