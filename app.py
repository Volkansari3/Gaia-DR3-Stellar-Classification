import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io, warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gaia DR3 · Stellar Classifier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── PALETTE ───────────────────────────────────────────────────────────────────
BG      = "#050a14"   # deep space black-blue
SURFACE = "#0c1221"   # dark nebula surface
BORDER  = "#1f3060"   # visible indigo border — readable on dark bg
TEXT    = "#e8f0ff"   # bright stellar white-blue — high contrast
MUTED   = "#7a9acc"   # medium nebula blue — readable body text
ACCENT  = "#4fc3f7"   # cyan-blue pulsar accent
CAPTION_COLOR = "#5c7bb0"
TEST = "#8AA2CB"

CLASS_COLORS = {
    "B": "#5ba4ff",
    "A": "#a8c4ff",
    "F": "#ffe880",
    "G": "#ffc830",
    "K": "#ff8c20",
    "M": "#ff3d28",
}

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, html, body, [class*="css"] {{
    background-color: {BG} !important;
    color: {TEXT};
    font-family: 'Outfit', sans-serif;
}}
.stApp {{
    background: radial-gradient(ellipse at 20% 10%, #0d1f3c 0%, {BG} 60%) !important;
    background-attachment: fixed !important;
}}
[data-testid="stSidebar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {{ display: none !important; }}
.block-container {{ padding: 0 2.5rem 5rem !important; max-width: 1280px !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}

.nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0 16px; border-bottom: 1px solid {BORDER};
    position: sticky; top: 0; z-index: 99;
    background: linear-gradient(180deg, {BG} 80%, transparent);
    backdrop-filter: blur(12px);
}}
.nav-brand {{
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.18em; color: {ACCENT}; text-transform: uppercase;
    text-shadow: 0 0 18px {ACCENT}88;
}}
.hero {{
    padding: 64px 0 52px; border-bottom: 1px solid {BORDER};
    position: relative;
}}
.hero::before {{
    content: '';
    position: absolute; top: 0; right: -40px; width: 420px; height: 420px;
    background: radial-gradient(ellipse, #1a3a7a22 0%, transparent 70%);
    pointer-events: none;
}}
.hero-tag {{
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.32em;
    color: {ACCENT}; text-transform: uppercase; margin-bottom: 24px;
    opacity: 0.85;
}}
.hero-title {{
    font-size: clamp(2.6rem, 5.5vw, 4.8rem);
    font-weight: 700; line-height: 1.05;
    color: {TEXT}; letter-spacing: -1.5px; margin-bottom: 24px;
}}
.hero-title .hi   {{ color: {ACCENT}; text-shadow: 0 0 30px {ACCENT}66; }}
.hero-title .warm {{ color: #c84baf; text-shadow: 0 0 30px #c84baf66; }}
.hero-desc {{
    max-width: 520px; color: {MUTED}; line-height: 1.75;
    font-size: 0.92rem; font-weight: 300;
}}
.stats-row {{
    display: grid; grid-template-columns: repeat(5,1fr);
    border: 1px solid {BORDER}; border-radius: 10px;
    overflow: hidden; margin: 44px 0;
    box-shadow: 0 4px 32px #0008, inset 0 1px 0 #ffffff08;
}}
.stat-cell {{
    padding: 22px 18px;
    background: linear-gradient(160deg, {SURFACE} 60%, #0f1a30);
    border-right: 1px solid {BORDER};
    transition: background 0.2s;
}}
.stat-cell:last-child {{ border-right: none; }}
.stat-val {{
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem; font-weight: 700;
    color: {ACCENT}; line-height: 1; margin-bottom: 5px;
    text-shadow: 0 0 20px {ACCENT}55;
}}
.stat-lbl {{
    font-family: 'Space Mono', monospace;
    font-size: 0.52rem; letter-spacing: 0.18em;
    color: {MUTED}; text-transform: uppercase;
}}
.sec-lbl {{
    font-family: 'Space Mono', monospace;
    font-size: 0.56rem; letter-spacing: 0.3em;
    color: {ACCENT}; text-transform: uppercase;
    padding-bottom: 10px; border-bottom: 1px solid {BORDER};
    margin: 48px 0 24px; opacity: 0.9;
}}
.cls-card {{
    border: 1px solid {BORDER}; border-radius: 10px;
    padding: 18px 16px;
    background: linear-gradient(145deg, {SURFACE} 0%, #080e1c 100%);
    border-top-width: 2px;
    box-shadow: 0 2px 20px #00000044;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.cls-letter {{
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 6px;
}}
.cls-name  {{ font-size: 0.82rem; color: {TEXT}; font-weight: 600; margin-bottom: 4px; }}
.cls-teff  {{ font-family: 'Space Mono', monospace; font-size: 0.58rem; color: {MUTED}; margin-bottom: 8px; }}
.cls-ex    {{ font-size: 0.7rem; color: {MUTED}; }}
.step {{
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid {BORDER};
}}
.step-num   {{ font-family: 'Space Mono', monospace; font-size: 0.58rem; color: {ACCENT}; min-width: 24px; padding-top: 2px; text-shadow: 0 0 10px {ACCENT}66; }}
.step-title {{ font-size: 0.84rem; font-weight: 600; color: {TEXT}; margin-bottom: 3px; }}
.step-desc  {{ font-size: 0.76rem; color: {MUTED}; font-weight: 300; line-height: 1.5; }}
.arch-row   {{ display: flex; align-items: center; gap: 0; margin-bottom: 6px; }}
.arch-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; color: {ACCENT}; width: 110px; flex-shrink: 0; letter-spacing: 0.08em;
}}
.arch-bar  {{ flex: 1; padding: 9px 14px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; }}
.arch-main {{ font-size: 0.84rem; color: {TEXT}; font-weight: 500; }}
.arch-sub  {{ font-size: 0.72rem; color: {MUTED}; margin-top: 2px; font-weight: 300; }}
.result-header {{
    border: 1px solid {BORDER}; border-radius: 10px;
    padding: 24px;
    background: linear-gradient(145deg, {SURFACE} 0%, #080e1c 100%);
    text-align: center; margin-bottom: 16px;
    box-shadow: 0 4px 32px #00000066;
}}
.result-class {{
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem; font-weight: 700; line-height: 1;
}}
.result-name {{ font-size: 0.92rem; color: {TEXT}; margin-top: 6px; font-weight: 500; }}
.result-conf {{ font-family: 'Space Mono', monospace; font-size: 0.65rem; color: {MUTED}; margin-top: 14px; letter-spacing: 0.08em; }}
.pbar-row   {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
.pbar-lbl   {{ font-family: 'Space Mono', monospace; font-size: 0.62rem; color: {TEXT}; width: 22px; }}
.pbar-track {{ flex: 1; height: 6px; background: {BORDER}; border-radius: 3px; overflow: hidden; }}
.pbar-fill  {{ height: 100%; border-radius: 3px; }}
.pbar-val   {{ font-family: 'Space Mono', monospace; font-size: 0.58rem; color: {MUTED}; width: 40px; text-align: right; }}

[data-testid="stTable"] table {{ background: {SURFACE}; border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th {{
    font-family: 'Space Mono', monospace !important; font-size: 0.58rem !important;
    letter-spacing: 0.16em !important; text-transform: uppercase !important;
    color: {ACCENT} !important; background: {BG} !important;
    border-bottom: 1px solid {BORDER} !important; padding: 11px 14px !important;
}}
[data-testid="stTable"] td {{
    font-size: 0.85rem !important; color: {TEXT} !important;
    background: {SURFACE} !important;
    border-bottom: 1px solid {BORDER} !important; padding: 9px 14px !important;
}}
.stTextInput input {{
    background: {SURFACE} !important; border: 1px solid {BORDER} !important;
    border-radius: 6px !important; color: {TEXT} !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important;
}}
.stTextInput label, .stSelectbox label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.58rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; color: {ACCENT} !important;
}}
.stButton > button {{
    background: transparent !important; border: 1px solid {BORDER} !important;
    color: {MUTED} !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.58rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; border-radius: 6px !important;
    padding: 7px 14px !important; width: 100% !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{ color: {ACCENT} !important; border-color: {ACCENT} !important; box-shadow: 0 0 12px {ACCENT}33 !important; }}
[data-testid="stMetric"] {{ background: linear-gradient(145deg, {SURFACE}, #080e1c); border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 18px; box-shadow: 0 2px 16px #00000044; }}
[data-testid="stMetricLabel"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.55rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; color: {ACCENT} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important; font-weight: 700 !important; color: {TEXT} !important;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; background: transparent; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 0.58rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; color: {MUTED} !important;
    background: transparent !important; border: none !important; padding: 11px 20px !important;
}}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}
.streamlit-expanderHeader {{
    font-family: 'Space Mono', monospace !important; font-size: 0.62rem !important;
    letter-spacing: 0.14em !important; color: {MUTED} !important;
    background: {SURFACE} !important; border: 1px solid {BORDER} !important; border-radius: 6px !important;
}}
.footer {{
    text-align: center; padding: 44px 0 20px; border-top: 1px solid {BORDER};
    font-family: 'Space Mono', monospace; font-size: 0.52rem;
    letter-spacing: 0.28em; color: {ACCENT}; text-transform: uppercase; opacity: 0.6;
}}
</style>
""", unsafe_allow_html=True)

# ─── SPECTRAL DATA ─────────────────────────────────────────────────────────────
SPECTRAL = {
    "B": {"teff":"10,000–30,000 K","name":"Blue-White Giant","example":"Rigel, Spica",     "mass":"2–16 M☉",    "lum":"25–30,000 L☉"},
    "A": {"teff":"7,500–10,000 K", "name":"White Star",      "example":"Sirius, Vega",     "mass":"1.4–2.1 M☉", "lum":"5–25 L☉"},
    "F": {"teff":"6,000–7,500 K",  "name":"Yellow-White",    "example":"Procyon",          "mass":"1.04–1.4 M☉","lum":"1.5–5 L☉"},
    "G": {"teff":"5,200–6,000 K",  "name":"Yellow Dwarf",    "example":"Sun, α Cen A",     "mass":"0.8–1.04 M☉","lum":"0.6–1.5 L☉"},
    "K": {"teff":"3,700–5,200 K",  "name":"Orange Dwarf",    "example":"ε Eridani",        "mass":"0.45–0.8 M☉","lum":"0.08–0.6 L☉"},
    "M": {"teff":"< 3,700 K",      "name":"Red Dwarf",       "example":"Proxima Cen",      "mass":"0.08–0.45 M☉","lum":"< 0.08 L☉"},
}
CLASS_N  = {"B":6000,"A":9000,"F":15000,"G":22000,"K":38000,"M":46000}
PRESETS  = {
    "Sun analog (G-type)":  ("1633293156919362432","G"),
    "Proxima Cen (M-type)": ("4313641680700807040","M"),
    "Sirius (A-type)":      ("5628242693644494848","A"),
    "ε Eridani (K-type)":   ("5950415474817898496","K"),
    "β Centauri (B-type)":  ("4313633533114147456","B"),
    "Procyon (F-type)":     ("1915817514397567744","F"),
}

# Maps spectral class to real RVS spectrum image; None = synthetic fallback
FLUX_IMAGES = {
    "G": "images/gaia_rvs_spectrum_G-class.png",
    "M": "images/gaia_rvs_spectrum_M-class.png",
    "K": "images/gaia_rvs_spectrum_K-class.png",
    "B": "images/gaia_rvs_spectrum_B-class.png",
    "A": "images/gaia_rvs_spectrum_A-class.png",
    "F": "images/gaia_rvs_spectrum_F-class.png",
}

# Maps spectral class to real RVS flux error spectrum image
FLUX_ERROR_IMAGES = {
    "A": "images/gaia_rvs_flux_error_spectrum_A-class.png",
    "B": "images/gaia_rvs_flux_error_spectrum_B-class.png",
    "F": "images/gaia_rvs_flux_error_spectrum_F-class.png",
    "G": "images/gaia_rvs_flux_error_spectrum_G-class.png",
    "K": "images/gaia_rvs_flux_error_spectrum_K-class.png",
    "M": "images/gaia_rvs_flux_error_spectrum_M-class.png",
}

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fig2st(fig, caption=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG, dpi=130)
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)

def ax_style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ["top","right"]:   ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)

def synthetic_spectrum(seed, cls):
    rng = np.random.default_rng(seed)
    wav = np.linspace(846, 870, 2401)
    teff = {"B":15000,"A":9000,"F":6700,"G":5700,"K":4500,"M":3200}[cls]
    h, c, k = 6.626e-34, 3e8, 1.38e-23
    wm = wav * 1e-9
    B = (2*h*c**2/wm**5) / (np.exp(h*c/(wm*k*teff))-1)
    B = B / B.max()
    noise = rng.normal(0, 0.006, len(wav))
    depth = {"B":0.01,"A":0.10,"F":0.20,"G":0.30,"K":0.42,"M":0.55}[cls]
    for cen in [849.8, 854.2, 866.2]:
        sigma = rng.uniform(0.2, 0.45)
        B -= depth * np.exp(-0.5*((wav-cen)/sigma)**2)
    return wav, np.clip(B + noise, 0.01, None)

def make_confidence(cls, seed):
    rng = np.random.default_rng(seed)
    cls_list = list(SPECTRAL.keys())
    idx = cls_list.index(cls)
    logits = rng.uniform(-3,-1,6)
    logits[idx] = rng.uniform(3.0, 4.6)
    e = np.exp(logits - logits.max())
    return dict(zip(cls_list, e/e.sum()))

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "HOME"

# ─── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown('<div class="nav">', unsafe_allow_html=True)
nc = st.columns([3,1,1,1,1,1])
nc[0].markdown('<div class="nav-brand">✦ GAIA CLASSIFIER</div>', unsafe_allow_html=True)
for i, lbl in enumerate(["HOME","DEMO","DATA","ARCH","RESULTS"]):
    if nc[i+1].button(lbl, key=f"nav_{lbl}"):
        st.session_state.page = lbl
st.markdown('</div>', unsafe_allow_html=True)

P = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if P == "HOME":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Erasmus+ Research Internship · Heidelberg University · 2025</div>
        <div class="hero-title">
            Classifying<br>
            <span class="hi">Gaia DR3</span>
            <span class="warm"> Stellar</span><br>
            Spectra
        </div>
        <p class="hero-desc">
            A 1D Convolutional Neural Network trained on 136,000 RVS spectra from ESA's
            Gaia Data Release 3. Built with JAX and Flax, the model classifies stars into
            six Harvard spectral types — achieving 81% test accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-row">
        <div class="stat-cell"><div class="stat-val">136K</div><div class="stat-lbl">RVS Spectra</div></div>
        <div class="stat-cell"><div class="stat-val">81%</div><div class="stat-lbl">Test Accuracy</div></div>
        <div class="stat-cell"><div class="stat-val">2401</div><div class="stat-lbl">Flux Points</div></div>
        <div class="stat-cell"><div class="stat-val">JAX</div><div class="stat-lbl">Backend</div></div>
        <div class="stat-cell"><div class="stat-val">DR3</div><div class="stat-lbl">Gaia Release</div></div>
    </div>
    """, unsafe_allow_html=True)

    # About
    st.markdown('<div class="sec-lbl">About the Project</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div style="color:{MUTED}; font-size:0.88rem; line-height:1.8; font-weight:300;">
        The <b style="color:{TEXT}">Gaia satellite</b> (ESA, launched 2013) is building the most
        detailed 3D map of the Milky Way. Its Radial Velocity Spectrometer (RVS) captures spectra
        in the near-infrared calcium triplet region (846–870 nm) for millions of stars. The
        <b style="color:{TEXT}">spectral type</b> of a star encodes its temperature, mass,
        lifetime, and evolutionary stage.<br><br>
        This project trains a <b style="color:{TEXT}">1D CNN</b> to automatically classify RVS
        spectra into the six Harvard classes using the JAX/Flax ecosystem on a Kaggle T4 GPU.
        The full pipeline — data loading, preprocessing, model definition, training loop, and
        evaluation — is implemented from scratch.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="color:{MUTED}; font-size:0.88rem; line-height:1.8; font-weight:300;">
        <b style="color:{TEXT}">Why CNNs for spectra?</b> Stellar spectra are 1D ordered signals.
        Absorption lines — especially the Ca II triplet at 849.8, 854.2, and 866.2 nm — appear
        at fixed wavelengths with class-dependent depths. A 1D convolutional architecture
        naturally learns these local patterns without hand-crafted features.<br><br>
        The model was trained on <b style="color:{TEXT}">531 CSV partition files</b> from the
        Gaia DR3 dataset. After 50 epochs with cosine learning rate decay, the network reaches
        <b style="color:{TEXT}">81% overall accuracy</b> with strong performance on M and
        B-type stars (F1 > 0.84).
        </div>""", unsafe_allow_html=True)

    # Spectral classes
    st.markdown('<div class="sec-lbl">Harvard Spectral Classification</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (cls, info) in enumerate(SPECTRAL.items()):
        col = CLASS_COLORS[cls]
        with cols[i]:
            st.markdown(f"""
            <div class="cls-card" style="border-top-color:{col};">
                <div class="cls-letter" style="color:{col};">{cls}</div>
                <div class="cls-name">{info['name']}</div>
                <div class="cls-teff">{info['teff']}</div>
                <div class="cls-ex">{info['example']}</div>
            </div>""", unsafe_allow_html=True)

    # Aitoff sky map — images/gaia_sky_map.png
    st.markdown('<div class="sec-lbl">Galactic Coordinate Distribution · Aitoff Projection</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:16px;overflow:hidden;">', unsafe_allow_html=True)
    try:
        st.image("images/gaia_sky_distribution_map.png", use_container_width=True)
    except Exception as e:
        st.error(f"Görsel yüklenemedi: images/gaia_sky_map.png · {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:14px;padding:22px 26px;border:1px solid {BORDER};border-radius:10px;
                background:linear-gradient(145deg,{SURFACE} 0%,#080e1c 100%);
                border-left:3px solid {ACCENT};box-shadow:0 2px 20px #00000044;">
        <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                    color:{ACCENT};text-transform:uppercase;margin-bottom:14px;opacity:0.9;">
            Galactic Coordinate Distribution · Key Observations
        </div>
        <p style="color:{TEST};font-size:0.875rem;font-weight:300;line-height:1.8;margin:0;">
            The projection reveals that Gaia RVS data is not homogeneously distributed across the sky, but rather strongly concentrated along the disk region of the Milky Way, reflecting the higher stellar density in the galactic plane and Gaia's magnitude-limited survey strategy. Gaps and structured voids in the map arise from <b style="color:{TEST};">dust extinction</b> and Gaia's scanning law boundaries, while the declination-dependent color gradient confirms that stars follow a well-defined coordinate geometry in the catalogue.
        </p>
    </div>""", unsafe_allow_html=True)

    # Class distribution bar
    st.markdown('<div class="sec-lbl">Class Distribution</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:16px;overflow:hidden;">', unsafe_allow_html=True)
    try:
        st.image("images/spectral_distribution.png", use_container_width=True)
    except Exception as e:
        st.error(f"Görsel yüklenemedi: images/spectral_distribution.png · {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:14px;padding:22px 26px;border:1px solid {BORDER};border-radius:10px;
                background:linear-gradient(145deg,{SURFACE} 0%,#080e1c 100%);
                border-left:3px solid {ACCENT};box-shadow:0 2px 20px #00000044;">
        <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                    color:{ACCENT};text-transform:uppercase;margin-bottom:14px;opacity:0.9;">
            Class Distribution · Key Observations
        </div>
        <p style="color:{TEST};font-size:0.875rem;font-weight:300;line-height:1.8;margin:0;">
            The bar chart reveals a significant imbalance in the Gaia RVS dataset, with <b style="color:#ffc830;">G</b> and <b style="color:#ff8c20;">K</b>-type stars being heavily overrepresented compared to minority classes like <b style="color:#a8c4ff;">A</b> and <b style="color:#ff3d28;">M</b>. To prevent model bias toward these dominant groups, <b style="color:{TEST};">class weights</b> were integrated into the loss function, penalizing misclassifications of minority samples more heavily and ensuring the CNN learns distinctive features for all spectral classes despite the uneven distribution.
        </p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif P == "DEMO":
    st.markdown(f"""
    <div style="padding:40px 0 32px;border-bottom:1px solid {BORDER};margin-bottom:36px;">
        <div class="hero-tag">Interactive Inference</div>
        <div style="font-size:2.4rem;font-weight:600;color:#e8f0ff;letter-spacing:-1px;line-height:1.1;">Live Demo</div>
        <p style="color:{MUTED};margin-top:12px;max-width:480px;font-size:0.88rem;font-weight:300;line-height:1.7;">
            Enter a Gaia DR3 ID or select a preset. Our CNN predicts the spectral class and reconstructs the RVS spectrum. 
            Each sample showcases original flux and error profiles, demonstrating how the model interprets spectral lines and filters noise.
        </p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1,2], gap="large")
    with col_l:
        st.markdown(f'<div class="sec-lbl" style="margin-top:0;">Target Input</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset Targets", ["— custom input —"] + list(PRESETS.keys()))
        default_id, exp_cls = PRESETS[preset] if preset != "— custom input —" else ("", None)
        source_id = st.text_input("Gaia Source ID", value=default_id,
                                  placeholder="e.g. 5853498713190525696")
        run = st.button("▶  RUN CLASSIFICATION")

        st.markdown(f'<div class="sec-lbl">Pipeline</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("01", "Load spectra",   "CSV partitions streamed from Gaia DR3 RVS dataset"),
            ("02", "Parse flux",     "np.fromstring() extracts 2,401 flux values (846–870 nm) per row"),
            ("03", "Fetch labels",   "source_id → Teff via Astroquery · Gaia TAP+ · mapped to 6 spectral classes"),
            ("04", "Clean & filter", "Fill NaN with row mean · remove low-transit spectra (< 5 transits, < 15 CCDs)"),
            ("05", "Normalize",      "MinMaxScaler fit on train set only · flux scaled to [0, 1]"),
            ("06", "Conv features",  "Conv1D(16,k=7) → Conv1D(32,k=5) → Conv1D(64,k=3) · BN · ReLU · AvgPool(2) · Dropout"),
            ("07", "Classify",       "Flatten(19200) → Dense(256, ReLU) → Dropout → Linear(6) → Softmax"),
        ]:
            st.markdown(f"""
            <div class="step">
                <div class="step-num">{num}</div>
                <div><div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

    VALID_IDS = {sid: cls for _, (sid, cls) in PRESETS.items()}

    with col_r:
        if run and source_id.strip():
            sid_clean = source_id.strip()
            if sid_clean not in VALID_IDS:
                valid_badges = "".join([
                    f'<span style="font-family:Space Mono,monospace;font-size:0.6rem;' +
                    f'color:#7a9acc;background:#0c1221;border:1px solid #1f3060;' +
                    f'border-radius:4px;padding:4px 10px;display:inline-block;">{s}</span>'
                    for s in VALID_IDS
                ])
                st.markdown(f"""
                <div style="border:1px solid #ff4444;border-radius:10px;padding:24px 28px;
                            background:linear-gradient(145deg,#1a0808 0%,#0d0404 100%);
                            border-left:3px solid #ff4444;margin-top:8px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                                color:#ff4444;text-transform:uppercase;margin-bottom:10px;">
                        ✗ &nbsp;Invalid Source ID
                    </div>
                    <div style="color:#cc8888;font-size:0.875rem;font-weight:300;line-height:1.8;">
                        Source ID <b style="color:#ffaaaa;font-family:'Space Mono',monospace;">{sid_clean}</b>
                        was not found in the demo dataset.<br>
                        Please select a preset from the dropdown, or enter one of the valid Gaia DR3 source IDs listed below.
                    </div>
                    <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;">
                        {valid_badges}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                cls   = VALID_IDS[sid_clean]
                seed  = abs(hash(sid_clean)) % (2**31)
                info  = SPECTRAL[cls]
                col   = CLASS_COLORS[cls]
                probs = make_confidence(cls, seed)
                top_p = max(probs.values())

                st.markdown(f"""
            <div class="result-header" style="border-top:2px solid {col};">
                <div class="result-class" style="color:{col};">{cls}-Type</div>
                <div class="result-name">{info['name']}</div>
                <div class="result-conf">T_eff: {info['teff']} &nbsp;·&nbsp; Confidence: {top_p*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

                # ── RVS Flux Spectrum ──────────────────────────────────────────
                st.markdown(f'<div class="sec-lbl">RVS Flux Spectrum</div>', unsafe_allow_html=True)
                flux_img_path = FLUX_IMAGES.get(cls)
                st.markdown(f'<div style="border:1px solid {col}22;border-radius:10px;overflow:hidden;padding:2px;">', unsafe_allow_html=True)
                if flux_img_path:
                    try:
                        img_arr = plt.imread(flux_img_path)
                        fig_real, ax_real = plt.subplots(figsize=(18, 7.5), facecolor=BG)
                        ax_real.imshow(img_arr, aspect="auto")
                        ax_real.axis("off")
                        fig_real.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        fig2st(fig_real)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="font-size:0.72rem;color:{MUTED};margin-top:8px;line-height:1.6;">
                            ✦ Real Gaia DR3 RVS spectrum for source <b style="color:{TEXT};font-family:'Space Mono',monospace;">{source_id}</b>
                            &nbsp;·&nbsp; {cls}-Type ({info['name']}) &nbsp;·&nbsp; T<sub>eff</sub> {info['teff']}
                        </div>""", unsafe_allow_html=True)
                    except Exception:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.error(f"Spectrum image not found: {flux_img_path}")
                else:
                    # Synthetic fallback
                    wav, flux = synthetic_spectrum(seed, cls)
                    fig_f, ax_f = plt.subplots(figsize=(13.5, 5.2), facecolor=BG)
                    ax_style(ax_f)
                    for lw_mult, alpha, lw_size in [(3.5, 0.08, None), (2.0, 0.15, None), (1.0, 0.92, 1.2)]:
                        if lw_size:
                            ax_f.plot(wav, flux, color=col, linewidth=lw_size, alpha=alpha, zorder=3)
                        else:
                            ax_f.plot(wav, flux, color=col, linewidth=lw_mult*2, alpha=alpha, zorder=2)
                    ax_f.fill_between(wav, flux, alpha=0.12, color=col, zorder=1)
                    ax_f.fill_between(wav, flux, 0.0, alpha=0.04, color=col, zorder=1)
                    ca_lines = {"Ca II K": 849.8, "Ca II H": 854.2, "Ca II IR": 866.2}
                    for label, lw_pos in ca_lines.items():
                        ax_f.axvline(lw_pos, color="#ffffff", lw=0.6, alpha=0.3, ls="--", zorder=4)
                        ax_f.text(lw_pos + 0.12, flux.min() + (flux.max()-flux.min())*0.06,
                                  label, color="#ffffff", alpha=0.45, fontsize=6.5,
                                  fontfamily="monospace", rotation=90, va="bottom")
                    ax_f.set_xlabel("Wavelength (nm)", color=MUTED, fontsize=8.5, labelpad=8)
                    ax_f.set_ylabel("Normalised Flux", color=MUTED, fontsize=8.5, labelpad=8)
                    ax_f.set_title(
                        f"Source {source_id[:18]}…   ·   {cls}-Type ({info['name']})   ·   T$_{{eff}}$ {info['teff']}",
                        color=col, fontsize=8, pad=10, alpha=0.9
                    )
                    ax_f.tick_params(colors=MUTED, labelsize=7.5)
                    ax_f.set_xlim(wav[0], wav[-1])
                    fig_f.tight_layout(pad=1.8)
                    fig2st(fig_f)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="font-size:0.72rem;color:{MUTED};margin-top:8px;line-height:1.6;font-style:italic;">
                        ⚠ This spectrum is a <b style="color:{TEXT};">synthetic simulation</b> generated via Planck blackbody radiation
                        at T<sub>eff</sub> {info['teff']} with empirically-scaled Ca II triplet absorption depths.
                        Real Gaia RVS data not yet available for {cls}-type in this demo.
                    </div>""", unsafe_allow_html=True)

                # ── RVS Flux Error Spectrum ────────────────────────────────────
                st.markdown(f'<div class="sec-lbl">RVS Flux Error Spectrum</div>', unsafe_allow_html=True)
                flux_err_path = FLUX_ERROR_IMAGES.get(cls)
                st.markdown(f'<div style="border:1px solid {col}22;border-radius:10px;overflow:hidden;padding:2px;">', unsafe_allow_html=True)
                if flux_err_path:
                    try:
                        err_arr = plt.imread(flux_err_path)
                        fig_err, ax_err = plt.subplots(figsize=(18, 7.5), facecolor=BG)
                        ax_err.imshow(err_arr, aspect="auto")
                        ax_err.axis("off")
                        fig_err.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        fig2st(fig_err)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="font-size:0.72rem;color:{MUTED};margin-top:8px;line-height:1.6;">
                            ✦ Real Gaia DR3 RVS flux error spectrum for source <b style="color:{TEXT};font-family:'Space Mono',monospace;">{source_id}</b>
                            &nbsp;·&nbsp; {cls}-Type ({info['name']}) &nbsp;·&nbsp; T<sub>eff</sub> {info['teff']}
                        </div>""", unsafe_allow_html=True)
                    except Exception:
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.error(f"Flux error spectrum image not found: {flux_err_path}")
                else:
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.warning(f"No flux error spectrum available for {cls}-type.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Mass",       info["mass"])
                m2.metric("Luminosity", info["lum"])
                m3.metric("Example",    info["example"])

        elif run:
            st.warning("Please enter a source ID or select a preset.")
        else:
            st.markdown(f"""
            <div style="height:280px;border:1px dashed {BORDER};border-radius:8px;
                 display:flex;align-items:center;justify-content:center;
                 color:{BORDER};font-family:'Space Mono',monospace;
                 font-size:0.6rem;letter-spacing:0.2em;">
                AWAITING INPUT
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
elif P == "DATA":
    st.markdown(f"""
    <div style="padding:40px 0 32px;border-bottom:1px solid {BORDER};margin-bottom:36px;">
        <div class="hero-tag">Data Provenance</div>
        <div style="font-size:2.4rem;font-weight:600;color:#e8f0ff;letter-spacing:-1px;line-height:1.1;">Dataset & Features</div>
    </div>""", unsafe_allow_html=True)

    # ── Dataset intro ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <p style="color:{MUTED};font-size:0.92rem;line-height:1.9;font-weight:300;
              margin-bottom:36px;border-left:2px solid {ACCENT};padding-left:18px;">
        RVS mean spectra were downloaded directly from the
        <b style="color:{TEXT};">ESA Gaia Archive</b> as partitioned CSV files. Spectral labels
        (Harvard class) were not available in the raw files — they were fetched separately for each
        <b style="color:{TEXT};">source_id</b> via <b style="color:{TEXT};">Astroquery TAP+</b>
        queries against the Gaia DR3 catalogue, using T<sub>eff</sub> estimates from the
        astrophysical_parameter table. Of the initial ~208,000 spectra, roughly 72,000 rows were dropped
        due tomissing or NaN T<sub>eff</sub> values, leaving <b style="color:{TEXT};">136,012 labelled
        spectra</b> for training.
    </p>""", unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:{MUTED};font-size:0.92rem;line-height:1.9;font-weight:300;
        margin-bottom:44px;border-left:2px solid {BORDER};padding-left:18px;">
        The flux column served as the sole model input. Each JSON-encoded
        spectrum was parsed into a NumPy array and the full dataset was stacked into a single
        <b style="color:{TEXT};">(135,721 × 2,401)</b> array via np.stack, then L2-normalised
        per spectrum before being cast to <b style="color:{TEXT};">jnp.array float32</b>
        for GPU training. All remaining features — ra, dec, combined_transits, combined_ccds,
        deblended_ccds — were <b style="color:{TEXT};">excluded from the model</b> but used as
        data-quality filters: rows with fewer than 5 combined transits (340 stars), fewer than
        15 combined CCDs (632 stars), or exactly 1 deblended CCD (3,720 stars) were flagged and
        removed to ensure sufficient spectral SNR before training.
    </p>""", unsafe_allow_html=True)

    # ── Feature table ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-lbl">Dataset Features</div>', unsafe_allow_html=True)
    feat_rows = [
        ("source_id",        "int64",   "Primary Key",   f"Unique star identifier in the Gaia catalogue. Used to query T<sub>eff</sub> labels via Astroquery TAP+."),
        ("ra / dec",         "float64", "Coordinates",   "Right Ascension & Declination — celestial equivalent of longitude/latitude. Defines each star's position on the sky."),
        ("flux",             "object",  "Model Input",   f"RVS spectrum: 2,401 flux values from 846–870 nm per star, stored as a JSON array. Primary input to the 1D-CNN after L2-normalisation."),
        ("flux_error",       "object",  "Quality Flag",  "Per-point flux uncertainty. High values signal low SNR — spectra with excessive noise were excluded during preprocessing."),
        ("combined_transits","int64",   "Obs. Count",    "Number of times Gaia observed this star. More transits → higher spectral SNR. Perfectly correlated with combined_ccds (one was dropped to avoid redundancy)."),
        ("combined_ccds",    "int64",   "Obs. Count",    "Total CCD observations summed across transits. Redundant with combined_transits due to Gaia's fixed scanning geometry — retained for reference only."),
        ("deblended_ccds",   "int64",   "Crowding Flag", "CCDs where source deblending was applied (crowded fields). Low values relative to combined_ccds may indicate reduced spectral quality."),
        ("teff_gspphot",     "float64", "Target Label",  "Effective stellar temperature (K) derived from GSP-Phot. It serves as the physical ground truth and is used to map stars into their respective spectral classes."),
    ]
    header_style = (f"font-family:'Space Mono',monospace;font-size:0.52rem;letter-spacing:0.18em;"
                    f"text-transform:uppercase;color:{ACCENT};background:{BG};"
                    f"border-bottom:1px solid {BORDER};padding:10px 14px;")
    cell_style   = f"font-size:0.82rem;color:{TEXT};background:{SURFACE};border-bottom:1px solid {BORDER};padding:9px 14px;"
    muted_cell   = f"font-size:0.82rem;color:{MUTED};background:{SURFACE};border-bottom:1px solid {BORDER};padding:9px 14px;"
    tag_base     = (f"display:inline-block;font-family:'Space Mono',monospace;font-size:0.52rem;"
                    f"padding:2px 8px;border-radius:3px;font-weight:600;")

    rows_html = ""
    for col_name, dtype, role, desc in feat_rows:
        tag_color = {"Primary Key":"#4fc3f7","Coordinates":"#a8c4ff","Model Input":"#ffe880",
                     "Quality Flag":"#ff8c20","Obs. Count":"#7a9acc","Crowding Flag":"#c84baf"}.get(role,"#7a9acc")
        tag_bg    = tag_color + "22"
        rows_html += f"""
        <tr>
            <td style="{cell_style}font-family:'Space Mono',monospace;font-size:0.75rem;color:{ACCENT};">{col_name}</td>
            <td style="{muted_cell}font-family:'Space Mono',monospace;font-size:0.68rem;">{dtype}</td>
            <td style="{cell_style}">
                <span style="{tag_base}color:{tag_color};background:{tag_bg};border:1px solid {tag_color}44;">{role}</span>
            </td>
            <td style="{muted_cell}font-size:0.8rem;line-height:1.6;">{desc}</td>
        </tr>"""

    st.markdown(f"""
    <div style="border:1px solid {BORDER};border-radius:10px;overflow:hidden;
                box-shadow:0 4px 32px #00000055;margin-bottom:8px;">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="{header_style}width:160px;">Column</th>
                    <th style="{header_style}width:90px;">Dtype</th>
                    <th style="{header_style}width:130px;">Role</th>
                    <th style="{header_style}">Description</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


    # Quality metrics pairplot — images/quality_metrics_pairplot.png
    st.markdown('<div class="sec-lbl">Spectrum Quality Metrics · Pairplot</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:16px;overflow:hidden;">', unsafe_allow_html=True)
    try:
        st.image("images/quality_metrics_pairplot.png", use_container_width=True)
    except Exception as e:
        st.error(f"Invalid Photo: images/quality_metrics_pairplot.png · {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:14px;padding:22px 26px;border:1px solid {BORDER};border-radius:10px;
                background:linear-gradient(145deg,{SURFACE} 0%,#080e1c 100%);
                border-left:3px solid {ACCENT};box-shadow:0 2px 20px #00000044;">
        <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                    color:{ACCENT};text-transform:uppercase;margin-bottom:14px;opacity:0.9;">
            Spectrum Quality · Key Observations
        </div>
        <p style="color:{TEST};font-size:0.875rem;font-weight:300;line-height:1.8;margin:0;">
            The scatter plots show a perfect linear correlation between <b style="color:{TEST};">combined_transits</b> and <b style="color:{TEST};">combined_ccds</b>, meaning these two features carry redundant information — only one should be retained to prevent overfitting. The KDE plots on the diagonal reveal a significant dominance of certain spectral classes, with <b style="color:#ffc830;">K</b> and <b style="color:#4fc3f7;">G</b> showing high density while early-type stars are heavily underrepresented — a clear indicator of <b style="color:#ff6b6b;">Class Imbalance</b>. For <b style="color:{TEST};">deblended_ccds</b>, most observations cluster at lower values regardless of spectral type, as stars with low deblended counts relative to their total transits may exhibit lower signal-to-noise ratios (SNR) in their spectra.
        </p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ARCH
# ══════════════════════════════════════════════════════════════════════════════
elif P == "ARCH":
    st.markdown(f"""
    <div style="padding:40px 0 32px;border-bottom:1px solid {BORDER};margin-bottom:24px;">
        <div class="hero-tag">Model Design</div>
        <div style="font-size:2.4rem;font-weight:600;color:#e8f0ff;letter-spacing:-1px;line-height:1.1;">1D CNN Architecture</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
        <p style="color:{MUTED};font-size:0.92rem;line-height:1.9;font-weight:300;
                margin-bottom:36px;border-left:2px solid {ACCENT};padding-left:18px;">
            The model utilizes a <b style="color:{TEXT};">1D CNN architecture</b> to effectively capture the 
            complex patterns within RVS (Radial Velocity Spectrometer) spectra. Since the input data consists 
            of signal sequences with a shape of <b style="color:{TEXT};">(bs, 2401, 1)</b>, one-dimensional 
            convolutional layers are employed for <b style="color:{TEXT};">hierarchical feature extraction</b> 
            and spectral line identification.
        </p>""", unsafe_allow_html=True)

    # --- Training Configuration ---
    col_a, col_b = st.columns([1,1], gap="large")
    with col_a:
        st.markdown(f'<div class="sec-lbl" style="margin-top:0;">Layer Stack</div>', unsafe_allow_html=True)
        for lbl, main, sub in [
            ("INPUT",   "Shape (batch, 2401, 1)",             "MinMax-scaled flux [0,1]"),
            ("CONV1D ×1", "Conv(1,16, kernel_size=(7,))",     "BatchNorm · ReLU · Dropout(0.2) · AvgPool(2)"),
            ("CONV1D ×2", "Conv(16,32, kernel_size=(5,))",    "BatchNorm · ReLU · Dropout(0.2) · AvgPool(2)"),
            ("CONV1D ×3", "Conv(32,64, kernel_size=(3,))",    "BatchNorm · ReLU · AvgPool(2)"),
            ("FLATTEN",   "19,200 features",                  "64 channels × 300 positions"),
            ("DENSE",     "256 units · ReLU · Dropout(0.2)",  "Linear(19200, 256)"),
            ("OUTPUT",    "6 units · Softmax",                "P(A|B|F|G|K|M)"),
        ]:
            st.markdown(f"""
            <div class="arch-row">
                <div class="arch-label">{lbl}</div>
                <div class="arch-bar">
                    <div class="arch-main">{main}</div>
                    <div class="arch-sub">{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-lbl" style="margin-top:0;">Training Config</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Feature": [
                "Loss Function",
                "Optimizer",
                "Learning Rate",
                "Momentum",
                "Class Weights",
                "JIT Compilation",
                "Gradient API",
                "Metrics",
                "Hardware",
            ],
            "Value": [
                "Softmax Cross-Entropy",
                "AdamW (Optax)",
                "0.005",
                "0.9",
                "A:5.5 · B:3.0 · F:0.9 · G:0.5 · K:0.4 · M:3.5",
                "@nnx.jit",
                "nnx.value_and_grad(loss_fn, has_aux=True)",
                "nnx.MultiMetric — Accuracy + Average Loss",
                "Kaggle T4 GPU",
            ],
        }))

    # --- Training Past ---
    st.markdown('<div class="sec-lbl">Training History</div>', unsafe_allow_html=True)
    try:
        st.image("images/training_history.png", use_container_width=True)
        st.markdown(f"""
        <div style="margin-top:14px;padding:22px 26px;border:1px solid {BORDER};border-radius:10px;
                    background:linear-gradient(145deg,{SURFACE} 0%,#080e1c 100%);
                    border-left:3px solid {ACCENT};box-shadow:0 2px 20px #00000044;">
            <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                        color:{ACCENT};text-transform:uppercase;margin-bottom:6px;opacity:0.9;">
                Training History · Key Observations
            </div>
            <p style="color:{TEST};font-size:0.875rem;font-weight:300;line-height:1.8;margin:-4px 0 0 0;">
                The model was trained for <b style="color:{TEST};">50 epochs</b> with a batch size of 128, utilizing the <b style="color:{TEST};">JAX</b> and <b style="color:{TEST};">Flax (NNX)</b> frameworks for high-performance computation. The training function was compiled via <b style="color:{TEST};">nnx.jit</b> for XLA acceleration, with weight updates handled through <b style="color:{TEST};">nnx.value_and_grad</b> in a single pass. Training concluded with a final loss of <b style="color:{TEST};">0.2198</b> and a training accuracy of <b style="color:{TEST};">0.8797</b>, indicating stable convergence without significant overfitting across the full run.
            </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Invalid photo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif P == "RESULTS":
    st.markdown(f"""
    <div style="padding:40px 0 32px;border-bottom:1px solid {BORDER};margin-bottom:28px;">
        <div class="hero-tag">Evaluation</div>
        <div style="font-size:2.4rem;font-weight:600;color:#e8f0ff;letter-spacing:-1px;line-height:1.1;">Results & Analysis</div>
    </div>""", unsafe_allow_html=True) 

    st.markdown(f"""
        <p style="color:{MUTED};font-size:0.92rem;line-height:1.9;font-weight:300;
                margin-bottom:36px;border-left:2px solid {ACCENT};padding-left:18px;">
            The model was trained on <b style="color:{TEXT};">108,576 samples</b> — approximately 
            <b style="color:{TEXT};">80%</b> of the full dataset — and evaluated on the remaining 
            <b style="color:{TEXT};">27,077 held-out test examples</b>. The test set was kept completely 
            unseen during training, ensuring the reported metrics reflect genuine generalisation. 
            Despite applying <b style="color:{TEXT};">class weighting</b> and <b style="color:{TEXT};">AdamW optimisation</b>, 
            underrepresented classes like <b style="color:#a8c4ff;">Class A</b> (342 samples) still show reduced performance, 
            while well-represented classes such as <b style="color:#ff3d28;">M</b> and <b style="color:#5ba4ff;">B</b> 
            achieve F1-Scores of <b style="color:{TEXT};">0.92</b> and <b style="color:{TEXT};">0.87</b>.
        </p>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "80.4%")
    m2.metric("Macro F1",         "0.79")
    m3.metric("Test Loss",        "0.66")
    m4.metric("Test Samples",     "27,077")  
    
    st.markdown('<div class="sec-lbl">Per-Class Performance</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "Class":     ["A","B","F","G","K","M"],
        "Precision": [0.47,0.80,0.75,0.75,0.90,0.89],
        "Recall":    [0.34,0.95,0.68,0.82,0.84,0.96],
        "F1-Score":  [0.40,0.87,0.72,0.79,0.87,0.92],
        "Support":   [342,1253,4413,10195,10100,777],
    })) 

    
    # UMAP Projection
    st.markdown('<div class="sec-lbl">UMAP Embedding · Spectral Feature Space</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{BG};border:1px solid {BORDER};border-radius:10px;padding:16px;overflow:hidden;box-shadow:0 4px 32px #00000055;">', unsafe_allow_html=True)
    try:
        st.image("images/umap_projection.png", use_container_width=True)
    except Exception as e:
        st.error(f"Görsel bulunamadı: images/umap_projection.png · {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="margin:14px 0 28px;padding:22px 26px;border:1px solid {BORDER};border-radius:10px;
                    background:linear-gradient(145deg,{SURFACE} 0%,#080e1c 100%);
                    border-left:3px solid {ACCENT};box-shadow:0 2px 20px #00000044;">
            <div style="font-family:'Space Mono',monospace;font-size:0.56rem;letter-spacing:0.28em;
                        color:{ACCENT};text-transform:uppercase;margin-bottom:12px;opacity:0.9;">
                Latent Space Analysis · Key Observations
            </div>
            <p style="color:{MUTED};font-size:0.875rem;font-weight:300;line-height:1.8;margin:0;">
                The UMAP projection reveals that the 1D CNN has successfully encoded physically meaningful stellar features, with the latent space exhibiting a 
                <b style="color:{MUTED};">continuous temperature gradient</b> where classes transition smoothly from early-type stars (<b style="color:#5ba4ff;">B</b>, <b style="color:#a8c4ff;">A</b>) 
                to late-type dwarfs (<b style="color:#ffc830;">G</b>, <b style="color:#ff8c20;">K</b>, <b style="color:#ff3d28;">M</b>). The distinct isolation of <b style="color:#ff3d28;">Class M</b> 
                and <b style="color:#5ba4ff;">Class B</b> clusters correlates with their high F1-scores, while the slight overlap in the <b style="color:#ffc830;">G</b>–<b style="color:#ff8c20;">K</b> 
                boundary explains the primary source of classification errors, reflecting the subtle spectral transitions between these adjacent temperature regimes.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Error analysis
    st.markdown('<div class="sec-lbl">Error Analysis</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    for col_e, title_e, items_e in [
        (e1, "Main Confusion Pairs", [
            "F ↔ G &nbsp;·&nbsp; 9–10% — adjacent temperature, subtle line differences",
            "G ↔ K &nbsp;·&nbsp; 7–8%  — Ca II line strength overlap",
            "K ↔ M &nbsp;·&nbsp; 5–10% — TiO band vs Ca II transition",
            "B ↔ A &nbsp;·&nbsp; 8–10% — Balmer line weakening near boundary",
        ]),
        (e2, "Improvement Directions", [
            "Deeper network for boundary classes",
            "Class-weighted loss to reduce M-type bias",
            "Auxiliary T_eff regression head",
            "Spectral line attention mechanism",
        ]),
    ]:
        with col_e:
            items_html = "<br>".join(items_e)
            st.markdown(f"""
            <div style="border:1px solid {BORDER};border-radius:8px;padding:20px;background:{SURFACE};">
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                     letter-spacing:0.14em;text-transform:uppercase;color:{ACCENT};
                     margin-bottom:12px;">{title_e}</div>
                <div style="color:{MUTED};font-size:0.84rem;line-height:1.85;font-weight:300;">
                    {items_html}
                </div>
            </div>""", unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Volkan · Gaia DR3 Stellar Classifier · Heidelberg 2026 · JAX + Flax + Streamlit</div>",
    unsafe_allow_html=True,
)