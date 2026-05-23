# -*- coding: utf-8 -*-
"""
app.py — Dashboard Interactivo Streamlit para Análisis Estadístico IA vs Humanos.
"""

import streamlit as st
import polars as pl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json

# Importación de módulos propios
from text_processor import build_consolidated_df
from contingency_analysis import build_contingency_table, prob_simple, prob_conjunta, prob_condicional
from bayes_engine import BayesTree, _IA_SOURCES_NAMES

# ──────────────────────────────────────────────
# 1. Configuración y Estilos de la App
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluación Significativa 2do. Momento Evaluativo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Share+Tech+Mono&display=swap');

    /* ── VARIABLES ── */
    :root {
        --bg:        #0B0F19;
        --bg2:       #131929;
        --bg3:       #1A2235;
        --cyan:      #00F2FE;
        --cyan-dim:  rgba(0, 242, 254, 0.15);
        --violet:    #9D4EDD;
        --violet-dim:rgba(157, 78, 221, 0.15);
        --text:      #E8EAF0;
        --text-dim:  #8892AA;
        --border:    rgba(0, 242, 254, 0.18);
        --border-v:  rgba(157, 78, 221, 0.28);
        --glass:     rgba(19, 25, 41, 0.75);
        --glow-c:    0 0 18px rgba(0, 242, 254, 0.35);
        --glow-v:    0 0 18px rgba(157, 78, 221, 0.35);
    }

    /* ── BASE ── */
    .stApp,
    section[data-testid="stMain"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }

    /* patrón de fondo sutil */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,242,254,.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,242,254,.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── HEADER ── */
    .main-header {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, var(--cyan) 0%, #a78bfa 60%, var(--violet) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        text-shadow: none;
    }

    /* ── MÉTRICAS — tarjeta cristal cyan ── */
    div[data-testid="stMetric"] {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-top: 2px solid var(--cyan) !important;
        border-radius: 6px !important;
        padding: 1.1rem 1.3rem !important;
        box-shadow: var(--glow-c), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] * { color: var(--text) !important; }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--cyan) !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text) !important;
    }

    /* ── TABS ── */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: var(--bg2) !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 4px;
        padding: 0 4px;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.6rem 1.2rem !important;
        transition: color 0.2s, border-color 0.2s !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        border-bottom-color: var(--violet) !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid var(--cyan) !important;
        box-shadow: 0 4px 12px rgba(0,242,254,0.2) !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: var(--text-dim) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: var(--cyan) !important;
    }

    /* ── BOTONES ── */
    div.stButton > button {
        background: transparent !important;
        color: var(--cyan) !important;
        border: 1px solid var(--cyan) !important;
        border-radius: 4px !important;
        padding: 0.45rem 1.2rem !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.06em;
        box-shadow: var(--glow-c) !important;
        transition: all 0.2s !important;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background: var(--cyan-dim) !important;
        box-shadow: 0 0 28px rgba(0,242,254,0.55) !important;
        color: #fff !important;
    }

    /* ── SELECTBOXES ── */
    div[data-baseweb="select"] > div {
        background: var(--bg3) !important;
        border: 1px solid var(--border-v) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] * { color: var(--text) !important; }
    div[data-baseweb="popover"] ul {
        background: var(--bg3) !important;
        border: 1px solid var(--border-v) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: var(--violet-dim) !important;
    }

    /* ── DATAFRAMES ── */
    [data-testid="stDataFrame"] > div {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    [data-testid="stDataFrame"] th {
        background: var(--bg3) !important;
        color: var(--cyan) !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] td {
        color: var(--text) !important;
        background: transparent !important;
        border-bottom: 1px solid rgba(0,242,254,0.06) !important;
    }

    /* ── CÓDIGO ── */
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code,
    .stCode pre, .stCode code {
        background: var(--bg3) !important;
        color: var(--cyan) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.88rem !important;
    }

    /* ── ALERTAS ── */
    div[data-testid="stAlert"] {
        border-radius: 4px !important;
        backdrop-filter: blur(6px);
    }
    /* info → cyan */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
    div[data-testid="stAlert"]:has(svg[data-testid="stAlertDynamicIcon-info"]) {
        background: rgba(0,242,254,0.08) !important;
        border-left: 3px solid var(--cyan) !important;
    }
    /* success → verde apagado */
    div[data-testid="stAlert"]:has(svg[data-testid="stAlertDynamicIcon-success"]) {
        background: rgba(0,255,153,0.07) !important;
        border-left: 3px solid #00ff99 !important;
    }
    /* warning → violeta */
    div[data-testid="stAlert"]:has(svg[data-testid="stAlertDynamicIcon-warning"]) {
        background: var(--violet-dim) !important;
        border-left: 3px solid var(--violet) !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] * { color: var(--text) !important; }

    /* ── SLIDERS ── */
    [data-testid="stSlider"] label p {
        color: var(--cyan) !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    [data-testid="stSlider"] [data-testid="stTickBar"] span { color: var(--text-dim) !important; }

    /* ── EXPANDER — tarjeta cristal violeta ── */
    details {
        background: var(--glass) !important;
        border: 1px solid var(--border-v) !important;
        border-left: 3px solid var(--violet) !important;
        border-radius: 6px !important;
        padding: 0.2rem 0.8rem !important;
        box-shadow: var(--glow-v) !important;
        backdrop-filter: blur(8px);
        margin-bottom: 0.6rem;
    }
    details summary {
        color: var(--violet) !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em;
        cursor: pointer;
        padding: 0.5rem 0;
    }
    details > div * { color: var(--text) !important; }

    /* ── PROGRESS BAR ── */
    [data-testid="stProgress"] > div {
        background: var(--bg3) !important;
        border-radius: 4px !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
        border-radius: 4px !important;
        box-shadow: var(--glow-c);
    }

    /* ── SUBHEADERS / TEXT ── */
    h1, h2, h3 { color: var(--text) !important; }
    h3 { color: var(--cyan) !important; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
    p, span, label { color: var(--text) !important; }
    strong { color: var(--cyan) !important; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb {
        background: var(--violet);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

    /* ── SPINNER ── */
    [data-testid="stSpinner"] * { color: var(--cyan) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# 2. Carga y Caché de Datos
# ──────────────────────────────────────────────
@st.cache_data
def get_dataframes():
    """Carga de datos tabulares (DataFrame y Contingencia)"""
    df = build_consolidated_df(top_n=15)
    ct = build_contingency_table(df)
    return df, ct

@st.cache_resource
def get_bayesian_tree():
    """Inicialización del motor Bayesiano que procesa el corpus"""
    return BayesTree.from_corpus()

try:
    with st.spinner("Analizando corpus y procesando datos..."):
        df_cons, ct = get_dataframes()
        tree = get_bayesian_tree()
except Exception as e:
    st.error(f"Error crítico al cargar los datos. Verifica la estructura local: {e}")
    st.stop()


# ──────────────────────────────────────────────
# 3. Interfaz Principal
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Árbol de probabilidades interactivo (Plotly)
# ──────────────────────────────────────────────
_CYAN   = "#00F2FE"
_VIOLET = "#9D4EDD"
_SOURCES_ALL = ["Humano", "ChatGPT", "Gemini", "Grok"]
_NODE_COLOR: dict[str, str] = {
    "Humano":  "#94A3B8",   # gris slate
    "ChatGPT": "#3B82F6",   # azul
    "Gemini":  "#00F2FE",   # cyan
    "Grok":    "#F43F5E",   # rojo-violeta
}


def _build_prob_tree_html(
    ct: pl.DataFrame,
    l1_dim: str,
    l1_sel: list[str],
    l2_sel: list[str],
) -> tuple[str, int]:
    CYAN, VIOLET = "#00F2FE", "#9D4EDD"
    ct_data    = ct.filter(pl.col("palabra") != "Total")
    total_row  = ct.filter(pl.col("palabra") == "Total")
    gran_total = int(total_row.select("Total").item())
    f_totals   = {s: int(total_row.select(s).item()) for s in _SOURCES_ALL}
    p_totals   = {r["palabra"]: int(r["Total"]) for r in ct_data.to_dicts()}

    nl1, nl2  = len(l1_sel), len(l2_sel)
    n_leaves  = nl1 * nl2
    SVG_W     = 680
    SVG_H     = max(500, nl1 * 130)
    X0, X1, X2 = 70, 265, 490
    leaf_ys   = [48 + (k + 0.5) * (SVG_H - 96) / n_leaves for k in range(n_leaves)]
    root_y    = SVG_H / 2

    e_svg = ""   # bezier edges
    el_svg = ""  # edge probability labels
    n_svg = ""   # nodes + text
    js_data: list[dict] = []

    def _bez(x1: float, y1: float, x2: float, y2: float) -> str:
        cx = x1 + (x2 - x1) * 0.42
        dx = x2 - (x2 - x1) * 0.42
        return (
            f'<path d="M{x1},{y1:.1f} C{cx:.1f},{y1:.1f} {dx:.1f},{y2:.1f} {x2},{y2:.1f}" '
            f'stroke="rgba(0,242,254,0.2)" stroke-width="1.4" fill="none"/>'
        )

    def _node(x: float, y: float, label: str, color: str, r: int, tip: str) -> None:
        nonlocal n_svg
        glow = "0,242,254" if color == CYAN else "157,78,221"
        nid  = f"n{len(js_data)}"
        n_svg += (
            f'<circle id="{nid}" cx="{x}" cy="{y:.1f}" r="{r}" fill="{color}" '
            f'fill-opacity="0.88" stroke="rgba(255,255,255,0.18)" stroke-width="1.4" '
            f'style="filter:drop-shadow(0 0 7px rgba({glow},.8));cursor:pointer"/>'
            f'<text x="{x + r + 5}" y="{y + 4:.1f}" fill="#E8EAF0" font-size="11" '
            f'font-family="Inter,sans-serif" font-weight="600">{label}</text>'
        )
        js_data.append({"id": nid, "r": r, "tip": tip})

    def _elabel(x: float, y: float, txt: str) -> None:
        nonlocal el_svg
        el_svg += (
            f'<text x="{x:.0f}" y="{y:.0f}" fill="rgba(0,242,254,0.6)" '
            f'font-size="9" text-anchor="middle" font-family="monospace">{txt}</text>'
        )

    # raíz
    _node(X0, root_y, "Corpus", CYAN, 22, f"Gran Total\n{gran_total:,}")

    for i, l1 in enumerate(l1_sel):
        band   = leaf_ys[i * nl2: (i + 1) * nl2]
        l1_y   = sum(band) / nl2
        l1_tot = f_totals[l1] if l1_dim == "fuente" else p_totals.get(l1, 0)
        p_l1   = l1_tot / gran_total if gran_total else 0.0

        e_svg += _bez(X0, root_y, X1, l1_y)
        _elabel((X0 + X1) / 2, (root_y + l1_y) / 2 - 8, f"{p_l1:.1%}")
        _node(X1, l1_y, l1, VIOLET, 18, f"Total: {l1_tot:,}\nP({l1}) = {p_l1:.4%}")

        for j, l2 in enumerate(l2_sel):
            l2_y   = band[j]
            if l1_dim == "fuente":
                row  = ct_data.filter(pl.col("palabra") == l2)
                freq = int(row.select(l1).item()) if not row.is_empty() else 0
                tip  = (f"'{l2}' | {l1}\nfreq = {freq}\n"
                        f"P(cond) = {freq/l1_tot:.4%}\nP(∩) = {freq/gran_total:.4%}")
                col2 = _NODE_COLOR.get(l1, CYAN)   # color por fuente padre
            else:
                row  = ct_data.filter(pl.col("palabra") == l1)
                freq = int(row.select(l2).item()) if not row.is_empty() else 0
                tip  = (f"{l2} | '{l1}'\nfreq = {freq}\n"
                        f"P(cond) = {freq/l1_tot:.4%}\nP(∩) = {freq/gran_total:.4%}")
                col2 = _NODE_COLOR.get(l2, VIOLET)  # color por fuente hoja
            p_cond = freq / l1_tot if l1_tot else 0.0
            e_svg += _bez(X1, l1_y, X2, l2_y)
            _elabel((X1 + X2) / 2, (l1_y + l2_y) / 2 - 7, f"{p_cond:.1%}")
            _node(X2, l2_y, l2, col2, 14, tip)

    # cabeceras nivel
    for lbl, xp in [("Corpus", X0), ("Nivel 1", X1), ("Nivel 2", X2)]:
        n_svg += (
            f'<text x="{xp}" y="26" fill="#8892AA" font-size="11" font-weight="700" '
            f'font-family="Inter,sans-serif" text-anchor="middle">{lbl}</text>'
        )

    js_json = json.dumps(js_data, ensure_ascii=False)

    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        f'*{{margin:0;padding:0;box-sizing:border-box}}'
        f'body{{background:transparent}}'
        f'#tt{{position:absolute;background:#0D1B2A;border:1px solid #00F2FE;'
        f'border-radius:4px;padding:8px 12px;color:#E8EAF0;font-size:12px;'
        f'font-family:Inter,sans-serif;pointer-events:none;display:none;'
        f'max-width:230px;white-space:pre;line-height:1.7;z-index:9;'
        f'box-shadow:0 0 14px rgba(0,242,254,.3)}}'
        f'</style></head><body>'
        f'<div id="w" style="position:relative;display:inline-block;'
        f'background:rgba(11,15,25,.55);border:1px solid rgba(0,242,254,.18);'
        f'border-radius:8px;overflow:hidden">'
        f'<svg width="{SVG_W}" height="{SVG_H}">{e_svg}{el_svg}{n_svg}</svg>'
        f'<div id="tt"></div></div>'
        f'<script>(function(){{'
        f'var d={js_json},'
        f'tt=document.getElementById("tt"),'
        f'w=document.getElementById("w");'
        f'd.forEach(function(n){{'
        f'var el=document.getElementById(n.id),r0=n.r;'
        f'if(!el)return;'
        f'el.addEventListener("mouseenter",function(){{'
        f'tt.innerText=n.tip;tt.style.display="block";el.setAttribute("r",r0*1.4);'
        f'}});'
        f'el.addEventListener("mousemove",function(e){{'
        f'var b=w.getBoundingClientRect(),'
        f'tx=e.clientX-b.left+16,ty=e.clientY-b.top-10;'
        f'if(tx+240>w.offsetWidth)tx=e.clientX-b.left-256;'
        f'tt.style.left=tx+"px";tt.style.top=ty+"px";'
        f'}});'
        f'el.addEventListener("mouseleave",function(){{'
        f'tt.style.display="none";el.setAttribute("r",r0);'
        f'}});'
        f'}});}})();</script></body></html>'
    )
    return html, SVG_H


st.markdown('<div class="main-header">Evaluación Significativa 2do. Momento Evaluativo</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "Dashboard Frecuencias", 
    "Tablas de Contingencia", 
    "Inferencia Bayesiana", 
    "Cadenas de Markov"
])


# ──────────────────────────────────────────────
# TAB 1: Dashboard de Frecuencias
# ──────────────────────────────────────────────
with tabs[0]:
    st.subheader("Dashboard Global del Corpus")
    
    # Métricas Globales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Palabras Corpus", f"{tree.total_corpus:,}")
    col2.metric("Palabras Humano", f"{tree.total_humano:,}")
    col3.metric("Palabras IA", f"{tree.total_ia:,}")
    col4.metric("Probabilidad A Priori Humano", f"{tree.p_humano:.2%}")
    
    st.markdown("### Top 15 Palabras en Común")
    
    # Preparar DataFrame para Gráfico de Barras Nativas
    # Pivotar para st.bar_chart: Index = palabra, Columns = Fuentes
    df_pivot = (
        df_cons.select(["palabra", "fuente", "conteo"])
        .pivot(on="fuente", index="palabra", values="conteo")
        .fill_null(0)
    )
    
    # Ordenar por el total para el gráfico
    pd_pivot = df_pivot.to_pandas().set_index("palabra")
    pd_pivot["Total"] = pd_pivot.sum(axis=1)
    pd_pivot = pd_pivot.sort_values(by="Total", ascending=False).drop(columns=["Total"])
    
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.markdown("**Frecuencias Absolutas Cruzadas**")
        st.bar_chart(pd_pivot, height=450)
        
    with col_table:
        st.markdown("**Dataset Consolidado (Frecuencia Relativa)**")
        # Mostrar el dataframe original con frecuencia relativa
        pd_cons = df_cons.to_pandas()
        st.dataframe(
            pd_cons.style.format({"freq_relativa": "{:.5f}"}), 
            width="stretch", 
            height=450
        )


# ──────────────────────────────────────────────
# TAB 2: Tablas de Contingencia
# ──────────────────────────────────────────────
with tabs[1]:
    st.subheader("Análisis de Probabilidad Dinámico")
    st.markdown("#### Tabla de Contingencia (Top 15 + Totales)")
    pd_ct = ct.to_pandas().set_index("palabra")
    st.dataframe(pd_ct.style.highlight_max(axis=1, color="#0D3348"), width="stretch")
    
    # Palabras y fuentes disponibles
    palabras_list = tree.palabras
    fuentes_list = ["Humano", "ChatGPT", "Gemini", "Grok"]
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown("#### Ejercicio 1: P. Simple P(A)")
        word_simple = st.selectbox("Selecciona Palabra", palabras_list, key="w1")
        if st.button("Calcular P(A)", key="b1"):
            res = prob_simple(ct, word_simple)
            st.info(f"**Fórmula:** {res.formula}\n\n**Fracción:** {res.fraccion}\n\n**Resultado:** {res.valor:.4%} \n\n{res.descripcion}")
            
    with colB:
        st.markdown("#### Ejercicio 2: P. Conjunta P(A ∩ B)")
        word_conj = st.selectbox("Selecciona Palabra", palabras_list, key="w2", index=1)
        src_conj = st.selectbox("Selecciona Fuente", fuentes_list, key="s2", index=1)
        if st.button("Calcular P(A ∩ B)", key="b2"):
            res = prob_conjunta(ct, word_conj, src_conj)
            st.success(f"**Fórmula:** {res.formula}\n\n**Fracción:** {res.fraccion}\n\n**Resultado:** {res.valor:.4%} \n\n{res.descripcion}")
            
    with colC:
        st.markdown("#### Ejercicio 3: P. Condicional P(A | B)")
        word_cond = st.selectbox("Selecciona Palabra", palabras_list, key="w3", index=2)
        src_cond = st.selectbox("Dado la Fuente", fuentes_list, key="s3", index=0)
        if st.button("Calcular P(A | B)", key="b3"):
            res = prob_condicional(ct, word_cond, src_cond)
            st.warning(f"**Fórmula:** {res.formula}\n\n**Fracción:** {res.fraccion}\n\n**Resultado:** {res.valor:.4%} \n\n{res.descripcion}")

    # ── Árbol interactivo ──────────────────────
    st.divider()
    st.markdown("### Árbol de Probabilidades Interactivo")

    _col_tree, _col_ctrl = st.columns([2, 1])

    # controles derecha — se ejecutan primero para tener los valores
    with _col_ctrl:
        st.markdown("##### Configuración")
        _l1_dim_label = st.radio(
            "Dimensión Nivel 1",
            ["Fuente → Palabra", "Palabra → Fuente"],
            key="tree_dim",
        )
        _l1_dim = "fuente" if _l1_dim_label == "Fuente → Palabra" else "palabra"

        if _l1_dim == "fuente":
            _l1_opts, _l1_def, _l1_lbl = _SOURCES_ALL, _SOURCES_ALL, "Fuentes (N1)"
            _l2_opts, _l2_def, _l2_lbl = palabras_list, palabras_list[:5], "Palabras (N2)"
        else:
            _l1_opts, _l1_def, _l1_lbl = palabras_list, palabras_list[:4], "Palabras (N1)"
            _l2_opts, _l2_def, _l2_lbl = _SOURCES_ALL, _SOURCES_ALL, "Fuentes (N2)"

        _l1_sel = st.multiselect(_l1_lbl, _l1_opts, default=_l1_def, key="tree_l1")
        _l2_sel = st.multiselect(_l2_lbl, _l2_opts, default=_l2_def, key="tree_l2")

        # métricas rápidas
        if _l1_sel and _l2_sel:
            st.markdown("---")
            _DIM = '<p style="color:#94A3B8;font-size:0.72rem;margin:0;line-height:1.4">'
            st.metric("Nodos N1", len(_l1_sel))
            st.markdown(_DIM + "Palabras activas en el análisis</p>",
                        unsafe_allow_html=True)
            st.metric("Nodos N2", len(_l2_sel))
            st.markdown(_DIM + "Modelos de lenguaje cruzados</p>",
                        unsafe_allow_html=True)
            _leaves = len(_l1_sel) * len(_l2_sel)
            st.metric("Hojas totales", _leaves)
            st.markdown(_DIM + "Caminos estocásticos calculados</p>",
                        unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#94A3B8;font-size:0.72rem;margin:4px 0 0 0">'
                "∑ P(Ramas) = 1.0000 ✓</p>",
                unsafe_allow_html=True,
            )

    # árbol izquierda
    with _col_tree:
        if not _l1_sel or not _l2_sel:
            st.warning("Selecciona al menos 1 nodo en cada nivel.")
        elif len(_l1_sel) * len(_l2_sel) > 40:
            st.warning("Demasiados nodos (máx 40 hojas). Reduce la selección.")
        else:
            _html, _h = _build_prob_tree_html(ct, _l1_dim, _l1_sel, _l2_sel)
            st.components.v1.html(_html, height=_h + 20, scrolling=False)


# ──────────────────────────────────────────────
# TAB 3: Inferencia Bayesiana
# ──────────────────────────────────────────────
def create_bayesian_network_fig(tree: BayesTree, word: str):
    """Genera la figura de la Red Bayesiana usando NetworkX y Matplotlib (3 Estados)"""
    G = nx.DiGraph()
    
    sources = ["Humano", "ChatGPT", "Gemini", "Grok"]
    node_w = f"'{word}'"
    node_not_w = f"No '{word}'"
    word_nodes = [node_w, node_not_w]
    dest_nodes = ["IA", "H"]
    
    # Añadir Nodos
    for s in sources: G.add_node(s)
    for w in word_nodes: G.add_node(w)
    for d in dest_nodes: G.add_node(d)
        
    # Posicionamiento Estricto
    pos = {}
    pos["Humano"]  = (0, 4)
    pos["ChatGPT"] = (0, 3)
    pos["Gemini"]  = (0, 2)
    pos["Grok"]    = (0, 1)
    
    pos[node_w]     = (1, 3.2)
    pos[node_not_w] = (1, 1.8)
    
    pos["IA"] = (2, 3.2)
    pos["H"]  = (2, 1.8)
    
    # Arcos Estado 0 -> Estado 1 (Fuentes -> Palabra)
    for s in sources:
        p_w = tree._p_palabra_dado_fuente(word, s)
        p_nw = 1.0 - p_w
        G.add_edge(s, node_w, label=f"{p_w:.2%}")
        G.add_edge(s, node_not_w, label=f"{p_nw:.2%}")
        
    # Arcos Estado 1 -> Estado 2 (Palabra -> Autor)
    for w in word_nodes:
        for d in dest_nodes:
            G.add_edge(w, d)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color="#FFFFFF", edgecolors="#000000", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="#000000", font_size=10, font_weight="bold")
    
    # Dibujar arcos
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#000000", alpha=1.0, arrows=True, arrowsize=15, width=1.5
    )
    
    # Etiquetas de probabilidad
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if "label" in d}
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=edge_labels, font_color="#000000", font_size=9, label_pos=0.28,
        bbox=dict(boxstyle="square,pad=0.2", fc="#FFFFFF", ec="#000000", alpha=1.0)
    )
    
    ax.set_title(f"Red Bayesiana para '{word}'", color="#000000", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig

with tabs[2]:
    st.subheader("Teorema de Bayes en Redes Bayesianas")
    
    # Controles de Selección
    c1, c2 = st.columns(2)
    word_bayes = c1.selectbox("Palabra Observada", tree.palabras, key="wb")
    tool_bayes = c2.selectbox("Herramienta (Hipótesis)", _IA_SOURCES_NAMES, key="tb")

    col_graph, col_bayes = st.columns([1, 1])
    
    with col_graph:
        # Visualizar Grafo Dinámicamente sin archivos estáticos
        st.pyplot(create_bayesian_network_fig(tree, word_bayes), transparent=False)
        
        # Diagrama de Árbol en texto plano
        with st.expander("Ver Diagrama de Árbol (Probabilidades A Priori)"):
            st.code(tree.print_tree(), language="markdown")
            
    with col_bayes:
        st.markdown("#### Inversión Bayesiana: P(Herramienta | IA ∩ Palabra)")
        st.write("Dado que sabemos que **una IA generó una palabra específica**, ¿cuál es la probabilidad de que haya sido una herramienta concreta (Gemini, ChatGPT o Grok)?")
        
        res = tree.bayes_fuente_dado_ia_y_palabra(tool_bayes, word_bayes)
        
        # Renderizado atractivo del resultado
        st.markdown(f"### Resultado: **{res.resultado:.2%}**")
        st.progress(res.resultado)
        st.info(res.interpretacion)
        
        with st.expander("Desglose Matemático del Teorema", expanded=True):
            # fórmula general
            st.markdown("**Fórmula General**")
            st.latex(
                r"P\!\left(\text{" + tool_bayes + r"} \mid IA \cap \textit{``" + word_bayes + r"''}\right) = "
                r"\frac{"
                r"P\!\left(\textit{``" + word_bayes + r"''} \mid \text{" + tool_bayes + r"}\right)"
                r"\cdot P\!\left(\text{" + tool_bayes + r"} \mid IA\right)"
                r"}{"
                r"\displaystyle\sum_{j \,\in\, \{\text{ChatGPT},\,\text{Gemini},\,\text{Grok}\}}"
                r"P\!\left(\textit{``" + word_bayes + r"''} \mid F_j\right) \cdot P\!\left(F_j \mid IA\right)"
                r"}"
            )

            # numerador
            _pw_f  = tree._p_palabra_dado_fuente(word_bayes, tool_bayes)
            _pf_ia = tree._p_fuente_dado_ia(tool_bayes)
            st.markdown("**Numerador (Conjunta)**")
            st.latex(
                r"P\!\left(\textit{``" + word_bayes + r"''} \mid \text{" + tool_bayes + r"}\right)"
                r"\cdot P\!\left(\text{" + tool_bayes + r"} \mid IA\right) = "
                f"{_pw_f:.6f} \\times {_pf_ia:.6f} = {res.numerador_valor:.8f}"
            )

            # denominador aligned
            _parts = [
                (src, tree._p_palabra_dado_fuente(word_bayes, src), tree._p_fuente_dado_ia(src))
                for src in ["ChatGPT", "Gemini", "Grok"]
            ]
            _denom_body = r" \\ &\quad + ".join(
                r"P\!\left(\textit{``" + word_bayes + r"''} \mid \text{" + s
                + r"}\right) \cdot P\!\left(\text{" + s
                + r"} \mid IA\right) = "
                + f"{pw:.6f} \\times {ps:.6f} = {pw*ps:.8f}"
                for s, pw, ps in _parts
            )
            st.markdown("**Denominador — Probabilidad Total IA**")
            st.latex(
                r"\begin{aligned} D &= "
                + _denom_body
                + r" \\ &= " + f"{res.denominador_valor:.8f}"
                + r"\end{aligned}"
            )

            # resultado final
            st.markdown("**Resultado Final**")
            st.latex(
                r"\therefore \quad P\!\left(\text{" + tool_bayes
                + r"} \mid IA \cap \textit{``" + word_bayes + r"''}\right) = "
                r"\frac{" + f"{res.numerador_valor:.8f}" + r"}{"
                + f"{res.denominador_valor:.8f}" + r"} \;=\; "
                + f"{res.resultado:.6f}"
                + r"\;\approx\; \mathbf{"
                + f"{res.resultado*100:.4f}"
                + r"\%}"
            )


# ──────────────────────────────────────────────
# TAB 4: Cadenas de Markov
# ──────────────────────────────────────────────
_MK_COLORS = ["#94A3B8", "#00F2FE", "#3B82F6"]   # Humano, Gemini, IA-Otros
_MK_GLOW   = ["148,163,184", "0,242,254", "59,130,246"]
_MK_LABELS = ["S1: Humano", "S2: Gemini", "S3: IA-Otros"]
_MK_STATES = ["Humano", "Gemini", "IA-Otros"]


def run_markov_simulation(tree: BayesTree, steps: int = 20, inertia: float = 0.75):
    pi_1 = tree.p_humano
    pi_2 = tree.p_ia * tree.p_gemini_dado_ia
    pi_3 = tree.p_ia * (tree.p_chatgpt_dado_ia + tree.p_grok_dado_ia)
    pi   = np.array([pi_1, pi_2, pi_3])
    T    = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = (1.0 - inertia) * pi[j]
            if i == j:
                T[i, j] += inertia
    v = np.array([0.0, 1.0, 0.0])
    history = [v.copy()]
    for _ in range(steps):
        v = v @ T
        history.append(v.copy())
    return T, np.array(history), pi


def _markov_diagram_html(T: np.ndarray, pi: np.ndarray) -> str:
    import math
    W, H, NR = 600, 310, 40
    # triángulo: S1 izq-centro, S2 top-der, S3 bot-der
    pos = [(115, 155), (460, 65), (460, 245)]

    markers = "".join(
        f'<marker id="mk{i}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
        f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{_MK_COLORS[i]}" opacity="0.9"/></marker>'
        for i in range(3)
    )

    # halo texto: stroke oscuro detrás para legibilidad
    _TS = 'style="paint-order:stroke fill;stroke:#0A0E18;stroke-width:3px;stroke-linejoin:round"'

    parts: list[str] = []

    # self-loops
    for i, (xi, yi) in enumerate(pos):
        w  = max(2.0, T[i][i] * 6)
        ly = yi - NR - 28
        parts.append(
            f'<path d="M{xi - NR*0.48:.1f},{yi - NR*0.75:.1f} '
            f'A 24,24 0 1,1 {xi + NR*0.48:.1f},{yi - NR*0.75:.1f}" '
            f'stroke="{_MK_COLORS[i]}" stroke-width="{w:.1f}" fill="none" '
            f'stroke-opacity="0.85" marker-end="url(#mk{i})"/>'
            f'<text x="{xi}" y="{ly}" fill="{_MK_COLORS[i]}" font-size="11" '
            f'font-family="monospace" font-weight="700" text-anchor="middle" {_TS}>'
            f'{T[i][i]:.3f}</text>'
        )

    # transiciones
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            xi, yi = pos[i];  xj, yj = pos[j]
            ang  = math.atan2(yj - yi, xj - xi)
            perp = ang + math.pi / 2
            ox, oy = math.cos(perp) * 11, math.sin(perp) * 11
            sx = xi + math.cos(ang) * NR + ox
            sy = yi + math.sin(ang) * NR + oy
            ex = xj - math.cos(ang) * (NR + 8) + ox
            ey = yj - math.sin(ang) * (NR + 8) + oy
            mx = (sx + ex) / 2 + math.cos(perp) * 14
            my = (sy + ey) / 2 + math.sin(perp) * 14
            w  = max(1.2, T[i][j] * 22)
            op = min(0.9, max(0.2, T[i][j] * 6))
            parts.append(
                f'<path d="M{sx:.1f},{sy:.1f} Q{mx:.1f},{my:.1f} {ex:.1f},{ey:.1f}" '
                f'stroke="{_MK_COLORS[i]}" stroke-width="{w:.1f}" fill="none" '
                f'stroke-opacity="{op:.2f}" marker-end="url(#mk{i})"/>'
                f'<text x="{mx:.0f}" y="{my:.0f}" fill="{_MK_COLORS[i]}" font-size="10" '
                f'font-family="monospace" font-weight="700" text-anchor="middle" {_TS}>'
                f'{T[i][j]:.3f}</text>'
            )

    # nodos encima
    for i, (xi, yi) in enumerate(pos):
        parts.append(
            f'<circle cx="{xi}" cy="{yi}" r="{NR}" fill="{_MK_COLORS[i]}" fill-opacity="0.13" '
            f'stroke="{_MK_COLORS[i]}" stroke-width="2.5" '
            f'style="filter:drop-shadow(0 0 10px rgba({_MK_GLOW[i]},.7))"/>'
            f'<text x="{xi}" y="{yi - 6}" fill="{_MK_COLORS[i]}" font-size="13" '
            f'font-family="Inter,sans-serif" font-weight="800" text-anchor="middle" {_TS}>'
            f'S{i+1}</text>'
            f'<text x="{xi}" y="{yi + 11}" fill="{_MK_COLORS[i]}" font-size="10" '
            f'font-family="Inter,sans-serif" font-weight="600" text-anchor="middle" {_TS}>'
            f'{_MK_STATES[i]}</text>'
            f'<text x="{xi}" y="{yi + NR + 16}" fill="rgba(148,163,184,.7)" font-size="9" '
            f'font-family="monospace" text-anchor="middle">π = {pi[i]:.4f}</text>'
        )

    body = "\n".join(parts)
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<style>*{{margin:0;padding:0}}body{{background:transparent}}</style>'
        f'</head><body>'
        f'<svg width="{W}" height="{H}" style="background:rgba(11,15,25,.55);'
        f'border:1px solid rgba(0,242,254,.18);border-radius:8px;display:block">'
        f'<defs>{markers}</defs>{body}</svg>'
        f'</body></html>'
    )


def _markov_line_fig(history: np.ndarray, steps: int, pi: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0D1629')
    x = np.arange(steps + 1)
    for i, (color, lbl) in enumerate(zip(_MK_COLORS, _MK_LABELS)):
        name = lbl.split(": ")[1]
        ax.plot(x, history[:, i], color=color, linewidth=2.4)
        ax.axhline(pi[i], color=color, linewidth=0.7, linestyle="--", alpha=0.35)
        # etiqueta sobre curva
        y_end = history[-1, i]
        ax.annotate(
            name, xy=(steps, y_end),
            xytext=(steps + steps * 0.01, y_end),
            color=color, fontsize=8.5, va="center",
            fontfamily="sans-serif", fontweight="bold",
        )
    # neon grid doble capa
    ax.grid(True, color="#00F2FE", linewidth=0.25, alpha=0.06)
    ax.grid(True, color="#1A2235", linewidth=0.5,  alpha=0.65)
    ax.set_xlim(0, steps * 1.14)
    ax.set_ylim(-0.02, 1.05)
    for sp in ax.spines.values():
        sp.set_color('#1A2235')
    ax.tick_params(colors='#8892AA', labelsize=8)
    ax.set_xlabel("Iteración (n)", color='#8892AA', fontsize=9)
    ax.set_ylabel("P(Estado)",     color='#8892AA', fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig


# tarjeta glass reutilizable
def _glass(title: str, color: str = "#00F2FE") -> None:
    st.markdown(
        f'<p style="color:{color};font-size:0.72rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.1em;'
        f'border-left:3px solid {color};padding-left:0.5rem;margin:0.8rem 0 0.3rem 0">'
        f'{title}</p>',
        unsafe_allow_html=True,
    )


with tabs[3]:
    st.subheader("Simulación de Convergencia: Cadenas de Markov")
    st.write(r"Cadena de Markov de 3 estados independientes. $T$ converge al vector estacionario $\pi$ según la inercia configurada.")

    col_izq, col_der = st.columns([1.2, 2])

    # ── Columna izquierda ────────────────────────
    with col_izq:
        _glass("⚙  Parámetros de Simulación")
        steps   = st.slider("Iteraciones (n)", min_value=5, max_value=50, value=25)
        inertia = st.slider("Inercia", min_value=0.1, max_value=0.99, value=0.75, step=0.05)

        T, history, pi = run_markov_simulation(tree, steps=steps, inertia=inertia)
        estado_final   = history[-1]
        error          = float(np.abs(estado_final - pi).sum())

        # convergencia aproximada
        converge_at = steps
        for _k in range(1, len(history)):
            if np.abs(history[_k] - pi).sum() < 0.01:
                converge_at = _k
                break

        _glass("∑  Matriz de Transición T")
        st.latex(
            r"T = \begin{bmatrix}"
            + f"{T[0,0]:.3f} & {T[0,1]:.3f} & {T[0,2]:.3f} \\\\ "
            + f"{T[1,0]:.3f} & {T[1,1]:.3f} & {T[1,2]:.3f} \\\\ "
            + f"{T[2,0]:.3f} & {T[2,1]:.3f} & {T[2,2]:.3f}"
            + r"\end{bmatrix}"
        )

        _glass("π  Vector Estacionario")
        st.latex(
            r"\boldsymbol{\pi} = \begin{bmatrix}"
            + f"{pi[0]:.4f} \\\\ {pi[1]:.4f} \\\\ {pi[2]:.4f}"
            + r"\end{bmatrix}^{\!T}"
        )
        st.markdown(
            '<p style="color:#94A3B8;font-size:0.72rem;margin:0">Distribución natural del corpus</p>',
            unsafe_allow_html=True,
        )

        # insight card
        _glass("💡  Insight Clave", color="#9D4EDD")
        st.markdown(
            f'<div style="background:rgba(157,78,221,0.08);border:1px solid rgba(157,78,221,0.25);'
            f'border-radius:6px;padding:0.75rem 1rem;margin-top:0.3rem">'
            f'<p style="color:#E8EAF0;font-size:0.85rem;margin:0 0 0.4rem 0">'
            f'Convergencia en <strong style="color:#9D4EDD">iteración ≈ {converge_at}</strong></p>'
            f'<p style="color:#94A3B8;font-size:0.78rem;margin:0">'
            f'Error residual: <strong style="color:#00F2FE">{error:.6f}</strong></p>'
            f'<p style="color:#94A3B8;font-size:0.72rem;margin:0.3rem 0 0 0">'
            f'Estado final: [{estado_final[0]:.3f}, {estado_final[1]:.3f}, {estado_final[2]:.3f}]</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Columna derecha ──────────────────────────
    with col_der:
        _glass("◉  Grafo de Transición de Estados")
        st.components.v1.html(_markov_diagram_html(T, pi), height=325, scrolling=False)

        _glass("〜  Trayectoria de Convergencia")
        st.pyplot(_markov_line_fig(history, steps, pi), transparent=True)
