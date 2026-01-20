import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# CONFIG PAGE
# -----------------------------
st.set_page_config(
    page_title="📈 Analyse de Séries Temporelles",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 24px; color: #1E88E5 !important; font-weight: bold; }
.stAlert { border-radius: 10px; }
h1, h2, h3 { color: #0D47A1; }
.stButton>button { width: 100%; border-radius: 6px; background-color: #1E88E5; color: white; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "ts" not in st.session_state:
    st.session_state.ts = None
if "date_col" not in st.session_state:
    st.session_state.date_col = None
if "val_col" not in st.session_state:
    st.session_state.val_col = None
if "freq" not in st.session_state:
    st.session_state.freq = "D"
if "period" not in st.session_state:
    st.session_state.period = 12

if "ts_final" not in st.session_state:
    st.session_state.ts_final = None
if "diff_d" not in st.session_state:
    st.session_state.diff_d = 0
if "diff_D" not in st.session_state:
    st.session_state.diff_D = 0

if "model_type" not in st.session_state:
    st.session_state.model_type = "ARIMA"
if "model_result" not in st.session_state:
    st.session_state.model_result = None
if "train" not in st.session_state:
    st.session_state.train = None
if "test" not in st.session_state:
    st.session_state.test = None
if "test_pred" not in st.session_state:
    st.session_state.test_pred = None
if "test_conf_int" not in st.session_state:
    st.session_state.test_conf_int = None


# -----------------------------
# UTILS
# -----------------------------
def load_data(file, sep=","):
    if file.name.endswith(".csv"):
        return pd.read_csv(file, sep=sep)
    if file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    raise ValueError("Format non supporté (CSV / Excel)")

def to_numeric_series(s: pd.Series) -> pd.Series:
    # gère "1,23" -> "1.23"
    s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def check_stationarity(series: pd.Series):
    series = series.dropna()
    adf_res = adfuller(series)
    kpss_res = kpss(series, regression="c", nlags="auto")
    return {
        "adf_stat": adf_res[0],
        "adf_p": adf_res[1],
        "kpss_stat": kpss_res[0],
        "kpss_p": kpss_res[1]
    }

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

def plot_ts_plotly(ts: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Série"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Valeur",
        hovermode="x unified",
        height=420
    )
    fig.update_yaxes(autorange=True, fixedrange=False)
    return fig

def safe_lags(ts_len: int, desired: int) -> int:
    # lags ne doit pas dépasser len(ts)-1
    return max(1, min(desired, ts_len - 1))


# -----------------------------
# SIDEBAR WORKFLOW
# -----------------------------
st.sidebar.title("📊 Workflow d’Analyse")

steps = [
    "1. Préparation des Données",
    "2. Visualisation & Décomposition",
    "3. Stationnarité & Différenciation",
    "4. Modélisation & Évaluation",
    "5. Diagnostic des Résidus",
    "6. Prévisions Futures"
]
progress_map = {s: (i + 1) / len(steps) for i, s in enumerate(steps)}
step = st.sidebar.radio("Navigation", steps)
st.sidebar.markdown(f"### 📍 Progression : **{int(progress_map[step]*100)}%**")
st.sidebar.progress(progress_map[step])

st.title("📈 Application d’Analyse de Séries Temporelles")
st.markdown("---")


# -----------------------------
# 1) PREPARATION
# -----------------------------
if step == "1. Préparation des Données":
    st.header("1️⃣ Préparation des Données")
    st.info("Charger, nettoyer et structurer votre série temporelle.")

    uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                sep = st.radio("Séparateur CSV", [",", ";", "\t"], horizontal=True)
                df = load_data(uploaded_file, sep=sep)
            else:
                df = load_data(uploaded_file)

            st.session_state.raw_df = df

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.subheader("Aperçu")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

            with c2:
                st.subheader("Configuration")
                date_col = st.selectbox("Colonne Date", df.columns, index=0)
                val_col = st.selectbox("Colonne Valeur", df.columns, index=min(1, len(df.columns)-1))

                freq_label = st.selectbox(
                    "Fréquence (resampling)",
                    ["D (Journalier)", "W (Hebdomadaire)", "M (Mensuel)", "Q (Trimestriel)", "Y (Annuel)"]
                )
                freq = freq_label.split(" ")[0]

                agg = st.selectbox("Agrégation (si plusieurs points dans la période)", ["mean", "sum", "median"])
                fill_method = st.selectbox("Valeurs manquantes", ["interpolate", "ffill", "bfill", "drop"])

                st.session_state.date_col = date_col
                st.session_state.val_col = val_col
                st.session_state.freq = freq

            if st.button("✅ Valider la Série", type="primary"):
                try:
                    df = df.copy()
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df.dropna(subset=[date_col])

                    df[val_col] = to_numeric_series(df[val_col])
                    df = df.dropna(subset=[val_col])

                    df = df.sort_values(date_col).set_index(date_col)

                    # resample + aggregate
                    if agg == "mean":
                        ts = df[val_col].resample(freq).mean()
                    elif agg == "sum":
                        ts = df[val_col].resample(freq).sum()
                    else:
                        ts = df[val_col].resample(freq).median()

                    # fill missing
                    if fill_method == "interpolate":
                        ts = ts.interpolate()
                    elif fill_method == "ffill":
                        ts = ts.ffill()
                    elif fill_method == "bfill":
                        ts = ts.bfill()
                    else:
                        ts = ts.dropna()

                    st.session_state.ts = ts.dropna()
                    st.session_state.ts_final = st.session_state.ts.copy()

                    # reset model state on new data
                    st.session_state.model_result = None
                    st.session_state.train = None
                    st.session_state.test = None
                    st.session_state.test_pred = None
                    st.session_state.test_conf_int = None

                    st.success("✅ Série temporelle prête !")
                    st.plotly_chart(plot_ts_plotly(st.session_state.ts, "Série temporelle (après préparation)"),
                                    use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur : {e}")

    else:
        st.info("👈 Importez un fichier pour commencer.")


# -----------------------------
# 2) EXPLORATION & DECOMPOSITION
# -----------------------------
elif step == "2. Visualisation & Décomposition":
    st.header("2️⃣ Visualisation & Décomposition")

    if st.session_state.ts is None:
        st.warning("⚠️ Charge d’abord tes données à l’étape 1.")
    else:
        ts = st.session_state.ts

        st.subheader("Série temporelle")
        st.plotly_chart(plot_ts_plotly(ts, "Série temporelle originale"), use_container_width=True)

        st.divider()
        st.subheader("Décomposition")

        colA, colB = st.columns([1, 2.5])
        with colA:
            period_options = {
                "7 (hebdo)": 7,
                "12 (mensuel)": 12,
                "4 (trimestriel)": 4,
                "52 (hebdo)": 52,
                "365 (annuel)": 365
            }
            selected_period = st.selectbox("Période saisonnière", list(period_options.keys()), index=1)
            period = period_options[selected_period]
            st.session_state.period = period

            method = st.selectbox("Méthode", ["STL (robuste)", "Additive", "Multiplicative"])
        with colB:
            try:
                if method == "STL (robuste)":
                    decomp = STL(ts, period=period, robust=True).fit()
                    trend = decomp.trend
                    seasonal = decomp.seasonal
                    resid = decomp.resid
                else:
                    model = method.lower()
                    decomp2 = seasonal_decompose(ts, model=model, period=period)
                    trend = decomp2.trend
                    seasonal = decomp2.seasonal
                    resid = decomp2.resid

                fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
                ts.plot(ax=axs[0], title="Original")
                trend.plot(ax=axs[1], title="Tendance")
                seasonal.plot(ax=axs[2], title="Saisonnalité")
                resid.plot(ax=axs[3], title="Résidus", style=".")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur décomposition : {e}")


# -----------------------------
# 3) STATIONARITE & DIFFERENCIATION
# -----------------------------
elif step == "3. Stationnarité & Différenciation":
    st.header("3️⃣ Stationnarité & Différenciation")

    if st.session_state.ts is None:
        st.warning("⚠️ Charge d’abord tes données à l’étape 1.")
    else:
        ts = st.session_state.ts

        st.subheader("Tests stationnarité (série originale)")
        res0 = check_stationarity(ts)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("ADF p-value", f"{res0['adf_p']:.4f}")
            st.write("✅ Stationnaire (ADF)" if res0["adf_p"] < 0.05 else "❌ Non-stationnaire (ADF)")
        with c2:
            st.metric("KPSS p-value", f"{res0['kpss_p']:.4f}")
            st.write("✅ Stationnaire (KPSS)" if res0["kpss_p"] > 0.05 else "❌ Non-stationnaire (KPSS)")

        st.divider()

        st.subheader("Différenciation")
        modele_choisi = st.radio("Préparation pour", ["ARIMA", "SARIMA"], horizontal=True)
        d = st.slider("Ordre différenciation (d)", 0, 2, st.session_state.diff_d)
        D = 0
        if modele_choisi == "SARIMA":
            D = st.slider("Différenciation saisonnière (D)", 0, 2, st.session_state.diff_D)

        ts_final = ts.copy()
        if d > 0:
            for _ in range(d):
                ts_final = ts_final.diff().dropna()
        if D > 0:
            ts_final = ts_final.diff(st.session_state.period).dropna()

        st.session_state.ts_final = ts_final
        st.session_state.diff_d = d
        st.session_state.diff_D = D

        if d > 0 or D > 0:
            res1 = check_stationarity(ts_final)
            st.write(f"**Après différenciation (d={d}, D={D})**")
            r1, r2 = st.columns(2)
            r1.write(f"ADF p-value : **{res1['adf_p']:.4f}**")
            r2.write(f"KPSS p-value : **{res1['kpss_p']:.4f}**")

            if res1["adf_p"] < 0.05 and res1["kpss_p"] > 0.05:
                st.success("✅ Série beaucoup plus stationnaire (bon signe).")
            else:
                st.warning("⚠️ Pas encore idéale. Tu peux ajuster d/D ou envisager une transformation (log, etc.).")

        st.divider()

        st.subheader("ACF & PACF (sur la série finale)")
        lags_desired = st.session_state.period * 5
        lags = safe_lags(len(st.session_state.ts_final), lags_desired)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(st.session_state.ts_final, lags=lags, ax=ax1)
        plot_pacf(st.session_state.ts_final, lags=lags, ax=ax2, method="ywm")
        plt.tight_layout()
        st.pyplot(fig)


# -----------------------------
# 4) MODELISATION & EVALUATION
# -----------------------------
elif step == "4. Modélisation & Évaluation":
    st.header("4️⃣ Modélisation & Évaluation")

    if st.session_state.ts is None:
        st.warning("⚠️ Charge d’abord tes données à l’étape 1.")
    else:
        ts = st.session_state.ts

        st.subheader("Split train / test")
        test_size = st.slider("Taille du test (%)", 5, 30, 20) / 100
        split_idx = int(len(ts) * (1 - test_size))
        train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

        st.session_state.train = train
        st.session_state.test = test

        cA, cB = st.columns([1, 1])
        with cA:
            st.caption("Train")
            st.plotly_chart(plot_ts_plotly(train, "Train"), use_container_width=True)
        with cB:
            st.caption("Test")
            st.plotly_chart(plot_ts_plotly(test, "Test"), use_container_width=True)

        st.divider()
        st.subheader("Choix du modèle")

        model_type = st.selectbox("Modèle", ["ARIMA", "SARIMA"])
        st.session_state.model_type = model_type

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Paramètres non-saisonniers (p, d, q)**")
            p = st.number_input("p", 0, 10, 1)
            d_val = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 10, 1)

        if model_type == "SARIMA":
            with col2:
                st.write("**Paramètres saisonniers (P, D, Q, s)**")
                P = st.number_input("P", 0, 10, 0)
                D_val = st.number_input("D", 0, 2, 0)
                Q = st.number_input("Q", 0, 10, 0)
                s = st.number_input("s (période)", value=int(st.session_state.period), min_value=2, max_value=365)

        if st.button(f"🚀 Entraîner {model_type}", type="primary"):
            with st.spinner("Entraînement du modèle..."):
                try:
                    if model_type == "ARIMA":
                        model = ARIMA(train, order=(p, d_val, q))
                        result = model.fit()
                    else:
                        model = SARIMAX(train, order=(p, d_val, q), seasonal_order=(P, D_val, Q, s))
                        result = model.fit(disp=False)

                    # IMPORTANT: on sauvegarde le modèle fitted pour étapes 5/6
                    st.session_state.model_result = result

                    # forecast sur la taille du test
                    fc = result.get_forecast(steps=len(test))
                    preds = fc.predicted_mean
                    conf_int = fc.conf_int()

                    st.session_state.test_pred = preds
                    st.session_state.test_conf_int = conf_int

                    mae, rmse, mape = calculate_metrics(test, preds)

                    st.subheader("Résultats")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:.3f}")
                    m2.metric("RMSE", f"{rmse:.3f}")
                    m3.metric("MAPE", f"{mape:.2%}")

                    with st.expander("📋 Résumé du modèle"):
                        st.text(result.summary())

                    # plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
                    fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
                    fig.add_trace(go.Scatter(x=test.index, y=preds, name="Prédictions", line=dict(dash="dash")))

                    # IC
                    fig.add_trace(go.Scatter(
                        x=test.index, y=conf_int.iloc[:, 0],
                        line_color="rgba(0,0,0,0)", showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=test.index, y=conf_int.iloc[:, 1],
                        fill="tonexty", fillcolor="rgba(30,136,229,0.15)",
                        line_color="rgba(0,0,0,0)", name="IC"
                    ))
                    fig.update_layout(
                        title="Train/Test + Prédictions (avec IC)",
                        hovermode="x unified",
                        height=520
                    )
                    fig.update_yaxes(autorange=True, fixedrange=False)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Erreur entraînement : {e}")


# -----------------------------
# 5) DIAGNOSTIC RESIDUS
# -----------------------------
elif step == "5. Diagnostic des Résidus":
    st.header("5️⃣ Diagnostic des Résidus")

    if st.session_state.model_result is None:
        st.warning("⚠️ Ajuste un modèle à l’étape 4 avant.")
    else:
        result = st.session_state.model_result
        resid = result.resid.dropna()

        st.subheader("Tests")
        c1, c2 = st.columns(2)

        with c1:
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
            st.metric("Ljung-Box p-value", f"{lb_p:.4f}")
            st.write("✅ Bruit blanc" if lb_p > 0.05 else "❌ Autocorrélation détectée")

        with c2:
            adf_p_resid = float(adfuller(resid)[1])
            st.metric("ADF p-value (résidus)", f"{adf_p_resid:.4f}")
            st.write("✅ Résidus stationnaires" if adf_p_resid < 0.05 else "❌ Résidus non stationnaires")

        st.divider()
        st.subheader("Graphiques")

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # 1) Residus
        axs[0, 0].plot(resid.values)
        axs[0, 0].axhline(0, color="black", linestyle="--", alpha=0.6)
        axs[0, 0].set_title("Résidus")
        axs[0, 0].grid(True, alpha=0.2)

        # 2) Hist + normale
        axs[0, 1].hist(resid.values, bins=30, density=True, alpha=0.7)
        x_axis = np.linspace(resid.min(), resid.max(), 200)
        axs[0, 1].plot(x_axis, norm.pdf(x_axis, np.mean(resid), np.std(resid)))
        axs[0, 1].set_title("Distribution (hist + normale)")
        axs[0, 1].grid(True, alpha=0.2)

        # 3) QQ plot
        qqplot(resid, line="s", ax=axs[1, 0])
        axs[1, 0].set_title("Q-Q plot")

        # 4) ACF residus
        lags = safe_lags(len(resid), 20)
        plot_acf(resid, lags=lags, ax=axs[1, 1])
        axs[1, 1].set_title("ACF résidus")

        plt.tight_layout()
        st.pyplot(fig)


# -----------------------------
# 6) FORECAST FUTUR
# -----------------------------
elif step == "6. Prévisions Futures":
    st.header("6️⃣ Prévisions Futures")

    if st.session_state.model_result is None or st.session_state.ts is None:
        st.warning("⚠️ Ajuste un modèle à l’étape 4 avant.")
    else:
        result = st.session_state.model_result
        ts = st.session_state.ts

        horizon = st.number_input("Horizon de prévision", min_value=1, max_value=365, value=24)

        if st.button("🔮 Générer les prévisions", type="primary"):
            with st.spinner("Prévision en cours..."):
                try:
                    fc = result.get_forecast(steps=horizon)
                    y_pred = fc.predicted_mean
                    conf_int = fc.conf_int()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historique"))

                    fig.add_trace(go.Scatter(
                        x=y_pred.index, y=y_pred.values,
                        name="Prévision", line=dict(dash="dash")
                    ))

                    fig.add_trace(go.Scatter(
                        x=y_pred.index, y=conf_int.iloc[:, 0],
                        line_color="rgba(0,0,0,0)", showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=y_pred.index, y=conf_int.iloc[:, 1],
                        fill="tonexty", fillcolor="rgba(30,136,229,0.15)",
                        line_color="rgba(0,0,0,0)", name="IC"
                    ))

                    fig.update_layout(
                        title="Prévisions futures (avec intervalle de confiance)",
                        hovermode="x unified",
                        height=540
                    )
                    fig.update_yaxes(autorange=True, fixedrange=False)

                    st.plotly_chart(fig, use_container_width=True)

                    out = pd.DataFrame({
                        "date": y_pred.index,
                        "forecast": y_pred.values,
                        "ci_low": conf_int.iloc[:, 0].values,
                        "ci_high": conf_int.iloc[:, 1].values
                    })

                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "📥 Télécharger CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="forecast.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Erreur prévision : {e}")
