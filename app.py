# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Analyse de Séries Temporelles",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Application d'Analyse de Séries Temporelles")
st.caption("Upload CSV/Excel • Visualisation • STL • ADF • ACF/PACF • ARIMA/SARIMA • Train/Test • Prévisions • Export CSV")
st.markdown("---")


# -----------------------------
# Utils
# -----------------------------
def load_data(file):
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            # tentative: séparateur auto + encodage commun
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, sep=";")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            st.error("Format non supporté. Utilisez CSV ou Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Convertit en numérique en gérant virgules décimales + espaces."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    s2 = s.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def adf_test(series: pd.Series):
    series = series.dropna()
    if len(series) < 10 or series.nunique() < 3:
        return None
    res = adfuller(series)
    return {
        "ADF Statistic": res[0],
        "p-value": res[1],
        "Critical Values": res[4],
        "stationnaire": res[1] < 0.05
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_infer_freq(idx: pd.DatetimeIndex):
    try:
        return pd.infer_freq(idx)
    except Exception:
        return None


def make_forecast_index(last_date, steps: int, freq: str):
    # start = last_date; periods = steps+1 then drop first to avoid repeating last point
    return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]


def plot_series_plotly(x, y, name, dash=None):
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(width=2, dash=dash if dash else "solid")
    )


# -----------------------------
# Session state
# -----------------------------
for k in ["data", "prepared_df", "model_fitted", "forecast_future", "forecast_future_index",
          "train", "test", "pred_test", "metrics", "stl_df", "validation_df", "summary_df"]:
    if k not in st.session_state:
        st.session_state[k] = None


# -----------------------------
# Sidebar - Upload
# -----------------------------
st.sidebar.header("1️⃣ Chargement des Données")
uploaded_file = st.sidebar.file_uploader(
    "Téléchargez votre fichier (CSV ou Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    st.session_state.data = load_data(uploaded_file)

if st.session_state.data is None:
    st.info("👆 Veuillez télécharger un fichier CSV ou Excel pour commencer.")
    st.markdown("""
## 📝 Instructions
1. Téléchargez un CSV/Excel contenant une **colonne Date** et une **colonne Valeur**.  
2. Sélectionnez les colonnes, vérifiez la série, lancez STL/ADF, puis ARIMA/SARIMA.  
3. Optionnel : activez Train/Test, ACF/PACF, export CSV.  
""")
    st.markdown("---")
    st.markdown("**Application d'Analyse de Séries Temporelles** | Streamlit")
    st.stop()

df_raw = st.session_state.data.copy()
st.sidebar.success("✅ Données chargées!")


# -----------------------------
# Preview
# -----------------------------
with st.expander("📊 Aperçu des données", expanded=True):
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.write(f"**Dimensions:** {df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes")


# -----------------------------
# Column selection & preparation
# -----------------------------
st.sidebar.header("2️⃣ Configuration")
date_col = st.sidebar.selectbox("Colonne de dates", options=df_raw.columns.tolist())

value_col = st.sidebar.selectbox(
    "Colonne de valeurs",
    options=[c for c in df_raw.columns.tolist() if c != date_col]
)

# Prepare data
try:
    df = df_raw[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = coerce_numeric(df[value_col])
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    df = df.set_index(date_col)

    # Remove duplicate dates by aggregating (mean) to avoid index issues
    if df.index.duplicated().any():
        df = df.groupby(df.index).mean(numeric_only=True)

    st.session_state.prepared_df = df

except Exception as e:
    st.error(f"Erreur lors de la préparation : {e}")
    st.info("Vérifiez que la colonne date est correcte et que la colonne valeur est numérique.")
    st.stop()

df = st.session_state.prepared_df

if len(df) < 10:
    st.warning("Série très courte : certaines analyses (ADF/ARIMA/STL) peuvent être instables.")

if df[value_col].nunique() < 5:
    st.warning("Série peu variable : les résultats peuvent être peu fiables.")


# -----------------------------
# Frequency (robust for online)
# -----------------------------
st.sidebar.header("3️⃣ Fréquence")
inferred = safe_infer_freq(df.index)
freq_choice = st.sidebar.selectbox(
    "Fréquence de la série",
    options=["Auto", "D", "W", "M", "MS", "H"],
    help="Auto tente de détecter; sinon choisis une fréquence."
)
freq = inferred if (freq_choice == "Auto" and inferred is not None) else ("D" if freq_choice == "Auto" else freq_choice)

if inferred is None and freq_choice == "Auto":
    st.sidebar.warning("Auto n'a pas trouvé la fréquence → fallback sur D (jour).")


# -----------------------------
# Visualization
# -----------------------------
st.header("📈 Visualisation de la Série Temporelle")

fig = go.Figure()
fig.add_trace(plot_series_plotly(df.index, df[value_col], "Série temporelle"))
fig.update_layout(
    title=f"Série Temporelle : {value_col}",
    xaxis_title="Date",
    yaxis_title=value_col,
    hovermode="x unified",
    height=420
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# STL
# -----------------------------
st.header("🔍 Décomposition STL")

c1, c2 = st.columns([1, 3])
with c1:
    period = st.number_input(
        "Période de saisonnalité (STL)",
        min_value=2,
        max_value=365,
        value=12,
        help="Nombre d'observations dans un cycle saisonnier."
    )

if st.button("🔄 Calculer la décomposition STL"):
    with st.spinner("Calcul STL..."):
        try:
            stl = STL(df[value_col], period=int(period))
            res = stl.fit()

            fig_stl, axes = plt.subplots(4, 1, figsize=(12, 10))
            axes[0].plot(df.index, df[value_col])
            axes[0].set_title("Décomposition STL")
            axes[0].set_ylabel("Original")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(res.trend.index, res.trend)
            axes[1].set_ylabel("Tendance")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(res.seasonal.index, res.seasonal)
            axes[2].set_ylabel("Saisonnalité")
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(res.resid.index, res.resid)
            axes[3].set_ylabel("Résidus")
            axes[3].set_xlabel("Date")
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_stl)

            stl_df = pd.DataFrame({
                "Date": df.index,
                "observed": df[value_col].values,
                "trend": res.trend.values,
                "seasonal": res.seasonal.values,
                "resid": res.resid.values
            })
            st.session_state.stl_df = stl_df

            st.download_button(
                "⬇️ Télécharger la décomposition STL (CSV)",
                data=to_csv_bytes(stl_df),
                file_name="stl_decomposition.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erreur STL : {e}")


# -----------------------------
# ADF
# -----------------------------
st.header("📊 Test de Stationnarité (ADF)")

if st.button("🧪 Effectuer le test ADF"):
    with st.spinner("Calcul ADF..."):
        adf_res = adf_test(df[value_col])
        if adf_res is None:
            st.warning("ADF impossible (série trop courte ou trop peu variable).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Statistique ADF", f"{adf_res['ADF Statistic']:.4f}")
            c2.metric("p-value", f"{adf_res['p-value']:.4f}")
            if adf_res["stationnaire"]:
                c3.success("✅ Stationnaire")
            else:
                c3.warning("⚠️ Non stationnaire")

            st.write("**Valeurs critiques**")
            for k, v in adf_res["Critical Values"].items():
                st.write(f"- {k}: {v:.4f}")

            if adf_res["stationnaire"]:
                st.info("Interprétation : p-value < 0.05 → série stationnaire.")
            else:
                st.info("Interprétation : p-value ≥ 0.05 → série non stationnaire (différenciation souvent utile).")


# -----------------------------
# ACF / PACF
# -----------------------------
st.header("📌 ACF / PACF (pour aider au choix de p et q)")

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    use_diff = st.checkbox("Afficher sur série différenciée (d=1)", value=False)
with colB:
    nlags = st.slider("Nombre de lags", 10, 60, 30)

if st.button("📉 Afficher ACF/PACF"):
    series_for_corr = df[value_col].dropna()
    if use_diff:
        series_for_corr = series_for_corr.diff().dropna()

    if len(series_for_corr) < 20:
        st.warning("Série trop courte pour ACF/PACF.")
    else:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        plot_acf(series_for_corr, lags=int(nlags), ax=ax1)
        ax1.set_title("ACF")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        plot_pacf(series_for_corr, lags=int(nlags), ax=ax2, method="ywm")
        ax2.set_title("PACF")
        st.pyplot(fig2)


# -----------------------------
# Modeling + Train/Test + Future Forecast
# -----------------------------
st.header("🤖 Modélisation, Validation et Prédiction")

st.subheader("🧩 Découpage Train/Test (optionnel mais recommandé)")
enable_split = st.checkbox("Activer la validation Train/Test", value=True)

train = df[value_col]
test = None

if enable_split:
    split_mode = st.radio(
        "Méthode de split",
        ["Pourcentage", "Nombre de points (test)"],
        horizontal=True
    )

    if split_mode == "Pourcentage":
        train_ratio = st.slider("Part du train (%)", 50, 95, 80)
        split_idx = int(len(df) * train_ratio / 100)
    else:
        test_size = st.number_input(
            "Taille du test (nombre de points)",
            min_value=5,
            max_value=max(5, min(365, len(df) - 5)),
            value=min(30, max(5, len(df) - 5))
        )
        split_idx = len(df) - int(test_size)

    train = df[value_col].iloc[:split_idx]
    test = df[value_col].iloc[split_idx:]
    st.caption(f"Train: {len(train)} points | Test: {len(test)} points")


st.subheader("⚙️ Choix du modèle et paramètres")
col1, col2 = st.columns(2)
with col1:
    model_type = st.selectbox("Modèle", ["ARIMA", "SARIMA"])
with col2:
    horizon_future = st.number_input(
        "Horizon de prédiction future",
        min_value=1, max_value=365, value=30,
        help="Nombre de pas dans le futur pour la prédiction finale."
    )

if model_type == "ARIMA":
    c1, c2, c3 = st.columns(3)
    with c1:
        p = st.number_input("p (AR)", 0, 10, 1)
    with c2:
        d = st.number_input("d (diff)", 0, 2, 1)
    with c3:
        q = st.number_input("q (MA)", 0, 10, 1)

    # placeholders SARIMA
    P = D = Q = s = None

else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p = st.number_input("p (AR)", 0, 10, 1)
    with c2:
        d = st.number_input("d (diff)", 0, 2, 1)
    with c3:
        q = st.number_input("q (MA)", 0, 10, 1)
    with c4:
        s = st.number_input("s (période saisonnière)", 2, 365, 12)

    c5, c6, c7 = st.columns(3)
    with c5:
        P = st.number_input("P (AR saisonnier)", 0, 10, 1)
    with c6:
        D = st.number_input("D (diff saisonnière)", 0, 2, 1)
    with c7:
        Q = st.number_input("Q (MA saisonnier)", 0, 10, 1)


def fit_model(series_train: pd.Series):
    if model_type == "ARIMA":
        model = ARIMA(series_train, order=(int(p), int(d), int(q)))
        fitted = model.fit()
    else:
        model = SARIMAX(
            series_train,
            order=(int(p), int(d), int(q)),
            seasonal_order=(int(P), int(D), int(Q), int(s))
        )
        fitted = model.fit(disp=False)
    return fitted


if st.button("🚀 Entraîner / Valider / Prédire", type="primary"):
    with st.spinner("Entraînement en cours..."):
        try:
            # 1) Fit on train
            fitted_train = fit_model(train)
            st.session_state.model_fitted = fitted_train

            # 2) Predict on test (validation)
            metrics = None
            validation_df = None
            pred_test = None

            if enable_split and test is not None and len(test) > 0:
                pred_test = fitted_train.forecast(steps=len(test))
                st.session_state.pred_test = pred_test

                mae = mean_absolute_error(test, pred_test)
                rmse = mean_squared_error(test, pred_test, squared=False)
                # MAPE safe
                denom = test.replace(0, np.nan)
                mape = (np.abs((test - pred_test) / denom)).mean() * 100

                metrics = {"MAE": float(mae), "RMSE": float(rmse), "MAPE_percent": float(mape)}
                st.session_state.metrics = metrics

                validation_df = pd.DataFrame({
                    "Date": test.index,
                    "y_true": test.values,
                    "y_pred": pred_test.values if hasattr(pred_test, "values") else np.array(pred_test)
                })
                st.session_state.validation_df = validation_df

            # 3) Fit on full series for final future forecast
            fitted_full = fit_model(df[value_col])
            # use get_forecast for possible conf_int
            pred_res = fitted_full.get_forecast(steps=int(horizon_future))
            forecast_future = pred_res.predicted_mean
            conf_int = pred_res.conf_int()

            future_index = make_forecast_index(df.index[-1], int(horizon_future), freq=freq)

            st.session_state.forecast_future = forecast_future
            st.session_state.forecast_future_index = future_index

            # 4) summary df export
            summary_df = pd.DataFrame([{
                "model_type": model_type,
                "p": int(p), "d": int(d), "q": int(q),
                "P": int(P) if model_type == "SARIMA" else np.nan,
                "D": int(D) if model_type == "SARIMA" else np.nan,
                "Q": int(Q) if model_type == "SARIMA" else np.nan,
                "s": int(s) if model_type == "SARIMA" else np.nan,
                "freq_used": freq,
                "train_test_enabled": bool(enable_split),
                "MAE": metrics["MAE"] if metrics else np.nan,
                "RMSE": metrics["RMSE"] if metrics else np.nan,
                "MAPE_percent": metrics["MAPE_percent"] if metrics else np.nan
            }])
            st.session_state.summary_df = summary_df

            st.success("✅ Entraînement terminé (validation + prédiction future générées).")

            with st.expander("📋 Résumé du modèle (fit sur train)"):
                st.text(fitted_train.summary())

            # store conf_int in session for plotting/export
            st.session_state["conf_int_future"] = remember_conf = conf_int.copy()

        except Exception as e:
            st.error(f"❌ Erreur entraînement/prédiction : {e}")


# -----------------------------
# Results display
# -----------------------------
if st.session_state.model_fitted is not None and st.session_state.forecast_future is not None:
    st.markdown("---")
    st.header("📊 Résultats")

    # Metrics
    if enable_split and st.session_state.metrics is not None:
        st.subheader("✅ Validation Train/Test")
        m = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.3f}")
        c2.metric("RMSE", f"{m['RMSE']:.3f}")
        c3.metric("MAPE (%)", f"{m['MAPE_percent']:.2f}")

        # Plot validation
        train_series = train
        test_series = test
        pred_test = st.session_state.pred_test

        fig_val = go.Figure()
        fig_val.add_trace(plot_series_plotly(train_series.index, train_series, "Train"))
        fig_val.add_trace(plot_series_plotly(test_series.index, test_series, "Test (vrai)"))
        fig_val.add_trace(plot_series_plotly(test_series.index, pred_test, "Prévision sur test", dash="dash"))
        fig_val.update_layout(title="Validation temporelle (Train/Test)", hovermode="x unified", height=450)
        st.plotly_chart(fig_val, use_container_width=True)

        # Export validation CSV
        validation_df = st.session_state.validation_df
        if validation_df is not None:
            st.download_button(
                "⬇️ Télécharger validation (test vs prédiction) (CSV)",
                data=to_csv_bytes(validation_df),
                file_name="validation_test_predictions.csv",
                mime="text/csv"
            )

    # Future forecast plot
    st.subheader("📈 Prédiction Future")
    future_index = st.session_state.forecast_future_index
    forecast_future = st.session_state.forecast_future
    conf_int = st.session_state.get("conf_int_future", None)

    fig_f = go.Figure()
    fig_f.add_trace(plot_series_plotly(df.index, df[value_col], "Historique"))
    fig_f.add_trace(plot_series_plotly(future_index, forecast_future, "Prévision future", dash="dash"))

    # Optional conf int
    if conf_int is not None and isinstance(conf_int, pd.DataFrame) and conf_int.shape[1] >= 2:
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        fig_f.add_trace(go.Scatter(x=future_index, y=lower, mode="lines", name="Borne basse", line=dict(dash="dot"), opacity=0.5))
        fig_f.add_trace(go.Scatter(x=future_index, y=upper, mode="lines", name="Borne haute", line=dict(dash="dot"), opacity=0.5))

    fig_f.update_layout(title="Historique + Prévision Future", hovermode="x unified", height=520)
    st.plotly_chart(fig_f, use_container_width=True)

    # Zoom recent
    st.subheader("🔎 Zoom sur la période récente (historique + future)")
    recent_period = st.slider(
        "Nombre de points historiques à afficher",
        min_value=10,
        max_value=len(df),
        value=min(60, len(df))
    )

    recent_data = df[value_col].tail(int(recent_period))
    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(x=recent_data.index, y=recent_data, mode="lines+markers", name="Observations"))
    fig_zoom.add_trace(go.Scatter(x=future_index, y=forecast_future, mode="lines+markers", name="Prévision future", line=dict(dash="dash")))
    fig_zoom.update_layout(title="Zoom : Observations récentes vs Prévision", hovermode="x unified", height=420)
    st.plotly_chart(fig_zoom, use_container_width=True)

    # Forecast table + export
    with st.expander("📋 Valeurs prédites (future)"):
        forecast_df = pd.DataFrame({
            "Date": future_index,
            "Prediction": forecast_future.values if hasattr(forecast_future, "values") else np.array(forecast_future)
        })
        st.dataframe(forecast_df, use_container_width=True)

        st.download_button(
            "⬇️ Télécharger les prédictions futures (CSV)",
            data=to_csv_bytes(forecast_df),
            file_name="predictions_forecast_future.csv",
            mime="text/csv"
        )

    # Export summary
    with st.expander("📤 Export global (résumé params + métriques)"):
        summary_df = st.session_state.summary_df
        if summary_df is not None:
            st.dataframe(summary_df, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger le résumé (CSV)",
                data=to_csv_bytes(summary_df),
                file_name="model_summary_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("Le résumé sera disponible après un entraînement.")


st.markdown("---")
st.markdown("**Application d'Analyse de Séries Temporelles** | Projet Streamlit 2024-2025")
