import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Séries Temporelles",
    page_icon="📈",
    layout="wide"
)

# Titre de l'application
st.title("📈 Application d'Analyse de Séries Temporelles")
st.markdown("---")

# Fonction pour charger les données
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Format de fichier non supporté. Utilisez CSV ou Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour le test ADF
def adf_test(series):
    result = adfuller(series.dropna())
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'stationnaire': result[1] < 0.05
    }

# Fonction pour la décomposition STL
def perform_stl(series, period=12):
    stl = STL(series, period=period)
    result = stl.fit()
    return result

# Sidebar pour le chargement des données
st.sidebar.header("1️⃣ Chargement des Données")
uploaded_file = st.sidebar.file_uploader(
    "Téléchargez votre fichier (CSV ou Excel)",
    type=['csv', 'xlsx', 'xls']
)

# Initialisation de session_state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

# Chargement des données
if uploaded_file is not None:
    st.session_state.data = load_data(uploaded_file)

if st.session_state.data is not None:
    df = st.session_state.data

    st.sidebar.success("✅ Données chargées avec succès!")

    # Affichage des premières lignes
    with st.expander("📊 Aperçu des données", expanded=True):
        st.dataframe(df.head(10))
        st.write(f"**Dimensions:** {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # Sélection des colonnes
    st.sidebar.header("2️⃣ Configuration")

    date_col = st.sidebar.selectbox(
        "Colonne de dates",
        options=df.columns.tolist()
    )

    value_col = st.sidebar.selectbox(
        "Colonne de valeurs",
        options=[col for col in df.columns.tolist() if col != date_col]
    )

    # Préparation des données
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df[[date_col, value_col]].dropna()
        df.set_index(date_col, inplace=True)

        # Visualisation de la série temporelle originale
        st.header("📈 Visualisation de la Série Temporelle")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[value_col],
            mode='lines',
            name='Série Temporelle',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title=f"Série Temporelle : {value_col}",
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Décomposition STL
        st.header("🔍 Décomposition STL")

        col1, col2 = st.columns([1, 3])
        with col1:
            period = st.number_input(
                "Période de saisonnalité",
                min_value=2,
                max_value=365,
                value=12,
                help="Nombre d'observations dans un cycle saisonnier"
            )

        if st.button("🔄 Calculer la décomposition STL"):
            with st.spinner("Calcul en cours..."):
                stl_result = perform_stl(df[value_col], period=period)

                fig, axes = plt.subplots(4, 1, figsize=(12, 10))

                # Série originale
                axes[0].plot(df.index, df[value_col], color='#1f77b4')
                axes[0].set_ylabel('Original')
                axes[0].set_title('Décomposition STL')
                axes[0].grid(True, alpha=0.3)

                # Tendance
                axes[1].plot(stl_result.trend.index, stl_result.trend, color='#ff7f0e')
                axes[1].set_ylabel('Tendance')
                axes[1].grid(True, alpha=0.3)

                # Saisonnalité
                axes[2].plot(stl_result.seasonal.index, stl_result.seasonal, color='#2ca02c')
                axes[2].set_ylabel('Saisonnalité')
                axes[2].grid(True, alpha=0.3)

                # Résidus
                axes[3].plot(stl_result.resid.index, stl_result.resid, color='#d62728')
                axes[3].set_ylabel('Résidus')
                axes[3].set_xlabel('Date')
                axes[3].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

        # Test de stationnarité
        st.header("📊 Test de Stationnarité (ADF)")

        if st.button("🧪 Effectuer le test ADF"):
            with st.spinner("Calcul du test ADF..."):
                adf_result = adf_test(df[value_col])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Statistique ADF", f"{adf_result['ADF Statistic']:.4f}")

                with col2:
                    st.metric("P-value", f"{adf_result['p-value']:.4f}")

                with col3:
                    if adf_result['stationnaire']:
                        st.success("✅ Série STATIONNAIRE")
                    else:
                        st.warning("⚠️ Série NON STATIONNAIRE")

                st.write("**Valeurs Critiques:**")
                for key, value in adf_result['Critical Values'].items():
                    st.write(f"- {key}: {value:.4f}")

                if adf_result['p-value'] < 0.05:
                    st.info("💡 **Interprétation:** La p-value est inférieure à 0.05, on rejette l'hypothèse nulle. La série est stationnaire.")
                else:
                    st.info("💡 **Interprétation:** La p-value est supérieure à 0.05, on ne peut pas rejeter l'hypothèse nulle. La série n'est pas stationnaire. Considérez une différenciation.")

        # Modélisation
        st.header("🤖 Modélisation et Prédiction")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "Sélectionner le modèle",
                options=["ARIMA", "SARIMA"]
            )

        with col2:
            forecast_steps = st.number_input(
                "Horizon de prédiction",
                min_value=1,
                max_value=365,
                value=30,
                help="Nombre de périodes à prédire"
            )

        # Paramètres du modèle
        st.subheader("⚙️ Configuration du Modèle")

        if model_type == "ARIMA":
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (ordre AR)", min_value=0, max_value=10, value=1)
            with col2:
                d = st.number_input("d (ordre de différenciation)", min_value=0, max_value=2, value=1)
            with col3:
                q = st.number_input("q (ordre MA)", min_value=0, max_value=10, value=1)
        else:  # SARIMA
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                p = st.number_input("p (ordre AR)", min_value=0, max_value=10, value=1)
            with col2:
                d = st.number_input("d (différenciation)", min_value=0, max_value=2, value=1)
            with col3:
                q = st.number_input("q (ordre MA)", min_value=0, max_value=10, value=1)
            with col4:
                s = st.number_input("s (période saisonnière)", min_value=2, max_value=365, value=12)

            col5, col6, col7 = st.columns(3)
            with col5:
                P = st.number_input("P (AR saisonnier)", min_value=0, max_value=10, value=1)
            with col6:
                D = st.number_input("D (diff. saisonnière)", min_value=0, max_value=2, value=1)
            with col7:
                Q = st.number_input("Q (MA saisonnier)", min_value=0, max_value=10, value=1)

        # Entraînement du modèle
        if st.button("🚀 Entraîner le Modèle et Prédire", type="primary"):
            with st.spinner("Entraînement du modèle en cours..."):
                try:
                    if model_type == "ARIMA":
                        model = ARIMA(df[value_col], order=(p, d, q))
                        st.session_state.model_fitted = model.fit()
                        st.success(f"✅ Modèle ARIMA({p},{d},{q}) entraîné avec succès!")
                    else:  # SARIMA
                        model = SARIMAX(
                            df[value_col],
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s)
                        )
                        st.session_state.model_fitted = model.fit(disp=False)
                        st.success(f"✅ Modèle SARIMA({p},{d},{q})×({P},{D},{Q},{s}) entraîné avec succès!")

                    # Prédictions
                    forecast = st.session_state.model_fitted.forecast(steps=forecast_steps)
                    st.session_state.forecast = forecast

                    # Résumé du modèle
                    with st.expander("📋 Résumé du Modèle"):
                        st.text(st.session_state.model_fitted.summary())

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement : {e}")

        # Affichage des résultats
        if st.session_state.model_fitted is not None and st.session_state.forecast is not None:
            st.header("📊 Résultats de la Prédiction")

            # Graphique des prédictions
            fig = go.Figure()

            # Série historique
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[value_col],
                mode='lines',
                name='Données Historiques',
                line=dict(color='#1f77b4', width=2)
            ))

            # Prédictions
            forecast_index = pd.date_range(
                start=df.index[-1],
                periods=forecast_steps + 1,
                freq=pd.infer_freq(df.index)
            )[1:]

            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=st.session_state.forecast,
                mode='lines',
                name='Prédictions',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))

            fig.update_layout(
                title="Série Temporelle avec Prédictions",
                xaxis_title="Date",
                yaxis_title=value_col,
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Zoom sur la période récente
            st.subheader("🔎 Zoom sur la Période Récente")

            recent_period = st.slider(
                "Nombre de périodes historiques à afficher",
                min_value=10,
                max_value=len(df),
                value=min(50, len(df))
            )

            fig_zoom = go.Figure()

            # Données récentes
            recent_data = df[value_col].tail(recent_period)
            fig_zoom.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data,
                mode='lines+markers',
                name='Observations',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=5)
            ))

            # Prédictions
            fig_zoom.add_trace(go.Scatter(
                x=forecast_index,
                y=st.session_state.forecast,
                mode='lines+markers',
                name='Prédictions',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=5)
            ))

            fig_zoom.update_layout(
                title="Comparaison Observations vs Prédictions (Période Récente)",
                xaxis_title="Date",
                yaxis_title=value_col,
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig_zoom, use_container_width=True)

            # Tableau des prédictions
            with st.expander("📋 Valeurs Prédites"):
                forecast_df = pd.DataFrame({
                    'Date': forecast_index,
                    'Prédiction': st.session_state.forecast
                })
                st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")
        st.info("Assurez-vous que la colonne de dates est au bon format et que la colonne de valeurs contient des nombres.")

else:
    st.info("👆 Veuillez télécharger un fichier CSV ou Excel pour commencer l'analyse.")

    # Instructions
    st.markdown("""
    ## 📝 Instructions d'utilisation

    1. **Téléchargez vos données** : Utilisez le panneau latéral pour charger un fichier CSV ou Excel
    2. **Sélectionnez les colonnes** : Choisissez la colonne de dates et la colonne de valeurs
    3. **Visualisez** : Explorez votre série temporelle et sa décomposition STL
    4. **Testez la stationnarité** : Effectuez le test ADF pour vérifier la stationnarité
    5. **Choisissez un modèle** : ARIMA ou SARIMA selon vos besoins
    6. **Configurez et entraînez** : Ajustez les paramètres et lancez l'entraînement
    7. **Analysez les résultats** : Visualisez les prédictions et comparez avec les données historiques

    ### 💡 Conseils
    - Pour une série avec saisonnalité, utilisez SARIMA
    - Si la série n'est pas stationnaire (test ADF), augmentez le paramètre d
    - Expérimentez avec différents paramètres pour améliorer les prédictions
    """)

# Footer
st.markdown("---")
st.markdown("**Application d'Analyse de Séries Temporelles** | Projet Streamlit 2024-2025")
