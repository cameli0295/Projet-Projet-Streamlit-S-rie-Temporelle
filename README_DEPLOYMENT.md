# Guide de Déploiement - Application Streamlit

## 🚀 Déploiement sur Streamlit Cloud (Recommandé)

### Prérequis
- Un compte GitHub (gratuit)
- Un compte Streamlit Cloud (gratuit) : https://streamlit.io/cloud

### Étapes

1. **Préparer le Repository GitHub**
   - Assurez-vous que votre repository contient :
     - `app.py` (application principale)
     - `requirements.txt` (dépendances)
     - `Air_quality_projet.csv` (données exemple)

2. **Connecter à Streamlit Cloud**
   - Allez sur https://share.streamlit.io/
   - Cliquez sur "New app"
   - Connectez votre compte GitHub

3. **Configurer le Déploiement**
   - Repository: Sélectionnez votre repository
   - Branch: `main` ou `claude/analyze-repository-Wtcll`
   - Main file path: `app.py`

4. **Déployer**
   - Cliquez sur "Deploy!"
   - L'application sera disponible en quelques minutes
   - Vous obtiendrez une URL publique du type : `https://[app-name].streamlit.app`

## 🌐 Autres Options de Déploiement

### Render.com

1. Créez un compte sur https://render.com/
2. Créez un nouveau "Web Service"
3. Connectez votre repository GitHub
4. Configurez :
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### HuggingFace Spaces

1. Créez un compte sur https://huggingface.co/
2. Créez un nouveau Space (type: Streamlit)
3. Clonez le repository Space et poussez vos fichiers
4. L'application sera automatiquement déployée

## 🧪 Test Local

Pour tester l'application localement avant le déploiement :

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📋 Checklist avant Soumission

- [ ] Application déployée et accessible en ligne
- [ ] URL publique fonctionnelle
- [ ] Test avec le fichier Air_quality_projet.csv
- [ ] Toutes les fonctionnalités sont opérationnelles :
  - [ ] Upload de fichiers CSV/Excel
  - [ ] Visualisation de la série temporelle
  - [ ] Décomposition STL
  - [ ] Test ADF de stationnarité
  - [ ] Modèles ARIMA et SARIMA
  - [ ] Prédictions et visualisations
- [ ] PDF de soumission préparé avec noms et URL

## 📄 Format du PDF de Soumission

```
PROJET ANALYSE DE SÉRIES TEMPORELLES

Équipe :
- [Nom Prénom 1]
- [Nom Prénom 2]
- [Nom Prénom 3] (optionnel)

URL de l'application :
https://[votre-app].streamlit.app

Date : [Date de soumission]
```

## ⚠️ Points Importants

- **Deadline** : Vendredi 23 janvier 2026, 18h00
- L'application DOIT être accessible en ligne
- Une application non accessible = non soumis
- Seul le PDF avec l'URL est à soumettre (pas le code)
