# Classi Fruits

Classi Fruits est une application Streamlit d'analyse de fraicheur de fruits basee sur un modele de classification d'images. Le projet detecte si une image ou un flux webcam montre un fruit comestible ou non, puis affiche un verdict, un niveau de confiance et une analyse interpretable.

## Apercu

- Classification de 6 classes: pommes, bananes et oranges, fraiches ou pourries.
- Mode photo pour analyser une image importee.
- Mode webcam pour un usage en temps reel si `streamlit-webrtc` est disponible.
- Interface Streamlit personnalisee avec affichage du verdict, des probabilites par classe et d'une analyse supplementaire pour les fruits reconnus comme deteriores.

## Stack technique

- Python 3.11
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Plotly

## Arborescence utile

- `app.py`: application principale Streamlit.
- `best_model_phase1.keras`: modele utilise a l'execution.
- `best_model_phase1.h5`: variante du modele conservee en backup.
- `requirements.txt`: dependances Python.
- `Dockerfile`: image de production pour l'execution du service.
- `cloudbuild.yaml`: deploiement Cloud Build vers Cloud Run.
- `runtime.txt`: version Python ciblee pour les environnements compatibles.

## Lancer en local

1. Creer et activer un environnement virtuel.
2. Installer les dependances:

```powershell
pip install -r requirements.txt
```

3. Demarrer l'application:

```powershell
streamlit run app.py
```

L'application charge automatiquement `best_model_phase1.keras` depuis la racine du projet.

## Utilisation

- Ouvre l'application dans le navigateur.
- Choisis le mode `Photo` pour televerser une image.
- Choisis le mode `Temps reel (webcam)` si l'environnement supporte `streamlit-webrtc`.
- Lis le verdict affiche, la confiance associee et les probabilites de toutes les classes.

## Deploiement Cloud Run

Le projet contient un pipeline de build/deploiement base sur Cloud Build et Cloud Run.

```powershell
gcloud builds submit --config cloudbuild.yaml `
  --substitutions=_REGION=europe-west1,_SERVICE_NAME=classi-fruits
```

Le fichier `runtime.txt` est configure pour Python 3.11 afin de rester compatible avec TensorFlow et les wheels binaires utilises par le projet.

## Notes sur les fichiers exclus

Les livrables de presentation et les jeux de donnees locaux ne sont pas destines au versionnage principal du projet. Ils sont exclus via `.gitignore` et `.gcloudignore` pour garder le depot propre et limiter la taille des envois.

## Remarque

Le modele et l'interface sont concus pour aider a l'usage ou au prototype. Ils ne remplacent pas une verification humaine de l'etat reel du fruit.
