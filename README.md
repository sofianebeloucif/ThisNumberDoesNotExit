# 🎨 Générateur MNIST avec KDE

Projet de génération d'images de chiffres manuscrits (MNIST) utilisant PCA (Principal Component Analysis) et KDE (Kernel Density Estimation).

## 📋 Description

Ce projet implémente une approche générative pour créer de nouvelles images de chiffres manuscrits en:


1. **Réduction de dimensionnalité** : Transformation des images MNIST (784 dimensions) en 50 dimensions avec PCA
2. **Estimation de densité** : Utilisation de KDE pour modéliser la distribution des données dans l'espace réduit
3. **Génération** : Échantillonnage depuis le KDE et reconstruction via PCA inverse

## 🏗️ Structure du projet

```
mnist-kde-generator/
├── notebooks/
│   └── train_kde_model.ipynb      # Entraînement des modèles
├── models/
│   ├── pca_model.pkl               # Modèle PCA sauvegardé
│   └── kde_model.pkl               # Modèle KDE sauvegardé
├── app/
│   ├── app.py                      # Application Flask
│   └── templates/
│       └── index.html              # Interface web
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/votre-username/mnist-kde-generator.git
cd mnist-kde-generator
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Sur Linux/Mac
source venv/bin/activate

# Sur Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📊 Étape 1 : Entraînement des modèles

### Lancer Jupyter Notebook

```bash
jupyter notebook
```

### Ouvrir et exécuter le notebook

1. Ouvrir `notebooks/train_kde_model.ipynb`
2. Exécuter toutes les cellules (Cell → Run All)
3. Les modèles seront sauvegardés dans le dossier `models/`

Le notebook effectue :
- Chargement des données MNIST (60,000 images d'entraînement)
- Réduction de 784 → 50 dimensions avec PCA (~95% de variance conservée)
- Optimisation du bandwidth KDE par validation croisée
- Entraînement du KDE sur 10,000 échantillons
- Visualisation et évaluation des résultats
- Sauvegarde des modèles

**Note** : L'entraînement prend environ 2-5 minutes selon votre machine.

## 🌐 Étape 2 : Lancer l'application web

### Démarrer le serveur Flask

```bash
cd app
python app.py
```

### Accéder à l'interface

Ouvrir votre navigateur à l'adresse : **http://localhost:5000**

## 🎮 Utilisation de l'application web

L'interface permet de :

1. **Visualiser les statistiques** du modèle (composantes PCA, variance expliquée, bandwidth)
2. **Choisir le nombre d'images** à générer (1-16)
3. **Générer de nouvelles images** en cliquant sur le bouton

Les images générées sont affichées dans une galerie interactive.

## 🔬 Méthodologie

### PCA (Principal Component Analysis)

- **Input** : Images 28×28 = 784 dimensions
- **Output** : 50 dimensions
- **Avantage** : Réduit drastiquement la dimensionnalité tout en conservant ~95% de l'information

### KDE (Kernel Density Estimation)

- **Kernel** : Gaussien
- **Bandwidth** : Optimisé par validation croisée
- **Échantillonnage** : Génération de nouveaux points depuis la distribution estimée

### Processus de génération

```
1. KDE.sample() → Vecteur 50D
2. PCA.inverse_transform() → Vecteur 784D
3. Reshape(28, 28) → Image MNIST
```

## 📈 Résultats attendus

- **Variance expliquée** : ~95% avec 50 composantes
- **Qualité visuelle** : Images reconnaissables mais légèrement floues
- **Diversité** : Grande variété de chiffres générés

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **NumPy** : Calculs numériques
- **scikit-learn** : PCA et KDE
- **TensorFlow/Keras** : Chargement de MNIST
- **Flask** : Application web
- **Matplotlib** : Visualisations
- **Pillow** : Traitement d'images

## 📝 API Endpoints

### `GET /`
Page d'accueil de l'application

### `POST /generate`
Génère des images MNIST

**Body** :
```json
{
  "n_samples": 4
}
```

**Response** :
```json
{
  "success": true,
  "images": ["data:image/png;base64,..."],
  "count": 4
}
```

### `GET /stats`
Retourne les statistiques des modèles

**Response** :
```json
{
  "pca_components": 50,
  "variance_explained": 0.95,
  "kde_bandwidth": 1.2,
  "original_dim": 784,
  "reduced_dim": 50
}
```

## 🔧 Personnalisation

### Modifier le nombre de composantes PCA

Dans `train_kde_model.ipynb` :
```python
pca = PCA(n_components=100)  # Au lieu de 50
```

### Ajuster le bandwidth KDE

```python
kde = KernelDensity(bandwidth=2.0)  # Valeur plus élevée = images plus floues
```

### Changer le kernel

```python
kde = KernelDensity(kernel='exponential')  # Autres options: 'gaussian', 'tophat', 'epanechnikov'
```

## 🐛 Troubleshooting

### Erreur : "Models not found"
→ Assurez-vous d'avoir exécuté le notebook d'entraînement

### Les images sont trop floues
→ Augmentez le nombre de composantes PCA ou ajustez le bandwidth

### Erreur de mémoire
→ Réduisez `n_samples_kde` dans le notebook (actuellement 10,000)

## 🤝 Contributions

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer des améliorations via pull request
- Partager vos résultats

## 📄 Licence

MIT License - Libre d'utilisation et de modification

## 👨‍💻 Auteur

Créé avec ❤️ pour explorer les méthodes génératives classiques

## 📚 Références

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Kernel Density Estimation](https://scikit-learn.org/stable/modules/density.html)
- [PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

**Bon amusement avec la génération d'images ! 🎨**