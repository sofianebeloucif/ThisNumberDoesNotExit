from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Charger les modèles au démarrage
print("Chargement des modèles...")
with open('../models/pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

with open('../models/kde_model.pkl', 'rb') as f:
    kde_model = pickle.load(f)

with open('../models/rejection_params.pkl', 'rb') as f:
    rejection_params = pickle.load(f)

print("✓ Modèles chargés avec succès")
print(f"✓ Rejection sampling activé (percentiles: 10%, 25%, 50%)")


def generate_samples_with_rejection(n_samples=1, threshold_percentile=25,
                                    max_attempts=10000, random_state=None):
    """
    Génère des images MNIST avec rejection sampling pour améliorer la qualité

    Args:
        n_samples: Nombre d'échantillons à générer
        threshold_percentile: Percentile de log-densité minimum (10, 25, ou 50)
        max_attempts: Nombre maximum de tentatives
        random_state: Seed pour reproductibilité

    Returns:
        samples_images: Images acceptées (n_samples, 28, 28)
        acceptance_rate: Taux d'acceptation
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Sélectionner le seuil approprié
    threshold_key = f'percentile_{threshold_percentile}'
    threshold = rejection_params.get(threshold_key, rejection_params['percentile_25'])

    accepted_samples = []
    attempts = 0

    while len(accepted_samples) < n_samples and attempts < max_attempts:
        # Générer un batch de candidats
        batch_size = min(n_samples * 5, 1000)
        candidates_pca = kde_model.sample(batch_size)

        # Calculer les log-densités
        log_densities = kde_model.score_samples(candidates_pca)

        # Accepter les échantillons au-dessus du seuil
        accepted_mask = log_densities >= threshold
        accepted_batch = candidates_pca[accepted_mask]

        accepted_samples.extend(accepted_batch[:n_samples - len(accepted_samples)])
        attempts += batch_size

    accepted_samples = np.array(accepted_samples[:n_samples])

    # Reconstruire les images
    samples_flat = pca_model.inverse_transform(accepted_samples)
    samples_flat = np.clip(samples_flat, 0, 1)
    samples_images = samples_flat.reshape(-1, 28, 28)

    acceptance_rate = len(accepted_samples) / attempts if attempts > 0 else 0

    return samples_images, acceptance_rate


def generate_samples_no_rejection(n_samples=1, random_state=None):
    """Génère des images MNIST sans rejection sampling (méthode de base)"""
    # Échantillonner depuis le KDE
    samples_pca = kde_model.sample(n_samples, random_state=random_state)

    # Reconstruire avec PCA inverse
    samples_flat = pca_model.inverse_transform(samples_pca)

    # Clipper les valeurs entre 0 et 1
    samples_flat = np.clip(samples_flat, 0, 1)

    # Reshape en images 28x28
    samples_images = samples_flat.reshape(-1, 28, 28)

    return samples_images


def image_to_base64(img_array):
    """Convertit un array numpy en base64 pour l'affichage web"""
    # Normaliser à 0-255
    img_array = (img_array * 255).astype(np.uint8)

    # Créer une image PIL
    img = Image.fromarray(img_array, mode='L')

    # Redimensionner pour meilleur affichage (280x280)
    img = img.resize((280, 280), Image.NEAREST)

    # Convertir en base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint pour générer des images"""
    try:
        data = request.get_json()
        n_samples = int(data.get('n_samples', 1))
        use_rejection = data.get('use_rejection', True)
        percentile = int(data.get('percentile', 25))

        # Limiter le nombre d'échantillons
        n_samples = min(max(1, n_samples), 16)

        # Valider le percentile
        if percentile not in [10, 25, 50]:
            percentile = 25

        # Générer les images
        if use_rejection:
            images, acceptance_rate = generate_samples_with_rejection(
                n_samples, threshold_percentile=percentile)
        else:
            images = generate_samples_no_rejection(n_samples)
            acceptance_rate = 1.0  # Pas de rejection

        # Convertir en base64
        images_b64 = [image_to_base64(img) for img in images]

        return jsonify({
            'success': True,
            'images': images_b64,
            'count': len(images_b64),
            'acceptance_rate': float(acceptance_rate),
            'use_rejection': use_rejection,
            'percentile': percentile if use_rejection else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats')
def stats():
    """Retourne des statistiques sur les modèles"""
    return jsonify({
        'pca_components': pca_model.n_components_,
        'variance_explained': float(pca_model.explained_variance_ratio_.sum()),
        'kde_bandwidth': float(kde_model.bandwidth),
        'original_dim': 784,
        'reduced_dim': 50,
        'rejection_sampling': {
            'available': True,
            'percentiles': [10, 25, 50],
            'default': rejection_params.get('default_percentile', 25),
            'thresholds': {
                '10': float(rejection_params['percentile_10']),
                '25': float(rejection_params['percentile_25']),
                '50': float(rejection_params['percentile_50'])
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)