"""
Prétraitement et post-traitement pour améliorer la qualité des images générées
"""
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA


class ImageCleaner:
    """
    Nettoie les images générées pour réduire les artefacts
    """

    @staticmethod
    def threshold(images, threshold=0.3):
        """
        Seuillage simple : pixels en dessous du seuil → 0

        Args:
            images: Images (N, 28, 28) ou (28, 28)
            threshold: Seuil (0-1)

        Returns:
            Images nettoyées
        """
        result = images.copy()
        result[result < threshold] = 0
        return result

    @staticmethod
    def otsu_threshold(images):
        """
        Seuillage automatique d'Otsu (trouve le meilleur seuil)

        Args:
            images: Images (N, 28, 28)

        Returns:
            Images binarisées
        """
        from skimage.filters import threshold_otsu

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            thresh = threshold_otsu(img)
            result[i] = img > thresh

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def denoise_bilateral(images, sigma_spatial=1.0, sigma_intensity=0.1):
        """
        Débruitage bilatéral : lisse tout en préservant les bords

        Args:
            images: Images (N, 28, 28)
            sigma_spatial: Contrôle le voisinage spatial
            sigma_intensity: Contrôle la similarité d'intensité

        Returns:
            Images débruitées
        """
        from skimage.restoration import denoise_bilateral

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            result[i] = denoise_bilateral(
                img,
                sigma_color=sigma_intensity,
                sigma_spatial=sigma_spatial,
                channel_axis=None
            )

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def morphological_closing(images, iterations=1):
        """
        Fermeture morphologique : remplit les petits trous

        Args:
            images: Images (N, 28, 28)
            iterations: Nombre d'itérations

        Returns:
            Images nettoyées
        """
        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            # Binariser d'abord
            binary = img > 0.3
            # Fermeture morphologique
            closed = ndimage.binary_closing(binary, iterations=iterations)
            # Restaurer les intensités
            result[i] = closed * img

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def remove_small_components(images, min_size=10):
        """
        Supprime les petites composantes connexes (artefacts isolés)

        Args:
            images: Images (N, 28, 28)
            min_size: Taille minimale d'une composante

        Returns:
            Images nettoyées
        """
        from skimage.morphology import remove_small_objects

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            # Binariser
            binary = img > 0.3
            # Supprimer petites composantes
            cleaned = remove_small_objects(binary, min_size=min_size)
            # Restaurer les intensités
            result[i] = cleaned * img

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def combined_cleaning(images, method='aggressive'):
        """
        Pipeline de nettoyage combiné

        Args:
            images: Images (N, 28, 28)
            method: 'light', 'medium', 'aggressive'

        Returns:
            Images nettoyées
        """
        result = images.copy()

        if method == 'light':
            # Juste un seuillage doux
            result = ImageCleaner.threshold(result, threshold=0.2)

        elif method == 'medium':
            # Seuillage + suppression petites composantes
            result = ImageCleaner.threshold(result, threshold=0.25)
            result = ImageCleaner.remove_small_components(result, min_size=8)

        elif method == 'aggressive':
            # Pipeline complet
            result = ImageCleaner.denoise_bilateral(result, sigma_spatial=1.0, sigma_intensity=0.1)
            result = ImageCleaner.threshold(result, threshold=0.3)
            result = ImageCleaner.remove_small_components(result, min_size=10)
            result = ImageCleaner.morphological_closing(result, iterations=1)

        # Normaliser entre 0 et 1
        if result.max() > 0:
            result = result / result.max()

        return result


class ImprovedRejectionSampling:
    """
    Rejection sampling amélioré avec critères de qualité supplémentaires
    """

    @staticmethod
    def compute_quality_metrics(images):
        """
        Calcule des métriques de qualité pour chaque image

        Returns:
            dict avec 'sparsity', 'contrast', 'connectivity'
        """
        metrics = {
            'sparsity': [],  # Proportion de pixels actifs
            'contrast': [],  # Écart-type des intensités
            'connectivity': []  # Nombre de composantes connexes
        }

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        for img in images:
            # Sparsité (on veut entre 10% et 30% de pixels actifs)
            sparsity = (img > 0.1).sum() / img.size
            metrics['sparsity'].append(sparsity)

            # Contraste
            contrast = img.std()
            metrics['contrast'].append(contrast)

            # Connectivité (on veut idéalement 1 composante principale)
            binary = img > 0.3
            labeled, n_components = ndimage.label(binary)
            metrics['connectivity'].append(n_components)

        return metrics

    @staticmethod
    def is_good_quality(img, strict=False):
        """
        Détermine si une image est de bonne qualité

        Args:
            img: Image (28, 28)
            strict: Critères plus stricts

        Returns:
            bool
        """
        # Sparsité : entre 10% et 40% de pixels actifs
        sparsity = (img > 0.1).sum() / img.size
        if strict:
            if sparsity < 0.12 or sparsity > 0.35:
                return False
        else:
            if sparsity < 0.08 or sparsity > 0.45:
                return False

        # Contraste suffisant
        contrast = img.std()
        if strict:
            if contrast < 0.15:
                return False
        else:
            if contrast < 0.10:
                return False

        # Pas trop de composantes (artefacts)
        binary = img > 0.3
        labeled, n_components = ndimage.label(binary)
        if strict:
            if n_components > 3:  # Maximum 3 composantes
                return False
        else:
            if n_components > 5:  # Maximum 5 composantes
                return False

        return True


def process_training_data(X, method='standard'):
    """
    Prétraite les données d'entraînement avant PCA

    Args:
        X: Données (N, 784)
        method: 'standard', 'normalized', 'whitened'

    Returns:
        X_processed, processor (pour inverser)
    """
    X_processed = X.copy()
    processor = {'method': method}

    if method == 'standard':
        # Juste centrer
        mean = X.mean(axis=0)
        X_processed = X - mean
        processor['mean'] = mean

    elif method == 'normalized':
        # Centrer et normaliser
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_processed = (X - mean) / std
        processor['mean'] = mean
        processor['std'] = std

    elif method == 'whitened':
        # Centrer et blanchir (PCA whitening)
        mean = X.mean(axis=0)
        X_centered = X - mean

        # PCA pour blanchir
        pca_whitening = PCA(whiten=True, random_state=42)
        X_processed = pca_whitening.fit_transform(X_centered)

        processor['mean'] = mean
        processor['pca'] = pca_whitening

    return X_processed, processor
