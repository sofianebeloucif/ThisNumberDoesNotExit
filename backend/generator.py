"""
Classes de génération pour MNIST avec PCA + KDE
Deux approches: modèle global vs modèles par classe
"""
import numpy as np
import pickle
from typing import BinaryIO
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class GlobalGenerator:
    """
    Générateur global : un seul KDE entraîné sur tous les chiffres
    """

    def __init__(self, n_components=50, bandwidth=1.2):
        """
        Initialise le générateur global

        Args:
            n_components: Nombre de composantes PCA
            bandwidth: Bandwidth du KDE
        """
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.pca = None
        self.kde = None
        self.rejection_thresholds = None
        self.is_trained = False

    def fit(self, X, n_samples_kde=10000):
        """
        Entraîne le générateur sur les données

        Args:
            X: Données d'entraînement (N, 784)
            n_samples_kde: Nombre d'échantillons pour le KDE
        """
        print(f"[GlobalGenerator] Entraînement sur {len(X)} images...")

        # PCA
        print("  → PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"    Variance expliquée: {variance:.2%}")

        # KDE sur un sous-ensemble
        n_samples_kde = min(n_samples_kde, len(X_pca))
        indices = np.random.choice(len(X_pca), n_samples_kde, replace=False)
        X_kde = X_pca[indices]

        print(f"  → KDE ({n_samples_kde} échantillons)...")
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(X_kde)

        # Calculer les seuils de rejection sampling
        print("  → Calcul des seuils...")
        densities = self.kde.score_samples(X_pca[:1000])
        self.rejection_thresholds = {
            10: np.percentile(densities, 10),
            25: np.percentile(densities, 25),
            50: np.percentile(densities, 50)
        }

        self.is_trained = True
        print("  ✓ Entraînement terminé")

    def generate(self, n_samples=1, use_rejection=True, percentile=25,
                 clean_images=True, cleaning_method='medium'):
        """
        Génère des images

        Args:
            n_samples: Nombre d'images à générer
            use_rejection: Utiliser rejection sampling
            percentile: Niveau de filtrage (10, 25 ou 50)
            clean_images: Appliquer le nettoyage post-génération
            cleaning_method: 'light', 'medium', 'aggressive'

        Returns:
            images: Images générées (n_samples, 28, 28)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné d'abord (appeler .fit())")

        if not use_rejection:
            samples_pca = self.kde.sample(n_samples)
        else:
            threshold = self.rejection_thresholds[percentile]
            samples_pca = self._rejection_sampling(n_samples, threshold)

        # Reconstruire
        images_flat = self.pca.inverse_transform(samples_pca)
        images_flat = np.clip(images_flat, 0, 1)
        images = images_flat.reshape(-1, 28, 28)

        # Nettoyage post-génération
        if clean_images:
            from image_processing import ImageCleaner
            images = ImageCleaner.combined_cleaning(images, method=cleaning_method)

        return images

    def _rejection_sampling(self, n_samples, threshold):
        """Rejection sampling interne"""
        accepted = []
        attempts = 0
        max_attempts = n_samples * 200

        while len(accepted) < n_samples and attempts < max_attempts:
            batch_size = min(n_samples * 10, 1000)
            candidates = self.kde.sample(batch_size)
            densities = self.kde.score_samples(candidates)

            mask = densities >= threshold
            accepted_batch = candidates[mask]
            if len(accepted_batch) > 0:
                accepted.extend(accepted_batch)
            attempts += batch_size

        # Compléter si nécessaire
        if len(accepted) < n_samples:
            remaining = n_samples - len(accepted)
            additional = self.kde.sample(remaining)
            accepted.extend(additional)

        return np.array(accepted[:n_samples])

    def save(self, filepath):
        """Sauvegarde le modèle"""
        data = {
            'pca': self.pca,
            'kde': self.kde,
            'rejection_thresholds': self.rejection_thresholds,
            'n_components': self.n_components,
            'bandwidth': self.bandwidth,
            'bandwidth_optimized': True  # Marqueur pour indiquer si optimisé
        }
        with open(filepath, 'wb') as f:
            f: BinaryIO
            pickle.dump(data, f)
        print(f"✓ Modèle sauvegardé: {filepath}")
        print(f"  Bandwidth: {self.bandwidth:.3f}")

    @classmethod
    def load(cls, filepath):
        """Charge un modèle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        gen = cls(n_components=data['n_components'], bandwidth=data['bandwidth'])
        gen.pca = data['pca']
        gen.kde = data['kde']
        gen.rejection_thresholds = data['rejection_thresholds']
        gen.is_trained = True

        print(f"✓ Modèle chargé: {filepath}")
        print(f"  Bandwidth: {data['bandwidth']:.3f}")
        if data.get('bandwidth_optimized', False):
            print(f"  (Optimisé par cross-validation)")
        return gen


class ConditionalGenerator:
    """
    Générateur conditionnel : un KDE par classe (chiffre)
    """

    def __init__(self, n_components=50, bandwidth=1.2):
        """
        Initialise le générateur conditionnel

        Args:
            n_components: Nombre de composantes PCA
            bandwidth: Bandwidth du KDE
        """
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.pca = None
        self.kde_models = {}  # Un KDE par classe
        self.rejection_thresholds = {}  # Seuils par classe
        self.is_trained = False

    def fit(self, X, y, n_samples_kde=5000):
        """
        Entraîne le générateur sur les données avec labels

        Args:
            X: Données d'entraînement (N, 784)
            y: Labels (N,)
            n_samples_kde: Nombre max d'échantillons par KDE
        """
        print(f"[ConditionalGenerator] Entraînement sur {len(X)} images...")

        # PCA global
        print("  → PCA global...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"    Variance expliquée: {variance:.2%}")

        # Entraîner un KDE par classe
        classes = np.unique(y)
        print(f"  → Entraînement de {len(classes)} KDE (un par classe)...")

        for cls in classes:
            # Extraire les données de cette classe
            mask = y == cls
            X_cls = X_pca[mask]

            # Sous-échantillonner si nécessaire
            n_samples = min(len(X_cls), n_samples_kde)
            if len(X_cls) > n_samples:
                indices = np.random.choice(len(X_cls), n_samples, replace=False)
                X_cls = X_cls[indices]

            # Entraîner le KDE
            kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
            kde.fit(X_cls)
            self.kde_models[int(cls)] = kde

            # Calculer les seuils
            densities = kde.score_samples(X_cls[:min(1000, len(X_cls))])
            self.rejection_thresholds[int(cls)] = {
                10: np.percentile(densities, 10),
                25: np.percentile(densities, 25),
                50: np.percentile(densities, 50)
            }

            print(f"    Classe {cls}: {len(X_cls)} échantillons")

        self.is_trained = True
        print("  ✓ Entraînement terminé")

    def generate(self, digit, n_samples=1, use_rejection=True, percentile=25,
                 clean_images=True, cleaning_method='medium'):
        """
        Génère des images d'un chiffre spécifique

        Args:
            digit: Chiffre à générer (0-9)
            n_samples: Nombre d'images à générer
            use_rejection: Utiliser rejection sampling
            percentile: Niveau de filtrage (10, 25 ou 50)
            clean_images: Appliquer le nettoyage post-génération
            cleaning_method: 'light', 'medium', 'aggressive'

        Returns:
            images: Images générées (n_samples, 28, 28)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné d'abord (appeler .fit())")

        if digit not in self.kde_models:
            raise ValueError(f"Chiffre {digit} non disponible. Classes: {list(self.kde_models.keys())}")

        kde = self.kde_models[digit]

        if not use_rejection:
            samples_pca = kde.sample(n_samples)
        else:
            threshold = self.rejection_thresholds[digit][percentile]
            samples_pca = self._rejection_sampling(kde, n_samples, threshold)

        # Reconstruire
        images_flat = self.pca.inverse_transform(samples_pca)
        images_flat = np.clip(images_flat, 0, 1)
        images = images_flat.reshape(-1, 28, 28)

        # Nettoyage post-génération
        if clean_images:
            from image_processing import ImageCleaner
            images = ImageCleaner.combined_cleaning(images, method=cleaning_method)

        return images

    def _rejection_sampling(self, kde, n_samples, threshold):
        """Rejection sampling interne"""
        accepted = []
        attempts = 0
        max_attempts = n_samples * 200

        while len(accepted) < n_samples and attempts < max_attempts:
            batch_size = min(n_samples * 10, 1000)
            candidates = kde.sample(batch_size)
            densities = kde.score_samples(candidates)

            mask = densities >= threshold
            accepted_batch = candidates[mask]
            if len(accepted_batch) > 0:
                accepted.extend(accepted_batch)
            attempts += batch_size

        # Compléter si nécessaire
        if len(accepted) < n_samples:
            remaining = n_samples - len(accepted)
            additional = kde.sample(remaining)
            accepted.extend(additional)

        return np.array(accepted[:n_samples])

    def generate_all(self, n_samples_per_digit=1, use_rejection=True, percentile=25):
        """
        Génère des images de tous les chiffres

        Args:
            n_samples_per_digit: Nombre d'images par chiffre
            use_rejection: Utiliser rejection sampling
            percentile: Niveau de filtrage

        Returns:
            dict: {digit: images}
        """
        result = {}
        for digit in self.kde_models.keys():
            result[digit] = self.generate(digit, n_samples_per_digit, use_rejection, percentile)
        return result

    def save(self, filepath):
        """Sauvegarde le modèle"""
        data = {
            'pca': self.pca,
            'kde_models': self.kde_models,
            'rejection_thresholds': self.rejection_thresholds,
            'n_components': self.n_components,
            'bandwidth': self.bandwidth,
            'bandwidth_optimized': True
        }
        with open(filepath, 'wb') as f:
            f: BinaryIO
            pickle.dump(data, f)
        print(f"✓ Modèle sauvegardé: {filepath}")
        print(f"  Bandwidth: {self.bandwidth:.3f}")

    @classmethod
    def load(cls, filepath):
        """Charge un modèle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        gen = cls(n_components=data['n_components'], bandwidth=data['bandwidth'])
        gen.pca = data['pca']
        gen.kde_models = data['kde_models']
        gen.rejection_thresholds = data['rejection_thresholds']
        gen.is_trained = True

        print(f"✓ Modèle chargé: {filepath}")
        print(f"  Bandwidth: {data['bandwidth']:.3f}")
        print(f"  Classes disponibles: {list(data['kde_models'].keys())}")
        if data.get('bandwidth_optimized', False):
            print(f"  (Optimisé par cross-validation)")
        return gen


# Fonction utilitaire pour comparer les deux approches
def compare_generators(global_gen, conditional_gen, n_samples=10):
    """
    Compare visuellement les deux générateurs

    Args:
        global_gen: GlobalGenerator entraîné
        conditional_gen: ConditionalGenerator entraîné
        n_samples: Nombre d'échantillons à générer

    Returns:
        dict: Résultats de la comparaison
    """
    import time

    results = {
        'global': {},
        'conditional': {}
    }

    # Test global
    start = time.time()
    images_global = global_gen.generate(n_samples)
    results['global']['time'] = time.time() - start
    results['global']['images'] = images_global

    # Test conditionnel (générer des 7 par exemple)
    start = time.time()
    images_cond = conditional_gen.generate(digit=7, n_samples=n_samples)
    results['conditional']['time'] = time.time() - start
    results['conditional']['images'] = images_cond

    return results


if __name__ == "__main__":
    """Test des classes"""
    from tensorflow import keras

    print("="*70)
    print("TEST DES CLASSES DE GÉNÉRATION")
    print("="*70)

    # Charger MNIST
    print("\nChargement de MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_flat = x_train.reshape(-1, 784) / 255.0

    # Test GlobalGenerator
    print("\n" + "="*70)
    print("Test GlobalGenerator")
    print("="*70)
    global_gen = GlobalGenerator(n_components=50, bandwidth=1.2)
    global_gen.fit(x_train_flat[:10000])  # Sous-ensemble pour le test
    images = global_gen.generate(n_samples=5)
    print(f"✓ Généré {len(images)} images, shape: {images.shape}")

    # Test ConditionalGenerator
    print("\n" + "="*70)
    print("Test ConditionalGenerator")
    print("="*70)
    cond_gen = ConditionalGenerator(n_components=50, bandwidth=1.2)
    cond_gen.fit(x_train_flat[:10000], y_train[:10000])
    images_7 = cond_gen.generate(digit=7, n_samples=5)
    print(f"✓ Généré {len(images_7)} images du chiffre 7, shape: {images_7.shape}")

    print("\n" + "="*70)
    print("✅ Tous les tests réussis!")
    print("="*70)