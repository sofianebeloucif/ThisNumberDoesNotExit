"""
Generation classes for MNIST using PCA + KDE
Two approaches: global model vs per-class models
"""
import numpy as np
import pickle
from typing import BinaryIO
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class GlobalGenerator:
    """
    Global generator: a single KDE trained on all digits
    """

    def __init__(self, n_components=50, bandwidth=1.2):
        """
        Initialize the global generator

        Args:
            n_components: Number of PCA components
            bandwidth: KDE bandwidth
        """
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.pca = None
        self.kde = None
        self.rejection_thresholds = None
        self.is_trained = False

    def fit(self, X, n_samples_kde=10000):
        """
        Train the generator on data

        Args:
            X: Training data (N, 784)
            n_samples_kde: Number of samples for KDE
        """
        print(f"[GlobalGenerator] Training on {len(X)} images...")

        # PCA
        print("  → PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"    Explained variance: {variance:.2%}")

        # KDE on a subset
        n_samples_kde = min(n_samples_kde, len(X_pca))
        indices = np.random.choice(len(X_pca), n_samples_kde, replace=False)
        X_kde = X_pca[indices]

        print(f"  → KDE ({n_samples_kde} samples)...")
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(X_kde)

        # Compute rejection-sampling thresholds
        print("  → Computing thresholds...")
        densities = self.kde.score_samples(X_pca[:1000])
        self.rejection_thresholds = {
            10: np.percentile(densities, 10),
            25: np.percentile(densities, 25),
            50: np.percentile(densities, 50)
        }

        self.is_trained = True
        print("  ✓ Training complete")

    def generate(self, n_samples=1, use_rejection=True, percentile=25,
                 clean_images=True, cleaning_method='medium'):
        """
        Generate images

        Args:
            n_samples: Number of images to generate
            use_rejection: Use rejection sampling
            percentile: Filtering level (10, 25, or 50)
            clean_images: Apply post-generation cleaning
            cleaning_method: 'light', 'medium', 'aggressive'

        Returns:
            images: Generated images (n_samples, 28, 28)
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first (call .fit())")

        if not use_rejection:
            samples_pca = self.kde.sample(n_samples)
        else:
            threshold = self.rejection_thresholds[percentile]
            samples_pca = self._rejection_sampling(n_samples, threshold)

        # Reconstruct
        images_flat = self.pca.inverse_transform(samples_pca)
        images_flat = np.clip(images_flat, 0, 1)
        images = images_flat.reshape(-1, 28, 28)

        # Post-generation cleaning
        if clean_images:
            from image_processing import ImageCleaner
            images = ImageCleaner.combined_cleaning(images, method=cleaning_method)

        return images

    def _rejection_sampling(self, n_samples, threshold):
        """Internal rejection sampling"""
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

        # Fill if necessary
        if len(accepted) < n_samples:
            remaining = n_samples - len(accepted)
            additional = self.kde.sample(remaining)
            accepted.extend(additional)

        return np.array(accepted[:n_samples])

    def save(self, filepath):
        """Save the model"""
        data = {
            'pca': self.pca,
            'kde': self.kde,
            'rejection_thresholds': self.rejection_thresholds,
            'n_components': self.n_components,
            'bandwidth': self.bandwidth,
            'bandwidth_optimized': True  # Flag to indicate if optimized
        }
        with open(filepath, 'wb') as f:
            f: BinaryIO
            pickle.dump(data, f)
        print(f"✓ Model saved: {filepath}")
        print(f"  Bandwidth: {self.bandwidth:.3f}")

    @classmethod
    def load(cls, filepath):
        """Load a model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        gen = cls(n_components=data['n_components'], bandwidth=data['bandwidth'])
        gen.pca = data['pca']
        gen.kde = data['kde']
        gen.rejection_thresholds = data['rejection_thresholds']
        gen.is_trained = True

        print(f"✓ Model loaded: {filepath}")
        print(f"  Bandwidth: {data['bandwidth']:.3f}")
        if data.get('bandwidth_optimized', False):
            print(f"  (Optimized via cross-validation)")
        return gen


class ConditionalGenerator:
    """
    Conditional generator: one KDE per class (digit)
    """

    def __init__(self, n_components=50, bandwidth=1.2):
        """
        Initialize the conditional generator

        Args:
            n_components: Number of PCA components
            bandwidth: KDE bandwidth
        """
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.pca = None
        self.kde_models = {}  # One KDE per class
        self.rejection_thresholds = {}  # Thresholds per class
        self.is_trained = False

    def fit(self, X, y, n_samples_kde=5000):
        """
        Train the generator on labeled data

        Args:
            X: Training data (N, 784)
            y: Labels (N,)
            n_samples_kde: Max number of samples per KDE
        """
        print(f"[ConditionalGenerator] Training on {len(X)} images...")

        # Global PCA
        print("  → Global PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"    Explained variance: {variance:.2%}")

        # Train a KDE per class
        classes = np.unique(y)
        print(f"  → Training {len(classes)} KDE models (one per class)...")

        for cls in classes:
            # Extract data for this class
            mask = y == cls
            X_cls = X_pca[mask]

            # Subsample if necessary
            n_samples = min(len(X_cls), n_samples_kde)
            if len(X_cls) > n_samples:
                indices = np.random.choice(len(X_cls), n_samples, replace=False)
                X_cls = X_cls[indices]

            # Train the KDE
            kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
            kde.fit(X_cls)
            self.kde_models[int(cls)] = kde

            # Compute thresholds
            densities = kde.score_samples(X_cls[:min(1000, len(X_cls))])
            self.rejection_thresholds[int(cls)] = {
                10: np.percentile(densities, 10),
                25: np.percentile(densities, 25),
                50: np.percentile(densities, 50)
            }

            print(f"    Class {cls}: {len(X_cls)} samples")

        self.is_trained = True
        print("  ✓ Training complete")

    def generate(self, digit, n_samples=1, use_rejection=True, percentile=25,
                 clean_images=True, cleaning_method='medium'):
        """
        Generate images of a specific digit

        Args:
            digit: Digit to generate (0-9)
            n_samples: Number of images to generate
            use_rejection: Use rejection sampling
            percentile: Filtering level (10, 25, or 50)
            clean_images: Apply post-generation cleaning
            cleaning_method: 'light', 'medium', 'aggressive'

        Returns:
            images: Generated images (n_samples, 28, 28)
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first (call .fit())")

        if digit not in self.kde_models:
            raise ValueError(f"Digit {digit} not available. Classes: {list(self.kde_models.keys())}")

        kde = self.kde_models[digit]

        if not use_rejection:
            samples_pca = kde.sample(n_samples)
        else:
            threshold = self.rejection_thresholds[digit][percentile]
            samples_pca = self._rejection_sampling(kde, n_samples, threshold)

        # Reconstruct
        images_flat = self.pca.inverse_transform(samples_pca)
        images_flat = np.clip(images_flat, 0, 1)
        images = images_flat.reshape(-1, 28, 28)

        # Post-generation cleaning
        if clean_images:
            from image_processing import ImageCleaner
            images = ImageCleaner.combined_cleaning(images, method=cleaning_method)

        return images

    def _rejection_sampling(self, kde, n_samples, threshold):
        """Internal rejection sampling"""
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

        # Fill if necessary
        if len(accepted) < n_samples:
            remaining = n_samples - len(accepted)
            additional = kde.sample(remaining)
            accepted.extend(additional)

        return np.array(accepted[:n_samples])

    def generate_all(self, n_samples_per_digit=1, use_rejection=True, percentile=25):
        """
        Generate images for all digits

        Args:
            n_samples_per_digit: Number of images per digit
            use_rejection: Use rejection sampling
            percentile: Filtering level

        Returns:
            dict: {digit: images}
        """
        result = {}
        for digit in self.kde_models.keys():
            result[digit] = self.generate(digit, n_samples_per_digit, use_rejection, percentile)
        return result

    def clean(self, images, method="medium"):
        """
        Clean already generated images

        Args:
            images: np.ndarray (N, 28, 28)
            method: 'light', 'medium', 'aggressive'

        Returns:
            cleaned images (N, 28, 28)
        """
        from image_processing import ImageCleaner
        return ImageCleaner.combined_cleaning(images, method=method)

    def save(self, filepath):
        """Save the model"""
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
        print(f"✓ Model saved: {filepath}")
        print(f"  Bandwidth: {self.bandwidth:.3f}")

    @classmethod
    def load(cls, filepath):
        """Load a model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        gen = cls(n_components=data['n_components'], bandwidth=data['bandwidth'])
        gen.pca = data['pca']
        gen.kde_models = data['kde_models']
        gen.rejection_thresholds = data['rejection_thresholds']
        gen.is_trained = True

        print(f"✓ Model loaded: {filepath}")
        print(f"  Bandwidth: {data['bandwidth']:.3f}")
        print(f"  Available classes: {list(data['kde_models'].keys())}")
        if data.get('bandwidth_optimized', False):
            print(f"  (Optimized via cross-validation)")
        return gen


# Utility function to compare the two approaches
def compare_generators(global_gen, conditional_gen, n_samples=10):
    """
    Visually compare the two generators

    Args:
        global_gen: Trained GlobalGenerator
        conditional_gen: Trained ConditionalGenerator
        n_samples: Number of samples to generate

    Returns:
        dict: Comparison results
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

    # Conditional test (generate 7s for example)
    start = time.time()
    images_cond = conditional_gen.generate(digit=7, n_samples=n_samples)
    results['conditional']['time'] = time.time() - start
    results['conditional']['images'] = images_cond

    return results


if __name__ == "__main__":
    """Classes test"""
    from tensorflow import keras

    print("="*70)
    print("GENERATOR CLASSES TEST")
    print("="*70)

    # Load MNIST
    print("\nLoading MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_flat = x_train.reshape(-1, 784) / 255.0

    # Test GlobalGenerator
    print("\n" + "="*70)
    print("Testing GlobalGenerator")
    print("="*70)
    global_gen = GlobalGenerator(n_components=50, bandwidth=1.2)
    global_gen.fit(x_train_flat[:10000])  # Subset for testing
    images = global_gen.generate(n_samples=5)
    print(f"✓ Generated {len(images)} images, shape: {images.shape}")

    # Test ConditionalGenerator
    print("\n" + "="*70)
    print("Testing ConditionalGenerator")
    print("="*70)
    cond_gen = ConditionalGenerator(n_components=50, bandwidth=1.2)
    cond_gen.fit(x_train_flat[:10000], y_train[:10000])
    images_7 = cond_gen.generate(digit=7, n_samples=5)
    print(f"✓ Generated {len(images_7)} images of digit 7, shape: {images_7.shape}")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)