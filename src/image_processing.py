"""
Pre-processing and post-processing to improve the quality of generated images
"""
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from typing import Dict, Any


class ImageCleaner:
    """
    Clean generated images to reduce artifacts
    """

    @staticmethod
    def threshold(images, threshold=0.3):
        """
        Simple thresholding: pixels below the threshold → 0

        Args:
            images: Images (N, 28, 28) or (28, 28)
            threshold: Threshold (0-1)

        Returns:
            Cleaned images
        """
        result = images.copy()
        result[result < threshold] = 0
        return result

    @staticmethod
    def otsu_threshold(images):
        """
        Otsu automatic thresholding (finds the best threshold)

        Args:
            images: Images (N, 28, 28)

        Returns:
            Binarized images
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
        Bilateral denoising: smooths while preserving edges

        Args:
            images: Images (N, 28, 28)
            sigma_spatial: Controls spatial neighborhood
            sigma_intensity: Controls intensity similarity

        Returns:
            Denoised images
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
        Morphological closing: fills small holes

        Args:
            images: Images (N, 28, 28)
            iterations: Number of iterations

        Returns:
            Cleaned images
        """
        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            # First binarize
            binary = img > 0.3
            # Morphological closing
            closed = ndimage.binary_closing(binary, iterations=iterations)
            # Restore intensities
            result[i] = closed * img

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def remove_small_components(images, min_size=10):
        """
        Remove small connected components (isolated artifacts)

        Args:
            images: Images (N, 28, 28)
            min_size: Minimum component size

        Returns:
            Cleaned images
        """
        from skimage.morphology import remove_small_objects

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        result = np.zeros_like(images)
        for i, img in enumerate(images):
            # Binarize
            binary = img > 0.3
            # Remove small components
            cleaned = remove_small_objects(binary, min_size=min_size)
            # Restore intensities
            result[i] = cleaned * img

        return result.squeeze() if len(images) == 1 else result

    @staticmethod
    def combined_cleaning(images, method='aggressive'):
        """
        Combined cleaning pipeline

        Args:
            images: Images (N, 28, 28)
            method: 'light', 'medium', 'aggressive'

        Returns:
            Cleaned images
        """
        result = images.copy()

        if method == 'light':
            # Only a light thresholding
            result = ImageCleaner.threshold(result, threshold=0.2)

        elif method == 'medium':
            # Thresholding + remove small components
            result = ImageCleaner.threshold(result, threshold=0.25)
            result = ImageCleaner.remove_small_components(result, min_size=8)

        elif method == 'aggressive':
            # Full pipeline
            result = ImageCleaner.denoise_bilateral(result, sigma_spatial=1.0, sigma_intensity=0.1)
            result = ImageCleaner.threshold(result, threshold=0.3)
            result = ImageCleaner.remove_small_components(result, min_size=10)
            result = ImageCleaner.morphological_closing(result, iterations=1)

        # Normalize between 0 and 1
        if result.max() > 0:
            result = result / result.max()

        return result


class ImprovedRejectionSampling:
    """
    Improved rejection sampling with additional quality criteria
    """

    @staticmethod
    def compute_quality_metrics(images):
        """
        Compute quality metrics for each image

        Returns:
            dict with 'sparsity', 'contrast', 'connectivity'
        """
        metrics = {
            'sparsity': [],  # Proportion of active pixels
            'contrast': [],  # Standard deviation of intensities
            'connectivity': []  # Number of connected components
        }

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        for img in images:
            # Sparsity (we want between 10% and 30% active pixels)
            sparsity = (img > 0.1).sum() / img.size
            metrics['sparsity'].append(sparsity)

            # Contrast
            contrast = img.std()
            metrics['contrast'].append(contrast)

            # Connectivity (ideally 1 main component)
            binary = img > 0.3
            labeled, n_components = ndimage.label(binary)
            metrics['connectivity'].append(n_components)

        return metrics

    @staticmethod
    def is_good_quality(img, strict=False):
        """
        Determine whether an image is of good quality

        Args:
            img: Image (28, 28)
            strict: Stricter criteria

        Returns:
            bool
        """
        # Sparsity: between 10% and 40% active pixels
        sparsity = (img > 0.1).sum() / img.size
        if strict:
            if sparsity < 0.12 or sparsity > 0.35:
                return False
        else:
            if sparsity < 0.08 or sparsity > 0.45:
                return False

        # Sufficient contrast
        contrast = img.std()
        if strict:
            if contrast < 0.15:
                return False
        else:
            if contrast < 0.10:
                return False

        # Not too many components (artifacts)
        binary = img > 0.3
        labeled, n_components = ndimage.label(binary)
        if strict:
            if n_components > 3:  # Maximum 3 components
                return False
        else:
            if n_components > 5:  # Maximum 5 components
                return False

        return True


def process_training_data(X, method='standard'):
    """
    Preprocess training data before PCA

    Args:
        X: Data (N, 784)
        method: 'standard', 'normalized', 'whitened'

    Returns:
        X_processed, processor (to invert)
    """
    X_processed = X.copy()
    processor: Dict[str, Any] = {'method': method}

    if method == 'standard':
        # Just center
        mean = X.mean(axis=0)
        X_processed = X - mean
        processor['mean'] = mean

    elif method == 'normalized':
        # Center and normalize
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_processed = (X - mean) / std
        processor['mean'] = mean
        processor['std'] = std

    elif method == 'whitened':
        # Center and whiten (PCA whitening)
        mean = X.mean(axis=0)
        X_centered = X - mean

        # PCA to whiten
        pca_whitening = PCA(whiten=True, random_state=42)
        X_processed = pca_whitening.fit_transform(X_centered)

        processor['mean'] = mean
        processor['pca'] = pca_whitening

    return X_processed, processor
