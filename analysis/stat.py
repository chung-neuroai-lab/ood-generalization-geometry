import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def stat_analysis(X, threshold_scale=1e-3):
    """
    X: (num_samples, num_features)
    """
    # Activation sparsity
    eps = 1e-6
    sparsity = np.sum(np.abs(X) > eps) / (X.shape[0] * X.shape[1])

    # Covariance
    cov_matrix = compute_covariance_matrix(X)

    # Pairwise distances
    dist_matrix = compute_pairwise_distances(X)

    # Pairwise angles (in radians)
    angle_matrix = compute_feature_angles(X)

    return {
        'sparsity': sparsity,
        'mean_distance': np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]),
        'mean_angle': np.mean(angle_matrix[np.triu_indices_from(angle_matrix, k=1)]),
        'mean_covariance': np.mean(np.abs(cov_matrix[np.triu_indices_from(cov_matrix, k=1)])),
        'numerical_rank': compute_numerical_rank(X, threshold_scale=threshold_scale),
    }


def compute_covariance_matrix(features):
    return np.cov(features, rowvar=False)

def compute_pairwise_distances(features, metric='euclidean'):
    return pairwise_distances(features, metric=metric)

def compute_feature_angles(features):
    # Normalize each feature vector
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / np.clip(norms, a_min=1e-8, a_max=None)

    # Compute cosine similarity
    cosine_sim = np.dot(normed, normed.T)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Numerical stability

    # Convert to angles
    angle_matrix = np.arccos(cosine_sim)
    return angle_matrix

def compute_numerical_rank(features, threshold_scale=1e-3):
    # Compute PCA to access singular values (square roots of explained variance * (n_samples - 1))
    pca = PCA()
    pca.fit(features)

    # PCA explained variance = eigenvalues of covariance matrix = singular_values^2 / (n_samples - 1)
    # So singular values:
    singular_values = np.sqrt(pca.explained_variance_ * (features.shape[0] - 1))

    if singular_values.size == 0:
        return 0

    # Threshold = largest singular value * threshold_scale
    sigma1 = singular_values[0]
    threshold = sigma1 * threshold_scale

    # Count singular values above threshold
    numerical_rank = np.sum(singular_values >= threshold)

    return int(numerical_rank)
