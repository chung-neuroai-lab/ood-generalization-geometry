import numpy as np
from sklearn.decomposition import PCA
import glue
from glue import glue_analysis
from glue.preprocess import downsample_manifolds

def geo_analysis(X, num_classes, M, rcond=1e-3):
    """
    Geometry analysis of feature representations.

    X: (num_samples, num_features) — consistent with stat_analysis convention
    num_classes: number of classes C
    M: number of samples per class
    rcond: relative cutoff for tiny eigenvalues in NC1 computation

    Returns dict with participation_ratio and neural_collapse metrics.

    """
    manifolds = [X[i*M:(i+1)*M].T for i in range(num_classes)]  # GLUE expects (n_features, n_points)
    sub_manifolds = downsample_manifolds(manifolds, n_points=min(50, M))
    glue_ret = glue_analysis(sub_manifolds)

    return {
        'participation_ratio': compute_participation_ratio(X),
        'neural_collapse': neural_collapse_1(X.T, num_classes, M, rcond),
        'dimension': glue_ret['dimension'],
        'radius': glue_ret['radius'],
        'utility': glue_ret['utility'],
    }


def compute_participation_ratio(features):
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_

    # Participation ratio formula: (sum(λ))^2 / sum(λ^2)
    total_var = np.sum(explained_variance)
    participation_ratio = (total_var ** 2) / np.sum(explained_variance ** 2)
    return participation_ratio


def neural_collapse_1(features, num_classes, M, rcond=1e-3):
    """
    Compute NC1 / zero-collapse = (1/C) * tr(S_W * S_B^+)
    using a truncated pseudo-inverse of S_B.

    features: (num_features, num_samples)
    num_classes: C
    M: samples per class
    rcond: relative cutoff for tiny eigenvalues of Sigma_B
    """
    X_dict = {c: features[:, M*c:M*(c+1)] for c in range(num_classes)}
    X_centered_dict = {c: X_dict[c] - X_dict[c].mean(axis=1, keepdims=True) for c in range(num_classes)}
    Sigma_dict = {c: X_centered_dict[c] @ X_centered_dict[c].T for c in range(num_classes)}
    Sigma_W = sum(Sigma_dict.values()) / (num_classes * M)

    mu_dict = {c: np.mean(X_dict[c], axis=1) for c in range(num_classes)}
    mu_G = np.mean(features, axis=1)
    Sigma_B = sum([np.outer(mu_dict[c] - mu_G, mu_dict[c] - mu_G) for c in range(num_classes)]) / num_classes

    # eigendecomposition of symmetric PSD matrix
    eigvals, eigvecs = np.linalg.eigh(Sigma_B)  # eigvals sorted ascending

    # relative threshold: keep only "meaningful" between-class directions
    lam_max = eigvals.max()
    tol = rcond * lam_max
    keep = eigvals > tol

    if not np.any(keep):
        # fully collapsed between-class structure; NC1 undefined / numerically 0
        return 0.0

    eigvals_trunc = eigvals[keep]
    eigvecs_trunc = eigvecs[:, keep]

    # Construct truncated pseudo-inverse: U_r Λ_r^{-1} U_r^T
    Sigma_B_pinv = (eigvecs_trunc / eigvals_trunc) @ eigvecs_trunc.T

    # Enforce symmetry against numerical noise
    Sigma_B_pinv = 0.5 * (Sigma_B_pinv + Sigma_B_pinv.T)

    # NC1 = (1/C) * tr(S_W S_B^+)
    nc1 = np.trace(Sigma_W @ Sigma_B_pinv) / num_classes
    return nc1
