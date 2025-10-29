"""
PCT Processor (PCA on image bands) - core logic.
"""

import numpy as np

class PCTProcessor:
    def __init__(self):
        self.stack = None
        self.orig_dtype = None
        self.H = self.W = self.B = 0
        self.X = None
        self.mean = None
        self.cov = None
        self.eigvals = None
        self.eigvecs = None
        self.scores = None
        self.pcs = None

    def load_stack(self, stack):
        self.orig_dtype = stack.dtype
        self.stack = stack
        self.H, self.W, self.B = self.stack.shape
        self.X = self.stack.reshape(self.H * self.W, self.B).astype(np.float32)
        self._reset_results()

    def _reset_results(self):
        self.mean = self.cov = self.eigvals = self.eigvecs = self.scores = self.pcs = None

    def compute_mean(self):
        self.mean = self.X.mean(axis=0, keepdims=True).astype(np.float32)

    def compute_covariance(self):
        M = self.X.shape[0]
        Xc = self.X - self.mean
        self.cov = (Xc.T @ Xc) / float(max(1, M - 1))
        self.cov = self.cov.astype(np.float32)
        return self.cov

    def compute_eigendecomposition(self):
        vals, vecs = np.linalg.eigh(self.cov)
        order = np.argsort(vals)[::-1]
        self.eigvals = vals[order].astype(np.float32)
        self.eigvecs = vecs[:, order].astype(np.float32)
        return self.eigvals, self.eigvecs

    def compute_scores_and_pcs(self):
        Xc = (self.X - self.mean).astype(np.float32)
        self.scores = Xc @ self.eigvecs
        self.scores = self.scores.astype(np.float32)
        self.pcs = self.scores.reshape(self.H, self.W, self.B)
        return self.pcs

    def compute_pca(self):
        if self.stack is None:
            raise ValueError("No image stack loaded.")
        self.compute_mean()
        self.compute_covariance()
        self.compute_eigendecomposition()
        self.compute_scores_and_pcs()

    def get_pc_image(self, index):
        if self.pcs is None:
            raise ValueError("PCA not computed.")
        if index < 0 or index >= self.B:
            raise IndexError("PC index out of range.")
        return self.pcs[:, :, index]

    def reconstruct(self, k):
        if self.scores is None or self.eigvecs is None or self.mean is None:
            raise ValueError("PCA not computed.")
        Sk = self.scores[:, :k]
        Vk = self.eigvecs[:, :k]
        Xc_rec = Sk @ Vk.T
        X_rec = (Xc_rec + self.mean).reshape(self.H, self.W, self.B)
        return X_rec.astype(np.float32)

    def compute_mse(self, reconstructed):
        mse = np.mean((self.stack.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        return float(mse)
