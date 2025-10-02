"""
PCT Processor (PCA on image bands) - core logic.
 - Computes mean, covariance (explicitly), eigen-decomposition (numpy allowed),
 - Projects: scores = X_centered @ eigenvectors
 - Produces principal component images and performs reconstruction using top-k components.
"""
import numpy as np

class PCTProcessor:
    def __init__(self):
        self.stack = None      # H x W x B (float64)
        self.H = self.W = self.B = 0
        self.X = None          # M x B
        self.mean = None       # 1 x B
        self.cov = None        # B x B
        self.eigvals = None
        self.eigvecs = None
        self.scores = None     # M x B
        self.pcs = None        # H x W x B

    def load_stack(self, stack):
        """
        stack: H x W x B numpy array (float64 recommended)
        """
        self.stack = stack.astype(np.float64)
        self.H, self.W, self.B = self.stack.shape
        self.X = self.stack.reshape(self.H * self.W, self.B)
        self._reset_results()

    def _reset_results(self):
        self.mean = None
        self.cov = None
        self.eigvals = None
        self.eigvecs = None
        self.scores = None
        self.pcs = None

    def compute_mean(self):
        self.mean = self.X.mean(axis=0, keepdims=True)  # shape (1, B)

    def compute_covariance(self):
        """
        Explicit covariance computation (do not use any PCA helper).
        C = (Xc^T Xc) / (M - 1)
        """
        M = self.X.shape[0]
        Xc = self.X - self.mean
        self.cov = (Xc.T @ Xc) / float(max(1, M - 1))
        return self.cov

    def compute_eigendecomposition(self):
        """
        Compute eigenvalues and eigenvectors of covariance matrix.
        Use numpy.linalg.eigh (stable for symmetric matrices).
        Sort in descending eigenvalue order.
        """
        vals, vecs = np.linalg.eigh(self.cov)  # vals ascending
        order = np.argsort(vals)[::-1]
        self.eigvals = vals[order]
        self.eigvecs = vecs[:, order]  # columns are eigenvectors (B x B)
        return self.eigvals, self.eigvecs

    def compute_scores_and_pcs(self):
        Xc = self.X - self.mean
        self.scores = Xc @ self.eigvecs   # M x B
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
        """
        Reconstruct using top-k components.
        Returns image of shape H x W x B (float).
        """
        if self.scores is None or self.eigvecs is None or self.mean is None:
            raise ValueError("PCA not computed.")
        if k < 1 or k > self.B:
            raise ValueError("k must be between 1 and B.")
        Sk = self.scores[:, :k]         # M x k
        Vk = self.eigvecs[:, :k]        # B x k
        Xc_rec = Sk @ Vk.T              # M x B
        X_rec = Xc_rec + self.mean      # M x B
        return X_rec.reshape(self.H, self.W, self.B)

    def compute_mse(self, reconstructed):
        """
        Mean Squared Error between original stack and reconstructed array (same shape).
        """
        mse = np.mean((self.stack - reconstructed) ** 2)
        return float(mse)
