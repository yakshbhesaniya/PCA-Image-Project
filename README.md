# PCT (Principal Component Transform) — Student Project

## What
A from-scratch implementation of Principal Component Transform (PCT) on multi-band images.
- Computes covariance (explicitly) and eigenvectors (numpy.linalg).
- Displays principal component images and reconstructs using top-k components.
- GUI for selecting files, previewing PCs, and saving outputs.

## Setup
1. Create virtualenv and activate.
2. `pip install -r requirements.txt`
3. `python -m src.main` or `./run.sh`

## Inputs
- Single multi-channel image (e.g., RGB) OR multiple single-band grayscale images (stacked in selection order).

## Output
- `PC_images/PC_01.png ...` (principal components)
- `reconstructed_*` (bands, RGB if applicable)
- `reconstructed_stack.npy` (float)
- `PCA_report.txt` (MSE, eigenvalues, config)

## Notes
- No built-in PCA functions used (covariance computed manually).
- Allowed numerical helper: `numpy.linalg.eig` / `eigh` for eigenvalues.
