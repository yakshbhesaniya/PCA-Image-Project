# PCT (Principal Component Transform) 

## What
A from-scratch implementation of Principal Component Transform (PCT) on multi-band satelite images.
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
- Displays all computed principal components.
- Displays reconstructed bands (based on selected k) alongside original bands.
- Shows MSE and eigenvalue summary in the interface.

## Notes
- No built-in PCA functions used (covariance computed manually).
- Allowed numerical helper: `numpy.linalg.eig` / `eigh` for eigenvalues.
