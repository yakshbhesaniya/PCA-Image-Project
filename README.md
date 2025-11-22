# PCT (Principal Component Transform)

A from-scratch implementation of Principal Component Transform (PCT) on multi-band satellite images with a user-friendly GUI interface. This project manually computes the covariance matrix and generates principal components, using built-in functions only for eigenvalue/eigenvector computation.

## 📋 Overview

This project implements Principal Component Analysis (PCA) from scratch for multi-band satellite imagery:
- **Manual covariance computation** - Implemented using matrix multiplication formula
- **Built-in eigendecomposition** - Uses `numpy.linalg.eigh()` (permitted helper function)
- **Manual PC generation** - Principal components computed by transforming data with eigenvectors
- **Interactive GUI** - Visualize principal components and reconstructed images

## ✨ Features

- ✅ Manual covariance matrix computation (from scratch)
- ✅ Principal component visualization with thumbnails
- ✅ Image reconstruction using top-k principal components
- ✅ Mean Squared Error (MSE) calculation for reconstruction quality
- ✅ Support for multi-band TIFF files (e.g., Landsat satellite imagery)
- ✅ Support for multiple single-band images
- ✅ Side-by-side comparison of original vs reconstructed bands
- ✅ Eigenvalue summary and PCA report

## 🛠️ Requirements

- Python 3.7+
- Required packages (see `requirements.txt`):
  - `numpy` - Numerical computations and linear algebra
  - `imageio` - Reading various image formats
  - `Pillow` - Image processing and display
  - `tifffile` - Reading multi-band TIFF files
  - `scikit-image` - Image transformation utilities
  - `tkinter` - GUI framework (built-in with Python)

## 📦 Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Application

**Option 1: Using Python module**
```bash
python -m src.main
```

**Option 2: Using the shell script** (Linux/Mac)
```bash
./run.sh
```

### Workflow

1. **Select Files**: Click "Select Files" button to choose:
   - A single multi-band image file (e.g., GeoTIFF with multiple bands)
   - OR multiple single-band grayscale images

2. **Load & Compute PCA**: Click "Load & Compute PCA" to:
   - Load images into a 3D stack (Height × Width × Bands)
   - Compute covariance matrix manually
   - Perform eigendecomposition
   - Generate principal component images

3. **View Principal Components**: 
   - Thumbnail view of all PCs in the right panel
   - Click any thumbnail to see full-size PC
   - Eigenvalue report shown in the left panel

4. **Reconstruct Image**:
   - Enter `k` (number of components to use, e.g., 3)
   - Click "Reconstruct" to rebuild image using top-k PCs
   - View side-by-side comparison: Original vs Reconstructed
   - MSE value shown in the report

5. **Browse Bands**: Use the slider to browse through different bands in reconstruction mode

## 📁 Project Structure

```
SIP Project/
├── src/
│   ├── main.py              # Application entry point
│   ├── ui/
│   │   └── app.py           # GUI application class
│   └── pct/
│       ├── io.py            # Image input/output functions
│       ├── processor.py     # PCA/PCT computation logic
│       └── utils.py         # Utility functions for normalization
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── PROJECT_DOCUMENTATION.md # Detailed project documentation
```

## 🔬 Implementation Details

### Manual Covariance Computation
```python
# Computed manually using matrix multiplication
X_centered = X - mean
covariance = (X_centered^T @ X_centered) / (N - 1)
```

### Eigendecomposition
```python
# Using permitted built-in function
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
```

### Principal Component Generation
```python
# Manual computation by transforming data
principal_components = (X - mean) @ eigenvectors
```

### Reconstruction
```python
# Reconstruct using top-k components
X_reconstructed = (scores[:, :k] @ eigenvectors[:, :k]^T) + mean
```

## 📝 Supported Input Formats

- **Single Multi-band Image**:
  - GeoTIFF/TIFF files (e.g., Landsat, Sentinel satellite imagery)
  - RGB images (JPEG, PNG)
  
- **Multiple Single-band Images**:
  - Any number of grayscale images
  - Automatically resized to match dimensions if needed

## 📊 Output

- **Principal Component Images**: All computed PCs displayed as thumbnails
- **Reconstructed Bands**: Original vs reconstructed comparison
- **Eigenvalue Report**: Shows variance explained by each PC
- **MSE (Mean Squared Error)**: Quantifies reconstruction quality

## 📚 Documentation

For detailed documentation of every file, function, and implementation detail, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## ✅ Compliance

This implementation follows the requirement:
- ✅ **Covariance matrix**: Computed manually from scratch
- ✅ **Eigenvalues/Eigenvectors**: Uses permitted built-in function (`numpy.linalg.eigh`)
- ✅ **Principal Components**: Generated manually using matrix multiplication

No built-in PCA functions (e.g., `sklearn.decomposition.PCA`) or covariance functions (e.g., `np.cov`) are used.

## 🐛 Troubleshooting

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Image loading issues**: Check that image files are in supported formats (TIFF, JPEG, PNG)
- **Dimension mismatch**: Multi-file inputs are automatically resized to match the first file's dimensions

## 📄 License

This project is developed as part of the SIP (Semester Internship Program) course at IIT Bombay.

## 👤 Author
Yaksh Bhesaniya & Team
Developed for IIT Bombay SIP Project - Sem 1
