# PCT Project - Complete Documentation

## Project Overview

This project implements **Principal Component Transform (PCT)** on multi-band satellite images from scratch. It computes PCA manually (covariance matrix calculation) and provides a GUI application for visualizing principal components and reconstructing images using top-k components.

**Main Features:**
- Computes covariance matrix explicitly (not using built-in PCA functions)
- Uses `numpy.linalg.eigh` for eigenvalue decomposition (only numerical helper allowed)
- GUI interface for file selection, PCA computation, visualization, and reconstruction
- Supports both single multi-channel images and multiple single-band images

---

## Project Structure

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
└── README.md               # Project overview
```

---

## Libraries Used

### Core Libraries

1. **numpy** - Numerical computations, array operations, linear algebra
   - Used in: `processor.py`, `io.py`, `utils.py`
   - Functions: `np.linalg.eigh`, `np.mean`, `np.asarray`, array operations

2. **tkinter** - GUI framework (built-in with Python)
   - Used in: `main.py`, `ui/app.py`
   - Components: Windows, frames, buttons, labels, canvas, file dialogs

3. **PIL (Pillow)** - Image processing and display
   - Used in: `ui/app.py`
   - Functions: `Image.fromarray()`, `ImageTk.PhotoImage()`, image conversion

4. **imageio** - Reading various image formats
   - Used in: `pct/io.py`
   - Functions: `imageio.v2.imread()` for reading images

5. **tifffile** - Reading TIFF files (especially multi-band TIFF)
   - Used in: `pct/io.py`
   - Functions: `tifffile.imread()` for reading GeoTIFF/satellite images

6. **scikit-image** - Image processing utilities
   - Used in: `pct/io.py`
   - Functions: `skimage.transform.resize()` for resizing mismatched images

---

## File-by-File Breakdown

### 1. `src/main.py` - Application Entry Point

**Purpose:** Initializes the GUI application and starts the main event loop.

**Lines Explained:**

```python
# entry point for the GUI app
from .ui.app import PCAApp
import tkinter as tk
```
- **Line 1:** Comment explaining this is the entry point
- **Line 2:** Imports the `PCAApp` class from the UI module (the main GUI application)
- **Line 3:** Imports `tkinter` as `tk` - Python's built-in GUI library

```python
def main():
    root = tk.Tk()
    # set a good window size
    root.geometry("1100x700")
    app = PCAApp(root)
    root.mainloop()
```
- **Line 5:** `main()` function - the entry point function
- **Line 6:** Creates the root Tkinter window (main application window)
- **Line 8:** Sets window size to 1100 pixels wide × 700 pixels tall
- **Line 9:** Creates an instance of `PCAApp` class, passing the root window
- **Line 10:** Starts the Tkinter event loop - keeps the window open and responds to events

```python
if __name__ == "__main__":
    main()
```
- **Lines 12-13:** Standard Python idiom - only runs `main()` if this file is executed directly (not imported)

---

### 2. `src/ui/app.py` - GUI Application Class

**Purpose:** Implements the complete GUI interface with file selection, PCA computation, visualization, and reconstruction features.

**Libraries Used:**
- `os` - File path operations
- `tkinter` - GUI widgets
- `PIL (Pillow)` - Image processing and display
- `numpy` - Array operations

#### Class: `PCAApp`

**Instance Variables:**
- `root` - Reference to the main Tkinter window
- `processor` - Instance of `PCTProcessor` for PCA computations
- `filepaths` - List of selected image file paths
- `orig_dtype`, `orig_min`, `orig_max` - Original image metadata
- `_thumb_refs` - List to keep references to thumbnail images (prevents garbage collection)
- `_big_ref` - Reference to the large displayed image
- `_current_mode` - Either "pc" (principal components) or "recon" (reconstruction)
- `rec_bands` - Reconstructed image bands
- `current_band_idx` - Currently displayed band index

#### Functions:

##### `__init__(self, root)`
**Lines 12-25:**
- Initializes the application
- Sets window title to "PCT"
- Creates a `PCTProcessor` instance
- Initializes all instance variables
- Calls `_build_ui()` to create the interface

##### `_build_ui(self)`
**Lines 28-78:** Builds the entire user interface

**Lines 29-38:** Top toolbar frame
- **Line 29:** Creates a top frame with padding
- **Line 30:** Packs it to the top, filling horizontally
- **Line 32:** "Select Files" button - opens file dialog
- **Line 33:** "Load & Compute PCA" button - loads images and computes PCA
- **Line 34:** Label for "k" input
- **Line 35-36:** Entry field for number of components (default: 3)
- **Line 38:** "Reconstruct" button - reconstructs using k components

**Lines 41-54:** Left panel (file list and report)
- **Line 41:** Creates a horizontal paned window (splittable layout)
- **Line 45:** Left frame (220px wide)
- **Line 48:** Listbox showing selected filenames
- **Line 50:** Label showing image dimensions (H, W, B)
- **Line 53:** Text widget showing PCA report (eigenvalues, MSE, etc.)

**Lines 57-66:** Center panel (large image display)
- **Line 60:** Canvas for displaying large PC/reconstruction images (dark background)
- **Line 63:** Slider for browsing reconstructed bands
- **Line 65:** Label showing current band number

**Lines 69-74:** Right panel (thumbnails)
- **Line 71:** Label title for thumbnails
- **Line 73:** Frame containing thumbnail images

**Lines 77-78:** Status bar at bottom showing current operation

##### `set_status(self, txt)`
**Lines 81-83:**
- Updates the status bar text
- `update_idletasks()` forces immediate GUI update

##### `select_files(self)`
**Lines 86-94:**
- Opens file dialog for selecting image files
- Updates the file list in the left panel
- Shows only filenames (not full paths)
- Updates status bar with number of selected files

##### `load_and_compute(self)`
**Lines 97-112:**
- Loads selected images into a stack (3D array: Height × Width × Bands)
- Computes PCA using the processor
- Updates UI with image dimensions and PC thumbnails
- Shows first principal component in large view
- **Line 102:** Calls `read_images_as_stack()` to load images (from `io.py`)
- **Line 103:** Loads the stack into the processor
- **Line 106:** Computes PCA (covariance, eigenvalues, eigenvectors, scores)
- **Line 108:** Updates thumbnail display
- **Line 109:** Shows PC1 in large view
- **Line 110:** Updates report with eigenvalues

##### `update_report(self, extra_text="")`
**Lines 114-122:**
- Generates a text report showing:
  - Number of bands
  - Eigenvalues for each principal component (descending order)
  - Additional text (e.g., MSE for reconstruction)
- Displays in the report text widget

##### `update_thumbnails(self)`
**Lines 125-140:**
- Creates thumbnail images for all principal components
- Arranges them in a 2-column grid
- Each thumbnail is clickable to show the full-size PC
- **Line 133:** Gets PC image from processor
- **Line 134:** Normalizes to 0-255 for display
- **Line 136:** Resizes to thumbnail size (120×80)
- **Line 138:** Creates button with thumbnail image

##### `show_large_pc(self, idx)`
**Lines 142-152:**
- Displays a single principal component in large view
- Centers the image on the canvas
- Scales to fit available canvas size
- **Line 143:** Gets PC image at index `idx`
- **Line 144:** Normalizes to uint8
- **Line 148:** Resizes to fit canvas
- **Line 150:** Displays on canvas

##### `reconstruct_and_display(self)`
**Lines 155-168:**
- Reconstructs original image using top-k principal components
- Computes Mean Squared Error (MSE)
- Switches UI to reconstruction mode
- **Line 157:** Gets k value from entry field
- **Line 162:** Calls processor to reconstruct using k components
- **Line 163:** Computes MSE between original and reconstructed
- **Line 166:** Switches display to show original vs reconstructed

##### `display_reconstructed_mode(self)`
**Lines 170-198:**
- Updates right panel to show side-by-side original vs reconstructed thumbnails
- Sets up slider for browsing bands
- Shows comparison for each band
- **Lines 180-192:** Creates thumbnails showing Original (left) vs Reconstructed (right) for each band

##### `_on_slider_move(self, event=None)`
**Lines 200-203:**
- Callback when slider is moved
- Updates display to show the selected band

##### `show_reconstructed_band(self, idx)`
**Lines 205-227:**
- Displays side-by-side comparison of original and reconstructed band
- Left half: Original band
- Right half: Reconstructed band
- **Lines 211-212:** Normalize both images to uint8
- **Lines 214-215:** Resize to fit half canvas width each
- **Lines 222-223:** Display both images side by side
- **Lines 224-225:** Add text labels above each image

---

### 3. `src/pct/io.py` - Image Input/Output

**Purpose:** Handles reading images from files and converting them into a standardized 3D array format (Height × Width × Bands).

**Libraries Used:**
- `numpy` - Array operations and data type handling
- `imageio` - Reading various image formats (JPEG, PNG, etc.)
- `tifffile` - Reading TIFF files (especially multi-band satellite images)
- `skimage.transform` - Resizing images when dimensions don't match

#### Functions:

##### `read_images_as_stack(filepaths)`
**Lines 11-56:** Main function for reading images

**Purpose:** Reads one or more image files and returns them as a 3D numpy array (stack).

**Parameters:**
- `filepaths`: List of file paths (strings)

**Returns:**
- `stack`: 3D numpy array of shape (H, W, B) as float32
- `orig_dtype`: Original numpy data type of the images
- `orig_min`: Minimum pixel value in the stack
- `orig_max`: Maximum pixel value in the stack

**Single File Mode (Lines 25-38):**
- If only one file is provided:
  - **Line 26:** Gets file extension (lowercase)
  - **Line 27:** If TIFF file, uses `tifffile.imread()` (handles multi-band TIFF)
  - **Line 30:** Otherwise uses `imageio.v2.imread()` (handles JPEG, PNG, etc.)
  - **Line 31:** Converts to numpy array
  - **Line 32:** If 2D (grayscale), adds a third dimension to make it 3D
  - **Line 34:** If already 3D, uses as-is
  - **Line 37:** Raises error for unsupported shapes (1D, 4D, etc.)

**Multiple Files Mode (Lines 39-53):**
- If multiple files are provided:
  - **Line 40:** Initialize list to store bands
  - **Line 41:** Reference shape for resizing mismatched images
  - **Line 42-51:** Loop through each file:
    - **Line 43:** Read image using imageio
    - **Line 44:** Convert to numpy array
    - **Line 45-46:** If RGB (3D), convert to grayscale by averaging channels
    - **Line 47-48:** Set reference shape from first image
    - **Line 49-50:** If dimensions don't match, resize to match reference
    - **Line 51:** Convert to float32 and add to bands list
  - **Line 52:** Stack all bands along third dimension (axis=2)

**Final Processing (Lines 55-56):**
- **Line 55:** Convert entire stack to float32 (required for PCA calculations)
- **Line 56:** Return stack, original dtype, min, and max values

---

### 4. `src/pct/processor.py` - PCA/PCT Computation Logic

**Purpose:** Implements the core Principal Component Analysis algorithm from scratch. Computes covariance matrix manually and performs eigenvalue decomposition.

**Libraries Used:**
- `numpy` - All numerical computations, array operations, linear algebra

#### Class: `PCTProcessor`

**Instance Variables:**
- `stack` - Original image stack (H × W × B)
- `orig_dtype` - Original data type
- `H, W, B` - Height, Width, Number of Bands
- `X` - Flattened data matrix (N × B) where N = H×W
- `mean` - Mean vector for each band (1 × B)
- `cov` - Covariance matrix (B × B)
- `eigvals` - Eigenvalues (B,) sorted in descending order
- `eigvecs` - Eigenvectors (B × B), columns are eigenvectors
- `scores` - PC scores (N × B), transformed data
- `pcs` - Principal component images (H × W × B)

#### Functions:

##### `__init__(self)`
**Lines 8-18:**
- Initializes all instance variables to `None` or zero
- No data loaded initially

##### `load_stack(self, stack)`
**Lines 20-25:**
- Loads image stack and prepares for PCA
- **Line 21:** Saves original data type
- **Line 22:** Stores the 3D stack
- **Line 23:** Extracts dimensions: Height, Width, Bands
- **Line 24:** Reshapes 3D stack to 2D matrix (N × B) where N = H×W
  - Each row is one pixel, each column is one band
- **Line 25:** Resets previous PCA results

##### `_reset_results(self)`
**Lines 27-28:**
- Clears all PCA computation results
- Called when new data is loaded

##### `compute_mean(self)`
**Lines 30-31:**
- Computes mean of each band (across all pixels)
- Result shape: (1 × B)
- Required for centering data (subtracting mean)

##### `compute_covariance(self)`
**Lines 33-38:**
- **Manually computes covariance matrix** (this is the "from scratch" part)
- **Line 34:** Gets number of pixels (M)
- **Line 35:** Centers data by subtracting mean
- **Line 36:** Computes covariance: `(X_centered^T @ X_centered) / (M-1)`
  - This is the manual computation (not using `np.cov`)
- **Line 37:** Ensures float32 type
- Returns the covariance matrix

**Mathematical Formula:**
```
Covariance = (X^T @ X) / (N-1)
where X is the centered data matrix
```

##### `compute_eigendecomposition(self)`
**Lines 40-45:**
- Computes eigenvalues and eigenvectors of covariance matrix
- **Line 41:** Uses `np.linalg.eigh()` (only numerical helper allowed)
  - `eigh` is for symmetric matrices (covariance is always symmetric)
- **Line 42:** Sorts eigenvalues in descending order (largest first)
- **Line 43-44:** Reorders eigenvalues and eigenvectors accordingly
- Returns sorted eigenvalues and eigenvectors

**Note:** Eigenvectors define the principal component directions

##### `compute_scores_and_pcs(self)`
**Lines 47-52:**
- Projects centered data onto principal component space
- **Line 48:** Centers the data (subtract mean)
- **Line 49:** Projects: `scores = X_centered @ eigenvectors`
  - This transforms data to PC space
- **Line 51:** Reshapes scores back to image dimensions (H × W × B)
- Returns principal component images

**Mathematical Formula:**
```
Scores = (X - mean) @ Eigenvectors
```

##### `compute_pca(self)`
**Lines 54-60:**
- Main function that runs the complete PCA pipeline
- Calls all steps in order:
  1. Compute mean
  2. Compute covariance (manually)
  3. Compute eigendecomposition
  4. Compute scores and PC images
- Raises error if no data is loaded

##### `get_pc_image(self, index)`
**Lines 62-67:**
- Returns a single principal component image (2D array)
- **Line 65:** Validates index is in valid range
- Returns the PC at the specified index (0 = PC1, 1 = PC2, etc.)

##### `reconstruct(self, k)`
**Lines 69-76:**
- Reconstructs original image using only top-k principal components
- **Line 72:** Takes only first k scores (columns)
- **Line 73:** Takes only first k eigenvectors (columns)
- **Line 74:** Reconstructs: `X_rec = Scores_k @ Eigenvectors_k^T`
- **Line 75:** Adds mean back and reshapes to image dimensions
- Returns reconstructed image stack

**Mathematical Formula:**
```
Reconstruction = (Scores_k @ Eigenvectors_k^T) + mean
```

##### `compute_mse(self, reconstructed)`
**Lines 78-80:**
- Computes Mean Squared Error between original and reconstructed
- **Line 79:** MSE = mean((original - reconstructed)²)
- Used to measure reconstruction quality
- Lower MSE = better reconstruction

---

### 5. `src/pct/utils.py` - Utility Functions

**Purpose:** Provides helper functions for image normalization and data type conversion, especially for display purposes.

**Libraries Used:**
- `numpy` - Array operations, data type information

#### Functions:

##### `normalize_to_uint8(img)`
**Lines 8-18:**
- Normalizes a single-channel (2D) image to 0-255 range (uint8)
- Required for displaying images (most display functions expect uint8)

**Process:**
1. **Line 13:** Finds minimum and maximum values (ignoring NaN)
2. **Line 15:** If image is constant (min == max), returns all zeros
3. **Line 17:** Scales values to 0-1 range: `(img - min) / (max - min)`
4. **Line 18:** Multiplies by 255, clips to 0-255, converts to uint8

**Used in:** Display functions to convert float PC images to displayable format

##### `stack_to_uint8_images(stack)`
**Lines 21-31:**
- Converts a 3D stack (H × W × B) to a list of uint8 images
- Normalizes each band independently
- **Lines 28-30:** Loops through each band, normalizes separately
- Returns list of 2D uint8 arrays

##### `float_stack_to_scaled_uint8(stack, orig_min=None, orig_max=None)`
**Lines 34-52:**
- Converts float stack to uint8 with optional global scaling

**Two modes:**
1. **Global scaling (Lines 43-47):** If `orig_min` and `orig_max` provided
   - Uses same scale for all bands (preserves relative brightness)
   - Formula: `(value - orig_min) / (orig_max - orig_min) * 255`

2. **Per-band scaling (Lines 49-50):** If no min/max provided
   - Each band normalized independently (better contrast per band)
   - Uses `normalize_to_uint8()` for each band

##### `float_stack_to_dtype(stack_float, dtype)`
**Lines 55-66:**
- Safely converts float array to specified data type
- **Lines 60-62:** For integer types, clips to valid range before casting
- **Lines 63-65:** For float types, clips to valid range
- Prevents overflow/underflow errors

---

## How Everything Works Together

### 1. **Application Startup**
```
main.py → Creates Tkinter window → Initializes PCAApp → Builds UI
```

### 2. **User Workflow**

**Step 1: Select Files**
- User clicks "Select Files" button
- `select_files()` opens file dialog
- File paths stored in `self.filepaths`
- Filenames shown in listbox

**Step 2: Load & Compute PCA**
- User clicks "Load & Compute PCA"
- `load_and_compute()` is called:
  - Calls `read_images_as_stack()` to load images into 3D array
  - Calls `processor.load_stack()` to store data
  - Calls `processor.compute_pca()`:
    1. Computes mean of each band
    2. Computes covariance matrix (manually)
    3. Computes eigenvalues/eigenvectors
    4. Projects data to PC space
  - Updates UI with thumbnails and large view

**Step 3: View Principal Components**
- Thumbnails shown in right panel
- Clicking thumbnail shows full-size PC
- Report shows eigenvalues

**Step 4: Reconstruct (Optional)**
- User enters k (number of components)
- Clicks "Reconstruct"
- `reconstruct_and_display()`:
  - Calls `processor.reconstruct(k)` using top-k components
  - Computes MSE
  - Switches UI to show Original vs Reconstructed comparison
  - Slider allows browsing through bands

### 3. **Data Flow**

```
Image Files → io.py (read_images_as_stack)
           → 3D Stack (H×W×B) float32
           → processor.py (load_stack)
           → Flattened Matrix (N×B) where N=H×W
           → compute_pca():
               - Mean vector (1×B)
               - Covariance matrix (B×B) [MANUALLY COMPUTED]
               - Eigenvalues (B,)
               - Eigenvectors (B×B)
               - PC Scores (N×B)
               - PC Images (H×W×B)
           → UI Display:
               - normalize_to_uint8() converts to displayable format
               - PIL converts to PhotoImage for tkinter
               - Canvas displays image
```

---

## Key Mathematical Concepts

### Principal Component Analysis (PCA)

1. **Mean Centering:** `X_centered = X - mean`
   - Shifts data so it's centered at origin

2. **Covariance Matrix:** `Cov = (X_centered^T @ X_centered) / (N-1)`
   - Measures how bands vary together
   - This is computed **manually** (not using `np.cov`)

3. **Eigendecomposition:** `Cov @ eigenvector = eigenvalue @ eigenvector`
   - Eigenvectors = directions of maximum variance (principal components)
   - Eigenvalues = variance explained by each PC
   - Uses `np.linalg.eigh()` (only numerical helper)

4. **Transformation:** `Scores = X_centered @ Eigenvectors`
   - Projects data onto principal component space
   - First PC has most variance, second has second-most, etc.

5. **Reconstruction:** `X_rec = (Scores_k @ Eigenvectors_k^T) + mean`
   - Reconstructs using only top-k components
   - Lower k = more compression, higher error

---

## Important Notes

1. **From Scratch Implementation:** Covariance is computed manually, not using built-in functions
2. **Only Numerical Helper:** Only `numpy.linalg.eigh` is used from numerical libraries
3. **Image Formats:** Supports TIFF (multi-band satellite), JPEG, PNG, and multiple grayscale files
4. **Data Types:** All computations use float32 for consistency
5. **Display:** Images normalized to 0-255 (uint8) for display purposes
6. **Memory:** Large images are flattened to 2D for efficient computation

---

## Summary

This project implements a complete PCA pipeline for multi-band images:
- **Input:** Single multi-band image or multiple single-band images
- **Processing:** Manual covariance computation + eigendecomposition
- **Output:** Principal component images + reconstruction using k components
- **Interface:** User-friendly GUI for visualization and interaction

All core computations are done from scratch, making it educational and transparent about the PCA algorithm's inner workings.

