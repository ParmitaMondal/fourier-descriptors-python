# Fourier Descriptors for Shape Boundaries (Python)

This project implements **Fourier descriptors** to represent and reconstruct 2D shape boundaries.  
It follows the classic workflow:

1. Extract the **outer boundary** of a binary object.
2. **Uniformly resample** the boundary to a fixed number of points.
3. Compute **Fourier descriptors** (with optional translation/scale/rotation invariance).
4. Reconstruct the boundary using:
   - **50%** of descriptors
   - **1%** of descriptors  
   and visualize the results.

## Features
- Outer boundary extraction via OpenCV contours
- Arc-length **uniform resampling** to N boundary points
- Descriptor normalization for **translation, scale, and rotation** invariance
- Inverse reconstruction using a centered low-frequency passband
- Side-by-side plots of original boundary vs. 50% & 1% reconstructions

## Files
- `fourier_descriptors.py` — main script (compute + reconstruct + visualize)
- `requirements.txt` — Python dependencies

## Quick Start
```bash
git clone https://github.com/<your-username>/fourier-descriptors-python.git
cd fourier-descriptors-python
pip install -r requirements.txt
