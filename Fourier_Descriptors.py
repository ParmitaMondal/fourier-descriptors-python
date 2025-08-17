"""
Fourier Descriptors for 2D shape boundaries (translation/scale/rotation robust options).

Pipeline:
1) Load an image, make a binary mask, extract the OUTER boundary.
2) Uniformly resample the boundary to N points.
3) Compute Fourier descriptors z = fourierdescp(S).
4) Reconstruct the boundary with a limited number of descriptors:
   - 50% of descriptors
   - 1% of descriptors
5) Display original and reconstructions.

Dependencies: numpy, opencv-python, matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def largest_external_contour(binary_0_255):
    """Return the largest external contour as Nx2 (x,y) int array."""
    cnts, _ = cv2.findContours(binary_0_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No contours found.")
    c = max(cnts, key=cv2.contourArea).squeeze()
    if c.ndim != 2:
        c = c.reshape(-1, 2)
    return c.astype(np.float64)


def uniform_resample_polyline(points_xy, n=256, closed=True):
    """
    Uniformly resample an ordered polyline to n points using arc-length parameterization.
    points_xy: Nx2 (x,y)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if closed and (pts[0] != pts[-1]).any():
        pts = np.vstack([pts, pts[0]])

    # cumulative arc length
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)

    s_new = np.linspace(0, total, n, endpoint=not closed)

    # linear interpolation for x and y independently
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    res = np.stack([x, y], axis=1)

    if closed:
        # ensure closed
        if (res[0] != res[-1]).any():
            res[-1] = res[0]
    return res


# -----------------------------
# Fourier descriptors
# -----------------------------
def fourierdescp(S, normalize=True, invariances=("translate", "scale", "rotation")):
    """
    Compute Fourier descriptors from an ordered boundary S (Nx2).
    Returns complex spectrum Z of length N.

    invariances:
      - "translate": subtract spatial mean (remove DC in coordinates)
      - "scale": divide by L2 norm of first nonzero harmonic magnitude
      - "rotation": make rotation-invariant by removing global phase (align first harmonic)
    """
    S = np.asarray(S, dtype=np.float64)
    # complex sequence (x + i y)
    z = S[:, 0] + 1j * S[:, 1]

    if normalize and ("translate" in invariances):
        z = z - z.mean()

    Z = np.fft.fft(z)

    if normalize:
        # Scale invariance: normalize spectrum magnitude by first nonzero harmonic
        if "scale" in invariances:
            # find first non-zero harmonic (skip k=0)
            mags = np.abs(Z)
            idxs = np.where(mags > 1e-12)[0]
            idxs = idxs[idxs != 0]
            if len(idxs) > 0:
                Z = Z / mags[idxs[0]]

        # Rotation invariance: remove global phase by aligning first harmonic to real axis
        if "rotation" in invariances:
            # pick first nonzero harmonic k>0 and rotate spectrum
            mags = np.abs(Z)
            idxs = np.where(mags > 1e-12)[0]
            idxs = idxs[idxs != 0]
            if len(idxs) > 0:
                k1 = idxs[0]
                phi = np.angle(Z[k1])
                Z = Z * np.exp(-1j * phi)

    return Z


def ifourierdescp(Z, nd=None):
    """
    Inverse Fourier descriptors using only nd lowest frequencies (centered passband).
    Z: complex spectrum (length N)
    nd: number of descriptors to retain (1..N). If None, use all.

    Returns Nx2 array of reconstructed boundary (x,y).
    """
    N = len(Z)
    if nd is None or nd >= N:
        z_rec = np.fft.ifft(Z)
        return np.column_stack([z_rec.real, z_rec.imag])

    # keep 'nd' centered low-frequency components in FFT-shifted spectrum
    Zs = np.fft.fftshift(Z.copy())
    keep = nd // 2
    Zs[:(N // 2 - keep)] = 0
    Zs[(N // 2 + keep + (nd % 2)):] = 0
    Z_kept = np.fft.ifftshift(Zs)

    z_rec = np.fft.ifft(Z_kept)
    return np.column_stack([z_rec.real, z_rec.imag])


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    # ---- 1) Load and get boundary as binary ----
    # Replace 'shape.png' with your image path (grayscale or RGB).
    path = "shape.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # Smooth (optional) + Otsu threshold
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick the largest outer boundary
    contour = largest_external_contour(bw)  # Nx2

    # Visualize binary + boundary
    vis1 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis1, [contour.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)

    # ---- 2) Resample and compute Fourier descriptors ----
    N = 256  # number of boundary samples
    S = uniform_resample_polyline(contour, n=N, closed=True)  # Nx2

    Z = fourierdescp(S, normalize=True,
                     invariances=("translate", "scale", "rotation"))

    # ---- 3) Inverse using 50% and 1% of descriptors ----
    nd_50 = max(2, N // 2)                # 50%
    nd_1 = max(2, int(np.ceil(0.01 * N))) # 1%

    S_50 = ifourierdescp(Z, nd=nd_50)
    S_01 = ifourierdescp(Z, nd=nd_1)

    # Bring reconstructions back to image coordinates
    # If you normalized for translation/scale/rotation, they’re already aligned.
    # For visualization, we’ll translate to positive quadrant.
    def to_int_canvas(pts, margin=5):
        pts = pts.copy()
        pts -= pts.min(axis=0)  # translate to start at (0,0)
        pts += margin
        return pts.astype(np.int32)

    S_i = to_int_canvas(S)
    S_50i = to_int_canvas(S_50)
    S_01i = to_int_canvas(S_01)

    # Rasterize reconstructions
    H, W = bw.shape
    canvas50 = np.zeros((H, W), np.uint8)
    cv2.polylines(canvas50, [S_50i.reshape(-1, 1, 2)], True, 255, 1)

    canvas01 = np.zeros((H, W), np.uint8)
    cv2.polylines(canvas01, [S_01i.reshape(-1, 1, 2)], True, 255, 1)

    # ---- Display ----
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(vis1[..., ::-1])
    plt.title("Original boundary")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(canvas50, cmap="gray")
    plt.title(f"Reconstruction with {nd_50}/{N} (50%) descriptors")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(canvas01, cmap="gray")
    plt.title(f"Reconstruction with {nd_1}/{N} (1%) descriptors")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
