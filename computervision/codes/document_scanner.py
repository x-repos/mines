
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, feature, measure, transform, util
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

img_color = io.imread('input.jpg')              

H, W = img_color.shape[:2]

I = color.rgb2gray(img_color)
I_blur = filters.gaussian(I, sigma=1.0)
th = filters.threshold_otsu(I_blur)
mask = I_blur > th
mask = morphology.dilation(mask, morphology.square(7))

edges = feature.canny(I_blur, sigma=1.2, low_threshold=0.06, high_threshold=0.18)

filled = ndi.binary_fill_holes(edges | mask)
lab, nlab = ndi.label(filled)
if nlab == 0:
    raise RuntimeError("No regions found—check thresholds or lighting.")

sizes = ndi.sum(filled, lab, index=np.arange(1, nlab+1))
label_max = (np.argmax(sizes) + 1)
big = (lab == label_max)

# Get contour of the largest region (pick the longest one if multiple)
contours = measure.find_contours(big.astype(float), level=0.5)
contours.sort(key=lambda c: len(c), reverse=True)
cont = contours[0]  # (N, 2) in (row, col)

def approx_vertices(rc, tol):
    poly = measure.approximate_polygon(rc, tolerance=tol)
    # ensure closed polygon
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly

def force_n_vertices(rc, n=4, tol_lo=0.5, tol_hi=50.0, max_iter=30):
    """
    Binary search a tolerance for Douglas-Peucker to produce exactly n vertices.
    rc: (N,2) array in (row, col). Returns (n+1,2) closed polygon or None.
    """
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (tol_lo + tol_hi)
        poly = approx_vertices(rc, mid)
        # closed polygon has len = n+1
        k = len(poly) - 1
        if k == n:
            best = poly
            tol_lo = mid
        elif k > n:
            tol_lo = mid
        else:
            tol_hi = mid
    return best

quad = force_n_vertices(cont, n=4)
# Fallback: try convex hull if needed
if quad is None:
    hull = convex_hull_image(big)
    hull_contours = measure.find_contours(hull.astype(float), level=0.5)
    hull_contours.sort(key=lambda c: len(c), reverse=True)
    quad = force_n_vertices(hull_contours[0], n=4)
corners = quad[:-1]   
p = quad[:, [1, 0]]
s, d = p.sum(1), np.diff(p, 1).ravel()

names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
corners = [p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]]

for name, (x, y) in zip(names, corners):
    print(f"{name}: (x={x:.1f}, y={y:.1f})")


if quad is None:
    raise RuntimeError("Could not force exactly 4 vertices. Try different image or adjusting thresholds.")

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_color)
ax.plot(quad[:,1], quad[:,0], linewidth=3)  # (col=x, row=y)
ax.scatter(quad[:-1,1], quad[:-1,0], s=40)  # corner markers
ax.set_title("Forced 4-edge quadrilateral")
ax.axis('off')
plt.show()
