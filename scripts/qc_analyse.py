# reads images under data/images/**.jpg, computes:
# - Variance of Laplacian (blur)
# - High-frequency energy (FFT proportion)
# - RMS contrast
# saves qc_metrics.csv with columns: path, blur_var, hp_energy, contrast

# EDA
# Importing of Libraries needed
import matplotlib.pyplot as plt
from PIL import Image
import random, pandas as pd
from collections import Counter
from IPython.display import display

# Quick visual sanity‑check by randomly plotting images from each class.

def show_random_grid(n: int = 12):
    """Display an n‑image grid of randomly sampled ultrasound frames."""
    sample = random.sample(image_paths, n)
    plt.figure(figsize=(12, 6))
    for i, p in enumerate(sample):
        plt.subplot(3, 4, i + 1)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.title(p.split(os.sep)[-2], fontsize=8)
        plt.axis("off")
    plt.tight_layout()

show_random_grid()

# Review of Image resolution distribution
# Collect (width, height) for every image in the dataset
sizes = [Image.open(p).size for p in image_paths]  # (w, h)
size_counts = Counter(sizes)

# Tabulate the most frequent resolutions
resolution_df = pd.DataFrame(
    [{"width": w, "height": h, "count": c} for (w, h), c in size_counts.items()]
).sort_values("count", ascending=False)

print("\nTop 10 image resolutions (w × h):")
display(resolution_df.head(10))

# Creating a 2‑D histogram (heat‑map) of all widths vs heights
widths  = [w for (w, _) in sizes]
heights = [h for (_, h) in sizes]

plt.figure(figsize=(6, 5))
plt.hist2d(widths, heights, bins=[30, 30])
plt.xlabel("Width (px)")
plt.ylabel("Height (px)")
plt.title("Resolution distribution across dataset")
plt.colorbar(label="Number of images")
plt.tight_layout()
plt.show()

# Creating a Bubble plot
top_n = 10

# Count (width, height) pairs and take the most common
pairs = Counter(sizes).most_common(top_n)
w_top  = [wh[0][0] for wh in pairs]    # widths
h_top  = [wh[0][1] for wh in pairs]    # heights
counts = [wh[1]     for wh in pairs]   # frequencies

plt.figure(figsize=(8,6))
plt.scatter(w_top, h_top, s=[c*6 for c in counts], alpha=0.6)  # bubble size / count
for w, h, c in zip(w_top, h_top, counts):
    plt.text(w, h, str(c), ha="center", va="center", fontsize=8)

plt.xlabel("Width (px)")
plt.ylabel("Height (px)")
plt.title(f"Top-{top_n} resolutions (bubble size = #images)")
plt.grid(True, linestyle="--", linewidth=0.3)
plt.show()

# Grayscale vs. Colour analysis
import textwrap
# Scanning of every image (or sample, if ~10 k feels slow)
scan_all = True        # flip to False to sample 2 000 files for speed
sample_size = 2_000

to_scan = image_paths if scan_all else random.sample(image_paths, sample_size)

mode_counter   = Counter()   # e.g. "L", "RGB", "RGBA"
pseudo_gray    = 0           # RGB triplets where all channels are equal
failed_reads   = 0

for p in tqdm.tqdm(to_scan, desc="Scanning images"):
    try:
        with Image.open(p) as im:
            mode_counter[im.mode] += 1
            # Check for 3-channel “gray masquerading as colour”
            if im.mode == "RGB":
                arr = np.asarray(im)
                if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]):
                    pseudo_gray += 1
    except Exception as e:
        failed_reads += 1

# Report Creation
total = len(to_scan)
print("\nImage mode counts:")
for m, c in mode_counter.most_common():
    print(f"  {m:<4} : {c}  ({100*c/total:.1f} %)")

if pseudo_gray:
    print(f"\n  From the {mode_counter['RGB']} RGB files, {pseudo_gray} "
          f"({100*pseudo_gray/max(1, mode_counter['RGB']):.1f} %) are pseudo-grayscale "
          "(all three channels identical).")

if failed_reads:
    print(f"\n{failed_reads} files could not be read.")

# Decision helper
msg = []
if mode_counter.get("RGB", 0) - pseudo_gray == 0 and mode_counter.get("RGBA", 0) == 0:
    msg.append("All images are effectively single-channel,thus you can safely convert to 1-channel tensors.")
else:
    msg.append("There are genuine colour images,hence converting to grayscale will lose information.")
    if pseudo_gray:
        msg.append("However, many RGBs are pseudo-grayscale, meaning  you could auto-detect and drop extra channels only for those.")

print("\n" + textwrap.fill(" ".join(msg), width=100))

# Artefact / Noise Analysis
# Importing additionally needed libraries
import cv2
from PIL import ImageOps

def analyse_one(path):
    """Returns blur_var, hp_energy, contrast for a single image."""
    im  = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)

    # Blur: variance of Laplacian
    blur_var = cv2.Laplacian(arr, cv2.CV_32F).var()

    # High-frequency energy: proportion of FFT power outside centre
    f  = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(f)
    h, w = mag.shape
    centre = mag[h//4:3*h//4, w//4:3*w//4]
    hp_energy = 100 * (mag.sum() - centre.sum()) / mag.sum()

    # RMS contrast
    contrast = arr.std()
    return blur_var, hp_energy, contrast

# Scan
scan_lim   = None          # None = scan all, or set an int (e.g. 3000) to sample for speed
paths_scan = image_paths if scan_lim is None else random.sample(image_paths, scan_lim)

records = []
for p in tqdm.tqdm(paths_scan, desc="Analysing"):
    records.append((p, *analyse_one(p)))

df_qc = pd.DataFrame(records, columns=["path", "blur_var", "hp_energy", "contrast"])

# Visualising the results in Plots
plt.figure(figsize=(16,4))

# 1. Blur
plt.subplot(1,4,1)
plt.hist(df_qc.blur_var, bins=40)
plt.xlabel("Variance of Laplacian"); plt.title("Blur")
plt.axvline(df_qc.blur_var.median(), color="r", linestyle="--")

# 2. High-freq energy
plt.subplot(1,4,2)
plt.hist(df_qc.hp_energy, bins=40)
plt.xlabel("High-freq energy (%)"); plt.title("Noise")
plt.axvline(df_qc.hp_energy.median(), color="r", linestyle="--")

# 3. Contrast
plt.subplot(1,4,3)
plt.hist(df_qc.contrast, bins=40)
plt.xlabel("RMS contrast"); plt.title("Contrast")
plt.axvline(df_qc.contrast.median(), color="r", linestyle="--")

# 4. Blur vs. noise scatter plot
plt.subplot(1,4,4)
plt.scatter(df_qc.blur_var, df_qc.hp_energy, alpha=0.3, s=8)
plt.xlabel("Blur (↓)"); plt.ylabel("High-freq energy (↑)")
plt.title("Quality landscape")

plt.tight_layout(); plt.show()

# Show worst offenders
n_show = 6
worst = (df_qc.blur_var < df_qc.blur_var.quantile(0.05)) | (df_qc.hp_energy > df_qc.hp_energy.quantile(0.95))
bad_paths = df_qc[worst].sort_values(["blur_var", "hp_energy"]).head(n_show).path.tolist()

plt.figure(figsize=(11,4))
for i, p in enumerate(bad_paths):
    plt.subplot(1, n_show, i+1)
    plt.imshow(ImageOps.equalize(Image.open(p).convert("L")), cmap="gray")
    plt.title(f"{i+1}", fontsize=8); plt.axis("off")
plt.suptitle("Sample low-quality frames", y=1.05)
plt.show()
