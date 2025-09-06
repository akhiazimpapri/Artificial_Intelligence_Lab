import os
import zipfile
import numpy as np
from PIL import Image
import re
import shutil

def prepare_mnist_like_npz(
    zip_path="datasets.zip",
    output_file="mnist_like.npz",
    img_size=28,                 # 28x28
    test_ratio=0.2,              # 20% test
    seed=42,                     # reproducible split
    normalize=False,             # True করলে [0,1] স্কেলে যাবে (float32)
    one_hot=False                # True করলে y one-hot হবে, num_classes=10
):
    """
    Convert a ZIP of digit images (0..9) into an MNIST-like .npz:
    -> (x_train, y_train, x_test, y_test)

    সমর্থিত লেবেল ফরম্যাট:
    - ফোল্ডার নামই লেবেল: 0/, 1/, ..., 9/
    - ফাইলনেমের শুরুতে digit: '3_img.png', '7-scan.jpg' ইত্যাদি
    """

    # ---------- Step 1: Extract ZIP ----------
    extract_to = "_temp_mnist_build"
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # ---------- Helpers ----------
    def try_get_label(root, fname):
        # 1) parent folder digit?
        basename = os.path.basename(root)
        if re.fullmatch(r"[0-9]", basename):
            return int(basename)

        # 2) filename-এর শুরুতে digit?
        m = re.match(r"^([0-9])", fname)
        if m:
            return int(m.group(1))

        return None  # not found

    def load_and_process_image(path):
        # গ্রেস্কেল + 28x28 resize
        img = Image.open(path).convert("L")
        # চাইলে আগে স্কয়ারে pad করতে পারো; এখানে ডাইরেক্ট resize
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # (28,28)
        return arr

    # ---------- Step 2: Walk & collect ----------
    images, labels = [], []
    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

    for root, _, files in os.walk(extract_to):
        for fname in files:
            if not fname.lower().endswith(supported_ext):
                continue

            label = try_get_label(root, fname)
            if label is None or not (0 <= label <= 9):
                # 0..9 ছাড়া কিছু পেলে স্কিপ
                # চাইলে এখানে warning প্রিন্ট করতে পারো
                continue

            fpath = os.path.join(root, fname)
            try:
                arr = load_and_process_image(fpath)
            except Exception as e:
                print(f"❌ Skipping {fpath}: {e}")
                continue

            images.append(arr)
            labels.append(label)

    if not images:
        raise ValueError("❌ কোনো বৈধ ইমেজ পাওয়া যায়নি (0–9 লেবেল দরকার)।")

    # ---------- Step 3: Arrayify (N,28,28,1) ----------
    X = np.stack(images, axis=0).astype(np.uint8)                 # (N,28,28)
    X = X.reshape(-1, img_size, img_size, 1)                      # (N,28,28,1)

    y = np.array(labels, dtype=np.int64)                          # (N,)

    # ---------- Optional: normalize / one-hot ----------
    if normalize:
        X = X.astype("float32") / 255.0

    if one_hot:
        num_classes = 10
        y_oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
        y_oh[np.arange(y.shape[0]), y] = 1.0
        y_out = y_oh
    else:
        y_out = y

    # ---------- Step 4: Train/Test split ----------
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    split = int(len(idx) * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]

    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y_out[train_idx], y_out[test_idx]

    # ---------- Step 5: Save ----------
    np.savez_compressed(
        output_file,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    # ---------- Info ----------
    print(f"✅ Saved: {output_file}")
    print("x_train:", x_train.shape, "| y_train:", y_train.shape)
    print("x_test :", x_test.shape,  "| y_test :", y_test.shape)
    if not one_hot:
        print("🔎 label unique:", np.unique(y))
    else:
        # show a quick summary of one-hot labels
        counts = np.sum(y_train, axis=0).astype(int)
        print("🔎 train label counts (0..9):", counts)

if __name__ == "__main__":
    # তোমার ফাইলের নাম 'datasets.zip' ধরেই কল করা হলো
    # একদম MNISTের মতো sparse label (ইন্টিজার) চাইলে এভাবেই রাখো:
    prepare_mnist_like_npz(
        zip_path="all_datasets.zip",
        output_file="mnist_like.npz",
        img_size=28,
        test_ratio=0.2,
        seed=42,
        normalize=False,   # Keras-এ দিলে পরে /255.0 করে নেবে
        one_hot=False      # sparse labels => loss='sparse_categorical_crossentropy'
    )

