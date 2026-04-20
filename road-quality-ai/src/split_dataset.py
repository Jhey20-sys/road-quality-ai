import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "dataset"
DEST_DIR = "dataset_split"

# Ratios must sum to 1.0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

classes = ["good", "satisfactory", "poor", "very_poor"]
valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# -----------------------------
# Create output folders
# -----------------------------
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# -----------------------------
# Split each class
# -----------------------------
for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.exists(class_path):
        print(f"❌ Folder not found: {class_path}")
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.endswith(valid_exts)
    ]

    print(f"\n📁 {cls}: {len(images)} images found")

    if len(images) == 0:
        print(f"⚠️ WARNING: No images in '{cls}' — skipping")
        continue

    random.shuffle(images)

    train_end = int(len(images) * TRAIN_RATIO)
    val_end = train_end + int(len(images) * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "train", cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "val", cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "test", cls, img)
        )

    print(
        f"✅ {cls}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} val | "
        f"{len(test_images)} test"
    )

print("\n Dataset splitting complete!")