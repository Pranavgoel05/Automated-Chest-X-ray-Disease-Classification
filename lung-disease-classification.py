import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from os import makedirs
from os.path import join, exists, abspath
import matplotlib.pyplot as plt
from itertools import chain

# TensorFlow / Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
DATA_DIR = r"a:\UNIV\AI PROJECT\data"           # folder you attached (contains Data_Entry_2017.csv)
CSV_PATH = join(DATA_DIR, "Data_Entry_2017.csv") 
IMAGES_DIR = join(DATA_DIR, "images")            # will contain downloaded images
KAGGLE_DATASET = "nih-chest-xrays/data"          # dataset slug for kaggle api
DOWNLOAD_COUNT = 200                              # how many images to download (set small for testing)
RANDOM_STATE = 2025
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ---------- KAGGLE DOWNLOAD (optional) ----------
def ensure_kaggle_and_download(sample_image_ids):
    """
    Uses kaggle API to download the specific image files listed in sample_image_ids.
    Requires ~/.kaggle/kaggle.json to be present.
    Downloads files into IMAGES_DIR as <Image Index>.png
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("kaggle package not installed. Install it with: pip install kaggle")
        raise

    kaggle_json = os.path.expanduser(join("~", ".kaggle", "kaggle.json"))
    if not exists(kaggle_json):
        raise FileNotFoundError(f"kaggle.json not found. Place your kaggle credentials at: {kaggle_json}")

    api = KaggleApi()
    api.authenticate()

    if not exists(IMAGES_DIR):
        makedirs(IMAGES_DIR, exist_ok=True)

    print(f"Downloading {len(sample_image_ids)} images to {abspath(IMAGES_DIR)} ...")
    for i, img_name in enumerate(sample_image_ids, start=1):
        # Images inside the dataset are in folders images_001/... etc, but the API expects the path inside the dataset.
        # We'll attempt to find the file path by trying typical images_XXX folders. If that fails, skip.
        downloaded = False
        for folder_idx in range(1, 14):  # NIH dataset has multiple images_00x folders; try a reasonable range
            internal_path = f"images_{folder_idx:03d}/{img_name}"
            try:
                # dataset_download_file will download a zip by default on some versions; use force and unzip=False and then rename
                api.dataset_download_file(KAGGLE_DATASET, file_name=internal_path, path=IMAGES_DIR, force=False)
                # the API saves a zip file named like <internal_path>.zip; extract or rename if necessary
                zip_path = join(IMAGES_DIR, os.path.basename(internal_path) + ".zip")
                # Newer API directly saves file; check for png path
                saved_png = join(IMAGES_DIR, img_name)
                if exists(saved_png):
                    downloaded = True
                    break
                # If zip exists, extract
                if exists(zip_path):
                    import zipfile
                    with zipfile.ZipFile(zip_path, "r") as z:
                        # the zip usually contains the png with same basename
                        for zinfo in z.infolist():
                            if zinfo.filename.endswith(".png"):
                                z.extract(zinfo, IMAGES_DIR)
                    os.remove(zip_path)
                    if exists(saved_png):
                        downloaded = True
                        break
            except Exception:
                # try next folder
                continue
        if not downloaded:
            # If file not found in tried folders, inform and continue
            print(f"[{i}/{len(sample_image_ids)}] WARNING: could not download {img_name} (skipping)")
        else:
            if i % 50 == 0 or i == len(sample_image_ids):
                print(f"[{i}/{len(sample_image_ids)}] downloaded")

# ---------- MAIN WORKFLOW ----------
def main(download_needed=True):
    if not exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print("CSV loaded. Total records:", df.shape[0])

    # Keep only rows that have image filename
    df = df[df['Image Index'].notna()].copy()

    # If images folder already populated, we will map them directly
    existing_images = glob(join(IMAGES_DIR, "*.png"))
    if len(existing_images) == 0:
        print("No local images found in", IMAGES_DIR)
        if download_needed:
            # sample images to download (balanced by top label frequency)
            # We'll sample by most common labels to ensure variety
            df['PrimaryLabel'] = df['Finding Labels'].map(lambda x: x.split('|')[0] if isinstance(x, str) and len(x) else "NoFinding")
            # choose SAMPLE based on label stratification
            sample_df = df.groupby('PrimaryLabel', group_keys=False).apply(lambda d: d.sample(frac=1.0, random_state=RANDOM_STATE))
            sample_df = sample_df.reset_index(drop=True)
            sample_df = sample_df.head(DOWNLOAD_COUNT)
            sample_image_ids = sample_df['Image Index'].tolist()
            # call kaggle download
            ensure_kaggle_and_download(sample_image_ids)
        else:
            print("Set download_needed=True to attempt downloads via Kaggle API. Exiting.")
            sys.exit(0)
    else:
        print(f"Found {len(existing_images)} local images in {IMAGES_DIR}")

    # Build image path mapping and filter dataframe to available files
    image_map = {os.path.basename(p): p for p in glob(join(IMAGES_DIR, "**", "*.png"), recursive=True)}
    print("Mapped images found:", len(image_map))

    df['path'] = df['Image Index'].map(image_map.get)
    df = df[df['path'].notna()].copy()
    print("Records with available image files:", df.shape[0])
    if df.shape[0] == 0:
        print("No images available after download/mapping. Exiting.")
        sys.exit(1)

    # create multi-label columns
    df['Finding Labels'] = df['Finding Labels'].astype(str)
    df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', '').strip())
    labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    labels = [l for l in labels if len(l)]
    print("Discovered labels:", labels)

    for lbl in labels:
        df[lbl] = df['Finding Labels'].map(lambda x: 1.0 if lbl in x else 0.0)

    # create vector column for flow_from_dataframe y
    df['disease_vec'] = df.apply(lambda row: row[labels].values, axis=1).tolist()

    # split
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print("Train/Valid sizes:", train_df.shape[0], valid_df.shape[0])

    # data generators
    idg = ImageDataGenerator(samplewise_center=True,
                             samplewise_std_normalization=True,
                             horizontal_flip=True)

    def flow_from_df(gen, df_in, shuffle=True, batch_size=BATCH_SIZE):
        return gen.flow_from_dataframe(df_in,
                                       x_col='path',
                                       y_col='disease_vec',
                                       target_size=IMG_SIZE,
                                       batch_size=batch_size,
                                       class_mode='raw',
                                       shuffle=shuffle)

    train_gen = flow_from_df(idg, train_df, shuffle=True, batch_size=BATCH_SIZE)
    valid_gen = flow_from_df(idg, valid_df, shuffle=False, batch_size=BATCH_SIZE)

    # quick sanity: show a batch
    x_batch, y_batch = next(train_gen)
    print("Batch shapes:", x_batch.shape, y_batch.shape)

    # model (light)
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = Flatten()(base.output)
    out = Dense(len(labels), activation='sigmoid')(x)
    model = Model(base.input, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model ready. To start training, call model.fit with train_gen / valid_gen.")

if __name__ == "__main__":
    # set download_needed=True the first time to pull a small sample of images via Kaggle API
    main(download_needed=True)