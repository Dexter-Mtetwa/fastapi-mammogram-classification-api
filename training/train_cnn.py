
"""
CBIS-DDSM Breast Cancer Classification
=====================================

Trains an EfficientNetB0 CNN to classify cropped mammography images as:
    0 = BENIGN
    1 = MALIGNANT

Pipeline:
    1. Load CBIS-DDSM train/test CSV files.
    2. Resolve CSV image paths against the extracted JPEG dataset.
    3. Split training cases into training and validation sets by patient.
    4. Decode and resize images using tf.data.
    5. Train a classification head on a frozen EfficientNetB0.
    6. Fine-tune the upper layers of EfficientNetB0.
    7. Select the checkpoint with the best validation ROC-AUC.
    8. Evaluate once on the supplied test set.
    9. Save model.h5, metrics and label metadata.

Expected dataset:
    CBIS-DDSM/
        csv/
            mass_case_description_train_set.csv
            mass_case_description_test_set.csv
            calc_case_description_train_set.csv
            calc_case_description_test_set.csv
        jpeg/
            <image UID folders containing JPEG images>

Install:
    pip install tensorflow pandas scikit-learn numpy

Example:
    python train_cnn.py --data-dir "/path/to/CBIS-DDSM"

For a dataset extracted under another directory, pass its root
directory instead. The script searches recursively for CSV/JPEG files.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 12

HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

VALIDATION_FRACTION = 0.15
FINE_TUNE_LAST_LAYERS = 30

AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# GPU setup
# ============================================================

def configure_gpu():
    """Allow TensorFlow to allocate GPU memory as required."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print(
            "WARNING: TensorFlow cannot see a GPU. "
            "Training will run on the CPU."
        )
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not configure GPU memory growth: {exc}")

    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu.name}")


# ============================================================
# Dataset discovery and image-path resolution
# ============================================================

def find_csv_files(data_root):
    """Find the four official CBIS-DDSM case-description CSV files."""
    required = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]

    found = {}

    for path in data_root.rglob("*.csv"):
        filename = path.name.lower()

        if filename in required:
            if filename not in found:
                found[filename] = path

    missing = [name for name in required if name not in found]

    if missing:
        raise FileNotFoundError(
            "Could not find these required CSV files:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + f"\n\nDataset root searched: {data_root}"
        )

    print("\nCSV files:")
    for name, path in sorted(found.items()):
        print(f"  {name}: {path}")

    return found


def build_image_index(data_root):
    """
    Index JPEG images by (parent folder UID, filename) and filename.

    CSV paths can contain an old absolute path or a prefix such as
    CBIS-DDSM/jpeg/. Matching the UID and filename avoids depending
    on the original machine's directory structure.
    """
    by_uid_and_name = {}
    by_name = {}
    total = 0

    for path in data_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        total += 1
        filename = path.name.lower()
        uid = path.parent.name.lower()

        by_uid_and_name.setdefault((uid, filename), []).append(path)
        by_name.setdefault(filename, []).append(path)

    if total == 0:
        raise FileNotFoundError(
            f"No JPEG images were found under {data_root}"
        )

    print(f"\nIndexed {total} JPEG images.")
    return by_uid_and_name, by_name


def resolve_image_path(
    csv_path,
    by_uid_and_name,
    by_name,
):
    """Resolve one image path from a CBIS-DDSM CSV row."""
    if pd.isna(csv_path):
        return None

    raw = str(csv_path).strip().replace("\\", "/")

    if not raw:
        return None

    # First handle paths that are already valid on this machine.
    direct = Path(raw)
    if direct.is_file():
        return str(direct.resolve())

    parts = [part for part in raw.split("/") if part]
    if not parts:
        return None

    filename = parts[-1].lower()

    # Most CBIS-DDSM paths end in UID/filename.jpg.
    if len(parts) >= 2:
        uid = parts[-2].lower()
        matches = by_uid_and_name.get((uid, filename), [])

        if len(matches) == 1:
            return str(matches[0].resolve())

        if len(matches) > 1:
            print(
                f"WARNING: Ambiguous image path in UID {uid}: {raw}"
            )
            return None

    # A basename-only fallback is safe only when it is unique.
    matches = by_name.get(filename, [])

    if len(matches) == 1:
        return str(matches[0].resolve())

    if len(matches) > 1:
        print(f"WARNING: Ambiguous image basename: {raw}")

    return None


# ============================================================
# CSV parsing and label preparation
# ============================================================

def normalized_column_name(name):
    """Normalize CSV column names for robust matching."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def get_column(df, possible_names):
    """Return the first matching column name from a list of aliases."""
    lookup = {
        normalized_column_name(column): column
        for column in df.columns
    }

    for name in possible_names:
        match = lookup.get(normalized_column_name(name))
        if match is not None:
            return match

    return None


def encode_pathology(value):
    """
    Map the pathology labels to a binary classification target.

    BENIGN and BENIGN_WITHOUT_CALLBACK -> 0
    MALIGNANT                         -> 1
    """
    if pd.isna(value):
        return None

    label = str(value).strip().upper().replace(" ", "_")

    if "MALIGNANT" in label:
        return 1

    if label.startswith("BENIGN"):
        return 0

    return None


def load_split(
    csv_files,
    split,
    by_uid_and_name,
    by_name,
):
    """Load mass and calcification cases for a given official split."""
    frames = []

    for case_type in ("mass", "calc"):
        filename = f"{case_type}_case_description_{split}_set.csv"
        csv_path = csv_files[filename]

        df = pd.read_csv(csv_path)

        pathology_col = get_column(df, ["pathology"])

        # Prefer cropped lesion images for consistent input.
        # Do not silently mix cropped images and full mammograms.
        image_col = get_column(
            df,
            [
                "cropped image file path",
                "cropped_image_file_path",
            ],
        )

        patient_col = get_column(df, ["patient_id"])

        if pathology_col is None:
            raise ValueError(
                f"Missing pathology column in {csv_path}"
            )

        if image_col is None:
            raise ValueError(
                f"Missing cropped image file path column in {csv_path}"
            )

        records = []
        unresolved = 0
        ignored_labels = 0

        for _, row in df.iterrows():
            label = encode_pathology(row[pathology_col])

            if label is None:
                ignored_labels += 1
                continue

            image_path = resolve_image_path(
                row[image_col],
                by_uid_and_name,
                by_name,
            )

            if image_path is None:
                unresolved += 1
                continue

            # Prefer the dataset's patient identifier to avoid
            # putting images from the same patient in both splits.
            if patient_col is not None and pd.notna(row[patient_col]):
                patient_id = str(row[patient_col]).strip()
            else:
                # Fallback is image-level, not guaranteed patient-level.
                patient_id = f"image:{Path(image_path).name}"

            records.append(
                {
                    "image_path": image_path,
                    "label": int(label),
                    "patient_id": patient_id,
                    "case_type": case_type,
                }
            )

        print(
            f"\n{filename}: "
            f"{len(records)} usable rows, "
            f"{unresolved} unresolved images, "
            f"{ignored_labels} unsupported labels"
        )

        frames.extend(records)

    result = pd.DataFrame(frames)

    if result.empty:
        raise ValueError(
            f"No usable images were loaded for the {split} split."
        )

    # A physical image should not be counted twice.
    result = result.drop_duplicates(
        subset=["image_path"],
        keep="first",
    ).reset_index(drop=True)

    result["label"] = result["label"].astype(np.int32)

    print(f"\n{split.upper()} split after deduplication:")
    print(result["label"].map({0: "BENIGN", 1: "MALIGNANT"}).value_counts())

    return result


# ============================================================
# Patient-level training/validation split
# ============================================================

def split_by_patient(train_df):
    """
    Create a validation set using patient identifiers.

    If a patient_id column was unavailable in the source data,
    load_split uses an image-level fallback. Inspect the output
    and verify patient IDs before treating results as robust.
    """
    patient_labels = (
        train_df.groupby("patient_id")["label"]
        .max()
        .rename("group_label")
        .reset_index()
    )

    if len(patient_labels) < 2:
        raise ValueError(
            "Not enough distinct patient groups for validation."
        )

    group_counts = patient_labels["group_label"].value_counts()

    can_stratify = (
        len(group_counts) == 2
        and group_counts.min() >= 2
    )

    stratify = (
        patient_labels["group_label"]
        if can_stratify
        else None
    )

    train_patients, val_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=stratify,
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)

    train_part = train_df[
        train_df["patient_id"].isin(train_patients)
    ].copy()

    val_part = train_df[
        train_df["patient_id"].isin(val_patients)
    ].copy()

    if train_part.empty or val_part.empty:
        raise ValueError("The patient-level split produced an empty set.")

    if set(train_part["patient_id"]) & set(val_part["patient_id"]):
        raise RuntimeError("Patient leakage detected in validation split.")

    if train_part["label"].nunique() < 2:
        raise ValueError(
            "Training split contains only one class. "
            "Check the CSV labels and patient distribution."
        )

    if val_part["label"].nunique() < 2:
        raise ValueError(
            "Validation split contains only one class. "
            "A reliable ROC-AUC cannot be calculated."
        )

    print("\nFinal data split:")
    print(f"  Training images:   {len(train_part)}")
    print(f"  Validation images: {len(val_part)}")
    print(f"  Training patients: {train_part['patient_id'].nunique()}")
    print(f"  Validation patients: {val_part['patient_id'].nunique()}")

    return train_part, val_part


# ============================================================
# tf.data preprocessing pipeline
# ============================================================

def decode_and_resize(path, label):
    """Read a JPEG, convert to RGB, and resize it."""
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    # EfficientNetB0 includes its own input rescaling layers.
    # Keep pixel values in the 0-255 range here.
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.float32)

    return image, label


def make_dataset(df, training=False):
    """Create an efficient TensorFlow dataset."""
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# CNN architecture
# ============================================================

def build_model(weights="imagenet"):
    """
    Build an EfficientNetB0 transfer-learning classifier.

    Output:
        One sigmoid value representing the model's estimated
        probability of the MALIGNANT class.
    """
    if weights == "none":
        pretrained_weights = None
    else:
        pretrained_weights = "imagenet"

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="mammogram",
    )

    # Mild augmentation is applied during training only.
    x = tf.keras.layers.RandomFlip(
        mode="horizontal",
        seed=SEED,
    )(inputs)

    x = tf.keras.layers.RandomRotation(
        factor=0.03,
        seed=SEED,
    )(x)

    x = tf.keras.layers.RandomZoom(
        height_factor=0.08,
        width_factor=0.08,
        seed=SEED,
    )(x)

    x = tf.keras.layers.RandomContrast(
        factor=0.10,
        seed=SEED,
    )(x)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=pretrained_weights,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling=None,
    )

    # Initially train only the classification head.
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="classification_features",
    )(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="malignancy_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cbis_ddsm_efficientnetb0",
    )

    return model


def compile_model(model, learning_rate):
    """Compile the binary classifier."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ============================================================
# Training callbacks
# ============================================================

def make_callbacks(checkpoint_path, log_path):
    """Create callbacks for checkpointing and training control."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(log_path),
            append=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# ============================================================
# Validation checkpoint selection
# ============================================================

def validation_auc(model, validation_dataset, validation_df):
    """Calculate validation ROC-AUC from predicted probabilities."""
    probabilities = model.predict(
        validation_dataset,
        verbose=0,
    ).reshape(-1)

    labels = validation_df["label"].to_numpy()

    return float(roc_auc_score(labels, probabilities))


def select_best_checkpoint(
    stage1_path,
    stage2_path,
    validation_dataset,
    validation_df,
):
    """
    Compare the best checkpoints from both training phases.
    Select the one with the highest validation ROC-AUC.
    """
    candidates = []

    for checkpoint_path in (stage1_path, stage2_path):
        if not checkpoint_path.exists():
            continue

        candidate = tf.keras.models.load_model(
            checkpoint_path,
            compile=False,
        )

        score = validation_auc(
            candidate,
            validation_dataset,
            validation_df,
        )

        print(
            f"Checkpoint {checkpoint_path.name}: "
            f"validation ROC-AUC = {score:.4f}"
        )

        candidates.append((score, checkpoint_path))

        del candidate

    if not candidates:
        raise RuntimeError("No valid model checkpoint was saved.")

    best_score, best_path = max(
        candidates,
        key=lambda item: item[0],
    )

    print(
        f"\nSelected checkpoint: {best_path.name} "
        f"(validation ROC-AUC = {best_score:.4f})"
    )

    return tf.keras.models.load_model(
        best_path,
        compile=False,
    )


# ============================================================
# Final evaluation
# ============================================================

def evaluate_test_set(model, test_df, test_dataset, output_dir):
    """
    Evaluate the selected model on the untouched supplied test set.

    The default classification threshold is 0.5. Threshold selection
    for a real application requires a separate validation procedure
    and should reflect the consequences of false negatives/positives.
    """
    y_true = test_df["label"].to_numpy(dtype=np.int32)

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    ).reshape(-1)

    y_pred = (probabilities >= 0.5).astype(np.int32)

    metrics = {
        "test_images": int(len(y_true)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
    else:
        metrics["roc_auc"] = None

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nConfusion matrix:")
    print("Rows = actual [BENIGN, MALIGNANT]")
    print("Columns = predicted [BENIGN, MALIGNANT]")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "MALIGNANT"],
            zero_division=0,
        )
    )

    metrics["confusion_matrix"] = matrix.tolist()

    # Save test predictions for later inspection.
    predictions = test_df[
        ["image_path", "patient_id", "label"]
    ].copy()

    predictions["malignancy_probability"] = probabilities
    predictions["predicted_label"] = y_pred

    predictions["actual_class"] = predictions["label"].map(
        {0: "BENIGN", 1: "MALIGNANT"}
    )

    predictions["predicted_class"] = predictions[
        "predicted_label"
    ].map({0: "BENIGN", 1: "MALIGNANT"})

    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    with open(output_dir / "test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN on CBIS-DDSM mammography images."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory of the extracted CBIS-DDSM dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_output"),
        help="Directory for the trained model and training artifacts.",
    )

    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet pretrained weights or train from scratch.",
    )

    args = parser.parse_args()

    data_root = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_root.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {data_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = output_dir / "stage1_best.keras"
    stage2_path = output_dir / "stage2_best.keras"
    training_log = output_dir / "training_log.csv"
    final_model_path = output_dir / "model.h5"

    configure_gpu()

    # --------------------------------------------------------
    # 1. Find metadata and image files
    # --------------------------------------------------------

    csv_files = find_csv_files(data_root)

    by_uid_and_name, by_name = build_image_index(data_root)

    # --------------------------------------------------------
    # 2. Load the supplied training and test cases
    # --------------------------------------------------------

    train_df = load_split(
        csv_files,
        "train",
        by_uid_and_name,
        by_name,
    )

    test_df = load_split(
        csv_files,
        "test",
        by_uid_and_name,
        by_name,
    )

    if train_df["label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both benign and malignant cases."
        )

    if test_df["label"].nunique() != 2:
        raise ValueError(
            "Test data must contain both benign and malignant cases "
            "to calculate ROC-AUC."
        )

    # --------------------------------------------------------
    # 3. Create patient-level training/validation sets
    # --------------------------------------------------------

    train_part, val_part = split_by_patient(train_df)

    # --------------------------------------------------------
    # 4. Build tf.data input pipelines
    # --------------------------------------------------------

    print("\nPreparing TensorFlow datasets...")

    train_dataset = make_dataset(
        train_part,
        training=True,
    )

    val_dataset = make_dataset(
        val_part,
        training=False,
    )

    test_dataset = make_dataset(
        test_df,
        training=False,
    )

    # --------------------------------------------------------
    # 5. Account for class imbalance
    # --------------------------------------------------------

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_part["label"].to_numpy(),
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print(f"\nClass weights: {class_weights}")

    # --------------------------------------------------------
    # 6. Build the CNN
    # --------------------------------------------------------

    print("\nBuilding EfficientNetB0...")

    model = build_model(weights=args.weights)

    compile_model(
        model,
        learning_rate=HEAD_LEARNING_RATE,
    )

    model.summary()

    # --------------------------------------------------------
    # 7. Phase one: train the classification head
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING THE CLASSIFICATION HEAD")
    print("=" * 60)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(stage1_path, training_log),
        verbose=1,
    )

    # Use the best head-training checkpoint as the fine-tuning start.
    if stage1_path.exists():
        del model

        model = tf.keras.models.load_model(
            stage1_path,
            compile=False,
        )
    else:
        raise RuntimeError(
            "Phase one did not save a checkpoint. "
            "Check the dataset and training logs."
        )

    # --------------------------------------------------------
    # 8. Phase two: fine-tune the upper CNN layers
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING EFFICIENTNETB0")
    print("=" * 60)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze earlier feature-extraction layers.
    cutoff = max(
        0,
        len(backbone.layers) - FINE_TUNE_LAST_LAYERS,
    )

    for layer in backbone.layers[:cutoff]:
        layer.trainable = False

    # Keep Batch Normalization statistics stable during fine-tuning.
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(
        int(np.prod(variable.shape))
        for variable in model.trainable_weights
    )

    print(f"Trainable parameters during fine-tuning: {trainable_count:,}")

    compile_model(
        model,
        learning_rate=FINE_TUNE_LEARNING_RATE,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(stage2_path, training_log),
        verbose=1,
    )

    del model

    # --------------------------------------------------------
    # 9. Select the best validation checkpoint
    # --------------------------------------------------------

    best_model = select_best_checkpoint(
        stage1_path,
        stage2_path,
        val_dataset,
        val_part,
    )

    # --------------------------------------------------------
    # 10. Save the final HDF5 model
    # --------------------------------------------------------

    print(f"\nSaving final model to: {final_model_path}")

    best_model.save(
        str(final_model_path),
        include_optimizer=False,
    )

    # Store the input contract and label mapping.
    metadata = {
        "architecture": "EfficientNetB0",
        "image_size": IMAGE_SIZE,
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "input_format": "RGB JPEG, pixel values 0-255",
        "output_shape": [1],
        "output_activation": "sigmoid",
        "output_meaning": "probability of malignant class",
        "threshold_used_for_evaluation": 0.5,
        "classes": {
            "0": "BENIGN",
            "1": "MALIGNANT",
        },
        "dataset": "CBIS-DDSM",
        "pretrained_weights": args.weights,
        "seed": SEED,
    }

    with open(output_dir / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    # --------------------------------------------------------
    # 11. Evaluate the untouched test split
    # --------------------------------------------------------

    evaluate_test_set(
        best_model,
        test_df,
        test_dataset,
        output_dir,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model:       {final_model_path}")
    print(f"Metrics:     {output_dir / 'test_metrics.json'}")
    print(f"Predictions: {output_dir / 'test_predictions.csv'}")
    print(f"Metadata:    {output_dir / 'model_metadata.json'}")
    print(f"Training log: {training_log}")
    print("\nThe saved model returns one malignancy probability per image.")


if __name__ == "__main__":
    main()
