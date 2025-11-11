
import os
import glob
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D,
    LSTM, Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


PREPROCESSED_DIR = "data_preprocessed"
IMG_SIZE = (224, 224)
SEQ_LEN = 8
BATCH_SIZE = 8
PHASE_1_EPOCHS = 8
PHASE_2_EPOCHS = 15
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

KNOWN_PHASES = sorted([
    "ACLReconstruction",
    "Diagnosis",
    "FemoralTunnelCreation",
    "Preparation",
    "TibialTunnelCreation"
])


def build_sequence_list(root):
    label_to_int = {p: i for i, p in enumerate(KNOWN_PHASES)}
    int_to_label = {i: p for i, p in enumerate(KNOWN_PHASES)}
    all_seqs = []
    video_folders = sorted(glob.glob(os.path.join(root, "video*")))

    for vfolder in tqdm(video_folders, desc="Parsing videos"):
        imgs = sorted(glob.glob(os.path.join(vfolder, "*.jpg")))
        if len(imgs) < SEQ_LEN:
            continue
        for i in range(0, len(imgs) - SEQ_LEN, SEQ_LEN):
            seq_paths = imgs[i:i + SEQ_LEN]
            phase_name = os.path.basename(seq_paths[-1]).split("_")[1].split(".")[0]
            if phase_name not in label_to_int:
                continue
            label = label_to_int[phase_name]
            all_seqs.append((vfolder, seq_paths, label))
    return all_seqs, label_to_int, int_to_label



class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, seq_list, batch_size, img_size, n_classes,
                 shuffle=True, augment=False):
        self.seq_list = seq_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.seq_list) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.seq_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.seq_list[i] for i in batch_idx]
        X, y = self.__data_generation(batch)
        return X, y

    def _load_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def __data_generation(self, batch):
        X = np.empty((self.batch_size, SEQ_LEN, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=int)
        for i, (_, seq_paths, label) in enumerate(batch):
            frames = [self._load_image(p) for p in seq_paths]
            X[i] = np.stack(frames)
            y[i] = label
        return X, y



def build_model(input_shape, n_classes):
    cnn = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    cnn.trainable = False

    seq_input = Input(shape=(SEQ_LEN, *input_shape))
    x = TimeDistributed(cnn)(seq_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = Model(seq_input, out)
    return model, cnn



def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    if not os.path.exists(PREPROCESSED_DIR):
        print(f"Missing {PREPROCESSED_DIR}, run preprocess.py first.")
        return

    print("Building sequence list ...")
    seq_list, label_to_int, int_to_label = build_sequence_list(PREPROCESSED_DIR)
    print(f"Found sequences: {len(seq_list)}")
    print(f"Classes: {label_to_int}")


    groups = [v for v, _, _ in seq_list]
    labels = [l for _, _, l in seq_list]
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT,
                                 random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(seq_list, labels, groups))

    train_seqs = [seq_list[i] for i in train_idx]
    val_seqs = [seq_list[i] for i in val_idx]
    print(f"Train seq: {len(train_seqs)}  Val seq: {len(val_seqs)}")


    cw = compute_class_weight("balanced",
                              classes=np.arange(len(KNOWN_PHASES)),
                              y=np.array(labels))
    class_weights = dict(enumerate(cw))
    print(f"Class weights: {class_weights}")


    train_gen = SequenceGenerator(train_seqs, BATCH_SIZE, IMG_SIZE, len(KNOWN_PHASES),
                                  shuffle=True)
    val_gen = SequenceGenerator(val_seqs, BATCH_SIZE, IMG_SIZE, len(KNOWN_PHASES),
                                shuffle=False)


    model, cnn = build_model((*IMG_SIZE, 3), len(KNOWN_PHASES))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


    callbacks_phase1 = [
        ModelCheckpoint("model_phase1_lstm.keras", save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        TensorBoard(log_dir="logs_lstm")
    ]

    print("\n=== Phase 1: training head ===")
    model.fit(train_gen, validation_data=val_gen,
              epochs=PHASE_1_EPOCHS,
              class_weight=class_weights,
              callbacks=callbacks_phase1)


    cnn.trainable = True
    for layer in cnn.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks_phase2 = [
        ModelCheckpoint("model_phase2_lstm.keras", save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    print("\n=== Phase 2: fine-tuning ===")
    model.fit(train_gen, validation_data=val_gen,
              epochs=PHASE_2_EPOCHS,
              class_weight=class_weights,
              callbacks=callbacks_phase2)


    with open("label_map.json", "w") as f:
        json.dump({"label_to_int": label_to_int, "int_to_label": int_to_label}, f)
    print("\nTraining complete.  Best model saved as model_phase2_lstm.keras")


if __name__ == "__main__":
    main()
