
import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
import os

class SequenceDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, master_list, batch_size, img_size=(224,224), seq_len=16,
                 shuffle=True, augment=False, preprocess_fn=None):

        self.master_list = master_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.augment = augment
        self.preprocess_fn = preprocess_fn or (lambda x: efficientnet.preprocess_input(x.astype('float32')))
        self._build_index()
        self.on_epoch_end()

    def _frame_sort_key(self, path):
        # try to find a numeric frame index in the filename
        base = os.path.basename(path)
        import re
        m = re.search(r'(\d{4,})', base)
        if m:
            return int(m.group(1))

        return base

    def _build_index(self):
        # Group frames per video, sorted by frame index
        from collections import defaultdict
        groups = defaultdict(list)
        for video_folder, img_path, label in self.master_list:
            groups[video_folder].append((img_path, label))
        self.groups = []
        for video, items in groups.items():
            items_sorted = sorted(items, key=lambda x: self._frame_sort_key(x[0]))
            self.groups.append((video, items_sorted))

        self.indexes = []
        half = self.seq_len // 2
        for gi, (_, items) in enumerate(self.groups):
            L = len(items)
            for center in range(L):

                self.indexes.append((gi, center))

        self.group_paths = []
        self.group_labels = []
        for _, items in self.groups:
            paths = [it[0] for it in items]
            labels = [it[1] for it in items]
            self.group_paths.append(paths)
            self.group_labels.append(labels)

    def __len__(self):
        return math.floor(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        self.order = np.arange(len(self.indexes))
        if self.shuffle:
            np.random.shuffle(self.order)

    def _augment_image(self, image):

        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-8, 8)
            h,w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w,h), borderMode=cv2.BORDER_REFLECT)
        if np.random.rand() < 0.25:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-15, 15)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if np.random.rand() < 0.1:
            k = np.random.choice([3,5])
            image = cv2.GaussianBlur(image, (k,k), 0)
        return image

    def __getitem__(self, idx):

        batch_order = self.order[idx*self.batch_size:(idx+1)*self.batch_size]
        seqs = np.zeros((self.batch_size, self.seq_len, self.img_size[0], self.img_size[1], 3), dtype='float32')
        targets = np.zeros((self.batch_size,), dtype='int32')
        half = self.seq_len // 2

        for i, ord_idx in enumerate(batch_order):
            gi, center = self.indexes[ord_idx]
            paths = self.group_paths[gi]
            labels = self.group_labels[gi]
            L = len(paths)

            idxs = [min(max(0, center - half + k), L-1) for k in range(self.seq_len)]
            frames = []
            for j, fi in enumerate(idxs):
                p = paths[fi]
                img = cv2.imread(p)
                if img is None:

                    img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype='uint8')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
                if self.augment:
                    img = self._augment_image(img)
                frames.append(self.preprocess_fn(img))
            seqs[i] = np.stack(frames, axis=0)
            targets[i] = labels[center]
        return seqs, targets
