
# ✅ Phase 1: IMPORT LIBRARIES
import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import Sequence

tf.keras.mixed_precision.set_global_policy('float32')
print("✅ Phase 1 Completed: Libraries Imported")

# ✅ Phase 2: SETTINGS
DATA_PATH = "D:\mahesh intern\internship\Sign Recognition Dataset"
MODEL_SAVE_PATH = "final_efficientnetb0_model.h5"  # ✅ Custom .h5 save path
FRAME_HEIGHT, FRAME_WIDTH = 64, 64
MAX_SEQ_LEN = 32
BATCH_SIZE = 16
EPOCHS = 30
INITIAL_LR = 1e-3
print("✅ Phase 2 Completed: Settings Defined")

# ✅ Phase 3: LOAD VIDEO PATHS & LABELS
video_extensions = ('.mp4', '.avi', '.mov')
video_paths, labels = [], []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(video_extensions):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(os.path.dirname(path)))
            video_paths.append(path)
            labels.append(label)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
class_names = label_encoder.classes_.tolist()
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(encoded_labels), y=encoded_labels)
class_weights = dict(enumerate(class_weights_array))

print("✅ Phase 3 Completed: Class labels encoded")

# ✅ Phase 4: SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(
    video_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

print(f"✅ Phase 4 Completed: Train={len(X_train)}, Val={len(X_val)}")

# 🔍 Show CLASS-WISE VIDEO COUNTS
train_counts = Counter([label_encoder.inverse_transform([y])[0] for y in y_train])
val_counts = Counter([label_encoder.inverse_transform([y])[0] for y in y_val])
print("\n📌 CLASS-WISE VIDEO COUNTS:")
print(f"{'Class':<25} {'Train':<10} {'Validation':<10}")
print("-" * 50)
for cls in class_names:
    print(f"{cls:<25} {train_counts.get(cls, 0):<10} {val_counts.get(cls, 0):<10}")



# ✅ Phase 5: VIDEO GENERATOR
class VideoDataGenerator(Sequence):
    def __init__(self, paths, labels, batch_size, seq_len, shuffle=True):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.indices = np.arange(len(paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_paths = [self.paths[i] for i in batch_idx]
        batch_labels = [self.labels[i] for i in batch_idx]
        X = [self._load_video(p) for p in batch_paths]
        return np.array(X, dtype=np.float32), np.array(batch_labels)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = random.randint(0, max(0, total - self.seq_len))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        while len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            if random.random() < 0.5: frame = cv2.flip(frame, 1)
            if random.random() < 0.3:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 2] *= 0.6 + 0.4 * np.random.rand()
                hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        cap.release()
        while len(frames) < self.seq_len:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
        return np.array(frames)

print("✅ Phase 5 Completed: Data Generator Ready")

# ✅ Phase 6: BUILD MODEL
def attention_layer(inputs):
    score = layers.Dense(1, activation='tanh')(inputs)
    attention_weights = layers.Softmax(axis=1)(score)
    weighted = layers.Multiply()([inputs, attention_weights])
    return layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)

def build_model(seq_len, h, w, c, num_classes):
    inp = layers.Input((seq_len, h, w, c))
    x = layers.Conv3D(16, (3,3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((1,2,2))(x)

    x = layers.Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((seq_len, h//2, w//2, 32))(x)
    x = layers.TimeDistributed(layers.Conv2D(3, (1,1), activation='relu'))(x)
    x = layers.TimeDistributed(layers.Resizing(64, 64))(x)

    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    for layer in base.layers[:150]: layer.trainable = False
    x = layers.TimeDistributed(base)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = attention_layer(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)

model = build_model(MAX_SEQ_LEN, FRAME_HEIGHT, FRAME_WIDTH, 3, len(class_names))
model.summary()

# 📌 MODEL CONFIGURATION SUMMARY:
trainable_count = np.sum([np.prod(v.shape) for v in model.trainable_weights])
non_trainable_count = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
total_params = trainable_count + non_trainable_count
print("\n📌 MODEL CONFIGURATION SUMMARY:")
print(f"• Input Shape        : {MAX_SEQ_LEN}x{FRAME_HEIGHT}x{FRAME_WIDTH}x3")
print(f"• Total Params       : {total_params:,}")
print(f"• Trainable Params   : {trainable_count:,}")
print(f"• Non-Trainable      : {non_trainable_count:,}")
print(f"• Batch Size         : {BATCH_SIZE}")
print(f"• Epochs             : {EPOCHS}")
print(f"• Initial LR         : {INITIAL_LR}")
print(f"• Classes            : {len(class_names)}")



# ✅ Phase 7: COMPILE MODEL
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=INITIAL_LR, first_decay_steps=10, t_mul=2.0, m_mul=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# ✅ Phase 8: TRAINING
train_gen = VideoDataGenerator(X_train, y_train, BATCH_SIZE, MAX_SEQ_LEN)
val_gen = VideoDataGenerator(X_val, y_val, BATCH_SIZE, MAX_SEQ_LEN, shuffle=False)

class UnfreezeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 5:
            print("🔓 Unfreezing all layers...")
            for layer in model.layers:
                layer.trainable = True

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    UnfreezeCallback()
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ✅ Phase 9: SAVE MANUALLY TO .h5 FORMAT
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# ✅ Phase 10: EVALUATION
loss, acc = model.evaluate(val_gen)
print(f"✅ Final Validation Accuracy: {acc*100:.2f}%")

# ✅ Phase 11: PLOTS
def plot_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(), plt.grid(True), plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(), plt.grid(True), plt.title("Loss")

    plt.tight_layout()
    plt.show()

plot_history(history)

# ✅ Phase 12: CLASSIFICATION REPORT
y_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.array(y_val)[:len(y_pred)]

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ✅ Phase 13: SAMPLE PREDICTIONS
def show_predictions(num_samples=5):
    for _ in range(num_samples):
        path = random.choice(X_val)
        true_label = os.path.basename(os.path.dirname(os.path.dirname(path)))
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(MAX_SEQ_LEN):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        cap.release()
        while len(frames) < MAX_SEQ_LEN:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
        frames = np.array(frames).reshape(1, MAX_SEQ_LEN, FRAME_HEIGHT, FRAME_WIDTH, 3)

        pred = model.predict(frames)
        pred_class = class_names[np.argmax(pred)]
        conf = np.max(pred) * 100
        plt.imshow(frames[0][0])
        plt.title(f"True: {true_label} | Pred: {pred_class} ({conf:.1f}%)")
        plt.axis('off')
        plt.show()

show_predictions(5)
