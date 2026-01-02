import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ================= CONFIG =================
DATASET_PATH = "dynamic_dataset"
ACTIONS = ["hello", "thank_you"]
SEQUENCE_LENGTH = 30
FEATURES = 126
# ==========================================

print("📂 Loading dataset...")
X, y = [], []

for label, action in enumerate(ACTIONS):
    folder = os.path.join(DATASET_PATH, action)
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        if data.shape == (SEQUENCE_LENGTH, FEATURES):
            X.append(data)
            y.append(label)

X = np.array(X)
y = to_categorical(y)
print(f"✅ Dataset loaded with {X.shape[0]} sequences and {len(ACTIONS)} actions.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= MODEL =================
print("🤖 Building the model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(len(ACTIONS), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model is ready to train.")
print("📈 Training started...")

# ================= TRAIN =================
for epoch in range(30):
    history = model.fit(
        X_train,
        y_train,
        epochs=1,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0  # hide detailed tensorflow logs
    )
    train_acc = history.history['accuracy'][0] * 100
    val_acc = history.history['val_accuracy'][0] * 100
    print(f"Epoch {epoch+1}/30 done. Training accuracy: {train_acc:.1f}%, Validation accuracy: {val_acc:.1f}%")

# ================= SAVE =================
model.save("dynamic_sign_model.h5")
print("🎉 Training complete! Model saved as 'dynamic_sign_model.h5'.")
