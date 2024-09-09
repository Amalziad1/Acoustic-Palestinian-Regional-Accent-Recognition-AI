import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from keras import layers
import os

# Constants
DATA_DIR = r'C:\Users\Administrator\spoken project'
NUM_MFCC = 40  # Number of MFCCs
SAMPLE_RATE = 22050
MAX_LEN = 216  # Maximum length of the MFCC features (determined empirically)
CITIES = ['ramallah-reef', 'jerusalem', 'hebron', 'nablus']
CITY_TO_LABEL = {city: idx for idx, city in enumerate(CITIES)}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Data augmentation: adding noise
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    
    # Data augmentation: time stretching
    y_stretched = librosa.effects.time_stretch(y, rate=0.9)
    
    # Data augmentation: pitch shifting
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=NUM_MFCC)
    mfcc_stretched = librosa.feature.mfcc(y=y_stretched, sr=sr, n_mfcc=NUM_MFCC)
    mfcc_shifted = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=NUM_MFCC)
    
    # Combine the original and augmented MFCCs
    mfcc_combined = np.concatenate([mfcc, mfcc_stretched, mfcc_shifted], axis=1)
    
    if mfcc_combined.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc_combined.shape[1]
        mfcc_combined = np.pad(mfcc_combined, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_combined = mfcc_combined[:, :MAX_LEN]
    
    mfcc_combined = np.mean(mfcc_combined.T, axis=0)
    return mfcc_combined

def load_data(data_dir):
    features = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            label_str = file_name.split('_')[0]
            labels.append(CITY_TO_LABEL[label_str])
    return np.array(features), np.array(labels)

# Load training data
X_train, y_train = load_data(os.path.join(DATA_DIR, 'training'))

# Load testing data
X_test, y_test = load_data(os.path.join(DATA_DIR, 'testing'))

# Normalize the features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Define the model using CNN
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((NUM_MFCC, 1), input_shape=(NUM_MFCC,)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(CITIES), activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with a larger batch size and more epochs
model.fit(X_train, y_train, epochs=150, batch_size=8, validation_split=0.2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred, target_names=CITIES))