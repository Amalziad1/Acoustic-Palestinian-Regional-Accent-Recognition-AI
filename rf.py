import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Load data
X, y = load_data(os.path.join(DATA_DIR, 'training'))
X_test, y_test = load_data(os.path.join(DATA_DIR, 'testing'))

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
rf_model = RandomForestClassifier()


# Hyperparameter tuning for Random Forest
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(rf_model, rf_param_grid, refit=True, verbose=2, cv=5)
rf_grid.fit(X_train, y_train)

# Evaluate the models
rf_best = rf_grid.best_estimator_

# Predict on the validation set
rf_val_pred = rf_best.predict(X_val)

y_test_pred = rf_best.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)

print(f'Test Accuracy: {accuracy}')
print(classification_report(y_test, y_test_pred, target_names=CITIES))