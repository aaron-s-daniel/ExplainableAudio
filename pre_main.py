import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import h5py

def load_preprocessed_data(data_path):
    X = []
    y = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.startswith('AlexNet') and file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                with h5py.File(file_path, 'r') as f:
                    X.append(f['data'][:])
                    y.append(f['label'][:, 0])  # Only take the digit label, not gender
    
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def create_alexnet(input_shape=(227, 227, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Load preprocessed data
    data_path = '/home/mqa887/tmp/audiomnist/AudioMNIST-master'
    print("Loading data...")
    X, y = load_preprocessed_data(data_path)
    print(f"Loaded data shape: X: {X.shape}, y: {y.shape}")
    
    # Reshape X if necessary (should be [num_samples, 227, 227, 1])
    if X.shape[1:] != (227, 227, 1):
        X = X.reshape(-1, 227, 227, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_alexnet()
    model.compile(optimizer=optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=2,
                        validation_split=0.2,
                        verbose=1)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('alexnet_audio_model.h5')
    print("Model saved to alexnet_audio_model.h5")