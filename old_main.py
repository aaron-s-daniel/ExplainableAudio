data_path = '/home/mqa887/tmp/audiomnist'
import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau


def load_audiomnist(data_path, sr=16000, duration=1.0):
    audio_path = os.path.join(data_path, 'AudioMNIST-master', 'data')
    data = []
    labels = []
    target_length = int(sr * duration)

    for folder in range(1, 11):
        folder_name = f"{folder:02d}"
        folder_path = os.path.join(audio_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                audio, _ = librosa.load(file_path, sr=sr, duration=duration)
                
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, max(0, target_length - len(audio))))
                
                data.append(audio)
                labels.append(folder - 1)  # Subtract 1 to get labels 0-9

    return np.array(data), np.array(labels)

def create_spectrogram(audio, sr=16000, n_mels=227, time_steps=227):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Ensure the spectrogram has exactly 227 time steps
    if mel_spec_db.shape[1] < time_steps:
        pad_width = time_steps - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :time_steps]
    
    return mel_spec_db

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
class PrintProgress(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        print(f"Train Loss: {logs['loss']:.4f}, Train Accuracy: {logs['accuracy']:.4f}")
        print(f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")
        # Remove the learning rate printing for now
        sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
            sys.stdout.flush()

if __name__ == "__main__":
    try:
        print("Script started")
        sys.stdout.flush()

        # Load the dataset
        data_path = '/home/mqa887/tmp/audiomnist'
        X, y = load_audiomnist(data_path, duration=1.0)  # Load 1-second clips
        print(f"Dataset loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
        sys.stdout.flush()

        # Create spectrograms
        X_spec = np.array([create_spectrogram(audio) for audio in X])
        print(f"Spectrograms created. Shape of X_spec: {X_spec.shape}")
        sys.stdout.flush()

        # Reshape for CNN input
        X_spec = X_spec.reshape((-1, 227, 227, 1))
        print(f"Reshaped spectrograms. Shape of X_spec: {X_spec.shape}")
        sys.stdout.flush()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.2, random_state=42)
        print("Data split into train and test sets")
        sys.stdout.flush()

        # Create and compile model
        model = create_alexnet()
        model.compile(optimizer=optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model created and compiled")
        sys.stdout.flush()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        print_progress = PrintProgress()

        print("Starting model training")
        sys.stdout.flush()

        # Train model
        history = model.fit(X_train, y_train,
                            batch_size=64,
                            epochs=50,
                            callbacks=[reduce_lr, print_progress],
                            validation_split=0.2,
                            verbose=0)  # Set to 0 as we're using custom callback

        print("Model training completed")
        sys.stdout.flush()

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        sys.stdout.flush()

        # Save model
        model.save('alexnet_audio_model.h5')
        print("Model saved to alexnet_audio_model.h5")
        sys.stdout.flush()

        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

        print("Training history plot saved")
        print("Script completed successfully")
        sys.stdout.flush()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()