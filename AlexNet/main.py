import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import json
import datetime

class PrintLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        print(f"\nLearning rate for epoch {epoch + 1} is {lr.numpy():.6f}")

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
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history, save_path):
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
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = f"results_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)

    # Load preprocessed data
    data_path = '/home/mqa887/tmp/audiomnist/AudioMNIST-master'
    print("Loading data...")
    X, y = load_preprocessed_data(data_path)
    print(f"Loaded data shape: X: {X.shape}, y: {y.shape}")

    # Reshape X if necessary (should be [num_samples, 227, 227, 1])
    if X.shape[1:] != (227, 227, 1):
        X = X.reshape(-1, 227, 227, 1)

    # Normalize your data
    X = (X - np.mean(X)) / np.std(X)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile model
    model = create_alexnet()
    
    initial_learning_rate = 0.001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=2500, decay_rate=0.5, staircase=True
    )
    optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, clipvalue=5.0)

    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

 

    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=100,  # As per the paper
                        epochs=8,  # Increase epochs
                        validation_split=0.2,
                        verbose=1,
                        # Add this callback to your model.fit() call
                        callbacks=[PrintLearningRate()])

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot training history
    plot_training_history(history, os.path.join(results_folder, 'training_history.png'))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, os.path.join(results_folder, 'confusion_matrix.png'))

    # Generate classification report
    class_report = classification_report(y_test, y_pred_classes, output_dict=True)

    # Save metrics and results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'classification_report': class_report,
        'training_history': history.history
    }

    with open(os.path.join(results_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save model
    model.save(os.path.join(results_folder, 'alexnet_audio_model.h5'))
    print(f"Model and results saved in {results_folder}")