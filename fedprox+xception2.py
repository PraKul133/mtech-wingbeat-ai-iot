import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, BatchNormalization
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import librosa
import librosa.display

# Set random seeds for reproducibility
seed = 2018
np.random.seed(seed)
tf.random.set_seed(seed)

# Parameters
NUM_CLIENTS = 3
NUM_EPOCHS = 15  # Increased epochs
BATCH_SIZE = 32
INPUT_SHAPE = (129, 120, 3)
NUM_CLASSES = 6
MU = 0.1  # FedProx hyperparameter
SR = 8000
N_FFT = 256
HOP_LEN = N_FFT // 6

target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

# Function to load and preprocess audio files
def audio_to_spectrogram(file_path):
    data, _ = librosa.load(file_path, sr=SR)
    spectrogram = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
    spectrogram = np.flipud(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.repeat(spectrogram, 3, axis=-1)
    return spectrogram

def load_all_data(dataset_path):
    X_names = []
    y = []
    target_count = []

    # Loop through each target (species) and collect file paths
    for i, target in enumerate(target_names):
        target_count.append(0)
        path = os.path.join(dataset_path, target)
        for root, dirs, files in os.walk(path, topdown=False):
            for filename in files:
                name, ext = os.path.splitext(filename)
                if ext.lower() == '.wav':
                    file_path = os.path.join(root, filename)
                    y.append(i)
                    X_names.append(file_path)
                    target_count[i] += 1
        print(f"{target} #recs = {target_count[i]}")

    print(f"Total #recs = {len(y)}")
    return X_names, y

def prepare_client_data(X_names, y, num_clients=3, batch_size=32):
    # First shuffle the data
    X_names, y = shuffle(X_names, y, random_state=seed)
    
    # Split into train and test (80/20)
    X_train_names, X_test_names, y_train, y_test = train_test_split(
        X_names, y, stratify=y, test_size=0.20, random_state=seed)
    
    # Split train data among clients (non-IID distribution for realism)
    client_data = []
    for i in range(num_clients):
        # Get a portion of data for this client (stratified by class)
        client_X_train, _, client_y_train, _ = train_test_split(
            X_train_names, y_train, train_size=1/num_clients, stratify=y_train, random_state=seed+i)
        
        # Create data generators
        train_gen = data_generator(client_X_train, client_y_train, batch_size)
        test_gen = data_generator(X_test_names, y_test, batch_size)
        
        client_data.append((train_gen, test_gen, len(client_X_train), len(X_test_names)))
    
    return client_data

def data_generator(X_names, y, batch_size):
    while True:
        for i in range(0, len(X_names), batch_size):
            batch_X = []
            batch_y = []
            for j in range(i, min(i + batch_size, len(X_names))):
                spectrogram = audio_to_spectrogram(X_names[j])
                batch_X.append(spectrogram)
                batch_y.append(y[j])
            yield np.array(batch_X), to_categorical(np.array(batch_y), num_classes=NUM_CLASSES)

# Enhanced Xception model with custom input shape
def create_xception_model(input_shape, num_classes):
    img_input = Input(shape=input_shape)
    x = Lambda(lambda x: (x / 255.0) * 2.0 - 1.0)(img_input)
    xception_base = Xception(include_top=False, weights='imagenet', input_tensor=x)

    x = GlobalAveragePooling2D()(xception_base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)  # Increased units
    x = Dropout(0.3)(x)  # Adjusted dropout
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=output)
    return model

# Enhanced Federated Learning with FedProx
class FedProxClient:
    def __init__(self, client_id, train_gen, test_gen, train_size, test_size):
        self.client_id = client_id
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.train_size = train_size
        self.test_size = test_size
        
        # Create local model with enhanced architecture
        self.model = create_xception_model(INPUT_SHAPE, NUM_CLASSES)
        self.optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
        self.model.compile(optimizer=self.optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        # Store initial weights
        self.initial_weights = None
    
    def train(self, global_weights, epochs=1, mu=0.1):
        # Set initial weights from global model
        self.model.set_weights(global_weights)
        self.initial_weights = global_weights
        
        # Train the model
        history = self.model.fit(
            self.train_gen,
            steps_per_epoch=math.ceil(self.train_size / BATCH_SIZE),
            epochs=epochs,
            verbose=0
        )
        
        # Calculate proximal term and update weights
        if mu > 0:
            current_weights = self.model.get_weights()
            updated_weights = []
            
            for w, w0 in zip(current_weights, self.initial_weights):
                # Apply proximal term: w = w - lr * mu * (w - w0)
                updated_w = w - 0.0005 * mu * (w - w0)  # Match learning rate
                updated_weights.append(updated_w)
            
            self.model.set_weights(updated_weights)
        
        return history.history
    
    def evaluate(self):
        # Evaluate accuracy
        results = self.model.evaluate(
            self.test_gen,
            steps=math.ceil(self.test_size / BATCH_SIZE),
            verbose=0
        )
        
        # Get detailed metrics
        y_true = []
        y_pred = []
        for _ in range(math.ceil(self.test_size / BATCH_SIZE)):
            X_batch, y_batch = next(self.test_gen)
            preds = self.model.predict(X_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        
        # Calculate per-class metrics
        class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        class_metrics = {}
        for i, target in enumerate(target_names):
            class_metrics[target] = {
                'precision': class_report[target]['precision'],
                'recall': class_report[target]['recall'],
                'f1': class_report[target]['f1-score'],
                'support': class_report[target]['support']
            }
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'class_metrics': class_metrics
        }
        
        return results[1], metrics  # Return accuracy and full metrics
    
    def get_weights(self):
        return self.model.get_weights()
    
    def get_data_size(self):
        return self.train_size

def print_detailed_test_stats(metrics, title):
    print(f"\n=== {title} Test Statistics ===")
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall:    {metrics['recall']:.4f}")
    print(f"Weighted F1-score:  {metrics['f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    class_df = pd.DataFrame(metrics['class_metrics']).T
    print(class_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Main function
def main():
    # Specify dataset path
    dataset_path = 'E:/archive/Wingbeats'  # Replace with your actual dataset path
    
    # Load all data
    print("Loading dataset...")
    X_names, y = load_all_data(dataset_path)
    
    # Prepare client datasets
    print("\nPreparing data for clients...")
    client_data = prepare_client_data(X_names, y, num_clients=NUM_CLIENTS)
    
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        train_gen, test_gen, train_size, test_size = client_data[i]
        client = FedProxClient(i+1, train_gen, test_gen, train_size, test_size)
        clients.append(client)
        print(f"Client {i+1} prepared - Train: {train_size} samples, Test: {test_size} samples")
    
    # Create global model
    global_model = create_xception_model(INPUT_SHAPE, NUM_CLASSES)
    global_model.compile(optimizer=Adam(learning_rate=0.0005),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    
    # Initialize global weights
    global_weights = global_model.get_weights()
    
    # Training history
    train_acc_history = [[] for _ in range(NUM_CLIENTS)]
    test_acc_history = [[] for _ in range(NUM_CLIENTS)]
    client_metrics_history = [[] for _ in range(NUM_CLIENTS)]
    training_times = []
    
    # Training loop
    total_training_time = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_start_time = time.time()
        
        # Client training
        client_weights = []
        client_sizes = []
        
        for i, client in enumerate(clients):
            print(f"Training Client {client.client_id}...", end=' ')
            history = client.train(global_weights, epochs=1, mu=MU)
            train_acc_history[i].append(history['accuracy'][-1])
            
            # Evaluate client
            test_acc, metrics = client.evaluate()
            test_acc_history[i].append(test_acc)
            client_metrics_history[i].append(metrics)
            
            # Collect weights and data sizes for averaging
            client_weights.append(client.get_weights())
            client_sizes.append(client.get_data_size())
            
            print(f"Done (Train Acc: {history['accuracy'][-1]:.4f}, Test Acc: {test_acc:.4f})")
        
        # Federated averaging
        print("\nPerforming federated averaging...")
        new_global_weights = []
        for layer in range(len(global_weights)):
            # Weighted average based on client data sizes
            weighted_sum = np.zeros_like(global_weights[layer])
            total_size = sum(client_sizes)
            
            for client_idx in range(NUM_CLIENTS):
                weighted_sum += client_weights[client_idx][layer] * (client_sizes[client_idx] / total_size)
            
            new_global_weights.append(weighted_sum)
        
        global_weights = new_global_weights
        global_model.set_weights(global_weights)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        training_times.append(epoch_time)
        
        # Print epoch summary
        avg_train_acc = np.mean([train_acc_history[i][-1] for i in range(NUM_CLIENTS)])
        avg_test_acc = np.mean([test_acc_history[i][-1] for i in range(NUM_CLIENTS)])
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"- Avg Training Accuracy: {avg_train_acc:.4f}")
        print(f"- Avg Test Accuracy:    {avg_test_acc:.4f}")
        print(f"- Epoch Time:           {epoch_time:.2f}s")
    
    # Final evaluation with global model
    print("\n=== Final Global Model Evaluation ===")
    
    # Get test data from first client (all clients share the same test set)
    _, test_gen, _, test_size = client_data[0]
    
    # Collect all test predictions
    y_true = []
    y_pred = []
    test_batches = math.ceil(test_size / BATCH_SIZE)
    
    for _ in range(test_batches):
        X_batch, y_batch = next(test_gen)
        preds = global_model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    # Calculate global metrics
    class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    class_metrics = {}
    for i, target in enumerate(target_names):
        class_metrics[target] = {
            'precision': class_report[target]['precision'],
            'recall': class_report[target]['recall'],
            'f1': class_report[target]['f1-score'],
            'support': class_report[target]['support']
        }
    
    global_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'class_metrics': class_metrics
    }
    
    # Print global model statistics
    print_detailed_test_stats(global_metrics, "Global Model")
    
    # Print client statistics
    for i, client in enumerate(clients):
        print(f"\n=== Client {client.client_id} Final Test Statistics ===")
        print(f"Training Samples: {client.train_size}")
        print(f"Test Samples:     {client.test_size}")
        print_detailed_test_stats(client_metrics_history[i][-1], f"Client {client.client_id}")
    
    # Print timing information
    print("\nTraining Statistics:")
    print(f"- Total Training Time:   {total_training_time:.2f} seconds")
    print(f"- Average Epoch Time:    {np.mean(training_times):.2f} seconds")
    print(f"- Total Samples Trained: {sum([client.get_data_size() for client in clients])}")
    
     # Plot training and testing accuracy for each client - INDIVIDUAL PLOTS
    for i in range(NUM_CLIENTS):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_EPOCHS+1), train_acc_history[i], 'o-', color='blue', label='Training Accuracy')
        plt.plot(range(1, NUM_EPOCHS+1), test_acc_history[i], 's--', color='red', label='Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Client {i+1} Training Progress')
        plt.legend()
        plt.grid()
        plt.xticks(range(1, NUM_EPOCHS+1))
        plt.ylim(0.5, 1.0)  # Adjusted y-axis limits for better visualization
        plt.show()

    # Keep the combined plot (optional)
    plt.figure(figsize=(12, 6))
    for i in range(NUM_CLIENTS):
        plt.plot(range(1, NUM_EPOCHS+1), train_acc_history[i], 'o-', label=f'Client {i+1} Train')
        plt.plot(range(1, NUM_EPOCHS+1), test_acc_history[i], 's--', label=f'Client {i+1} Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('All Clients Training Progress (Federated Xception Model)')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, NUM_EPOCHS+1))
    plt.show()
    
    # Plot client metrics comparison
    final_metrics = []
    for i in range(NUM_CLIENTS):
        final_metrics.append(client_metrics_history[i][-1])
    
    metrics_df = pd.DataFrame(final_metrics, index=[f'Client {i+1}' for i in range(NUM_CLIENTS)])
    
    plt.figure(figsize=(12, 6))
    metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar')
    plt.title('Final Client Metrics Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y')
    plt.show()
    
    # Save the model
    global_model.save('mosquito_classification_xception.hdf5')
    print("\nModel saved as 'mosquito_classification_xception.hdf5'")

if __name__ == "__main__":
    main()