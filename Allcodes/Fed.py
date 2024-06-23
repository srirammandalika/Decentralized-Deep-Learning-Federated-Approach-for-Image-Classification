import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = np.load('/Users/srirammandalika/Downloads/fracturemnist3d.npz')

# List the keys in the dataset
print("Keys in the .npz file:", list(data.keys()))

# Inspect the shape of the images and labels
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
print("Shape of train_images:", train_images.shape)
print("Shape of train_labels:", train_labels.shape)

# Ensure labels are binary
train_labels = (train_labels > 0).astype(int)
val_labels = (val_labels > 0).astype(int)

# Plot a few samples
def plot_samples(images, labels, num_samples=3, slice_idx=14):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i, slice_idx], cmap='gray')
        plt.title(f'Label: {labels[i, 0]}')
        plt.axis('off')
    plt.show()

# Plotting 3 samples with a slice index of 14
plot_samples(train_images, train_labels, num_samples=3, slice_idx=14)

# Model creation function
def create_model():
    model = models.Sequential([
        layers.Input(shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3], 1)),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    return model

# Initialize the global model
global_model = create_model()

# Local adaptation function
def local_adaptation(model, data, labels, learning_rate=0.001, epochs=1):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()  # Assuming binary classification
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return model

# Example adaptation for one client
client_data = train_images[:10].reshape(-1, train_images.shape[1], train_images.shape[2], train_images.shape[3], 1)
client_labels = train_labels[:10]

adapted_model = local_adaptation(global_model, client_data, client_labels)

# Federated averaging function
def federated_averaging(models):
    avg_model = create_model()
    new_weights = [np.mean([model.get_weights()[i] for model in models], axis=0) for i in range(len(models[0].get_weights()))]
    avg_model.set_weights(new_weights)
    return avg_model

# Evaluation function
def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    predicted_labels = (predictions > 0.5).astype(int)
    true_labels = labels.astype(int)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall, f1

# Federated training pipeline
def federated_training(global_model, clients_data, clients_labels, val_data, val_labels, rounds=10, local_epochs=40):
    for round in range(rounds):
        local_models = []
        for client_data, client_labels in zip(clients_data, clients_labels):
            # Local adaptation
            client_data = client_data.reshape(-1, client_data.shape[1], client_data.shape[2], client_data.shape[3], 1)
            local_model = local_adaptation(global_model, client_data, client_labels, epochs=local_epochs)
            local_models.append(local_model)
        # Federated averaging
        global_model = federated_averaging(local_models)
        
        # Evaluate the model on validation data
        val_data = val_data.reshape(-1, val_data.shape[1], val_data.shape[2], val_data.shape[3], 1)
        accuracy, precision, recall, f1 = evaluate_model(global_model, val_data, val_labels)
        print(f'Round {round + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}')
    
    return global_model

# Example data split for 4 clients
num_clients = 4
client_size = len(train_images) // num_clients
clients_data = [train_images[i*client_size:(i+1)*client_size] for i in range(num_clients)]
clients_labels = [train_labels[i*client_size:(i+1)*client_size] for i in range(num_clients)]

# Federated training with evaluation
final_global_model = federated_training(global_model, clients_data, clients_labels, val_images, val_labels)
