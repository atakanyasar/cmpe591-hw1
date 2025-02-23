import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt  

# Configuration (Constants)
POSITIONS_FILE = "data/positions_0.pt"
ACTIONS_FILE = "data/actions_0.pt"
IMAGES_FILE = "data/imgs_observation_0.pt"
INPUT_SIZE = 4 + 3 * 128 * 128
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 5
MIN_DELTA = 1e-4
LEARNING_RATE_ADAM = 0.001
LEARNING_RATE_SGD = 0.01
MOMENTUM_SGD = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.positions = torch.load(POSITIONS_FILE)
        self.actions = torch.load(ACTIONS_FILE)
        self.images = torch.load(IMAGES_FILE).float() / 255.0

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        action_onehot = torch.zeros(4)
        action_onehot[int(action)] = 1
        image = self.images[idx].reshape(-1)
        x = torch.cat([action_onehot, image])
        y = self.positions[idx]
        return x, y

# Models
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, image_size=128, num_linear_features=4):
        super().__init__()
        self.image_size = image_size
        self.num_linear_features = num_linear_features
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        flattened_size = 32 * (image_size // 4) * (image_size // 4) + num_linear_features
        self.prediction_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        linear_features = x[:, :self.num_linear_features]
        image_features = x[:, self.num_linear_features:].view(-1, 3, self.image_size, self.image_size)
        extracted_features = self.convolutional_layers(image_features)
        combined_features = torch.cat([linear_features, extracted_features], dim=1)
        return self.prediction_layers(combined_features)

# Training Function (for a single fold)
def train_fold(model, train_loader, val_loader, criterion, optimizer, device, max_epochs, patience, min_delta):
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{max_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    return best_model_state, best_val_loss, train_losses, val_losses

# Main Training Loop (with K-fold cross-validation)
def train_model():
    dataset = CustomDataset()
    kfold = KFold(n_splits=5, shuffle=True)

    results = {  # Store results for all model/optimizer combinations
        "cnn_adam": {"fold_results": [], "train_losses": [], "val_losses": [], "best_model_state": None, "best_val_loss": float("inf")},
        "cnn_sgd": {"fold_results": [], "train_losses": [], "val_losses": [], "best_model_state": None, "best_val_loss": float("inf")},
    }

    for model_name in ["cnn"]:
        for optimizer_name in ["adam", "sgd"]:
            print(f"\nTraining with {model_name.upper()} and {optimizer_name.upper()} optimizer")
            print("=" * 50)

            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                print(f"\nFOLD {fold+1}/{5}")
                print("-" * 50)

                train_sampler = SubsetRandomSampler(train_ids)
                val_sampler = SubsetRandomSampler(val_ids)

                train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
                val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

                model = MLP(INPUT_SIZE).to(DEVICE) if model_name == "mlp" else CNN().to(DEVICE)
                criterion = nn.MSELoss()
                if optimizer_name == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_ADAM)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM_SGD)

                fold_best_state, fold_best_val_loss, train_losses, val_losses = train_fold(
                    model, train_loader, val_loader, criterion, optimizer, DEVICE, MAX_EPOCHS, PATIENCE, MIN_DELTA
                )

                key = f"{model_name}_{optimizer_name}"
                results[key]["fold_results"].append(fold_best_val_loss)
                results[key]["train_losses"].append(train_losses)
                results[key]["val_losses"].append(val_losses)
                if fold_best_val_loss < results[key]["best_val_loss"]:
                    results[key]["best_val_loss"] = fold_best_val_loss
                    results[key]["best_model_state"] = fold_best_state

            print(f"\n{model_name.upper()} with {optimizer_name.upper()} CROSS VALIDATION RESULTS")
            print("-" * 50)
            for fold, val_loss in enumerate(results[f"{model_name}_{optimizer_name}"]["fold_results"]):
                print(f"Fold {fold+1}: {val_loss:.6f}")
            mean_loss = np.mean(results[f"{model_name}_{optimizer_name}"]["fold_results"])
            std_loss = np.std(results[f"{model_name}_{optimizer_name}"]["fold_results"])
            print(f"Average validation loss: {mean_loss:.6f} (+- {std_loss:.6f})")

    best_model_key = min(results, key=lambda k: np.mean(results[k]["fold_results"]))

    print("\nFINAL COMPARISON")
    print("=" * 50)
    for key, data in results.items():
        mean_loss = np.mean(data["fold_results"])
        print(f"{key.upper()} average loss: {mean_loss:.6f}")

    print(f"\nBest model and optimizer: {best_model_key.upper()}")

    torch.save(
        {
            "model_state_dict": results[best_model_key]["best_model_state"],
            "input_size": INPUT_SIZE,
            "best_model_key": best_model_key,
            "results": results,
        },
        "complete_model_cnn.pth",
    )

    best_model = MLP(INPUT_SIZE).to(DEVICE) if best_model_key.startswith("mlp") else CNN().to(DEVICE)
    best_model.load_state_dict(results[best_model_key]["best_model_state"])
    return best_model, results

# Prediction Function
def load_and_predict(input_data):
    checkpoint = torch.load("complete_model_cnn.pth")
    model_key = checkpoint["best_model_key"]
    model = MLP(checkpoint["input_size"]).to(DEVICE) if model_key.startswith("mlp") else CNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_data = input_data.to(DEVICE)

    with torch.no_grad():
        prediction = model(input_data)
    return prediction

# Plotting Function
def plot_training_history(results):  
    """Plots training and validation loss for the best model."""
    checkpoint = torch.load("complete_model_cnn.pth")
    best_model_key = checkpoint["best_model_key"]

    n_folds = len(results[best_model_key]["train_losses"])
    fig, axes = plt.subplots(n_folds, 1, figsize=(8, n_folds * 4))
    if n_folds == 1:
        axes = [axes]
    for fold, (train_losses, val_losses) in enumerate(zip(results[best_model_key]["train_losses"],
                                                           results[best_model_key]["val_losses"])):
        axes[fold].plot(train_losses, label="Training Loss")
        axes[fold].plot(val_losses, label="Validation Loss")
        axes[fold].set_title(f"Fold {fold+1}")
        axes[fold].set_xlabel("Epoch")
        axes[fold].set_ylabel("Loss")
        axes[fold].legend()
    plt.tight_layout()
    plt.savefig(f"training_history_{best_model_key}_folds.png")
    plt.show()

    min_epochs = min(len(tl) for tl in results[best_model_key]["train_losses"])
    avg_train_losses = np.mean([tl[:min_epochs] for tl in results[best_model_key]["train_losses"]], axis=0)
    avg_val_losses = np.mean([vl[:min_epochs] for vl in results[best_model_key]["val_losses"]], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(avg_train_losses, label="Average Training Loss")
    ax.plot(avg_val_losses, label="Average Validation Loss")
    ax.set_title("Average Loss (All Folds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"training_history_{best_model_key}_average.png")
    plt.show()

def train():
    trained_model, best_results = train_model()
    plot_training_history(best_results)

def test():
    dataset = CustomDataset()
    test_input, test_target = dataset[0]
    test_input = test_input.unsqueeze(0).to(DEVICE)

    predicted_position = load_and_predict(test_input)
    
    print("\nTest Prediction:")
    print(f"Predicted position: {predicted_position.squeeze()}")
    print(f"Actual position: {test_target}")

if __name__ == "__main__":
    train()
    test()
