
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import matplotlib.pyplot as plt

# Configuration
OBSERVATION_IMAGES_FILE = "data/imgs_observation_0.pt"
RESULT_IMAGES_FILE = "data/imgs_result_0.pt"
ACTIONS_FILE = "data/actions_0.pt"
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10
MIN_DELTA = 1e-4
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SPLIT = 0.8  # 80% training, 20% test
LATENT_DIM = 64  # Latent dimension for better representation

# Dataset
class ReconstructionDataset(Dataset):
    def __init__(self):
        self.observations = torch.load(OBSERVATION_IMAGES_FILE).float() / 255.0
        self.results = torch.load(RESULT_IMAGES_FILE).float() / 255.0
        self.actions = torch.load(ACTIONS_FILE)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = self.actions[idx]
        action_onehot = torch.zeros(4)
        action_onehot[int(action)] = 1
        return self.observations[idx], action_onehot, self.results[idx]

# VAE Model with Encoder, Decoder, and Latent Space
class VAE(nn.Module):
    def __init__(self, image_size=128, num_linear_features=4):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space (mean and log variance for the reparameterization trick)
        encoded_size = 128 * (image_size // 8) * (image_size // 8)
        combined_size = encoded_size + num_linear_features

        self.fc_mu = nn.Linear(combined_size, LATENT_DIM)
        self.fc_logvar = nn.Linear(combined_size, LATENT_DIM)

        # Decoder
        self.fc_decoder = nn.Linear(LATENT_DIM + num_linear_features, 128 * (image_size // 8) * (image_size // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, action):
        encoded = self.encoder(obs)
        combined = torch.cat((encoded, action), dim=1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        z_combined = torch.cat((z, action), dim=1)
        decoded = self.fc_decoder(z_combined).view(z_combined.size(0), 128, obs.size(2) // 8, obs.size(3) // 8)
        output = self.decoder(decoded)
        return output, mu, logvar

# VAE Loss Function
def loss_function(output, x, mu, logvar, batch_size):
    recon_loss = F.mse_loss(output, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + 0.002 * kl_loss

# Training function
def train_model(train_loader, test_loader):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_test_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses, test_losses = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_train_loss = 0
        for obs, action, result in train_loader:
            obs, action, result = obs.to(DEVICE), action.to(DEVICE), result.to(DEVICE)
            output, mu, logvar = model(obs, action)
            loss = loss_function(output, result, mu, logvar, obs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for obs, action, result in test_loader:
                obs, action, result = obs.to(DEVICE), action.to(DEVICE), result.to(DEVICE)
                output, mu, logvar = model(obs, action)
                loss = loss_function(output, result, mu, logvar, obs.size(0))
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

        if avg_test_loss < best_test_loss - MIN_DELTA:
            best_test_loss = avg_test_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping!")
            break

    torch.save(best_model_state, "vae_model.pth")
    return model, train_losses, test_losses

# Compare function for generated images
def compare_images(model, test_loader, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    test_iter = iter(test_loader)

    for i in range(num_samples):
        image_input, linear_input, target_image = next(test_iter)
        image_input, linear_input, target_image = image_input.to(DEVICE), linear_input.to(DEVICE), target_image.to(DEVICE)

        with torch.no_grad():
            predicted_image, _, _ = model(image_input, linear_input)
            predicted_image = predicted_image.cpu()

        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]

        ax[0].imshow(image_input.cpu().squeeze().permute(1, 2, 0))
        ax[1].imshow(target_image.cpu().squeeze().permute(1, 2, 0))
        ax[2].imshow(predicted_image.squeeze().permute(1, 2, 0))

    plt.tight_layout()
    plt.savefig("vae_results.png")
    plt.show()

def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("vae_losses.png")
    plt.show()


def train():
    dataset = ReconstructionDataset()
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    trained_model, train_losses, test_losses = train_model(train_loader, test_loader)
    plot_losses(train_losses, test_losses)

def test():
    dataset = ReconstructionDataset()
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = VAE().to(DEVICE)
    model.load_state_dict(torch.load("vae_model.pth"))

    compare_images(model, test_loader, num_samples=5)


# Main Execution
if __name__ == "__main__":
    train()
    test()
