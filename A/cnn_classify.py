import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import BreastMNIST
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, image_size=50):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (image_size // 8) * (image_size // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class BreastMNIST_Classifier:
    def __init__(self, model_name=None):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BreastMNIST dataset info
        self.data_flag = "breastmnist"
        self.download = True
        self.dataset_dir = "Datasets/BreastMNIST/"
        self.info = INFO[self.data_flag]
        self.task = self.info["task"]
        self.n_channels = self.info["n_channels"]
        self.n_classes = len(self.info["label"])
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        # Load datasets
        self.train_dataset = BreastMNIST(
            split="train",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.val_dataset = BreastMNIST(
            split="val",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.test_dataset = BreastMNIST(
            split="test",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.image_size = self.train_dataset[0][0].shape[1]

        self.batch_size = 16
        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        # Initialize model, loss, optimizer, scheduler
        self.model_name = model_name
        self.model = SimpleCNN(self.image_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # Training and validation parameters
        self.num_epochs = 50
        self.best_val_loss = float("inf")
        self.early_stop_patience = 5
        # Initialize lists to store training and validation losses, validation accuracies
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # train_accuracy = 100 * correct / total
            # print(f'iteration {epoch * iterations + iter}, Train Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:

                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return running_loss / len(self.val_loader), accuracy

    def train_model(self):
        best_val_loss = float("inf")
        counter = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train()
            val_loss, val_accuracy = self.validate()
            # update learning rate scheduler
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

            # check early stop condition
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")  # save best model
                print("Validation loss improved. Saving model...")
            else:
                counter += 1
                print(
                    f"EarlyStopping counter: {counter} out of {self.early_stop_patience}"
                )
                if counter >= self.early_stop_patience:
                    print("Early stopping triggered. Stopping training.")
                    break

    def test(self):
        # Load the best model
        self.model.load_state_dict(torch.load("best_model_A.pth"))

        # Evaluate on test set
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_outputs += outputs.cpu().numpy().tolist()
                all_labels += labels.cpu().numpy().tolist()
        auc = roc_auc_score(all_labels, all_outputs)
        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100 * correct / total
        # write results to file without overwriting
        if not os.path.exists("A/results"):
            os.makedirs("A/results")
        with open(f"A/results/{self.model_name}_results.txt", "a") as f:
            f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, AUC: {auc:.4f}\n")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, AUC: {auc:.4f}")

    def plot_curves(self):
        # Plotting the training and validation loss curves
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        if not os.path.exists("A/plots"):
            os.makedirs("A/plots")
        plt.savefig(f"A/plots/loss_{self.model_name}.png")
        plt.show()

        # Plotting the validation accuracy curve
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.savefig(f"A/plots/valiadation_accuracy_{self.model_name}.png")
        plt.show()
