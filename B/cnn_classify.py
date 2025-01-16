import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import BreastMNIST, BloodMNIST
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, image_size=50, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
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
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class BloodMNIST_Classifier:
    def __init__(self, model_name=None):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BloodMNIST dataset info
        self.data_flag = "bloodmnist"
        self.download = True
        self.dataset_dir = "Datasets/"
        self.info = INFO[self.data_flag]
        self.task = self.info["task"]
        self.n_channels = self.info["n_channels"]
        self.n_classes = len(self.info["label"])
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        # Load datasets
        self.train_dataset = BloodMNIST(
            split="train",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.val_dataset = BloodMNIST(
            split="val",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.test_dataset = BloodMNIST(
            split="test",
            transform=self.transform,
            download=self.download,
            root=self.dataset_dir,
        )
        self.image_size = self.train_dataset[0][0].shape[1]

        self.batch_size = 64
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
        self.model = SimpleCNN(self.image_size, self.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
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
            labels = labels.squeeze(1)
            images, labels = images.to(self.device), labels.to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                labels = labels.squeeze(1)
                images, labels = images.to(self.device), labels.to(self.device).long()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
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
                torch.save(self.model.state_dict(), "best_model_B.pth")  # save best model
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
        self.model.load_state_dict(torch.load("best_model.pth"))

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                labels = labels.squeeze(1)
                images, labels = images.to(self.device), labels.to(self.device).long()

                # Model forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Accumulate metrics
                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Collect all outputs and labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        # Concatenate all collected outputs and labels
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Convert labels to one-hot encoding
        y_true_one_hot = torch.eye(self.n_classes)[all_labels]

        # Calculate AUC per class
        auc_scores = []
        for i in range(self.n_classes):
            auc = roc_auc_score(
                y_true_one_hot[:, i].numpy(),
                all_outputs[:, i].numpy()
            )
            auc_scores.append(auc)

        # Macro-Averaged AUC
        macro_auc = sum(auc_scores) / self.n_classes
        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100 * correct / total
        # write results to a file
        if not os.path.exists("B/results"):
            os.makedirs("B/results")
        with open(f"B/results/{self.model_name}_results.txt", "a") as f:
            f.write(
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Macro-AUC: {macro_auc:.4f}\n"
            )
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Macro-AUC: {macro_auc:.4f}")

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
        # save plots
        if not os.path.exists("B/plots"):
            os.makedirs("B/plots")
        plt.savefig(f"B/plots/loss_{self.model_name}.png")
        plt.show()

        # Plotting the validation accuracy curve
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.savefig(f"B/plots/validation_accuracy_{self.model_name}.png")
        plt.show()
