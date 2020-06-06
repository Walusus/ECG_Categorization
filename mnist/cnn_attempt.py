import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sklearn.metrics as mtr
import numpy as np
import seaborn as sns
import gc
import math
from mnist.cnn_net_model import CnnNet
from torchsummary import summary


# Check for cuda presence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import the dataset
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(
    "dataset/", train=True, download=False,
    transform=transforms.Lambda(lambda x: transform(x).to(device=device)),
    target_transform=transforms.Lambda(lambda x: torch.tensor(x, device=device))
)
test_set = torchvision.datasets.MNIST(
    "dataset/", train=False, download=False,
    transform=transforms.Lambda(lambda x: transform(x).to(device=device)),
    target_transform=transforms.Lambda(lambda x: torch.tensor(x, device=device))
)

# Present few data samples
plt.figure()
for i in range(4):
    x, y = test_set[i]
    plt.subplot(221 + i)
    plt.imshow(transforms.ToPILImage()(x.cpu()), cmap='gray')
    plt.title(f"Label: {y:d}")
plt.savefig("plots/class_samples.png")
plt.show()

# Initialize loaders
train_batch_size = 150
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=250, shuffle=True)

# Import network motel and print summary
net = CnnNet().to(device=device)
summary(net, input_data=(1, 28, 28))

# Load model weights
ans = input("Load model? [y/n]")
if ans is 'y':
    filename = input("File name: ")
    net.load_state_dict(torch.load("weights/" + filename + ".pt"))

# Choose loss criterion, optimiser and learning rate
learning_rate = 1e-3
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_batch_num = math.ceil(len(train_set) / train_batch_size)
test_loss_track = []
train_loss_track = []
epochs_num = 15
for epoch in range(epochs_num):
    # Train network
    net.train()
    for batch_num, (inputs, labels) in enumerate(train_loader):
        optim.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        train_loss_track.append(loss.item())

        # Printing log each 5th batch
        if batch_num % 5 == 0:
            print(f"Epoch: {epoch+1:d}/{epochs_num:d} ({(epoch+1)/epochs_num*100:.1f}%),"
                  f"\tMini-batch: {batch_num+1:d} ({(batch_num+1)/train_batch_num*100:.1f}%),"
                  f"\tLoss: {loss.item():f}")
        # Cleanup
        del inputs, outputs, labels, loss
        gc.collect()

    # Test network every epoch
    net.eval()
    loss_sum = 0
    accuracy_sum = 0
    num = 0
    for batch_num, (inputs, labels) in enumerate(test_loader):
        outputs = net(inputs)
        _, pred_labels = outputs.cpu().detach().max(1)
        loss_sum += criterion(outputs, labels).item()
        accuracy_sum += mtr.accuracy_score(labels.cpu(), pred_labels)
        num += 1
        # Cleanup
        del inputs, outputs, labels, pred_labels, _
        gc.collect()

    accuracy = accuracy_sum / num
    loss = loss_sum / num
    test_loss_track.append(loss)
    print(f"Test accuracy: {100*accuracy:.1f}%,\tTest loss: {loss:f}")

# Plot learning results.
plt.figure(figsize=(8, 4))
plt.subplot(111)
plt.plot(np.linspace(1, epochs_num, num=len(train_loss_track)), train_loss_track, label="Train loss")
plt.plot(np.linspace(1, epochs_num, num=len(test_loss_track)), test_loss_track, label="Test loss")
plt.title(f"Epochs: {epochs_num:d}, batch size: {train_batch_size:d}, lr: {learning_rate:.1e}")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig("plots/train_report.png")
plt.show()

# Test network
accuracy_sum = 0
num = 0
conf_mat = np.zeros((10, 10), dtype=np.int)
for batch_num, (inputs, labels) in enumerate(train_loader):
    outputs = net(inputs)
    _, pred_labels = outputs.cpu().detach().max(1)
    accuracy_sum += mtr.accuracy_score(labels.cpu(), pred_labels)
    num += 1
    for i in range(len(pred_labels)):
        conf_mat[labels[i], pred_labels[i]] += 1

accuracy = accuracy_sum / num
print(f"\nTest set accuracy: {100*accuracy:.1f}%")

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, cmap="YlGnBu", annot=True, fmt="d", cbar=False, square=True)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title(f"Test set accuracy: {100*accuracy:.1f}%")
plt.savefig("plots/conf_mat.png")
plt.show()

# Save model weights
ans = input("Save model? [y/n]")
if ans is 'y':
    filename = input("Save as: ")
    torch.save(net.state_dict(), "weights/" + filename + ".pt")
