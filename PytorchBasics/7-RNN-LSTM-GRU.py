import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary
#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
n_input_size = 28
n_seq_len = 28
n_layers = 2
n_hidden_size = 100
n_classes = 10
n_epochs = 10
batch_size = 4
learning_rate = 0.001

#load dataset
train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='/data', train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

samples, labels = next(iter(train_loader))
# print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2,3, i+1)
#     plt.imshow(samples[i][0], cmap='grey')
# plt.show()
# Many-to-one architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = layers
        self.rnn = nn.RNN(input_size, hidden_size, layers, batch_first=True) # X(shape) -> batch_size, seq_len, input/feature_size
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0) 
        # out(shape) = batch_size, seq_len, hidden_size 
        # out(shape) = (N, 28, 128)
        out = out[:, -1, :]
        # out(shape) = (N, 128)
        out = self.fc(out)
        return out
    
model = RNN(n_input_size, n_hidden_size, n_layers, n_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(summary(model, batch_dim=batch_size))

# Training loop
start_time = time.time()
n_total_steps = len(train_loader)
losses = []
print(f'\n {"="*30}')
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #reshape image: Flatten - 100, 1 , 28 ,28 -> 100, 748
        images = images.reshape(-1, n_seq_len, n_input_size).to(device)
        labels = labels.to(device)
        
        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # if(epoch + 1) % 10 == 0:
    losses.append(loss.item())
    print(f'| Epoch: {epoch+1}/{n_epochs}\t| Loss: {loss.item():.4f} |')
print(f' {"="*30}\n')
end_time = time.time() - start_time
#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, n_seq_len, n_input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    accuracy = n_correct * 100.0 / n_samples
    print(f'\tAccuracy: {accuracy:.2f}%')
print(f'\n {"="*30}\n')
print(f'  Training Time: {end_time/60:.2f} Minute(s)')
print(f'\n {"="*30}\n')
plt.plot(losses)
plt.show()