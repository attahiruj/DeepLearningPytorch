import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchinfo import summary
import matplotlib.pyplot as plt

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper params
n_epochs = 50
batch_size = 128
learning_rate = 0.01

# Transform dataset from PIL images [0, 1] to normalized tensors [-1, 1]
transform = transforms.Compose([
								transforms.ToTensor(), 
								transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
								])

# load datasets
train_data = CIFAR10(   root='/../data',
						train=True, 
						download=True,
						transform=transform)
						
test_data = CIFAR10(    root='/../data',
						train=False, 
						download=False,
						transform=transform)
						
train_loader = DataLoader(  dataset=train_data,
							batch_size=batch_size,
							shuffle=True)
							
test_loader = DataLoader(   dataset=test_data,
							batch_size=batch_size,
							shuffle=False)

classes = (	'plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse',
			'ship', 'truck')


# ConvNet
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(16*5*5, 120)   # (n + 2p -f)/s + 1
		self.fc2 = nn.Linear(120, 84)        
		self.fc3 = nn.Linear(84, 10)    
			
	def forward(self, x):
		out = self.pool(F.relu(self.conv1(x)))
		out = self.pool(F.relu(self.conv2(out)))
		out = out.view(-1, 16*5*5)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out
	
model = ConvNet().to(device)
losses = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

summary(model, batch_dim=batch_size)
print(f'\n Bacth_size: {batch_size} | Learning_rate: {learning_rate} | Device: {device}\n')
print(f'{"="*65}\n')

start_time = time.time()
# training loop
n_total_set = len(train_loader)
for epoch in range(n_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		
		#Foward Pass
		outputs = model(images)
		loss  =criterion(outputs, labels)
		
		# backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	losses.append(loss.item())
	print(f'| Epoch: {epoch+1}/{n_epochs}\t| Loss: {loss.item():.4f} |')
print(f' {"="*30}\n')
end_time = time.time() - start_time

#test
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	n_class_correct = [0 for i in range(10)]
	n_class_samples = [0 for i in range(10)]
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		
		_, predictions = torch.max(outputs, 1)
		n_samples += labels.size(0)
		n_correct += (predictions == labels).sum().item()
		
		for i in range(10):
			label = labels[i]
			predicted = predictions[i]
			if (label == predicted):
				n_class_correct[label] += 1
			n_class_samples[label] += 1
			
	accuracy = n_correct * 100.0 / n_samples
	print(f'  Accuracy of Network: {accuracy:.2f}%')
	print(f'\n {"="*30}\n')
	for i in range(10):
		class_accuracy = n_class_correct[i] * 100.0 / n_class_samples[i]
		print(f'  Accuracy of {classes[i]}: {class_accuracy:.2f}%')	

print(f'\n {"="*30}\n')
print(f'  Training Time: {end_time/60:.2f} Minute(s)')
print(f'\n {"="*30}\n')
plt.plot(losses)
plt.show()