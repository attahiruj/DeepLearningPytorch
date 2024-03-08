from msilib.schema import File
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data transforms
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transformer = {
	'train':
		transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]),
	'val':
		transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])
}
# load dataset from folder
data_dir = 'data/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transformer[x])
					for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
					for x in ['train', 'val']}

dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(f'\nData set class names: {class_names}')
print(f'\n{"="*50}\n')
# define training model
def train_model(model, criterion, optimizer, scheduler, n_epochs=25):
	start_time = time.time()
	best_model_weights = copy.deepcopy(model.state_dict())
	best_accuracy = 0.0

	for epoch in range(n_epochs):
		print(f'Epoch {epoch+1}/{n_epochs}')
		print('-' * 50)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# Foward
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, pred = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# Bacward > Optimize
					if phase == 'train':
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
				# Statistic
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(pred == labels.data)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_size[phase]
			epoch_accuracy = running_corrects.double() / dataset_size[phase]
			print(f'Phase: {phase}\t| Loss: {epoch_loss:.4f}\t| Accuracy: {epoch_accuracy:.4f}')

			# Deep copy model
			if phase == 'val' and epoch_accuracy > best_accuracy:
				best_accuracy = epoch_accuracy
				best_model_weights = copy.deepcopy(model.state_dict())
		print('-' * 50)
		print()

	end_time = time.time() - start_time
	print(f'\n{"="*50}\n')
	print(f' Training complete in {end_time//60:.0f}m {end_time%60:.0f}s')
	print(f' Best Accuracy: {best_accuracy:.4f}')
	print(f'\n{"="*50}\n')
	# Load best model weights
	model.load_state_dict(best_model_weights)
	return model

# loading a pre-trained ResNet-18 model
# from the torchvision library with pre-trained weights.
model = models.resnet18()
summary(model)
for param in model.parameters():
    param.requires_grad = False

n_features = model.fc.in_features
n_classes = 2
n_epoch = 1

model.fc = nn.Linear(n_features, n_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

bee_ant_model = train_model(model, criterion, optimizer, step_lr_scheduler, n_epoch)

#save model
FILE = "PytorchBasics/models/BeeAntClassifierModel.pth"
torch.save(bee_ant_model.state_dict(), FILE)
