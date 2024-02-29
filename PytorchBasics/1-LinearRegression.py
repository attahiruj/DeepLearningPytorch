import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#data preparation
X_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples , n_features = X.shape

#model setup
input_size = n_features
output_size = 1
learning_rate = 0.01
model = nn.Linear(input_size, output_size)

#Loss and Optimization
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


'''
Training Loop
    - Foward Pass
    - Backward Pass
    - Weight update
'''
n_epochs = 1000
for epoch in range(n_epochs):
    # Foward
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward
    loss.backward()
        
    # Update
    optimizer.step()
    optimizer.zero_grad()
    
    if(epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1}\t| Loss:{loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X, y, 'ro')
plt.plot(X, predicted, 'b')
plt.show()
