import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Prepare data
breast_cancer_data = datasets.load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 2. Design Model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 3. Loss and Optimize
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 4. Training Loop
n_epoch = 10000
print(f'\n{"="*30}')
for epoch in range(n_epoch):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if(epoch + 1) % 100 == 0:
            print(f'| Epoch: {epoch+1}\t| Loss:{loss.item():.4f} |')
print(f'{"="*30}\n')

# 5. Test
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_classes = y_predicted.round()
    accuracy =  y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'\tAccuracy: {accuracy*100:.2f}%')
print(f'\n{"="*30}\n')