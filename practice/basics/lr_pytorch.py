import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Read Dataset
data = pd.read_csv('./datasets/medical.csv')
print(data.sample(3))

X = data.iloc[:, 0:6]
y = data.iloc[:, 6]

# Label Encoding
label_encoder = LabelEncoder()
X['sex'] = label_encoder.fit_transform(X['sex'])
X['smoker'] = label_encoder.fit_transform(X['smoker'])
X['region'] = label_encoder.fit_transform(X['region'])

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Print the current dataset
print(X[0:5])

# Split the dataset into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train.values).float()
y_test = torch.from_numpy(y_test.values).float()

# Create Linear Regression class
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
    

# Create model
model = LinearRegression(6, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# set device
device = 'cpu'
model.to(device)

# Train the model
for epoch in range(100):

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test.unsqueeze(1))
    print(f"Mean Squared Error: {mse.item()}")

