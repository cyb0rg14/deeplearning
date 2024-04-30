import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('./datasets/medical.csv')
print(data.sample(3))

plt.plot(data['bmi'], data['charges'])
plt.show()

# Split the dataset into train & test set
X = data['bmi']
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create Linear Regression class
class LinearRegression:
    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        numerator = 0
        denominator = 0

        for i in range(X.shape[0]):
            numerator += (X[i] - X.mean()) * (y[i] - y.mean())
            denominator += (X[i] - X.mean()) ** 2

        self.coef_ = numerator / denominator
        self.intercept_ = y.mean() - self.coef_ * X.mean()

    def predict(self, X):
        return self.coef_ * X + self.intercept_
    

model = LinearRegression()
model.fit(X_train.values, y_train.values)

for i in range(10):
    input = X_test.values[i]
    pred = model.predict(input)

    print(f'Actual: {y_test.values[i]}, Predicted: {pred}')