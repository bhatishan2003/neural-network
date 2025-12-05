import numpy as np
import argparse
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


class NeuralNetork:
    def __init__(self, input_size, hidden_size, activation="relu", lr=0.01):
        self.lr = lr
        self.activation_name = activation

        # Select activation manually (no dict)
        if activation == "relu":
            self.activation = self.relu
            self.activation_deriv = self.relu_derivative
        elif activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_deriv = self.sigmoid_derivative
        elif activation == "tanh":
            self.activation = self.tanh
            self.activation_deriv = self.tanh_derivative
        elif activation == "leaky_relu":
            self.activation = self.leaky_relu
            self.activation_deriv = self.leaky_relu_derivative
        else:
            raise ValueError("Unknown activation function")

        # ----- SIMPLE RANDOM INITIALIZATION (no Xavier/He) -----
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros((1, 1))

    # ---------------- ACTIVATIONS ---------------- #

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def sigmoid(self, x):
        # PURE SIGMOID (no clipping)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # ---------------- LOSS + DERIVATIVE ---------------- #

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    # ---------------- FORWARD ---------------- #

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    # ---------------- BACKWARD ---------------- #

    def backward(self, X, y):
        y_pred = self.z2

        dZ2 = self.mse_derivative(y, y_pred)
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_deriv(self.z1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


# ---------------- MAIN SCRIPT ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu", "leaky_relu"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=32)
    args = parser.parse_args()

    # Load dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model
    model = NeuralNetork(
        input_size=X.shape[1],
        hidden_size=args.hidden,
        activation=args.activation,
        lr=args.lr
    )

    # Training loop
    for epoch in range(args.epochs):
        pred = model.forward(X)
        loss = model.mse(y, pred)
        model.backward(X, y)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f})")

    # Predicted vs actual
    print("\n=== Predicted vs Actual (first 10 samples) ===")
    final_pred = model.forward(X)

    for i in range(10):
        print(f"Pred: {final_pred[i][0]:.4f}   Actual: {y[i][0]:.4f}")


if __name__ == "__main__":
    main()
