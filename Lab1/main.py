import numpy as np
from neural_network import NeuralNetwork
from data_generate import generate_linear, generate_XOR_easy
from optimizer import SGD, StepLR

np.random.seed(42)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def BCE(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8) # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred_labels)

def train(nn, X, y, epochs=30000, lr=1, optimizer=None):
    print("\n======= Training Started =======")
    lr_scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = nn.forward(X)

        # Compute loss
        N = y.shape[0]
        loss = MSE(y, y_pred)
        dy = 2 * (y_pred - y) / N # Gradient of MSE loss w.r.t. predictions 
        losses.append(loss)

        # Backward pass
        optimizer.zero_grad()
        nn.backward(dy)

        # Update weights and biases
        optimizer.step()
        lr_scheduler.step(epoch)

        if epoch % 5000 == 0:
            print(f"epoch {epoch} loss: {loss}")
    print("======= Training Ended =======\n")
    return losses

def test(nn, X, y):
    print("\n======= Testing Started =======")
    y_pred = nn.forward(X)
    acc = accuracy(y, y_pred)
    for i in range(len(X)):
        print(f"Data {i}, Input: {X[i]}, True Label: {y[i][0]}, Predicted: {y_pred[i][0]:.4f}")
    print(f"Overall Accuracy: {acc:.4f}")
    print("======= Testing Ended =======\n")

def plot_loss_curve(losses_dict, figure_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Learning Curve', fontsize=18)
    for label, losses in losses_dict.items():
        plt.plot(range(len(losses)), losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()

def show_result(x, y, pred_y, figure_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.savefig(figure_name)

if __name__ == "__main__":
    # Generate data
    X_linear, y_linear = generate_linear(n=100)
    X_xor, y_xor = generate_XOR_easy()

    # Initialize neural network
    nn_linear = NeuralNetwork(h1_dim=4, h2_dim=4)
    nn_xor = NeuralNetwork(h1_dim=6, h2_dim=4)

    # Train on linear data
    print("Training on linear data...")
    optimizer_linear = SGD(parameters=nn_linear.parameters, lr=0.1)
    losses_linear = train(nn_linear, X_linear, y_linear, optimizer=optimizer_linear)
    test(nn_linear, X_linear, y_linear)
    y_pred_linear = (nn_linear.forward(X_linear) > 0.5).astype(int)
    show_result(X_linear, y_linear, y_pred_linear, "linear_pred_result.png")
    plot_loss_curve({"Linear": losses_linear}, "linear_loss_curve.png")

    # Train on XOR data
    print("\nTraining on XOR data...")
    optimizer_xor = SGD(parameters=nn_xor.parameters, lr=0.1)
    losses_xor = train(nn_xor, X_xor, y_xor, optimizer=optimizer_xor)
    test(nn_xor, X_xor, y_xor)
    y_pred_xor = (nn_xor.forward(X_xor) > 0.5).astype(int)
    show_result(X_xor, y_xor, y_pred_xor, "xor_pred_result.png")
    plot_loss_curve({"XOR": losses_xor}, "xor_loss_curve.png")