import numpy as np
import torch


def main():
    X = torch.randn((5, 6), requires_grad=True)
    W = torch.randn((6, 4), requires_grad=True)
    b = torch.randn((4,), requires_grad=True)
    in_grads = torch.randn((5, 4))

    Y = torch.matmul(X, W) + b
    print("Y")
    print(Y)
    Y.backward(in_grads)
    print("X.grad")
    print(X.grad)
    print("W.grad")
    print(W.grad)
    print("b.grad")
    print(b.grad)

    print("SELF")
    X_np = X.detach().numpy()
    W_np = W.detach().numpy()
    b_np = b.detach().numpy()
    in_grads_np = in_grads.detach().numpy()

    Y_np = np.matmul(X_np, W_np) + b_np
    print("Y_np")
    print(Y_np)
    b_grad = in_grads_np.sum(axis=0)
    X_grad = np.matmul(in_grads_np, W_np.transpose())
    W_grad = np.matmul(X_np.transpose(), in_grads_np)
    print("X_grad")
    print(X_grad)
    print("W_grad")
    print(W_grad)
    print("b_grad")
    print(b_grad)

    ratio = np.isclose(Y.detach().numpy(), Y_np).sum() / Y_np.size
    print("Y ratio {}".format(ratio))
    ratio = np.isclose(X.grad.detach().numpy(), X_grad).sum() / X_grad.size
    print("X grad ratio {}".format(ratio))
    ratio = np.isclose(W.grad.detach().numpy(), W_grad).sum() / W_grad.size
    print("W grad ratio {}".format(ratio))
    ratio = np.isclose(b.grad.detach().numpy(), b_grad).sum() / b_grad.size
    print("b grad ratio {}".format(ratio))


if __name__ == "__main__":
    main()
