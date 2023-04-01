import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchquantum.datasets import MNIST

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


class Net(nn.Module):
    def __init__(self, number):
        super(Net, self).__init__()
        self.number = number
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)



def get_loader(number):
    n_samples = 500

    X_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))

    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == number)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

    n_samples = 10

    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == number)[0][:n_samples])

    X_test.data = X_test.data[:n_samples]  # [idx]
    X_test.targets = X_test.targets[:n_samples]  # [idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train = False
    test = True
    if train:
        test_loader = []
        train_loader = []
        models = []

        for i in range(1, 10):
            a = get_loader(i)
            train_loader.append(a[0])
            test_loader.append(a[1])
            models.append(Net(i))

        for i, model in enumerate(models):

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_func = nn.NLLLoss()

            epochs = 10
            loss_list = []

            model.train()
            for epoch in range(epochs):
                total_loss = []
                for batch_idx, (data, target) in enumerate(train_loader[i]):

                    if target == torch.tensor([model.number]):
                        target = torch.tensor([1])
                    optimizer.zero_grad()
                    # Forward pass
                    output = model(data)
                    # Calculating loss
                    loss = loss_func(output, target)
                    # Backward pass
                    loss.backward()
                    # Optimize the weights
                    optimizer.step()

                    total_loss.append(loss.item())
                loss_list.append(sum(total_loss) / len(total_loss))
                print('Model {}, Training [{:.0f}%]\tLoss: {:.4f}'.format(i,100. * (epoch + 1) / epochs, loss_list[-1]))

            '''Save Model'''
            torch.save(model.state_dict(), "models/model{}".format(i))

        plt.plot(loss_list)
        plt.title('Hybrid NN Training Convergence')
        plt.xlabel('Training Iterations')
        plt.ylabel('Neg Log Likelihood Loss')
        # plt.show()
    else:
        test_loader = []
        train_loader = []
        models = []
        for i in range(0, 9):
            a = get_loader(i)
            train_loader.append(a[0])
            test_loader.append(a[1])
            model = Net(i)
            model.load_state_dict(torch.load("models/model{}".format(i)))
            models.append(model)
            print(test_loader[0])

    if test:
        # Concentrating on the first 100 samples
        n_samples = 100

        X_train = datasets.MNIST(root='./data', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

        # Leaving only labels 0 and 1
        idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                        np.where(X_train.targets == 1)[0][:n_samples]
                        )
        # idx = np.append(idx, np.where(X_train.targets == 2)[0][:n_samples])
        X_train.data = X_train.data[idx]
        X_train.targets = X_train.targets[idx]

        train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True);
        n_samples_show = 6

        data_iter = iter(train_loader)
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

        while n_samples_show > 0:
            images, targets = data_iter.__next__()

            axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))

            n_samples_show -= 1
        plt.show()

        predictions = []
        for i, model in enumerate(models):
            predictions.append([])
            model.eval()
            with torch.no_grad():
                correct = 0
                for _, (data, target) in enumerate(test_loader[0]):


                    output = model(data)
                    predictions[i].append(output)
                    print("Model", i)
                    print("Out: ", output)
                    print("Target: ", target)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    if target == torch.tensor([model.number]):
                        target = torch.tensor([1])

        print(predictions)

        for i, (_, target) in enumerate(test_loader[0]):
            pred = [sublist[i].numpy().tolist()[0] for sublist in predictions]
            print(pred)
            print(max(pred))
            print(pred.index(max(pred)))
            print("Target: ", target)
            print("---")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if count == n_samples_show:
                    break
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)

                axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

                axes[count].set_xticks([])
                axes[count].set_yticks([])
                axes[count].set_title('Predicted {}'.format(pred.item()))

                count += 1

        plt.show()
