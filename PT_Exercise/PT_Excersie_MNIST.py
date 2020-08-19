import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

batch_size = 30
num_epochs = 10
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.dp2(x)
        out = self.fc3(x)
        return out

    def fit(self, criterion=nn.CrossEntropyLoss(),
              learning_rate=0.01,
              optimizer='default'):
        if optimizer == 'default':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)      # exponential decay

        acc = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            net.train()
            for i, (images, labels) in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs = self(images)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            # Exponential Decay for leraning rate:
            lr = scheduler.get_last_lr()[0]         # latest pytorch 1.5+ uses get_last_lr,  previously it was get_lr iirc;
            lr1 = optimizer.param_groups[0]["lr"]   # either the above line or this, both should do the same thing
            scheduler.step()

            acc.append(self.predict())
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {} \n\t Learning rate - old {}, new {}'.format(epoch,
                                        loss.item(), acc[epoch], lr, lr1))
        plt.plot(range(num_epochs), acc)
        plt.title("Pytorch - NN with two 512 layers")
        plt.grid()
        plt.show()

    def predict(self):
        # Calculate Accuracy
        correct = 0
        total = 0
        net.eval()
        # Iterate through test dataset
        for images, labels in test_loader:
            # Forward pass only to get logits/output
            outputs = self(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum()

        return np.round(100 * (np.divide(correct, total)), 2)



if __name__ == '__main__':
    net = Net()
    net.fit(optimizer='default')
