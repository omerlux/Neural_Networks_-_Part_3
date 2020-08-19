import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self,
                     trainloader,
                     optimizer,
                     criterion=nn.CrossEntropyLoss(),
                     epoch_num=2):
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

    def predict_net(self,
                    testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

if __name__ == '__main__':
    net = Net()
    net.train_net(trainloader, optimizer=optim.Adam(net.parameters(), lr=1e-4), epoch_num=10)
    net.predict_net(testloader)

# # TESTING
# if __name__ == '__main__':
#     # get some random training images
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()
#
#     # show images
#     imshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#     # Building the model
#     net = Net()
#     # Define loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#     for epoch in range(2):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#     print('Finished Training')
#     PATH = './cifar_net.pth'
#     torch.save(net.state_dict(), PATH)
#
#     # Test the network
#     dataiter = iter(testloader)
#     images, labels = dataiter.next()
#
#     # print images
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#     # # Loading back the model
#     # net = Net()
#     # net.load_state_dict(torch.load(PATH))
#
#     # Feedforward
#     outputs = net(images)
#     _, predicted = torch.max(outputs, 1)
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                   for j in range(4)))
#
#     # Accuracy over Test:
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#             100 * correct / total))
