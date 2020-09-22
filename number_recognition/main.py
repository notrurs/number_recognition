import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def get_dataset():
    train = datasets.MNIST('', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ]))

    test = datasets.MNIST('', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ]))

    trainset = DataLoader(train, batch_size=10, shuffle=True)
    testset = DataLoader(test, batch_size=10, shuffle=False)

    return trainset, testset


def train_model(model, trainset):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = None
    for epoch in range(3):  # 3 full passes over the data
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            model.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = model(X.view(-1, 784))  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = loss_function(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!


def check_model_accuracy(model, testset):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testset:
            X, y = data
            output = model(X.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print(f'Accuracy: {round(correct / total, 3)}')


def prediction(model):
    image = cv2.imread('pred_data.png')
    gray_image = rgb2gray(image)
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()
    tensor_image = torch.tensor(gray_image)
    prediction_probabilities = model(tensor_image.float().view(-1, 784))[0]

    for i, prob in enumerate(prediction_probabilities):
        print(f'{i}: {prob}')

    print(f'Prediction: {int(torch.argmax(prediction_probabilities))}')


def main():
    trainset, testset = get_dataset()
    net = Net()

    try:
        net.load_state_dict(torch.load('model.model'))
    except FileNotFoundError:
        train_model(net, trainset)
        check_model_accuracy(net, testset)
        torch.save(net.state_dict(), 'model.model')
    else:
        print('Model found!')
    finally:
        prediction(net)


if __name__ == '__main__':
    main()
