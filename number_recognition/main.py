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
    """Neural network architecture"""
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
    """Returns prepared MNIST train and test dataset"""
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
    """Trains model"""
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = None
    for epoch in range(3):
        for data in trainset:
            X, y = data
            model.zero_grad()
            output = model(X.view(-1, 784))
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
        print(loss)


def check_model_accuracy(model, testset):
    """Validates trained model and prints accuracy"""
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
    """Get prediction"""
    image = cv2.imread('pred_data.png')
    gray_image = rgb2gray(image)

    # If yoy wanna to see you pred_data, uncomment two strings below
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()

    tensor_image = torch.tensor(gray_image)
    prediction_probabilities = model(tensor_image.float().view(-1, 784))[0]

    # Prints probs for each class
    for i, prob in enumerate(prediction_probabilities):
        print(f'{i}: {prob}')

    print(f'Prediction: {int(torch.argmax(prediction_probabilities))}')


def main():
    # Get train and test datasets
    trainset, testset = get_dataset()

    # Initialization NN model
    net = Net()

    # Try to load trained model
    try:
        net.load_state_dict(torch.load('model.model'))
    # If there is no model, train a new
    except FileNotFoundError:
        train_model(net, trainset)
        check_model_accuracy(net, testset)
        torch.save(net.state_dict(), 'model.model')
    # If there is model, print about it
    else:
        print('Model found!')
    # Get prediction
    finally:
        prediction(net)


if __name__ == '__main__':
    main()
