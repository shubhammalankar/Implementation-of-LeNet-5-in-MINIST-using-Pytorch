import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

import Lenet5Impl

batch_size = 64
learnRate = 0.001
numEpoch = 10
deviceAvailable = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainingData = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                          download=True)

testingData = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.1325,), std=(0.3105,))]),
                                         download=True)

trainingLoader = torch.utils.data.DataLoader(dataset=trainingData,
                                             batch_size=batch_size,
                                             shuffle=True)

testingLoader = torch.utils.data.DataLoader(dataset=testingData,
                                            batch_size=batch_size,
                                            shuffle=True)

modelLeNet5 = Lenet5Impl.LeNet().to(deviceAvailable)
print(modelLeNet5)
cost = nn.CrossEntropyLoss()
optimise = torch.optim.Adam(modelLeNet5.parameters(), lr=learnRate)
totalStep = len(trainingLoader)
for epoch in range(numEpoch):
    for i, (img, label) in enumerate(trainingLoader):
        img = img.to(deviceAvailable)
        label = label.to(deviceAvailable)
        output = modelLeNet5(img)
        loss = cost(output, label)
        optimise.zero_grad()
        loss.backward()
        optimise.step()
        if (i + 1) % 400 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, numEpoch, i + 1, totalStep, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testingLoader:
        images = images.to(deviceAvailable)
        labels = labels.to(deviceAvailable)
        outputs = modelLeNet5(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('From 10000 test images we get the accuracy of the network as: {} %'.format(100 * correct / total))

figure = plt.figure(figsize=(8, 8))
columns, rows = 9, 5
plt.axis("off")
figure.add_subplot(rows, columns, 1)
for i in range(1, columns * rows + 1):
    sampleIndex = torch.randint(len(testingData), size=(1,)).item()
    sampleImage, sampleLable = testingData[sampleIndex]
    figure.add_subplot(rows, columns, i)
    with torch.no_grad():
        modelLeNet5.eval()
        probs = modelLeNet5(testingData[sampleIndex][0].unsqueeze(0))
    title = f'{torch.argmax(probs)} ({torch.max(probs * 10):.0f}%)'
    plt.title(title, fontsize=7)
    plt.imshow(sampleImage.squeeze(), cmap="gray")
plt.show()
