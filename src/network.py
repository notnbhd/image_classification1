import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5) # 32 - 5/1 = 27 + 1 = 28 -> (12, 28, 28)
        self.pool = nn.MaxPool2d(2, 2) # (12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5) # 14 - 5 / 1 + 1 = 10 -> (24, 10, 10) -> (24, 5, 5) -> Flatten (24 * 5 *5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # output must be 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def image_classify(img):
    net = NeuralNet()
    net.load_state_dict(torch.load('trained_net.pth'))


    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    net.eval()
    with torch.no_grad():
        output = net(img)
        _, pred = torch.max(output, 1)
    
    return class_names[pred.item()]