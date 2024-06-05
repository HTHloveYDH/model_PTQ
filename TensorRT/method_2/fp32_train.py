import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


torch.manual_seed(42)

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=6)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.resnet(x)


def train(net, trainloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0

    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(trainloader)


def test(net, testloader, criterion, device): 
    net.eval() 
    correct = 0 
    total = 0 

    with torch.no_grad(): 
        for data in testloader: 
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = net(inputs) 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 

    accuracy = 100 * correct / total 
    return accuracy 


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    net = Net() 
    net.to(device) 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) 
    
    
    # Training loop
    num_epochs = 20 
    for epoch in range(num_epochs): 
        train_loss = train(net, trainloader, criterion, optimizer, device) 
        test_accuracy = test(net, testloader, criterion, device) 
    
        torch.save(net.state_dict(), "../output/fp32/net_%d_%f.pth"%(epoch, test_accuracy))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%") 
    
    print("Training finished")
    
