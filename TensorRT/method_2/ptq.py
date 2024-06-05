import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from utils import *
from fp32_train import Net, test


# Model
quant_modules.initialize()
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = Net()
print("Loading FP32 ckpt")
model.load_state_dict(torch.load("../output/fp32/net_18_76.680000.pth"))
model.cuda()


# Data
print("Preparing data")
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


# It is a bit slow since we collect histograms on CPU
print("Collecting histograms")
with torch.no_grad():
    collect_stats(model, trainloader, num_batches=2)
    compute_amax(model, method="percentile", percentile=99.99)


# Evaluate
criterion = nn.CrossEntropyLoss()   
device = "cuda" if torch.cuda.is_available() else "cpu" 
test_accuracy = test(model, testloader, criterion, device)
print(f"PTQ Acc: {test_accuracy:.2f}%")


# Save the model
torch.save(model.state_dict(), "../output/quant_resnet18-calibrated.pth")
print("Done")
