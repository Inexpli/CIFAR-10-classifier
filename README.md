# CIFAR-10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

<img width="590" height="448" alt="image" src="https://github.com/user-attachments/assets/4cac8db1-2a7a-4e44-a607-3d574e3ccc29" />

## Model with 92% Accuracy:

### Training variables:
```jupyter
Training dataset: 50000
Validation dataset: 5000
Test dataset: 5000 
```

```python
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_epochs)
torch.manual_seed(42)
num_of_epochs = 70
```

### Data Augmentation:
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Model Architecture:
```python
class MyResnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
```

![alt text](charts/image92-1.png)
![alt text](charts/image92-2.png)
![alt text](charts/image92-3.png)

<hr>
<br>

## Model with 89% Accuracy:

### Training variables:
```jupyter
Training dataset: 50000
Validation dataset: 5000
Test dataset: 5000 
```

```python
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
torch.manual_seed(42)
num_of_epochs = 55
```

### Data Augmentation:
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Model Architecture:
```python
class MyResnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
```

![alt text](charts/image89-1.png)
![alt text](charts/image89-2.png)
![alt text](charts/image89-3.png)
