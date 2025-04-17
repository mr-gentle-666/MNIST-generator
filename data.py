import torchvision
path = "./data"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_data = torchvision.datasets.MNIST(root=path,train=True,transform=transform,download=True)
test_data = torchvision.datasets.MNIST(root=path,train=False,transform=transform,download=True)