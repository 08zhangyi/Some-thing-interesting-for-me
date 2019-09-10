import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt


transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = torchvision.datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = (mean * image) + std
    plt.imshow(image, cmap='gray')


sample_data = next(iter(train_loader))
plot_img(sample_data[0][1])
plot_img(sample_data[0][2])