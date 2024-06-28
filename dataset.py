import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)

    print("the length of mnist:", len(mnist))

    img, id = mnist[10]
    img.show()

    tensor = ToTensor()(img)
    print(tensor.shape)  # torch.Size([1, 28, 28]) 单通道图片
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size):
    # 原始数据最大值为1，因为后续DDPM将图像和正态分布关联起来，希望图像颜色取值为-1，1，因此进行一个线性变化
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def get_img_shape():
    return (1, 28, 28)