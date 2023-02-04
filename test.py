import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 数据转换为 tensor 格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net里面定义的网络模型，将模型转到GPU/CPU
model = MyLeNet5().to(device)

# 加载模型
model.load_state_dict(torch.load("save_model/best_model.pth"))

# 获取结果

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
# 把 tensor 转换为图片,方便可视化
show = ToPILImage()

# 进行验证

for i in range(20):
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show()
    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)

        predicated, actual = classes[torch.argmax(pred[0])], classes[y]

        print(f'predicated:"{predicated}",actual:"{actual}"')