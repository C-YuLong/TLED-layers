import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import os

from torch.quantization import QuantStub, DeQuantStub

LENGTH = 8
HARDWARE_ADJUST = 1
HARDWARE_LAMBDA = 0.0005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lut_tensor = torch.zeros((256, 256), dtype=torch.uint8)

def gpu_show():
    for i in range(3):
        torch.cuda.set_device(i)

        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_cached(i)

        free_memory = total_memory - allocated_memory
        used_memory_percentage = (allocated_memory / total_memory) * 100

        print(f"GPU {i}:")
        print(f"  Total Memory: {total_memory / (1024**3):.2f} GB")
        print(f"  Allocated Memory: {allocated_memory / (1024**3):.2f} GB")
        print(f"  Cached Memory: {cached_memory / (1024**3):.2f} GB")
        print(f"  Free Memory: {free_memory / (1024**3):.2f} GB")
        print(f"  Used Memory Percentage: {used_memory_percentage:.2f}%\n")

def print_grad_hook(grad):
    print('Hook: Gradient:', grad)

cuda_module = load(
    name='Approximate_Dot',
    sources=['GPU-acclerate.cu', 'tled_layers.cpp'],
    verbose=True
)

def Approximate_Dot_(input, weight_t, approximate):
    height = input.size(0)
    k = input.size(1)
    assert k == weight_t.size(0)
    width = weight_t.size(1)
    output = torch.zeros(height, width, dtype=torch.float32).cuda()
    cuda_module.torch_Approximate_Dot(input, weight_t, output, height, k, width, approximate)
    return output

def loss_derivative(approximate, hardware_lambda):
    grad = 0
    if(math.floor(approximate) <= LENGTH):
        grad = (2.0*math.floor(approximate)+1.0) / (2.0*LENGTH*LENGTH) * hardware_lambda
    else:
        grad = (((2*LENGTH - 1) - 2*(math.floor(approximate)-LENGTH)) / (2.0*LENGTH*LENGTH)) * hardware_lambda
    return grad

def hardware_loss(approximate):
    if(torch.floor(approximate) <= LENGTH):
        loss = (1 - (approximate*(approximate+1) / (2*LENGTH*LENGTH))) * HARDWARE_LAMBDA
    else:
        loss = ( ((2*LENGTH-approximate)*(2*LENGTH-approximate-1)) / (2*LENGTH*LENGTH) ) * HARDWARE_LAMBDA
    return loss

class MyApproximate_Dot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, approximate):
        ctx.save_for_backward(input, weight)
        ctx.approximate = approximate
        return Approximate_Dot_(input, weight.t(), torch.floor(approximate))
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        approximate = ctx.approximate
        grad_input = grad_weight = grad_approximate_value = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(b)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(a)
        if ctx.needs_input_grad[2]:
            grad_approximate = torch.zeros(b.size(0), dtype=torch.float32).cuda()
            cuda_module.torch_Approximate_multiplication_derivative_GPU(b.t(), a, grad_approximate, a.size(0), a.size(1), b.size(0), approximate)
            grad_approximate_col = grad_approximate.unsqueeze(1)
            grad_approximate_value = torch.matmul(grad_output, grad_approximate_col)
            grad_approximate_value = (grad_approximate_value * HARDWARE_ADJUST)    
            mask = grad_approximate_value > 0.1
            grad_approximate_value[mask] = torch.where(
                grad_approximate_value[mask] == 0,  
                grad_approximate_value[mask],  
                (grad_approximate_value[mask] / (10 ** (torch.floor(torch.log10(grad_approximate_value[mask])) + 1))) * 0.01
            )
        return grad_input, grad_weight, -grad_approximate_value, None

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, approximate):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.approximate = approximate
        output = torch.zeros(input.size(0), weight.size(0), (input.size(2) + 2*padding[0] - weight.size(2))//stride + 1, (input.size(3) + 2*padding[1] - weight.size(3))//stride + 1, dtype=torch.float32)
        output = output.float().cuda()
        cuda_module.torch_Approximate_convolution(input, weight, output, stride, padding[0], padding[1], torch.floor(approximate))
        if bias is not None:
            output += bias.view(1, -1, 1, 1)  
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        approximate = ctx.approximate
        grad_input = grad_weight = grad_bias = grad_approximate = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        if ctx.needs_input_grad[5]:
            grad_approximate = torch.zeros(1, dtype=torch.float32).cuda()
            cuda_module.torch_Convolution_Parameter_Gradient_GPU(input, weight, grad_output, grad_approximate, stride, padding[0], padding[1], torch.floor(approximate))
            grad_approximate = grad_approximate * HARDWARE_ADJUST 
            mask = grad_approximate > 0.1
            grad_approximate[mask] = torch.where(
                grad_approximate[mask] == 0,  
                grad_approximate[mask],  
                
                (grad_approximate[mask] / (10 ** (torch.floor(torch.log10(grad_approximate[mask])) + 1))) * 0.01
            )

        return grad_input, grad_weight, grad_bias, None, None, -grad_approximate, None

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, approximate):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.approximate = nn.Parameter(torch.Tensor([1]))
        self.reset_parameters(approximate)
        self.custom_conv2d_function = CustomConv2dFunction.apply

    def reset_parameters(self, approximate_):
        nn.init.uniform_(self.weight, -1, 1)
        nn.init.constant_(self.approximate, approximate_)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        my_output = self.custom_conv2d_function(x, self.weight, self.bias, self.stride, self.padding, self.approximate)
        return my_output

class CustomDense(nn.Module):
    def __init__(self, in_features, out_features, approximate, bias=False):
        super(CustomDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.approximate_value = approximate
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) 
        self.approximate = nn.Parameter(torch.Tensor([1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.custom_dot = MyApproximate_Dot.apply

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1, 1)
        nn.init.constant_(self.approximate, self.approximate_value)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = self.custom_dot(input, self.weight, self.approximate)
        if self.bias is not None:
            output += self.bias.cuda()
        return output

class CustomQuanDense(nn.Module):
    def __init__(self, in_features, out_features, approximate, bias=False):
        super(CustomQuanDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.approximate_value = approximate
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

class CustomAnyDense(nn.Module):
    def __init__(self, in_features, out_features, lut_tensor, bias=True):
        super(CustomAnyDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lut_tensor = lut_tensor
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        weight_t = self.weight.t()
        height = input.size(0)
        k = input.size(1)
        assert k == weight_t.size(0)
        width = weight_t.size(1)
        output = torch.zeros(height, width, dtype=torch.float32).cuda()
        cuda_module.torch_Approximate_AnyDot(input, weight_t, output, height, k, width, self.lut_tensor)
        if self.bias is not None:
            output += self.bias
        return output

class CustomAnyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lut_tensor):
        super(CustomAnyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lut_tensor = lut_tensor
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        output = torch.zeros(input.size(0), self.weight.size(0), (input.size(2) + 2*self.padding[0] - self.weight.size(2))//self.stride + 1, (input.size(3) + 2*self.padding[1] - self.weight.size(3))//self.stride + 1, dtype=torch.float32)
        output = output.float().cuda()
        cuda_module.torch_Approximate_Anyconvolution(input, self.weight, output, self.stride, self.padding[0], self.padding[1], self.lut_tensor)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output

class TwoLayerFCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerFCModel, self).__init__()
        self.fc1 = CustomDense(input_size, hidden_size, approximate=2, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = CustomDense(hidden_size, output_size, approximate=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = CustomConv2d(1, 6, kernel_size=5, stride=1, padding=(2,2), approximate=2)
        self.conv2 = CustomConv2d(6, 16, kernel_size=5, stride=1, padding=(0,0), approximate=2)
        self.conv3 = CustomConv2d(16, 120, kernel_size=5, stride=1, padding=(0,0), approximate=2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)) 
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)

class FullLeNet(nn.Module):
    def __init__(self):
        super(FullLeNet, self).__init__()
        self.conv1 = CustomConv2d(1, 16, kernel_size=5, stride=1, padding=(2,2), approximate=2)
        self.conv2 = CustomConv2d(16, 32, kernel_size=5, stride=1, padding=(2,2), approximate=2)
        self.conv3 = CustomConv2d(32, 64, kernel_size=5, stride=1, padding=(2,2), approximate=2)
        self.fc1 = CustomDense(1024, 256, approximate=2, bias=True)
        self.fc2 = CustomDense(256, 10, approximate=2, bias=True)

    def forward(self, x):
        x_ = F.relu(self.conv1(x))
        x = F.max_pool2d(x_, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LeNetAny(nn.Module):
    def __init__(self, lut_tensor):
        super(LeNetAny, self).__init__()
        self.conv1 = CustomAnyConv2d(1, 16, kernel_size=5, stride=1, padding=(2,2), lut_tensor=lut_tensor)
        self.conv2 = CustomAnyConv2d(16, 32, kernel_size=5, stride=1, padding=(2,2), lut_tensor=lut_tensor)
        self.conv3 = CustomAnyConv2d(32, 64, kernel_size=5, stride=1, padding=(2,2), lut_tensor=lut_tensor)
        self.fc1 = CustomAnyDense(1024, 256, lut_tensor=lut_tensor)
        self.fc2 = CustomAnyDense(256, 10, lut_tensor=lut_tensor)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class preTrainLeNet(nn.Module):
    def __init__(self):
        super(preTrainLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=(2,2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=(2,2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2,2))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = self.relu2(self.conv2(x)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu3(self.conv3(x)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = torch.flatten(x, 1) 
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class PreTrainQuanLeNet(preTrainLeNet):
    def __init__(self):
        super(PreTrainQuanLeNet, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super(PreTrainQuanLeNet, self).forward(x)
        x = self.dequant(x)
        return x

def LeNetTrain():

    batch_size = 64
    file_name = './weights/lenet/'

    def load_approxflow_model(file_path, model):
        state_dict = torch.load(file_path)
        model.conv1.weight.data = state_dict['Conv1_weights'].data.cuda()
        model.conv1.bias.data = state_dict['Conv1_biases'].data.cuda()
        model.conv2.weight.data = state_dict['Conv2_weights'].data.cuda()
        model.conv2.bias.data = state_dict['Conv2_biases'].data.cuda()
        model.conv3.weight.data = state_dict['Conv3_weights'].data.cuda()
        model.conv3.bias.data = state_dict['Conv3_biases'].data.cuda()
        model.fc1.weight.data = state_dict['FC1_weights'].data.cuda()
        model.fc1.bias.data = state_dict['FC1_biases'].data.cuda()
        model.fc2.weight.data = state_dict['FC2_weights'].data.cuda()
        model.fc2.bias.data = state_dict['FC2_biases'].data.cuda()

    def custom_loss_function(outputs, targets, approximate1, approximate2, approximate3, approximate4, approximate5):
        total_loss_ = F.nll_loss(outputs, targets)
        total_loss = total_loss_ + hardware_loss(approximate1) + hardware_loss(approximate2) + hardware_loss(approximate3) + hardware_loss(approximate4) + hardware_loss(approximate5)
        return total_loss
    
    def train(model, device, train_loader, network_optimizer, approximate_optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            approximate_optimizer.zero_grad()
            network_optimizer.zero_grad()
            output = model(data)
            loss = custom_loss_function(output, target, model.conv1.approximate, model.conv2.approximate, model.conv3.approximate, model.fc1.approximate, model.fc2.approximate)
            loss.backward()
            approximate_optimizer.step()
            network_optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(model, device, test_loader, epoch, csv_path, weights_path):
        model.eval()
        test_loss = 0
        hardware_loss_value = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  
                hardware_loss_value += hardware_loss(model.conv1.approximate) + hardware_loss(model.conv2.approximate) + hardware_loss(model.conv3.approximate) + hardware_loss(model.fc1.approximate) + hardware_loss(model.fc2.approximate)
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        hardware_loss_value /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Hardware loss: {hardware_loss_value.item():.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
        print(f'conv1 approximate is {model.conv1.approximate}')
        print(f'conv2 approximate is {model.conv2.approximate}')
        print(f'conv3 approximate is {model.conv3.approximate}')
        print(f'fc1 approximate is {model.fc1.approximate}')
        print(f'fc2 approximate is {model.fc2.approximate}')

        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        torch.save(model.state_dict(), f'{weights_path}/lenet_epoch_{epoch}.pt')

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch', 'Test Loss', 'Hardware Loss', 'Accuracy', 'Conv1 Approximate', 'Conv2 Approximate', 'Conv3 Approximate', 'FC1 Approximate', 'FC2 Approximate'])
            writer.writerow([epoch, test_loss, hardware_loss_value.item(), accuracy, model.conv1.approximate.item(), model.conv2.approximate.item(), model.conv3.approximate.item(), model.fc1.approximate.item(), model.fc2.approximate.item()])

    def pre_train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Exact Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def pre_test(model, device, test_loader, epoch, weights_path):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f'\npreTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        torch.save(model.state_dict(), f'{weights_path}/lenet_pretrain_epoch_{epoch}.pt')

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=False)

    model = FullLeNet()

    approximate_parameters = [model.conv1.approximate, model.conv2.approximate, model.conv3.approximate, model.fc1.approximate, model.fc2.approximate]
    approximate_optimizer = optim.Adam(approximate_parameters, lr=0.001)

    network_parameters = [model.conv1.weight, model.conv2.weight, model.conv3.weight, model.fc1.weight, model.fc2.weight, model.conv1.bias, model.conv2.bias, model.conv3.bias, model.fc1.bias, model.fc2.bias]

    network_optimizer = optim.Adam(network_parameters, lr=0.0003)
    network_optimizer_scheduler = optim.lr_scheduler.ExponentialLR(network_optimizer, gamma=0.9)

    model = model.cuda()

    state_dict = torch.load('./weights/lenet5/lenet_pretrain_epoch_14.pt')

    model.conv1.weight.data = state_dict['conv1.weight'].data
    model.conv2.weight.data = state_dict['conv2.weight'].data
    model.conv3.weight.data = state_dict['conv3.weight'].data
    model.conv1.bias.data = state_dict['conv1.bias'].data
    model.conv2.bias.data = state_dict['conv2.bias'].data
    model.conv3.bias.data = state_dict['conv3.bias'].data
    model.fc1.weight.data = state_dict['fc1.weight'].data
    model.fc2.weight.data = state_dict['fc2.weight'].data
    model.fc1.bias.data = state_dict['fc1.bias'].data
    model.fc2.bias.data = state_dict['fc2.bias'].data

    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(15):
        train(model, device, train_loader, network_optimizer, approximate_optimizer, epoch)
        network_optimizer_scheduler.step()
        test(model, device, test_loader, epoch, csv_path=f'{file_name}data.csv', weights_path=f'{file_name}')

def LeNetTestAny():
    batch_size = 64
    def save_results_to_csv(result_path, weight_file, multiplier_file, accuracy):
        headers = ['Weight', 'Multiplier', 'Accuracy']
        file_exists = os.path.isfile(result_path)
        with open(result_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({'Weight': weight_file, 'Multiplier': multiplier_file, 'Accuracy': accuracy})

    def load_lut(multiplier_path):
        lut_tensor = torch.zeros((256, 256), dtype=torch.int32 )  
        with open(multiplier_path, 'r') as file:
            for i in range(256):
                line = next(file)
                lut_tensor[i] = torch.tensor([int(val) for val in line.split()], dtype=torch.long)
        return lut_tensor

    test_loader = DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=False)
    
    def load_weights(model, path):
        state_dict = torch.load(path)
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            original_model = model.module
        else:
            original_model = model

        custom_state_dict = original_model.state_dict()

        model.conv1.weight.data = state_dict['conv1.weight'].data
        model.conv2.weight.data = state_dict['conv2.weight'].data
        model.conv3.weight.data = state_dict['conv3.weight'].data
        model.conv1.bias.data = state_dict['conv1.bias'].data
        model.conv2.bias.data = state_dict['conv2.bias'].data
        model.conv3.bias.data = state_dict['conv3.bias'].data
        model.fc1.weight.data = state_dict['fc1.weight'].data
        model.fc2.weight.data = state_dict['fc2.weight'].data
        model.fc1.bias.data = state_dict['fc1.bias'].data
        model.fc2.bias.data = state_dict['fc2.bias'].data

    def test(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    multipliers_path = "./multipliers/"
    weights_path = "./weights/lenet7/"
    results_path = "./weights/LenetData/"
    os.makedirs(results_path, exist_ok=True)

    for multiplier_file in os.listdir(multipliers_path):
        lut_tensor = load_lut(os.path.join(multipliers_path, multiplier_file))
        
        print(f"Testing {multiplier_file}")
        for weight_file in os.listdir(weights_path):
            if weight_file.startswith("lenet_epoch_"):
                epoch_num = int(weight_file.split('epoch_')[1].split('.')[0])
                print(f"Testing {weight_file} with {multiplier_file}")
                test_model = LeNetAny(lut_tensor=lut_tensor).cuda()
                load_weights(test_model, os.path.join(weights_path, weight_file))
                accuracy = test(test_model, test_loader)
                result_filename = f"{multiplier_file}_with_{weight_file}_result.txt"
                with open(os.path.join(results_path, result_filename), "w") as result_file:
                    result_file.write(f"Weight: {weight_file}\n")
                    result_file.write(f"Multiplier: {multiplier_file}\n")
                    result_file.write(f"Accuracy: {accuracy}%\n")
                save_results_to_csv(os.path.join(results_path, "results.csv"), weight_file, multiplier_file, accuracy)

def SpecialLeNetTest():
    approximate_list = [0, 0, 0, 0, 0]
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=False)
    
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        hardware_loss_value = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  
                hardware_loss_value += hardware_loss(model.conv1.approximate) + hardware_loss(model.conv2.approximate) + hardware_loss(model.conv3.approximate) + hardware_loss(model.fc1.approximate) + hardware_loss(model.fc2.approximate)
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        hardware_loss_value /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Hardware loss: {hardware_loss_value.item():.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
        print(f'conv1 approximate is {model.conv1.approximate}')
        print(f'conv2 approximate is {model.conv2.approximate}')
        print(f'conv3 approximate is {model.conv3.approximate}')
        print(f'fc1 approximate is {model.fc1.approximate}')
        print(f'fc2 approximate is {model.fc2.approximate}')

    model = FullLeNet()

    for i in range(15):
        state_dict = torch.load(f'./weights/lenet7/lenet_epoch_{i}.pt')
        model.conv1.weight.data = state_dict['conv1.weight'].data
        model.conv2.weight.data = state_dict['conv2.weight'].data
        model.conv3.weight.data = state_dict['conv3.weight'].data
        model.conv1.bias.data = state_dict['conv1.bias'].data
        model.conv2.bias.data = state_dict['conv2.bias'].data
        model.conv3.bias.data = state_dict['conv3.bias'].data
        model.fc1.weight.data = state_dict['fc1.weight'].data
        model.fc2.weight.data = state_dict['fc2.weight'].data
        model.fc1.bias.data = state_dict['fc1.bias'].data
        model.fc2.bias.data = state_dict['fc2.bias'].data

        model.conv1.approximate = torch.nn.Parameter(torch.tensor(approximate_list[0], dtype=torch.float32), requires_grad=True)
        model.conv2.approximate = torch.nn.Parameter(torch.tensor(approximate_list[1], dtype=torch.float32), requires_grad=True)
        model.conv3.approximate = torch.nn.Parameter(torch.tensor(approximate_list[2], dtype=torch.float32), requires_grad=True)
        model.fc1.approximate = torch.nn.Parameter(torch.tensor(approximate_list[3], dtype=torch.float32), requires_grad=True)
        model.fc2.approximate = torch.nn.Parameter(torch.tensor(approximate_list[4], dtype=torch.float32), requires_grad=True)

        test(model, device, test_loader)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096),  
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ApproximateAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ApproximateAlexNet, self).__init__()
        self.features = nn.Sequential(
            CustomConv2d(3, 96, kernel_size=3, stride=1, padding=(1,1), approximate=7), 
            nn.ReLU(inplace=True),
            CustomConv2d(96, 96, kernel_size=3, stride=1, padding=(1,1), approximate=7), 
            nn.ReLU(inplace=True),
            CustomConv2d(96, 256, kernel_size=3, stride=1, padding=(1,1), approximate=7),   
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CustomConv2d(256, 384, kernel_size=3, stride=1, padding=(1,1), approximate=7), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CustomConv2d(384, 256, kernel_size=3, stride=1, padding=(1,1), approximate=7), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            CustomDense(256 * 4 * 4, 4096, approximate=7, bias=True), 
            nn.ReLU(inplace=True),
            CustomDense(4096, 4096, approximate=7, bias=True), 
            nn.ReLU(inplace=True),
            CustomDense(4096, num_classes, approximate=7, bias=True) 
        )

    def forward(self, x):
        x = self.features(x)
        
        x = self.classifier(x)
        return x

class AlexNetAny(nn.Module):
    def __init__(self, lut_tensor, num_classes=1000):
        super(AlexNetAny, self).__init__()
        self.features = nn.Sequential(
            CustomAnyConv2d(3, 96, kernel_size=3, stride=1, padding=(1,1), lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            CustomAnyConv2d(96, 96, kernel_size=3, stride=1, padding=(1,1), lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            CustomAnyConv2d(96, 256, kernel_size=3, stride=1, padding=(1,1), lut_tensor=lut_tensor),   
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CustomAnyConv2d(256, 384, kernel_size=3, stride=1, padding=(1,1), lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CustomAnyConv2d(384, 256, kernel_size=3, stride=1, padding=(1,1), lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            CustomAnyDense(256 * 4 * 4, 4096, lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            CustomAnyDense(4096, 4096, lut_tensor=lut_tensor), 
            nn.ReLU(inplace=True),
            CustomAnyDense(4096, num_classes, lut_tensor=lut_tensor) 
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def AlexnetTrain():
    multi_gpu = True
    batch_size = 32
    learning_rate = 0.01
    file_name = 'AlexNet14'
    weights_path = f'./weights/{file_name}/'
    
    def load_datasets():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return trainloader, testloader

    def load_weights(model, path):
        state_dict = torch.load(path)
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            original_model = model.module
        else:
            original_model = model
        custom_state_dict = original_model.state_dict()

        for name, param in state_dict.items():
            adjusted_name = name.replace('module.', '') if 'module.' in name else name
            if adjusted_name in custom_state_dict and param.size() == custom_state_dict[adjusted_name].size():
                
                custom_state_dict[adjusted_name].copy_(param)
            else:
                print(f"Skipping layer: {adjusted_name}, due to mismatch or non-existence in target model.")

        original_model.load_state_dict(custom_state_dict, strict=False)

    def custom_loss_function(outputs, targets, approximate1, approximate2, approximate3, approximate4, approximate5, approximate6, approximate7, approximate8):
        outputs = F.log_softmax(outputs, dim=1)
        total_loss_ = F.nll_loss(outputs, targets)
        total_loss = total_loss_ + hardware_loss(approximate1) + hardware_loss(approximate2) + hardware_loss(approximate3) + hardware_loss(approximate4) + hardware_loss(approximate5) + hardware_loss(approximate6) + hardware_loss(approximate7) + hardware_loss(approximate8)
        return total_loss
    
    def pre_train(model, device, train_loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        total_samples = len(train_loader.dataset)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            trained_samples = batch_idx * len(inputs)
            if batch_idx % 100 == 99 or batch_idx == num_batches - 1:  
                progress = 100. * (batch_idx + 1) / num_batches
                print(f'Epoch: {epoch + 1}: [{batch_idx + 1:5d}/{num_batches}] loss: {total_loss / total:.3f}, '
                    f'Progress: {progress:.2f}%, Samples: {trained_samples}/{total_samples}')

        return total_loss / len(train_loader), 100. * correct / total

    def pre_test(model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        print(f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')
        return test_loss, accuracy

    def train(model, train_loader, network_optimizer, approximate_optimizer, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        total_samples = len(train_loader.dataset)
        
        if multi_gpu:
            original_model = model.module
        else:
            original_model = model

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
            
            network_optimizer.zero_grad()
            approximate_optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = custom_loss_function(outputs, targets, original_model.features[0].approximate, original_model.features[2].approximate, original_model.features[4].approximate, original_model.features[7].approximate, original_model.features[10].approximate, original_model.classifier[1].approximate, original_model.classifier[3].approximate, original_model.classifier[5].approximate)
            loss.backward()
            network_optimizer.step()
            approximate_optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            trained_samples = batch_idx * len(inputs)
            if batch_idx % 100 == 99 or batch_idx == num_batches - 1:  
                progress = 100. * (batch_idx + 1) / num_batches
                print(f'Epoch: {epoch }: [{batch_idx + 1:5d}/{num_batches}] loss: {total_loss / total:.3f}, '
                    f'Progress: {progress:.2f}%, Samples: {trained_samples}/{total_samples}')

        return total_loss / len(train_loader), 100. * correct / total

    def test(model, test_loader, weights_path, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if multi_gpu:
            original_model = model.module
        else:
            original_model = model

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                
                loss = custom_loss_function(outputs, targets, original_model.features[0].approximate, original_model.features[2].approximate, original_model.features[4].approximate, original_model.features[7].approximate, original_model.features[10].approximate, original_model.classifier[1].approximate, original_model.classifier[3].approximate, original_model.classifier[5].approximate)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        print(f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')
        for name, module in model.named_children():
            print(f"Layer: {name}: ", end="")
            for param_name, param in module.named_parameters():
                if "appro" in param_name:
                    print(f"\tParam: {param_name}, Value: {param}")

        
        torch.save(model.state_dict(), weights_path + f'alexnet_epoch_appro{epoch}.pth')

        return test_loss, accuracy

    gpu_id = [0, 1, 2]
    model = ApproximateAlexNet(num_classes=10)
    model = nn.DataParallel(model, device_ids=gpu_id).to('cuda')
    params_with_appro = []  
    params_without_appro = []  

    for name, param in model.named_parameters():
        if "appro" in name:
            params_with_appro.append(param)
        else:
            params_without_appro.append(param)
    
    network_optimizer = optim.Adam(params_without_appro, lr=learning_rate*0.05)
    approximate_optimizer = optim.SGD(params_with_appro, lr=learning_rate*10)

    load_weights(model, './weights/AlexNet/alexnet_epoch12.pth')
    start_epoch = 0 
    
    train_loader, test_loader = load_datasets()

    os.makedirs(weights_path, exist_ok=True)
    results = []
    for epoch in range(20):  
        train_loss, train_acc = train(model, train_loader, network_optimizer, approximate_optimizer, start_epoch + epoch)
        print()
        test_loss, test_acc = test(model, test_loader, weights_path, start_epoch + epoch)
        print()

        current_result = [start_epoch + epoch, train_loss, train_acc, test_loss, test_acc, model.module.features[0].approximate.item(), model.module.features[2].approximate.item(), model.module.features[4].approximate.item(), model.module.features[7].approximate.item(), model.module.features[10].approximate.item(), model.module.classifier[1].approximate.item(), model.module.classifier[3].approximate.item(), model.module.classifier[5].approximate.item()]
        results.append(current_result)

        df = pd.DataFrame([current_result], columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'conv1.approximate', 'conv2.approximate', 'conv3.approximate', 'conv4.approximate', 'conv5.approximate', 'fc1.approximate', 'fc2.approximate', 'fc3.approximate'])

        df.to_csv(weights_path + 'data.csv', mode='a', index=False, header=(epoch==0))

def AlexnetPreTrain():
    learning_rate = 0.01
    batch_size = 16

    def pre_train(model, device, train_loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        total_samples = len(train_loader.dataset)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            trained_samples = batch_idx * len(inputs)
            if batch_idx % 100 == 99 or batch_idx == num_batches - 1:  
                progress = 100. * (batch_idx + 1) / num_batches
                print(f'Epoch: {epoch + 1}: [{batch_idx + 1:5d}/{num_batches}] loss: {total_loss / total:.3f}, '
                    f'Progress: {progress:.2f}%, Samples: {trained_samples}/{total_samples}')

        return total_loss / len(train_loader), 100. * correct / total

    def pre_test(model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        print(f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')
        torch.save(model.state_dict(), f'./weights/AlexNetAcc/alexnet_acc_epoch{epoch}.pth')
        return test_loss, accuracy

    def load_datasets():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return trainloader, testloader

    gpu_id = [0, 1, 2]
    model = CustomAlexNet(num_classes=10)
    model = nn.DataParallel(model, device_ids=gpu_id).to('cuda')

    network_optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.01)
    
    model.load_state_dict(torch.load('./weights/AlexNet0/alexnet_epoch12.pth'))

    train_loader, test_loader = load_datasets()
    criterion = nn.CrossEntropyLoss()

    os.makedirs('./weights/AlexNetAcc/', exist_ok=True)
    results = []
    for epoch in range(20):  
        train_loss, train_acc = pre_train(model, 'cuda:0', train_loader, network_optimizer, criterion, epoch)
        print()
        test_loss, test_acc = pre_test(model, 'cuda:0', test_loader, criterion)
        print()
        current_result = [epoch, train_loss, train_acc, test_loss, test_acc]
        df = pd.DataFrame([current_result], columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        df.to_csv('./weights/AlexNetAcc/data.csv', mode='a', index=False, header=(epoch==0))

def AlexnetAnyTest():

    def save_results_to_csv(result_path, weight_file, multiplier_file, accuracy):
        headers = ['Weight', 'Multiplier', 'Accuracy']
        file_exists = os.path.isfile(result_path)
        with open(result_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({'Weight': weight_file, 'Multiplier': multiplier_file, 'Accuracy': accuracy})

    def load_lut(multiplier_path):
        lut_tensor = torch.zeros((256, 256), dtype=torch.int32 )  
        with open(multiplier_path, 'r') as file:
            for i in range(256):
                line = next(file)
                lut_tensor[i] = torch.tensor([int(val) for val in line.split()], dtype=torch.long)

        return lut_tensor

    def load_weights(model, path):
        state_dict = torch.load(path)

        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            original_model = model.module
        else:
            original_model = model

        custom_state_dict = original_model.state_dict()

        for name, param in state_dict.items():
            adjusted_name = name.replace('module.', '') if 'module.' in name else name
            if adjusted_name in custom_state_dict and param.size() == custom_state_dict[adjusted_name].size():
                custom_state_dict[adjusted_name].copy_(param)
            else:
                print(f"Skipping layer: {adjusted_name}, due to mismatch or non-existence in target model.")

        original_model.load_state_dict(custom_state_dict, strict=False)

    def test(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
        return accuracy

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    multipliers_path = "./multipliers/"
    weights_path = "./weights/AlexNet10/"
    results_path = "./weights/AlexNetDataF/"
    os.makedirs(results_path, exist_ok=True)

    tested_multipliers = []

    for multiplier_file in os.listdir(multipliers_path):
        lut_tensor = load_lut(os.path.join(multipliers_path, multiplier_file))
        if multiplier_file in tested_multipliers:
            continue
        for weight_file in os.listdir(weights_path):
            if weight_file.startswith("alexnet_epoch_appro"):
                epoch_num = int(weight_file.split('epoch_appro')[1].split('.')[0])
                if epoch_num <= 12:
                    print(f"Testing {weight_file} with {multiplier_file}")
                    test_model = AlexNetAny(num_classes=10, lut_tensor=lut_tensor).cuda()
                    load_weights(test_model, os.path.join(weights_path, weight_file))
                    accuracy = test(test_model, test_loader)
                    result_filename = f"{multiplier_file}_with_{weight_file}_result.txt"
                    with open(os.path.join(results_path, result_filename), "w") as result_file:
                        result_file.write(f"Weight: {weight_file}\n")
                        result_file.write(f"Multiplier: {multiplier_file}\n")
                        result_file.write(f"Accuracy: {accuracy}%\n")
                    save_results_to_csv(os.path.join(results_path, "results.csv"), weight_file, multiplier_file, accuracy)

def AlexnetSpecialTest():
    
    multiplier_array = [0, 0, 0, 0, 0, 0, 0, 0]

    def set_approximate(model, multiplier_array):
        for name, module in model.named_children():
            for param_name, param in module.named_parameters():
                if "appro" in param_name:
                    param.data = torch.nn.Parameter(torch.tensor(multiplier_array[0], dtype=torch.float32), requires_grad=True)

    def test(model, test_loader):
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
        print(f"model conv1.approximate: {model.features[0].approximate.item()}")
        print(f"model conv2.approximate: {model.features[2].approximate.item()}")
        print(f"model conv3.approximate: {model.features[4].approximate.item()}")
        print(f"model conv4.approximate: {model.features[7].approximate.item()}")
        print(f"model conv5.approximate: {model.features[10].approximate.item()}")
        print(f"model fc1.approximate: {model.classifier[1].approximate.item()}")
        print(f"model fc2.approximate: {model.classifier[3].approximate.item()}")
        print(f"model fc3.approximate: {model.classifier[5].approximate.item()}")
        return accuracy

    def load_weights(model, path):
        state_dict = torch.load(path)
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            original_model = model.module
        else:
            original_model = model

        custom_state_dict = original_model.state_dict()

        for name, param in state_dict.items():
            adjusted_name = name.replace('module.', '') if 'module.' in name else name
            if adjusted_name in custom_state_dict and param.size() == custom_state_dict[adjusted_name].size():
                
                custom_state_dict[adjusted_name].copy_(param)
            else:
                print(f"Skipping layer: {adjusted_name}, due to mismatch or non-existence in target model.")
        original_model.load_state_dict(custom_state_dict, strict=False)

    def save_results_to_csv(result_path, weight_file, multiplier_file, accuracy):
        headers = ['Weight', 'Multiplier', 'Accuracy']
        file_exists = os.path.isfile(result_path)
        with open(result_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({'Weight': weight_file, 'Multiplier': multiplier_file, 'Accuracy': accuracy})

    model = ApproximateAlexNet(num_classes=10).cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    for i in range(12):
        print(f"Testing epoch{i} with all zeros")
        load_weights(model, f'./weights/AlexNet10/alexnet_epoch_appro{i}.pth')
        set_approximate(model, multiplier_array)
        accuracy = test(model, test_loader)
        save_results_to_csv("./weights/AlexNetDataF/results.csv", f"alexnet_epoch{i}.pth", "design_ware", accuracy)

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    print("PyTorch version:", torch.__version__)

    AlexnetSpecialTest()