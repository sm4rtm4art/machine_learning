import numpy as np
import torch
import torch.nn as nn

width = 800
height = 600

x = torch.zeros(3, width, height)


class TestModule(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=1)
        )

    def forward(self, x):
        return self.net(x)


print("original data shape")
original = x.shape[1]
print(original)

print("#####################")


#
def estimator(original, kernel_size, stride, padding, dilation):
    result =np.ceil((original + 2 * padding - dilation * (kernel_size - 1) - 1) / stride )
    return str(int(result))


for kernel_size in [1, 2, 3]:
    for stride in [1, 2, 3]:
        for padding in [0, 1, 2]:
            for dilation in [0, 1, 2]:
                # Use f-strings for easy formatting
                print()
                print(f" {kernel_size}  {stride}  {padding}")
                model = TestModule(kernel_size, stride, padding)
                print(model(x).shape[1])
                print(estimator(original, kernel_size, stride, padding, dilation))
