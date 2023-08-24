import torch
import torch.nn as nn

class ZeroConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConvBatchNorm, self).__init__()

        # Define a zero convolutional layer with batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv.weight.data.fill_(0)  # Initialize weights with zeros
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)  # Apply the zero convolution operation
        out = self.bn(out)   # Apply batch normalization
        return out

class IdentityMappingModule(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(IdentityMappingModule, self).__init__()

        layers = []
        for _ in range(6):  # Create 6 pairs of zero conv and batch norm layers
            zero_conv_bn = ZeroConvBatchNorm(hidden_channels, hidden_channels)
            layers.append(zero_conv_bn)
        self.identity_module = nn.Sequential(*layers)

    def forward(self, x):
        out = self.identity_module(x)  # Apply the sequence of zero conv and batch norm layers
        return out + x  # Implement skip connection by adding input tensor to output tensor

# Example usage
input_channels = 3
hidden_channels = 3
input_tensor = torch.randn(8, input_channels, 32, 32)  # Generate example input tensor with batch size 8

identity_module = IdentityMappingModule(input_channels, hidden_channels)  # Create an instance of the identity mapping module
output_tensor = identity_module(input_tensor)  # Apply the identity mapping module to the input tensor

# Print input and output tensor information
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
print("Input and output tensors are equal:", torch.all(input_tensor == output_tensor))
