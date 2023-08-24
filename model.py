import torch
import torch.nn as nn

class ZeroConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConvBatchNorm, self).__init__()

        # Define zero convolutional layer and batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv.weight.data.fill_(0)  # Set all weights to zero
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
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
        out = self.identity_module(x)
        return out + x  # Skip connection

# Example usage
input_channels = 3
hidden_channels = 3
input_tensor = torch.randn(8, input_channels, 32, 32)  # Example input tensor with batch size 8

identity_module = IdentityMappingModule(input_channels, hidden_channels)
output_tensor = identity_module(input_tensor)

print("Input tensor shape:", input_tensor)
print("Output tensor shape:", output_tensor)
print(input_tensor == output_tensor)

