import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size, eps=1e-8):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight) # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W) 
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape
        
        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w1 * w2

        # demodulate
        d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
        weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # convolution
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        x = F.conv2d(x, weight, stride=1, padding=1, groups=N)
        x = x.reshape(N, self.output_channels, H, W)

        return x
        
class NoiseInjection(nn.Module):
    """Some Information about NoiseInjection"""
    def __init__(self, gain=1.0):
        super(NoiseInjection, self).__init__()
        self.gain = gain
    def forward(self, x):
        x += (torch.randn(x.shape) * self.gain).to(x.device)
        return x

class PixelWiseBias(nn.Module):
    """Some Information about PixelWiseBias"""
    def __init__(self, channels):
        super(PixelWiseBias, self).__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))
        
    def forward(self, x):
        # x: (batch_size, channels, H, W)
        x = x + self.bias[None, :, None, None]
        return x
    
class BlurRGB(nn.Module):
    """Blur the 3x3 3 channels"""
    def __init__(self):
        super(BlurRGB, self).__init__()
        self.kernel = torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel.view(1, 1, 3, 3)
        self.kernel = self.kernel.repeat(3, 1, 1, 1)
    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])

class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self, input_channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, input_channels, output_channels, upsample=False, blur=False, noise_gain=0.1, style_dim=512):
        super(GeneratorBlock, self).__init__()
        self.upsample = upsample
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self._noise_gain = noise_gain
        
        self.affine1 = nn.Linear(style_dim, output_channels)
        self.conv1 = Conv2dMod(input_channels, output_channels, 3, eps=1e-8)
        self.bias1 = PixelWiseBias(output_channels)
        self.noise1 = NoiseInjection(gain=noise_gain)
        self.activation1 = nn.LeakyReLU()
        
        self.affine2 = nn.Linear(style_dim, output_channels)
        self.conv2 = Conv2dMod(output_channels, output_channels, 3, eps=1e-8)
        self.bias2 = PixelWiseBias(output_channels)
        self.noise2 = NoiseInjection(gain=noise_gain)
        self.activation1 = nn.LeakyReLU()
        
        self.to_rgb = ToRGB(output_channels)

    def forward(self, x, y):
        if self.upsample:
            x = self.upsample_layer(x)
            

        x = self.conv1(x, self.affine1(y))
        x = self.bias1(x)
        x = self.noise1(x)
        x = self.activation1(x)
        
        x = self.conv2(x, self.affine2(y))
        x = self.bias2(x)
        x = self.noise2(x)
        x = self.activation1(x)
        
        rgb = self.to_rgb(x)
        return x, rgb
    
    @property
    def noise_gain(self):
        return self._noise_gain

    @noise_gain.setter
    def noise_gain(self, value):
        self._noise_gain = value
        self.noise1.gain = value
        self.noise2.gain = value
    
def exists(thing):
    if thing:
        return True
    else:
        return False
    
class Generator(nn.Module):
    """Some Information about Generator"""
    """initial resolution: 4x4"""
    def __init__(self, initial_channels = 512):
        super(Generator, self).__init__()
        self.alpha = 0
        self.layers = nn.ModuleList([])
        self.last_channels = initial_channels
        self.first_layer = GeneratorBlock(initial_channels, initial_channels, upsample=False, blur=False)
        self.const = nn.Parameter(torch.zeros(initial_channels, 4, 4))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.blur = BlurRGB()
        
    def forward(self, y):
        num_layers = len(self.layers)
        x = self.const.repeat(y.shape[0], 1, 1, 1)
        x, out = self.first_layer(x, y)
        for i in range(num_layers):
            x, rgb = self.layers[i](x, y)
            out = self.blur(self.upsample(out)) + rgb
        return out / num_layers
        
    
    def add_layer(self, channels):
        self.layers.append(GeneratorBlock(self.last_channels, channels, upsample=True))
        self.last_channels = channels
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

G = Generator()
G.add_layer(256)
G.add_layer(128)

style = torch.randn(2, 512)
out = G(style)
print(out.shape)