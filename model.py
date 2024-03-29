from numpy import mod
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
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        
        
        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        
        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N)
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
    
class Blur(nn.Module):
    """Some Information about Blur"""
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x
    
class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self, input_channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x

class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, input_channels, output_channels, upsample=False, noise_gain=0.01, style_dim=512):
        super(GeneratorBlock, self).__init__()
        self.upsample = upsample
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self._noise_gain = noise_gain        
        self.affine1 = nn.Linear(style_dim, output_channels)
        self.conv1 = Conv2dMod(input_channels, output_channels, 3, eps=1e-8)
        self.bias1 = PixelWiseBias(output_channels)
        self.noise1 = NoiseInjection(gain=noise_gain)
        self.activation1 = nn.PReLU()
        
        self.affine2 = nn.Linear(style_dim, output_channels)
        self.conv2 = Conv2dMod(output_channels, output_channels, 3, eps=1e-8)
        self.bias2 = PixelWiseBias(output_channels)
        self.noise2 = NoiseInjection(gain=noise_gain)
        self.activation1 = nn.PReLU()
        
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
    def __init__(self, initial_channels = 512, style_dim=512, noise_gain=0.1):
        super(Generator, self).__init__()
        self.style_dim = style_dim
        self.alpha = 0
        self.layers = nn.ModuleList([])
        self.noise_gain = noise_gain
        self.last_channels = initial_channels
        self.first_layer = GeneratorBlock(initial_channels, initial_channels, upsample=False)
        self.const = nn.Parameter(torch.zeros(initial_channels, 4, 4))
        self.upsample = nn.Upsample(scale_factor=2)
        self.blur = Blur()
        
    def forward(self, y):
        if type(y) != list:
            y = [y for _ in range(len(self.layers)+1)]
        num_layers = len(self.layers)
        x = self.const.repeat(y[0].shape[0], 1, 1, 1)
        x, out = self.first_layer(x, y[0])
        for i in range(num_layers):
            x, rgb = self.layers[i](x, y[i+1])
            out = self.upsample(out)
            if i == num_layers - 1:
                out = self.blur(out) + rgb * self.alpha
            else:
                out += rgb
        return out
    
    def add_layer(self, channels):
        self.layers.append(GeneratorBlock(self.last_channels, channels, upsample=True, style_dim=self.style_dim))
        self.last_channels = channels
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
    
    @property
    def noise_gain(self):
        return self._noise_gain

    @noise_gain.setter
    def noise_gain(self, value):
        self._noise_gain = value
        for layer in self.layers:
            layer.noise_gain = value


class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, input_channels, output_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, 3, stride=1, padding=1)
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU(0.2)
        self.down_sample = nn.AvgPool2d(2, stride=2)
        self.channel_conv = nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0)
        self.from_rgb = nn.Conv2d(3, input_channels, 1, stride=1, padding=0)
        self.flag_downsample = downsample
    def forward(self, x):
        if self.flag_downsample:
            x_down = self.down_sample(x)
            x_down = self.channel_conv(x_down)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.flag_downsample:
            x = self.down_sample(x)
            x += x_down
        return x

class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, initial_channel=512):
        super(Discriminator, self).__init__()
        self.alpha = 0
        self.layers = nn.ModuleList([DiscriminatorBlock(initial_channel, initial_channel, downsample=False)])
        self.fc = nn.Linear(4*4*initial_channel+2, 1)
        self.last_channels = initial_channel
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, rgb):
        minibatch_std = torch.std(rgb, dim=(0)).mean().unsqueeze(0).repeat(rgb.shape[0], 1)
        color_std = torch.std(rgb, dim=1).mean((1, 2)).reshape(rgb.shape[0], 1)
        x = self.layers[0].from_rgb(rgb * self.alpha + self.upsample(self.downsample(rgb)) * (1 - self.alpha))
        for i, layer in enumerate(self.layers):
            if i == 1:
                x += self.layers[1].from_rgb(self.downsample(rgb) * (1 - self.alpha))
                x = layer(x)
            else:
                x = layer(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, minibatch_std, color_std), 1)
        x = self.fc(x)
        return x
    
    def add_layer(self, channels):
        self.layers.insert(0, DiscriminatorBlock(channels, self.last_channels))
        self.last_channels = channels
        
class EqualLinear(nn.Module):
    """Some Information about EqualLinear"""
    def __init__(self, input_dim, output_dim):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, latent_dim, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[EqualLinear(latent_dim, latent_dim) for _ in range(num_layers)])
    def forward(self, x):
        return self.seq(x)

class StyleBasedGAN(nn.Module):
    """Some Information about StyleBasedGAN"""
    def __init__(self, latent_dim, initial_channels=512, num_mapping_network_layers=8):
        super(StyleBasedGAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(initial_channels, style_dim=latent_dim)
        self.disccriminator = Discriminator(initial_channels)
        self.mapping_network = MappingNetwork(latent_dim, num_layers=num_mapping_network_layers)
    
    def add_layer(self, channels):
        self.generator.add_layer(channels)
        self.disccriminator.add_layer(channels)
        