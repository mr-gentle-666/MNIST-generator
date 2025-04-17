import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# 时间嵌入
def timestep_embedding(timesteps,dim,max_period=1000):
    # 一半sin一半cos
    half = dim // 2
    # 计算每个周期的频率
    freqs = torch.exp(
        -math.log(max_period)*torch.arange(start=0,end=half,dtype=torch.float32)/half
    ).to(device=timesteps.device)
    args = timesteps[:,None].float()*freqs[None]
    embedding = torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
    
    # 如果dim是奇数，则填充一个0
    if dim%2:
        embedding = torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
    
    return embedding

# 定义一个抽象类，用于表示时间步的块
class TimestepBlock(nn.Module):
    def forward(self,x,t):
        pass



class TimestepEmbedSequential(nn.Sequential,TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input
    """
    def forward(self,x,t):
        for layer in self:
            if isinstance(layer,TimestepBlock):
                x = layer(x,t)
            else:
                x = layer(x)
            # 如果是timestempBlock，则传入x,t，如果是普通，则传入x
        return x

# 归一化层
def norm_layer(channels):
    return nn.GroupNorm(32,channels) # 32个分组

class ResidualBlock(TimestepBlock):
    def __init__(self,in_channels,out_channels,time_channels,dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(), # 激活函数
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        )

        #   时间嵌入
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels,out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout), # 丢弃率
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        )

    
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1) # 通道数不匹配，使用1x1卷积
        else:
            self.shortcut = nn.Identity() # 如果输入通道数等于输出通道数，则使用Identity()

    def forward(self,x,t):
        h = self.conv1(x)
        h += self.time_emb(t)[:,:,None,None] # 添加时间嵌入匹配输入通道数
        h = self.conv2(h)
        return h + self.shortcut(x) # 残差连接,h是新的

# 注意力块 ？？？？？
class AttentionBlock(nn.Module):
    def __init__(self,channels,num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels,channels*3,kernel_size=1,bias=False)
        self.proj = nn.Conv2d(channels,channels,kernel_size=1)

    def forward(self,x):
        batch_size,channels,height,width = x.shape
        x = self.norm(x) # 归一化
        qkv = self.qkv(x) # 卷积得到qkv
        q,k,v = qkv.reshape(batch_size*self.num_heads,-1,height*width).chunk(3,dim=1)
        scale = 1./math.sqrt(math.sqrt(channels//self.num_heads))
        attn = torch.einsum("bct,bcs->bts",q * scale,k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct",attn,v)
        h = h.reshape(batch_size,channels,height,width)
        h = self.proj(h)
        return x + h





class Downsample(nn.Module):
    def __init__(self,channels,use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            # 卷积核大小为3，步长为2，填充为1 缩小为原来的一半
            self.op = nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=1)
        else:
            # 使用平均池化进行下采样 kernel_size默认=stride
            self.op = nn.AvgPool2d(stride=2)

    def forward(self,x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self,channels,use_conv):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    
    def forward(self,x):
        # 使用F.interpolate进行上采样
        # 插值核大小为2，使用最近邻插值模式
        x = F.interpolate(x,scale_factor=2,mode="nearest")

        # 如果use_conv为True，则使用卷积进行上采样
        if self.use_conv:
            x = self.conv(x)
        
        return x


class Unet(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """
    def __init__(
        self,
        in_channels=1,  # 输入通道数，默认为3（适用于RGB图像）
        model_channels=128,  # 模型通道数，默认为128
        out_channels=1,  # 输出通道数，默认为3（适用于RGB图像）
        num_res_blocks=2,  # 残差块的数量，默认为2
        attention_resolutions=(8, 16),  # 注意力分辨率的元组，默认为(8, 16)
        dropout=0,  # Dropout概率，默认为0（不使用Dropout）
        channel_mult=(1, 2, 2, 2),  # 通道倍增因子的元组，默认为(1, 2, 2, 2)
        conv_resample=True,  # 是否使用卷积重采样，默认为True
        num_heads=4  # 注意力头的数量，默认为4
    ):
        super().__init__()

        # 初始化模型的各种参数
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # 时间嵌入（用于处理时间信息的嵌入）
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 下采样块
        #所有的模块都是先定义，然后通过迭代的方式往模块里面加东西
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]  # 存储下采样块每一阶段的通道数
        ch = model_channels  # 当前通道数初始化为模型通道数 初始为128
        ds = 1  # 下采样的倍数，初始值为1
		# 遍历不同阶段的下采样块
		#channel_mult模块为（1，2，2，2），下采样块每层的块数
        for level, mult in enumerate(channel_mult):
        	#num_res_blocks为残差块的数量，表示每块需要的残差快的数量
            for _ in range(num_res_blocks):
                layers = [
                #ch为输入通道数，mult * model_channels为需要输出的维度数，time_embed_dim为时间嵌入的维度
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                    #初始化剩余块，让我们后续能用forward函数将时间嵌入到x中
                ]
                ch = mult * model_channels
                #ds为一个值，一开始为1，然后每次乘以2，这里如果ds为8或者16时需要加上一个注意力模块
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                #将加入了残差快和注意力块的层加入下采样块当中
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                #记录每一层采样的通道数
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # 最后一个阶段不使用下采样
           		#这里由于之前的ch*2 所以，下采样后又恢复到了 ch，所以，我们在下采样通道中加入的ch
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
             #整个流程的格式变换，128，128，64；256,256，128；256,256；

        # 中间块
        #中间块就是一个残差块+注意力块+残差块
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # 上采样块
        self.up_blocks = nn.ModuleList([])
        #反过来计算通道的情况（2,2,2，1）
        for level, mult in list(enumerate(channel_mult))[::-1]:
        	#反向时残差块的数目为3
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                #如果level不为0，并且，i为2时（最后一块时）,进行上采样
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # 输出层
        #只是一个正则化，激活后的再一次不改变通道数的卷积
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, timesteps):
        """Apply the model to an input batch.

        Args:
            x (Tensor): [N x C x H x W]
            timesteps (Tensor): a 1-D batch of timesteps.

        Returns:
            Tensor: [N x C x ...]
        """
        #记录每次下采样得到结果，用于后面上采样的copy and crop
        hs = []
        # 时间步嵌入
        #利用timesteps参数，计算时间步的嵌入
        #首先用timestep_embedding,将时间序列timesteps（1*n）转化为（n*model_channels）
        #然后用time_embed将之前的n*model_channels转化为 n*time_embed_dim（也就是原来的mocel_channels*4）
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        #最终得到一个时间步嵌入的矩阵

        # 下采样阶段
        h = x
        for module in self.down_blocks:
        	#每次用时间步嵌入的矩阵信息emb，更新并记录每次的h
            h = module(h, emb)
            hs.append(h)
        
        # 中间阶段
        h = self.middle_block(h, emb)
        
        # 上采样阶段
        for module in self.up_blocks:
            # 从 hs 中弹出一个张量
            skip_connection = hs.pop()
            
            # 填充 h 的尺寸以匹配 skip_connection
            if skip_connection.shape[2:] != h.shape[2:]:
                diff_h = skip_connection.shape[2] - h.shape[2]
                diff_w = skip_connection.shape[3] - h.shape[3]
                h = F.pad(h, (0, diff_w, 0, diff_h))
            
            # 拼接并继续上采样
            cat_in = torch.cat([h, skip_connection], dim=1)
            h = module(cat_in, emb)
        
        return self.out(h)


if __name__ == '__main__':
    model = Unet(in_channels=3, model_channels=128, out_channels=3, num_res_blocks=2, attention_resolutions=(8, 16), dropout=0.1, channel_mult=(1, 2, 2, 2), conv_resample=True, num_heads=4)
    print(model)
    print(sum(p.numel() for p in model.parameters()))