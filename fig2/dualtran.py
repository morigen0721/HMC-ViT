import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath

## 包含两个卷积层的卷积模块
class Conv_Layers(nn.Module):
    ## in_channels: 输入通道数
    def __init__(self,in_channels: int = 3):
        super().__init__()  # 调用父类的构造函数，初始化这个类
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, 3, stride = 1, padding = 1)  #参数：输入通道数，输出通道数，卷积核大小，步长，填充
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, 3, stride = 1, padding = 1)  #卷积核大小、步长、填充确保输出的空间尺寸与输入相同
    ## 定义前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:  #两个torch.Tensor是类型注释，去掉了也是一样的
        out = self.conv1(x)
        out = self.conv2(out)
        return out

        
## 将输入数据分割为patch，并将每个patch转换为嵌入向量，同时添加位置编码
class PatchEmbedding(nn.Module):
    ## in_channels: 输入通道数；patch_size: x和y方向patch大小默认为3；emb_size: 嵌入维度，这里是patch大小乘通道数；img_size: 原始图像的大小
    def __init__(self, in_channels: int = 3, patch_size_x: int = 9, patch_size_y: int = 9, emb_size: int = 248, img_size: int = 16):
        ## 把patch尺寸保存到类的属性中
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        ## 计算图像中patch的数量，加1是为了加入一个CLS token。这里假设了可以整除
        self.positions_num = (img_size - self.patch_size_x + 1) * (img_size - self.patch_size_y + 1) + 2
        super().__init__()
        ## 使用一个卷积层从原始图像中提取patch，并展平排列为适合transformer输入的形状。使用nn.Sequential将多个层或操作组合在一起。
        self.projection = nn.Sequential(
            ## 使用一个卷积层从原始图像中提取patch
            ## 使用patch_size大小的x、y、stride，使得卷积操作不会有重叠
            ## 原始图像为10通道，输出的特征图为emb_size个，emb_size也就是卷积核的个数
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_x, patch_size_y), stride = (1, 1)),
            nn.BatchNorm2d(emb_size),  #批归一化
            ## 展平，按指定方式重新排列张量维度
            ## (batch_size, emb_size, h, w) -> (batch_size, h*w, emb_size),h*w为patch_size
            ## 括号是为了指明需要操作的维度
            Rearrange('b e (h) (w) -> b (h w) e'),
        )       

        ## 用于表示整个输入的摘要信息，最后输出这个token进行分类。每个batch共享，只有一个标记，维度与patch维度一致
        self.cls_token_a = nn.Parameter(torch.randn(1, 1, emb_size)) 
        self.cls_token_b = nn.Parameter(torch.randn(1, 1, emb_size)) 
        ## 位置编码用于提供每个输入patch的位置信息，Transformer本身不具有位置信息感知能力。通过添加位置编码，可以让模型知道每个patch在原始图像中的位置
        ## self.positions_num：patch数量+一个cls_token
        ## 因为位置编码需要直接加在每一个patch上来表示每个patch的位置，所以有一个维度必须是emb_size
        self.positions = nn.Parameter(torch.randn(self.positions_num, emb_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape  #获取输入张量的形状，并提取batch_size
        x = self.projection(x)  #使用上面定义的方法，把原始数据转化为patch
        cls_tokens_a = repeat(self.cls_token_a, '() n e -> b n e', b=b)  #生成cls_token，并将其维度扩展至batch_size
        cls_tokens_b = repeat(self.cls_token_b, '() n e -> b n e', b=b)  #生成cls_token，并将其维度扩展至batch_size
        x = torch.cat([cls_tokens_a, x], dim=1)  #把cls_token和patch在第2个维度（patch）前进行拼接
        x = torch.cat([x, cls_tokens_b], dim=1)  #把cls_token和patch在第2个维度（patch）后进行拼接
        x += self.positions  #对所有patch加上位置编码，由于广播机制，这个操作会扩展到batch中的每一个数据上
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 248, num_heads: int = 8):
        super().__init__()
        self.emb_size = emb_size  #保存嵌入维度
        self.num_heads = num_heads  #保存多头注意力的头数
        self.qkv = nn.Linear(emb_size, emb_size * 3)  #定义一个线性层，用于把输入映射为qkv
        self.projection = nn.Linear(emb_size, emb_size)  #定义一个线性层，用于把多头注意力的输出映射为原始维度
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ## 得到qkv
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)  #计算qkv之后重组维度
        queries, keys, values = qkv[0], qkv[1], qkv[2]  #分别提取qkv

        ## 计算注意力分数
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  #定义乘法方式，计算QK之间的点积
        scaling = self.emb_size ** (1/2)  #计算缩放因子，对注意力得分进行缩放，通常是嵌入维度的平方根
        att = F.softmax(energy, dim=-1) / scaling  #对注意力得分进行softmax，将得分转换为概率分布
        
        ## 计算注意力输出
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  #使用注意力权重处理values
        out = rearrange(out, "b h n d -> b n (h d)")  #将多头的输出拼接在一起
        out = self.projection(out)  #将多头注意力的输出映射回原始维度
        return out
    
## 实现了一个残差连接
class ResidualAdd(nn.Module):
    ## fn是一个子网络模块或者一系列操作，总之是一个有输入输出的东西
    ## 将传入的fn保存为类的属性，从而在forward方法中使用
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    ## 定义残差块前向传播方法
    def forward(self, x, **kwargs):
        res = x  #保存输入为残差
        x = self.fn(x, **kwargs)  #调用fn，计算输出
        x += res  #输出+残差，得到残差块结果
        return x

## transformer编码器中的一个组件，前馈神经网络
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4):  #expansion: 前馈网络的扩展倍数
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),  #线性层，将输入映射到更高维度
            nn.GELU(),  #激活函数，gelu而不是relu
            nn.Linear(expansion * emb_size, emb_size),  #线性层，将高维度映射回原始维度
        )

## transformer编码器的基本组成部分，堆叠形成transformer编码器
## 包含一个多头注意力模块和一个前馈模块
class TransformerEncoderBlock(nn.Sequential):
    ## forward_expansion: 前馈模块的扩展倍数，指示了前馈网络中间层比输入层大几倍
    def __init__(self,
                 emb_size: int = 248,
                 forward_expansion: int = 4,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  #层归一化
                MultiHeadAttention(emb_size, **kwargs),  #多头注意力模块
                DropPath(0.2)  # 添加 DropPath
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  #层归一化
                FeedForwardBlock(emb_size, expansion=forward_expansion),    #前馈模块
                DropPath(0.2)  # 添加 DropPath
            )
            ))

## 继承自nn.Sequential，将多个transformerEncoderBlock堆叠到一起，形成transformerEncoder
class TransformerEncoder(nn.Sequential):
    ## depth定义了堆叠几个transformerEncoderBlock
    def __init__(self, depth: int = 4, **kwargs):
        ## __init__调用父类构造函数，将多个transformerEncoderBlock组合到一起
        ## 使用了列表生成式创建一个包含depth个TransformerEncoderBlock的列表。解包后传给父类的构造函数
        super().__init__(
            nn.Dropout(0.25),
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)],
            nn.LayerNorm(248)  #248是嵌入维度
        )

## 模型的最后一层，将transformer的输出映射到类别        
class ClassificationHead(nn.Module):
    def __init__(self, emb_size=248, num_classes=121, dropout_rate=0.25):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(emb_size * 2, emb_size * 8)  # 486 -> 1944
        self.fc2 = nn.Linear(emb_size * 8, emb_size * 4)  # 1944 -> 972
        self.final_fc = nn.Linear(emb_size * 4, num_classes)  # 121 -> num_classes

        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, emb_size)
        cls_token_first = x[:, 0, :]  # 提取第一个class token, shape: (batch_size, emb_size)
        cls_token_last = x[:, -1, :]  # 提取最后一个class token, shape: (batch_size, emb_size)

        # 拼接 class tokens
        x = torch.cat((cls_token_first, cls_token_last), dim=1)  # shape: (batch_size, emb_size * 2)

        # 通过全连接网络
        x = self.fc1(x)  # shape: (batch_size, 1944)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)  # shape: (batch_size, 972)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.final_fc(x)  # shape: (batch_size, num_classes)
        
        return x

class dualtran(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,  #输入通道数
                patch_size_x: int = 9,  #patch大小x
                patch_size_y: int = 9,  #patch大小y
                emb_size: int = 248,  #嵌入维度
                img_size: int = 16,  #原始图像大小
                depth: int = 4,  #transformer编码器深度
                n_classes: int = 121,  #分类类别数
                **kwargs):
        super().__init__(
            # Conv_Layers(),  #卷积模块
            PatchEmbedding(in_channels, patch_size_x, patch_size_y, emb_size, img_size),  #patch嵌入模块
            TransformerEncoder(depth, emb_size = emb_size, **kwargs),  #transformer编码器
            ClassificationHead(emb_size, n_classes)  #分类头
        )
