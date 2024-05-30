import torch
import torch.nn as nn
import math
import os
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath

from common import layers as L

from common.utils import nlc_to_nchw,nchw_to_nlc

class BaseMiTB(nn.Module):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[3, 4, 18, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,     #freeze，so dropout=0
                 act_cfg='GELU',
                 with_cp=False,
                 pretrained = None):
        
        """
        The backbone of Segformer.

        This backbone is the implementation of `SegFormer: Simple and
        Efficient Design for Semantic Segmentation with
        Transformers <https://arxiv.org/abs/2105.15203>`_.
        Args:
            in_channels (int): Number of input channels. Default: 3.
            embed_dims (int): Embedding dimension. Default: 768.
            num_layers (Sequence[int]): The layer number of each transformer encode
                layer. Default: [3, 4, 6, 3].
            num_heads (Sequence[int]): The attention heads of each transformer
                encode layer. Default: [1, 2, 4, 8].
            patch_sizes (Sequence[int]): The patch_size of each overlapped patch
                embedding. Default: [7, 3, 3, 3].
            strides (Sequence[int]): The stride of each overlapped patch embedding.
                Default: [4, 2, 2, 2].
            sr_ratios (Sequence[int]): The spatial reduction rate of each
                transformer encode layer. Default: [8, 4, 2, 1].
            out_indices (Sequence[int] | int): Output from which stages.
                Default: (0, 1, 2, 3).
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
                Default: 4.
            qkv_bias (bool): Enable bias for qkv if True. Default: True.
            drop_rate (float): Probability of an element to be zeroed.
                Default 0.0
            attn_drop_rate (float): The drop out rate for attention layer.
                Default 0.0
            drop_path_rate (float): stochastic depth rate. Default 0.0
            act_cfg (str): The activation config for FFNs.
                Default: 'GELU'.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed. Default: False.
            pretrained (str, optional): model pretrained path. Default: None.
        """
        
        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = 4
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        self.pretrained = pretrained
        if pretrained is not None: assert os.path.isfile(self.pretrained), "Pretrained file not exist! Please check if the path is correct!"
        assert self.num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        self._hidden_sizes = []
        cur = 0
        self.layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            stage = LayerStage(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                num_layer=num_layer,
                patch_size=patch_sizes[i],
                stride=strides[i],
                norm=True,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * embed_dims_i,
                cur=cur,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratios[i]
            )

            in_channels = embed_dims_i
            self._hidden_sizes.append(embed_dims_i)

            self.layers.append(stage)
            cur += num_layer
    
        self.init_weights()

    def init_weights(self):
        
        if self.pretrained is not None:
            pretrained_weights = torch.load(self.pretrained,"cpu")
            statedict = self.state_dict()
            classname = self.__class__.__name__.lower()
            from pathlib import Path
            folder = Path(__file__).parent
            with open(os.path.join(folder,"pretrained",
                f"pretrained_ordered_{classname}_statedict.txt"),"r") as file:
                pretrained_dict = file.read().splitlines()

            missing = []
            for k, (kn,vn) in zip(pretrained_dict,statedict.items()):
                original_size = vn.size()
                try:
                    v = pretrained_weights[k]
                    if len(v.size()) < 3 and list(vn.size()[2:]) == [1,1]:
                        vn = vn.squeeze()
                except:
                    pass
                if vn.size() == pretrained_weights[k].size():
                    statedict[kn] = pretrained_weights[k].reshape(original_size)
                else: missing.append(f"{k}:{kn}")

            if len(missing): print(missing)

            self.load_state_dict(statedict)
            return
       
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
        
    @property
    def hidden_sizes(self):
        return self._hidden_sizes
    
    def forward(self, x):
        outs = []

        for i, stage in enumerate(self.layers):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

class LayerStage(nn.Module):
    """Implements one stage block of Segformer 
        PatchEmbedding + TrensformerEncoder Layers.

    Args:
        in_channels (int): The number of input channels.
        embed_dims (int): The feature dimension.
        num_layer (int): The number of transformer encoder layers.
        patch_size (int): The stride of each overlapped patch embedding.
        stride (int): The stride with which patches are extracted.
        norm (bool): Indicates whether normalization is applied in PatchEmbed class.
        num_heads (int): The number of parallel attention heads.
        feedforward_channels (int): The hidden dimension size for the feed-forward networks (FFNs).
        cur (int): The current layer index, typically used for tracking purposes within a series of layers.
        drop_rate (float): The probability of an element being zeroed after the feed-forward layer. 
            Default is 0.0.
        attn_drop_rate (float): The dropout rate for the attention layer. Default is 0.0.
        drop_path_rate (float): The stochastic depth rate, which is a form of regularization. 
            Default is 0.0.
        qkv_bias (bool): If True, enables bias for the query, key, and value projections. 
            Default is True.
        act_cfg (str): The activation configuration for the feed-forward networks. 
            Default is 'GELU'.
        with_cp (bool): If True, enables checkpointing to save memory at the cost of slower training speed. Default is False.
        sr_ratio (int): The spatial reduction ratio for Efficient Multi-head Attention in Segformer. Default is 1.

    """
    def __init__(self,
                 in_channels:int, embed_dims:int,
                 num_layer:int, patch_size:int,
                 stride:int, norm:bool,
                 num_heads:int,feedforward_channels:int,
                 cur:int, drop_rate:float,
                 attn_drop_rate:float, drop_path_rate:float,
                 qkv_bias:bool, act_cfg:str,
                 with_cp:bool, sr_ratio:int
                 ):
        """
        Args:
        ---
        in_channels (int): The number of input channels.
        embed_dims (int): The feature dimension.
        num_layer (int): The number of transformer encoder layers.
        patch_size (int): The size of the patches extracted from the input.
        stride (int): The stride with which patches are extracted.
        norm (bool): Indicates whether normalization is applied in PatchEmbed class.
        num_heads (int): The number of parallel attention heads.
        feedforward_channels (int): The hidden dimension size for the feed-forward networks (FFNs).
        cur (int): The current layer index, typically used for tracking purposes within a series of layers.
        drop_rate (float): The probability of an element being zeroed after the feed-forward layer. 
            Default is 0.0.
        attn_drop_rate (float): The dropout rate for the attention layer. Default is 0.0.
        drop_path_rate (float): The stochastic depth rate, which is a form of regularization. 
            Default is 0.0.
        qkv_bias (bool): If True, enables bias for the query, key, and value projections. 
            Default is True.
        act_cfg (str): The activation configuration for the feed-forward networks. 
            Default is 'GELU'.
        with_cp (bool): If True, enables checkpointing to save memory at the cost of slower training speed. Default is False.
        sr_ratio (int): The spatial reduction ratio for Efficient Multi-head Attention in Segformer. Default is 1.

        """
        super().__init__()

        self.patch_embed = L.PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims,
                kernel_size=patch_size,
                stride=stride,
                padding= patch_size // 2,
                norm=norm)
        
        self.tr_block = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate[cur+idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratio) for idx in range(num_layer)
            ])
        
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self,x):

        x, hw_shape = self.patch_embed(x)
        for block in self.tr_block:
            x = block(x, hw_shape)
        x = self.norm(x)
        x = nlc_to_nchw(x, hw_shape)
        return x


class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            MultiheadAttention. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (str): The activation config for FFNs.
            Default: 'GELU'
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:ConfigDict): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:mmcv.ConfigDict): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg,
                 ffn_drop=0.,
                 dropout_layer:bool = None,
                 drop_path_rate=0.0):
        
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = getattr(nn,act_cfg)() #nn.GELU()

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(drop_path_rate) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class Attention(nn.Module):
    def __init__(self, embed_dims, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_dims % num_heads == 0, f"dim {embed_dims} should be divided by num_heads {num_heads}."

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        

    def forward(self, q, k, v):

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        return x

class EfficientMultiheadAttention(nn.Module):
    """An implementation of Efficient Multi-head Attention of Segformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after nn.MultiheadAttention.
            Default: 0.0.
        dropout_layer (obj:ConfigDict): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 drop_path_rate=0.0,
                 batch_first=True,
                 qkv_bias=False,
                 sr_ratio=1):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        proj_drop,
        dropout_layer=dropout_layer,
        
        self.attn = Attention(embed_dims,
                                num_heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_drop)
        
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(drop_path_rate) if dropout_layer else nn.Identity()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                bias=True)
            self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        out = self.attn(x_q, x_kv, x_kv)
        out = self.proj(out)
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (str): The activation config for FFNs.
            Default: 'GELU'.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg="GELU",
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            drop_path_rate=drop_path_rate,
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            sr_ratio=sr_ratio)
        
        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            drop_path_rate = drop_path_rate,
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class MiTB0(BaseMiTB):
    def __init__(self,):
        super().__init__(
            in_channels=3,
            embed_dims=32,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0., 
            pretrained="common/backbones/pretrained/mit_b0.pth"  
        )

class MiTB1(BaseMiTB):
    def __init__(self,):
        super().__init__(
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[2, 2, 2, 2],
                 num_heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,  
                 pretrained="common/backbones/pretrained/mit_b1.pth"
        )

class MiTB2(BaseMiTB):
    def __init__(self,):
        super().__init__(
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pretrained="common/backbones/pretrained/mit_b2.pth"  
        )

class MiTB3(BaseMiTB):
    def __init__(self,):
        super().__init__(
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[3, 4, 18, 3],
                 num_heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,  
                 pretrained="common/backbones/pretrained/mit_b3.pth"
        )

class MiTB4(BaseMiTB):
    def __init__(self,):
        super().__init__(
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[3, 8, 27, 3],
                 num_heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pretrained="common/backbones/pretrained/mit_b4.pth"   
        )

class MiTB5(BaseMiTB):
    def __init__(self,):
        super().__init__(
                 in_channels=3,
                 embed_dims=64,
                 num_layers=[3, 5, 40, 3],
                 num_heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pretrained="common/backbones/pretrained/mit_b5.pth"  
        )