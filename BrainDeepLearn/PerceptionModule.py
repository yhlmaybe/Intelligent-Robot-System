import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Iterable, Tuple, Any
from FunctionTools import GetParameterSScale, SiteSpec, BaseOnlineWrapper





def ProjectFroNorm(tensor: torch.Tensor, maxNorm: Optional[float]):
    if not maxNorm:
        return
    with torch.no_grad():
        n = torch.linalg.vector_norm(tensor, ord=2)
        if torch.isfinite(n) and (n > maxNorm):
            tensor.mul_(float(maxNorm) / (n + 1e-12))


def Norm2d(C: int, groups: int = 32, desiredCpg: int = 16, mincpg: int = 8) -> nn.Module:
    max_g = min(groups, C)

    candidates = [g for g in range(1, max_g + 1) if (C % g == 0) and (C // g >= mincpg)]

    if not candidates:
        candidates = [g for g in range(1, max_g + 1) if (C % g == 0)] 

    g = min(candidates, key=lambda d: abs((C // d) - desiredCpg))

    return nn.GroupNorm(num_groups=g, num_channels=C, affine=True)


class GrowableLoRAConv2d(nn.Module):
    def __init__(self, targetConv: nn.Conv2d):
        super().__init__()
        self.target = targetConv 
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

        w = self.target.weight
        self.cout, self.cin, self.kh, self.kw = w.shape

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        ksz = self.kh * self.kw
        if init is None: init = {}

        dev = self.target.weight.device
        dt = self.target.weight.dtype

        A = init.get("A", torch.randn(addRank, self.cin * ksz, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.randn(self.cout, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=A.device, dtype=A.dtype))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def DeltaWeight(self):
        if len(self.A_list) == 0:
            return None
        ksz = self.kh * self.kw
        delta = self.target.weight.new_zeros(self.cout, self.cin * ksz)
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            delta = delta + torch.tanh(s) * GetParameterSScale(s) * (B @ A)
        return delta.view(self.cout, self.cin, self.kh, self.kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.target.weight

        if hasattr(self.target, "Preprocess"):
            x = self.target.Preprocess(x)

        delta = self.DeltaWeight()
        if delta is not None:
            w = w + delta
        return F.conv2d(x, w, self.target.bias, stride=self.target.stride, padding=self.target.padding, dilation=self.target.dilation, groups=self.target.groups)


class GrowableConv1x1Adapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

        self.register_buffer("_anchor", torch.empty(0))

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        A = init.get("A", torch.randn(addRank, self.C, 1, 1) * 1e-4)
        B = init.get("B", torch.randn(self.C, addRank, 1, 1) * 1e-4)
        s = init.get("scale", 1e-3)

        dev, dt = self._anchor.device, self._anchor.dtype
        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.A_list) == 0:
            return x
        y = x
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            z = F.conv2d(x, A, bias=None, stride=1, padding=0)
            z = F.conv2d(z, B, bias=None, stride=1, padding=0)
            y = y + torch.tanh(s) * GetParameterSScale(s) * z
        return y


class GrowableTokenAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.D = dim
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

        self.register_buffer("_anchor", torch.empty(0))

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        
        A = init.get("A", torch.randn(addRank, self.D) * 1e-4)
        B = init.get("B", torch.randn(self.D, addRank) * 1e-4)
        s = init.get("scale", 1e-3)

        dev, dt = self._anchor.device, self._anchor.dtype
        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.A_list) == 0:
            return x
        y = x
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            z = torch.matmul(x, A.t())
            z = torch.matmul(z, B.t())
            y = y + torch.tanh(s) * GetParameterSScale(s) * z
        return y

class SheafGaugeConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        sheaf_alpha: float = 0.1, 
        sheaf_iters: int = 1,  
        gauge_groups: int = 1, 
        gauge_scale: float = 0.1,  
        gauge_bias_scale: float = 0.1,
        eps: float = 1e-5, ):
        factory = {"device": device, "dtype": dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,dilation, groups, bias, padding_mode, **factory)

        self.sheaf_alpha = nn.Parameter(torch.tensor(float(sheaf_alpha), **factory))
        self.sheaf_iters = int(sheaf_iters)
        self.eps = float(eps)
        self.sheaf_gain_h = nn.Parameter(torch.ones(in_channels, **factory))
        self.sheaf_gain_v = nn.Parameter(torch.ones(in_channels, **factory))

        assert in_channels % gauge_groups == 0, "gauge_groups must be divisible by in_channels"
        self.gauge_gamma = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=gauge_groups, bias=True, **factory)
        self.gauge_beta = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=gauge_groups, bias=True, **factory)

        nn.init.zeros_(self.gauge_gamma.weight)
        nn.init.zeros_(self.gauge_gamma.bias)
        nn.init.zeros_(self.gauge_beta.weight)
        nn.init.zeros_(self.gauge_beta.bias)

        self.gauge_scale = float(gauge_scale)
        self.gauge_bias_scale = float(gauge_bias_scale)

    @staticmethod
    def Shift(x: torch.Tensor, dim: int, step: int) -> torch.Tensor:
        if step == 0:
            return x
        y = torch.roll(x, shifts=step, dims=dim)
        if step > 0:
            sl = [slice(None)] * y.ndim
            sl[dim] = slice(0, step)
            y[tuple(sl)] = x[tuple(sl)]
        else:
            sl = [slice(None)] * y.ndim
            sl[dim] = slice(step, None)
            y[tuple(sl)] = x[tuple(sl)]
        return y

    def SheafStep(self, x: torch.Tensor) -> torch.Tensor:
        if self.sheaf_iters <= 0:
            return x
        g_h = self.sheaf_gain_h.view(1, -1, 1, 1)
        g_v = self.sheaf_gain_v.view(1, -1, 1, 1)

        x_cur = x
        for _ in range(self.sheaf_iters):
            left = self.Shift(x_cur, dim=-1, step=-1)
            right = self.Shift(x_cur, dim=-1, step=+1)
            up = self.Shift(x_cur, dim=-2, step=-1)
            down = self.Shift(x_cur, dim=-2, step=+1)

            h_mean = 0.5 * (left + right)
            v_mean = 0.5 * (up + down)

            msg = g_h * (h_mean - x_cur) + g_v * (v_mean - x_cur)
            x_cur = x_cur + self.sheaf_alpha * msg
        return x_cur

    def GaugeStep(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()

        gamma = torch.tanh(self.gauge_gamma(x_norm)) * self.gauge_scale
        beta  = torch.tanh(self.gauge_beta(x_norm))  * self.gauge_bias_scale
        return (1.0 + gamma) * x + beta

    def Preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self.SheafStep(x)
        x = self.GaugeStep(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Preprocess(x)
        return super().forward(x)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        extras = (f"sheaf_alpha={float(self.sheaf_alpha):.4f}, "
                  f"sheaf_iters={self.sheaf_iters}, "
                  f"gauge_groups={self.gauge_gamma.groups}")
        return base + ", " + extras



class HebbianConv2d(nn.Module):
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int,
        stride: int = 1,
        padding: int = 0,
        hebbRate: float = 1e-3,
        emaMomentum: float = 0.995,
        applyScale: float = 0.25,
        memNormCap: Optional[float] = 1.0,
        bias: bool = False,
        useHebbian: bool = False,):
        super().__init__()

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding=padding, bias=bias)
        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap
        self.enable_hebbian_updates = useHebbian
        self.register_buffer("hebb_memory", torch.zeros_like(self.conv.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_hebbian_updates:
            weight_eff = self.conv.weight + self.apply_scale * self.hebb_memory.detach()
        else:
            weight_eff = self.conv.weight

        out = F.conv2d(
            x, weight_eff, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups)

        if self.enable_hebbian_updates:
            with torch.no_grad():
                kH, kW = self.conv.kernel_size

                x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=self.conv.padding, stride=self.conv.stride)  # [B, Cin*kH*kW, L]

                out_unfold = out.view(out.size(0), out.size(1), -1)  # [B, Cout, L]

                # Hebb: y x^T；Decay: <y^2> * W
                hebb_term = torch.einsum('bik,bjk->ij', out_unfold, x_unfold)  # [Cout, Cin*kH*kW]
                weight_flat = self.conv.weight.view(self.conv.weight.size(0), -1)

                y2_sum = out_unfold.square().sum(dim=[0, 2])
                decay_term = y2_sum.unsqueeze(1) * weight_flat

                delta_w = self.hebb_rate * (hebb_term - decay_term)
                delta_w = delta_w.view_as(self.hebb_memory)

                self.hebb_memory.mul_(self.ema_alpha).add_(delta_w, alpha=(1.0 - self.ema_alpha))
                ProjectFroNorm(self.hebb_memory, self.mem_norm_cap)
        return out

    def ResetHebbianMemory(self):
        with torch.no_grad():
            self.hebb_memory.zero_()

class HebbianLinear(nn.Module):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,
        hebbRate: float = 1e-3,
        emaMomentum: float = 0.995,
        applyScale: float = 0.2,
        memNormCap: Optional[float] = 1.0,
        normalize: bool = False,
        weightConstraint: Optional[str] = None,
        bias: bool = True,
        useHebbian: bool = False,):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)
        self.bias = nn.Parameter(torch.zeros(outFeatures)) if bias else None

        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap
        self.normalize = normalize
        self.weight_constraint = weightConstraint
        self.enable_hebbian_updates = useHebbian

        self.register_buffer("hebb_memory", torch.zeros(outFeatures, inFeatures))
        if normalize:
            self.register_buffer("running_mean", torch.zeros(outFeatures))
            self.register_buffer("running_var", torch.ones(outFeatures))
            self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_hebbian_updates:
            w_eff = self.weight + self.apply_scale * self.hebb_memory.detach()
        else:
            w_eff = self.weight
        y = F.linear(x, w_eff, self.bias) 

        if self.normalize:
            if self.training:
                with torch.no_grad():
                    mean = y.mean(0)
                    var = y.var(0, unbiased=False)

                    self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
            y_hat = (y - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        else:
            y_hat = y

        if self.enable_hebbian_updates:
            with torch.no_grad():
                hebb_term = torch.einsum('bi,bj->ij', y_hat, x)
                y_sq = (y_hat ** 2).sum(dim=0)
                decay_term = y_sq.unsqueeze(1) * self.weight

                delta_w = self.hebb_rate * (hebb_term - decay_term)
                self.hebb_memory.mul_(self.ema_alpha).add_(delta_w, alpha=(1.0 - self.ema_alpha))

                if self.weight_constraint == 'clip':
                    self.hebb_memory.clamp_(-1.0, 1.0)
                elif self.weight_constraint == 'norm':
                    self.hebb_memory.copy_(F.normalize(self.hebb_memory, dim=1))

                ProjectFroNorm(self.hebb_memory, self.mem_norm_cap)
        return y_hat

    def ResetHebbianMemory(self):
        with torch.no_grad():
            self.hebb_memory.zero_()



class TransformerEncode(nn.Module):
    def __init__(self, modelDim: int, headNum: int, dimFeedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_atten = nn.MultiheadAttention(modelDim, headNum, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(modelDim, dimFeedforward)
        self.linear2 = nn.Linear(dimFeedforward, modelDim)
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, srcMask: Optional[torch.Tensor] = None, srcKeyPaddingMask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_norm1 = self.norm1(src)
        src2, _ = self.self_atten(
            src_norm1, src_norm1, src_norm1,
            attn_mask=srcMask,
            key_padding_mask=srcKeyPaddingMask,
            need_weights=False)
        
        src = src + self.dropout1(src2)

        src_norm2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(src2)
        return src
    

class ResidualBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1, useHebbian: bool = False):
        super().__init__()
        self.downsample = None
        if stride != 1 or inChannels != outChannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                Norm2d(outChannels))
            
        self.conv1 = HebbianConv2d(inChannels, outChannels, 3, stride=stride, padding=1,bias=False, useHebbian=useHebbian)
        self.bn1 = Norm2d(outChannels)
        self.conv2 = HebbianConv2d(outChannels, outChannels, 3, stride=1, padding=1,bias=False, useHebbian=useHebbian)
        self.bn2 = Norm2d(outChannels)
        self.relu = nn.SiLU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class CNNFeatureExtractor(nn.Module):
    def __init__(self, inChannels: int = 3, baseChannels: int = 64, useHebbian: bool = True):
        super().__init__()
        self.conv1 = HebbianConv2d(inChannels, baseChannels, 7, stride=2, padding=3,bias=False, useHebbian=useHebbian)
        
        self.bn1 = Norm2d(baseChannels)
        self.relu = nn.SiLU() 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(baseChannels, baseChannels, blocks=2, stride=1, useHebbian=useHebbian)
        self.layer2 = self.make_layer(baseChannels, baseChannels*2, blocks=2, stride=2, useHebbian=useHebbian)
        self.layer3 = self.make_layer(baseChannels*2, baseChannels*4, blocks=2, stride=2, useHebbian=useHebbian)
        self.layer4 = self.make_layer(baseChannels*4, baseChannels*8, blocks=2, stride=2, useHebbian=useHebbian)

        self.conv2 = HebbianConv2d(baseChannels*8, baseChannels*16, 3, stride=1, padding=1, bias=False, useHebbian=useHebbian)
        self.bn2 = Norm2d(baseChannels*16)

    def make_layer(self, inC, outC, blocks, stride, useHebbian):
        layers = [ResidualBlock(inC, outC, stride=stride, useHebbian=useHebbian)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(outC, outC, stride=1, useHebbian=useHebbian))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        return x  # [B, C, H', W']


class PerceiveExtractor(nn.Module):
    def __init__(
        self,
        imgSize: int = 512,
        patchSize: int = 1,
        embedDim: int = 512,
        numHeads: int = 8,
        numLayers: int = 6,
        hebbRate: float = 0.01,
        useHebbian: bool = True,
        baseChannels: int = 64,
        dropout: float = 0.1,
        posDrop: float = 0.1):
        super().__init__()

        assert embedDim % numHeads == 0, "embed_dim must be divisible by num_heads"

        self.img_size = imgSize
        self.patch_size = patchSize
        self.use_hebbian = useHebbian
        self.base_channels = baseChannels

        self.cnn_extractor = CNNFeatureExtractor(
            inChannels=3,
            baseChannels=baseChannels,
            useHebbian=useHebbian)

        with torch.no_grad():
            mods = [m for m in self.cnn_extractor.modules() if hasattr(m, "enable_hebbian_updates")]
            old = [bool(m.enable_hebbian_updates) for m in mods]
            for m in mods: m.enable_hebbian_updates = False

            dummy = torch.zeros(1, 3, imgSize, imgSize)
            fmap = self.cnn_extractor(dummy)
            Hf, Wf = fmap.shape[-2], fmap.shape[-1]

            for m, v in zip(mods, old):
                m.enable_hebbian_updates = v

        num_patches = (Hf // patchSize) ** 2

        cnn_feat_dim = baseChannels * 16

        self.patch_embed = SheafGaugeConv2d(
            in_channels=cnn_feat_dim,
            out_channels=embedDim,
            kernel_size=patchSize,
            stride=patchSize,
            bias=False,
            sheaf_alpha=0.1,
            sheaf_iters=1,
            gauge_groups=1, 
            gauge_scale=0.1,
            gauge_bias_scale=0.1)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedDim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedDim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=posDrop)


        self.cnn_feat_adapter = GrowableConv1x1Adapter(channels=cnn_feat_dim)
        self.patch_adapter = GrowableLoRAConv2d(self.patch_embed)
        self.token_adapters = nn.ModuleList([GrowableTokenAdapter(embedDim) for _ in range(numLayers)])

        self.transformer_layers = nn.ModuleList([
            TransformerEncode(
                modelDim=embedDim,
                headNum=numHeads,
                dimFeedforward=embedDim * 4,
                dropout=dropout
            ) for _ in range(numLayers)])
        
        self.encoder_norm = nn.LayerNorm(embedDim)

        hidden_dim = embedDim * 2
        layers = []

        layers.append(nn.Linear(embedDim, hidden_dim, bias=True))
        layers.append(nn.GELU())
        layers.append(HebbianLinear(hidden_dim, hidden_dim, hebbRate=hebbRate, useHebbian = useHebbian))
        layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, embedDim, bias=True))
        layers.append(nn.GELU())
        layers.append(HebbianLinear(embedDim, embedDim, hebbRate=hebbRate, useHebbian = useHebbian))
        layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

        self.adaptive_gate = nn.Sequential(
            nn.Linear(embedDim, embedDim // 4, bias=True),
            nn.SiLU(),
            nn.Linear(embedDim // 4, 1, bias=True),
            nn.Sigmoid())

        self.output_norm = nn.LayerNorm(embedDim, eps=1e-6)

        self.patch_aggregator = nn.Sequential(
            nn.Linear(embedDim, embedDim // 4, bias= True),
            nn.SiLU(),
            nn.Linear(embedDim // 4, 1, bias=True))

        self.InitWeights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        feat = self.cnn_extractor(x)  # [B, C, Hf, Wf]

        feat = self.cnn_feat_adapter(feat) 

        patches = self.patch_adapter(feat)  # [B, embed_dim, Ph, Pw]
        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            x = self.token_adapters[i](x)
        x = self.encoder_norm(x)

        cls_rep = x[:, 0, :] # [B, embed_dim]

        mlp_out = self.mlp(cls_rep)  # [B, embed_dim]

        gate = self.adaptive_gate(mlp_out)  # [B, 1]
        out = gate * mlp_out + (1 - gate) * cls_rep
        out = self.output_norm(out)  # [B, embed_dim]

        patch_tokens = x[:, 1:, :]
        patch_scores = self.patch_aggregator(patch_tokens)
        patch_scores = patch_scores.squeeze(-1)

        patch_weights = F.softmax(patch_scores, dim=1)

        global_patch = (patch_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)

        fuse_out = torch.cat([out, global_patch], dim=1)

        return fuse_out # [B, embed_dim * 2]


    def InitWeights(self):
        if isinstance(self.patch_embed, nn.Conv2d):
            nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
            if self.patch_embed.bias is not None:
                nn.init.zeros_(self.patch_embed.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm, nn.BatchNorm2d)):
                if getattr(m, "affine", True):
                    if hasattr(m, "weight") and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Linear):
                    nn.init.xavier_uniform_(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)
            
            elif isinstance(m, HebbianLinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def ResetHebbianMemory(self): 
        for name, m in self.named_children(): 
            if hasattr(m, "ResetHebbianMemory"): 
                m.ResetHebbianMemory()



class PerceptionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module, 
        initRankEach: int = 4, 
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankFeat: int = 64,
        maxRankPatch: int = 64,
        maxRankToken: int = 64,):
        self.maxRankFeat = int(maxRankFeat)
        self.maxRankPatch = int(maxRankPatch)
        self.maxRankToken = int(maxRankToken)
        super().__init__(base, initRankEach=initRankEach, autoRank=autoRank, evThreshold=evThreshold, gradEma=gradEma)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        C = int(self.base.patch_embed.in_channels)
        E = int(self.base.patch_embed.out_channels)
        kh, kw = self.base.patch_embed.kernel_size
        ksz = kh * kw
        D = int(self.base.pos_embed.size(-1))
        L = len(self.base.transformer_layers)

        def alloc_feat(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, C, device=device, dtype=dtype) * 1e-4) 
            B = nn.Parameter(torch.zeros(C, addRank, device=device, dtype=dtype) * 1e-4) 
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_feat(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        def alloc_patch(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, C * ksz, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(E, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_patch(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        def alloc_token(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, D, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(D, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_token(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        return {
            "feat": SiteSpec("feat", 1, C, C, self.maxRankFeat, alloc_feat, compose_feat),
            "patch": SiteSpec("patch", 1, C * ksz, E, self.maxRankPatch, alloc_patch, compose_patch),
            "token": SiteSpec("token", L, D, D, self.maxRankToken, alloc_token, compose_token),}

    def ForwardWithDeltas(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> torch.Tensor:
        feat = self.base.cnn_extractor(x)
        feat = self.base.cnn_feat_adapter(feat)

        deltaFeat2D = deltasPerLayer[0].get("feat", None)
        if deltaFeat2D is not None:
            C = deltaFeat2D.size(0)
            w1x1 = deltaFeat2D.view(C, C, 1, 1).to(device=feat.device, dtype=feat.dtype)
            feat = feat + F.conv2d(feat, w1x1, bias=None, stride=1, padding=0)

        feat_patch = feat
        if hasattr(self.base.patch_embed, "Preprocess"):
            feat_patch = self.base.patch_embed.Preprocess(feat_patch)

        W_eff = self.base.patch_embed.weight
        base_delta = self.base.patch_adapter.DeltaWeight()
        if base_delta is not None:
            W_eff = W_eff + base_delta

        deltaPatch2D = deltasPerLayer[0].get("patch", None)
        if deltaPatch2D is not None:
            E, Ckhw = deltaPatch2D.shape
            C_in = self.base.patch_embed.in_channels
            kh, kw = self.base.patch_embed.kernel_size
            W_eff = W_eff + deltaPatch2D.view(E, C_in, kh, kw).to(device=feat_patch.device, dtype=feat_patch.dtype)

        patches = F.conv2d(feat_patch,W_eff,bias=None,stride=self.base.patch_embed.stride,
            padding=self.base.patch_embed.padding,dilation=self.base.patch_embed.dilation,groups=self.base.patch_embed.groups,)

        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, "b c h w -> b (h w) c")
        cls_tokens = repeat(self.base.cls_token, "1 1 d -> b 1 d", b=B)
        xTok = torch.cat([cls_tokens, patches], dim=1)
        xTok = xTok + self.base.pos_embed
        xTok = self.base.pos_drop(xTok)

        for i, layer in enumerate(self.base.transformer_layers):
            xTok = layer(xTok)
            xTok = self.base.token_adapters[i](xTok)
            deltaTok2D = deltasPerLayer[i].get("token", None)
            if deltaTok2D is not None:
                xTok = xTok + (xTok @ deltaTok2D.t())

        xTok = self.base.encoder_norm(xTok)
        cls_rep = xTok[:, 0, :]
        mlp_out = self.base.mlp(cls_rep)
        gate = self.base.adaptive_gate(mlp_out)
        out = gate * mlp_out + (1 - gate) * cls_rep
        out = self.base.output_norm(out)

        patch_tokens = xTok[:, 1:, :]
        patch_scores = self.base.patch_aggregator(patch_tokens).squeeze(-1)
        patch_weights = F.softmax(patch_scores, dim=1)
        global_patch = (patch_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)

        return torch.cat([out, global_patch], dim=1)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        if site == "feat":
            if layerIdx != 0:
                return False
            r = a.size(0)
            C = self.base.cnn_feat_adapter.C 

            a2 = a.detach().clone().view(r, C, 1, 1) 
            b2 = b.detach().clone().view(C, r, 1, 1)

            init = {"A": a2, "B": b2, "scale": float(scale)}
            self.base.cnn_feat_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
            return True

        elif site == "patch":
            if layerIdx != 0:
                return False
            init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}
            self.base.patch_adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        elif site == "token":
            init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}
            self.base.token_adapters[layerIdx].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        else:
            raise ValueError(f"Unknown site: {site}")



class TestPerceptionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def AdapterRankAndParams(self, adapter) -> Tuple[int, int]:
        rank_sum = 0
        param_cnt = 0
        if not hasattr(adapter, "A_list"):
            return 0, 0
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            rank_sum += int(A.shape[0])
            param_cnt += int(A.numel() + B.numel() + 1)
        return rank_sum, param_cnt

    def TokenRanksAndParams(self, token_adapters: Iterable) -> Tuple[List[int], int]:
        per_layer_ranks = []
        total_params = 0
        for ta in token_adapters:
            r, p = self.AdapterRankAndParams(ta)
            per_layer_ranks.append(r)
            total_params += p
        return per_layer_ranks, total_params

    def DeltaFromConv1x1Adapter(self, adapter) -> torch.Tensor:
        if not hasattr(adapter, "A_list") or len(adapter.A_list) == 0:
            C = adapter.C if hasattr(adapter, "C") else 0
            return torch.zeros(C, C, device=self.device)
        C = adapter.C
        delta = torch.zeros(C, C, device=adapter.A_list[0].device, dtype=adapter.A_list[0].dtype)
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            A2 = A.view(A.size(0), C) 
            B2 = B.view(C, A.size(0)) 
            scale = torch.tanh(s.detach()) * GetParameterSScale(s.detach())
            delta = delta + scale * (B2 @ A2) 
        return delta

    def DeltaFromTokenAdapter(self, adapter) -> torch.Tensor:
        if not hasattr(adapter, "A_list") or len(adapter.A_list) == 0:
            D = adapter.D if hasattr(adapter, "D") else 0
            return torch.zeros(D, D, device=self.device)
        D = adapter.D
        delta = torch.zeros(D, D, device=adapter.A_list[0].device, dtype=adapter.A_list[0].dtype)
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            scale = torch.tanh(s.detach()) * GetParameterSScale(s.detach())
            delta = delta + scale * (B @ A)
        return delta

    def TestHebbianConv2d(self):
        try:
            conv = HebbianConv2d(inChannels=3, outChannels=16, kernelSize=3, stride=1, padding=1, useHebbian=True).to(self.device)
            x = torch.randn(4, 3, 32, 32, device=self.device)
            y = conv(x)
            assert y.shape == (4, 16, 32, 32), f"Output shape does not match: {y.shape}"
            conv.ResetHebbianMemory()
            print("HebbianConv2d test passed.")
            return True
        except AssertionError as e:
            print(f"HebbianConv2d test failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianConv2d test error: {e}")
            return False

    def TestHebbianLinear(self):
        try:
            lin = HebbianLinear(inFeatures=32, outFeatures=64, useHebbian=True).to(self.device)
            x = torch.randn(5, 32, device=self.device)
            y = lin(x)
            assert y.shape == (5, 64), f"Output shape does not match: {y.shape}"
            lin.ResetHebbianMemory()
            print("HebbianLinear test passed.")
            return True
        except AssertionError as e:
            print(f"HebbianLinear test failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianLinear test error: {e}")
            return False

    def TestPerceiveExtractor(self):
        try:
            model = PerceiveExtractor(imgSize=512, patchSize=1, embedDim=512, numHeads=8, numLayers=6, useHebbian=True).to(self.device)
            x = torch.randn(2, 3, 512, 512, device=self.device)
            out = model(x)
            expected_dim = 512 * 2
            assert out.shape == (2, expected_dim), f"Output shape does not match: {out.shape}"
            print("PerceiveExtractor forward passed.")
            return True
        except AssertionError as e:
            print(f"PerceiveExtractor test failed: {e}")
            return False
        except Exception as e:
            print(f"PerceiveExtractor test error: {e}")
            return False

    def TrainStepSmoke(self):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            model.train()
            head = nn.Linear(64 * 2, 16).to(self.device)
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 3, 64, 64, device=self.device)
            target = torch.randn(8, 16, device=self.device)

            out = model(x)
            pred = head(out)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            grads_ok = []
            for _, p in model.named_parameters():
                if p.grad is not None:
                    grads_ok.append(bool(torch.isfinite(p.grad).all().item()))
            grads_ok = all(grads_ok) and (head.weight.grad is not None) and bool(torch.isfinite(head.weight.grad).all().item())
            assert grads_ok, "There are parameters whose gradient is None or non-finite."

            opt.step()
            print("TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print(f"TrainStepSmoke failed: {e}")
            return False
        except Exception as e:
            print(f"TrainStepSmoke error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 30):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            head = nn.Linear(128, 16).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            for t in range(steps):
                x = torch.randn(8, 3, 64, 64, device=self.device)
                y = torch.randn(8, 16, device=self.device)
                pred = head(model(x))
                loss = F.mse_loss(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in list(model.named_parameters()) + list(head.named_parameters()):
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"step {t} Gradient not finite: {n}"
                opt.step()
            print("NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"NoNanAfterManySteps error: {e}")
            return False

    def ParamsActuallyChange(self, steps: int = 10):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            head = nn.Linear(128, 16).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            with torch.no_grad():
                key_params = {
                    "conv1.weight": next(p for n, p in model.cnn_extractor.named_parameters() if n == "conv1.conv.weight"),
                    "patch_embed.weight": model.patch_embed.weight,
                    "attn_any_0": next(p for p in model.transformer_layers[0].self_atten.parameters()),
                    "head.weight": head.weight}
                init_norms = {k: v.norm().item() for k, v in key_params.items()}

            for _ in range(steps):
                x = torch.randn(8, 3, 64, 64, device=self.device)
                y = torch.randn(8, 16, device=self.device)
                pred = head(model(x))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                new_norms = {k: v.norm().item() for k, v in key_params.items()}

            changed = any(abs(new_norms[k] - init_norms[k]) > 1e-6 for k in init_norms)
            assert changed, "Key parameters' norms barely changed; suspected no update."
            print("ParamsActuallyChange passed.")
            return True
        except AssertionError as e:
            print(f"ParamsActuallyChange failed: {e}")
            return False
        except Exception as e:
            print(f"ParamsActuallyChange error: {e}")
            return False

    def TestNormalTrainingConvergence(self, steps: int = 120, logEvery: int = 30):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            head = nn.Linear(128, 16).to(self.device)
            model.train(); head.train()

            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 16
            data_x = torch.randn(B, 3, 64, 64, device=self.device)
            data_y = torch.randn(B, 16, device=self.device)

            with torch.no_grad():
                start = F.mse_loss(head(model(data_x)), data_y).item()

            for t in range(1, steps + 1):
                pred = head(model(data_x))
                loss = F.mse_loss(pred, data_y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if (t % logEvery) == 0 or t == 1:
                    print(f"[PerceptionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(model(data_x)), data_y).item()

            print(f"\n[PerceptionTrain] loss start={start:.6f} -> end={end:.6f}")
            assert end <= 0.8 * start, "Training did not show sufficient convergence (<20% decline)."
            print("TestNormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print(f"TestNormalTrainingConvergence failed: {e}")
            return False
        except Exception as e:
            print(f"TestNormalTrainingConvergence error: {e}")
            return False

    def WrapperForwardEqualWhenNoInitRank(self):
        try:
            base = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()
            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.eval()

            x = torch.randn(3, 3, 64, 64, device=self.device)
            with torch.no_grad():
                y_base = base(x)
                y_wrap = wrapper(x)

            max_abs = (y_base - y_wrap).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper forward differs when ranks=0: max_abs={max_abs:.3e}"
            print("WrapperForwardEqualWhenNoInitRank passed.")
            return True
        except AssertionError as e:
            print(f"WrapperForwardEqualWhenNoInitRank failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperForwardEqualWhenNoInitRank error: {e}")
            return False

    def WrapperAPIBasics(self):
        try:
            base = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()
            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()

            r = wrapper.Update("ranks")["ranks"]
            assert r["sum"]["feat"] == 0 and r["sum"]["patch"] == 0 and all(row["token"] == 0 for row in r["perLayer"])

            wrapper.Update("grow", growFactor=2.0, addEach=1)
            r2 = wrapper.Update("ranks")["ranks"]
            assert r2["sum"]["feat"] >= 1 and r2["sum"]["patch"] >= 1 and all(row["token"] >= 1 for row in r2["perLayer"])

            wrapper.Update("accumulategrads")

            st = wrapper.Update("set", evThreshold=0.85, gradEma=0.8,
                                **{"maxRank:feat":64, "maxRank:patch":64, "maxRank:token":64})
            assert st["ok"]

            wrapper.Update("rollback")
            r3 = wrapper.Update("ranks")["ranks"]
            assert r3["sum"]["feat"] == 0 and r3["sum"]["patch"] == 0 and all(row["token"] == 0 for row in r3["perLayer"])

            print("WrapperAPIBasics passed.")
            return True
        except AssertionError as e:
            print("WrapperAPIBasics failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAPIBasics error:\n", e)
            return False

    def WrapperManualGrowTrainAndCommit(self):
        try:
            img_size = 64
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()

            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=4).to(self.device)
            wrapper.train()

            head = nn.Linear(128, 16).to(self.device).train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=3e-3)

            _ = wrapper.Update("grow", growFactor=2.0, addEach=0)

            for _ in range(8):
                x = torch.randn(8, 3, img_size, img_size, device=self.device)
                y = torch.randn(8, 16, device=self.device)
                pred = head(wrapper(x))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            expected = []
            for li in range(wrapper.layerCount):
                per = {}
                for site in ("feat", "patch", "token"):
                    per[site] = wrapper.ComposeOne(site, li).detach().clone()
                expected.append(per)

            res = wrapper.Update("commit")
            assert res["ok"] and res["commit_stats"]["committed_triples"] > 0, "Nothing committed."

            committed_triples = int(res["commit_stats"]["committed_triples"])
            committed_rank = int(res["commit_stats"]["committed_rank"])
            print(f"[Commit] committed_triples={committed_triples}, committed_rank={committed_rank}")

            feat_rank, feat_params = self.AdapterRankAndParams(base.cnn_feat_adapter)
            patch_rank, patch_params = self.AdapterRankAndParams(base.patch_adapter)
            token_ranks, token_params = self.TokenRanksAndParams(base.token_adapters)

            total_rank_injected = feat_rank + patch_rank + sum(token_ranks)
            total_params_injected = feat_params + patch_params + token_params

            print(
                "[Injected] feat: rank={}, params={}; patch: rank={}, params={}; "
                "token per-layer ranks={}, token_total_params={}; "
                "TOTAL rank={}, TOTAL params={}".format(
                    feat_rank, feat_params,
                    patch_rank, patch_params,
                    token_ranks, token_params,
                    total_rank_injected, total_params_injected))

            r = wrapper.Update("ranks")["ranks"]
            assert all(row["feat"] == 0 and row["patch"] == 0 and row["token"] == 0 for row in r["perLayer"])

            atol, rtol = 1e-6, 1e-4

            exp_feat = expected[0]["feat"]
            if not torch.allclose(exp_feat, torch.zeros_like(exp_feat)):
                got_feat = self.DeltaFromConv1x1Adapter(base.cnn_feat_adapter)
                assert torch.allclose(got_feat, exp_feat, atol=atol, rtol=rtol), f"feat delta mismatch: max_abs={(got_feat-exp_feat).abs().max().item():.3e}"

            exp_patch = expected[0]["patch"]
            if not torch.allclose(exp_patch, torch.zeros_like(exp_patch)):
                dw = base.patch_adapter.DeltaWeight()
                assert dw is not None, "patch adapter not injected"
                got_patch = dw.view(exp_patch.shape[0], -1)
                assert torch.allclose(got_patch, exp_patch, atol=atol, rtol=rtol), f"patch delta mismatch: max_abs={(got_patch-exp_patch).abs().max().item():.3e}"

            for li, ta in enumerate(base.token_adapters):
                exp_tok = expected[li]["token"]
                if torch.allclose(exp_tok, torch.zeros_like(exp_tok)):
                    continue
                got_tok = self.DeltaFromTokenAdapter(ta)
                assert torch.allclose(got_tok, exp_tok, atol=atol, rtol=rtol), f"token[{li}] delta mismatch: max_abs={(got_tok-exp_tok).abs().max().item():.3e}"

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 3, img_size, img_size, device=self.device)
            with torch.no_grad():
                y0 = base(x_chk)
                y1 = wrapper(x_chk)
            assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-4), "base vs wrapper mismatch after commit."

            print("WrapperManualGrowTrainAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperManualGrowTrainAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperManualGrowTrainAndCommit error:\n", e)
            return False

    def WrapperAutoGrowDecreaseRank(self):
        try:
            img_size = 64
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()

            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=4).to(self.device)
            wrapper.train()

            ranks_before = wrapper.Update("ranks")["ranks"]["sum"]["feat"]
            assert ranks_before >= 4, f"expected feat rank >=4, got {ranks_before}"

            wrapper.Update("accumulategrads")
            wrapper.Update("autogrow")
            ranks_after = wrapper.Update("ranks")["ranks"]["sum"]["feat"]
            assert ranks_after == 0, f"rank not decreased to 0: before={ranks_before}, after={ranks_after}"

            print("WrapperAutoGrowDecreaseRank passed.")
            return True
        except AssertionError as e:
            print("WrapperAutoGrowDecreaseRank failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAutoGrowDecreaseRank error:\n", e)
            return False

    def WrapperPipelineCompatible(self):
        try:
            img_size = 64
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=4).to(self.device)
            wrapper.train(); base.eval()

            head = nn.Linear(128, 16).to(self.device)
            opt = torch.optim.Adam(list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 3, img_size, img_size, device=self.device)
            target = torch.randn(8, 16, device=self.device)

            out = wrapper(x)
            pred = head(out)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head grad invalid with wrapper."
            opt.step()

            print("WrapperPipelineCompatible passed.")
            return True
        except AssertionError as e:
            print(f"WrapperPipelineCompatible failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperPipelineCompatible error: {e}")
            return False

    def WrapperAdaptiveGrowAndCommit(self):
        try:
            img_size = 64
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()

            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=0, autoRank=True).to(self.device)
            wrapper.train()

            wrapper.Update("set", evThreshold=0.97, gradEma=0.5, **{"maxRank:feat": 16, "maxRank:patch": 32, "maxRank:token": 24})

            wrapper.Update("autogrow")
            r0 = wrapper.Update("ranks")["ranks"]
            print(f"[Seed] ranks after initial autogrow -> feat={r0['sum']['feat']}, "f"patch={r0['sum']['patch']}, token per-layer={[row['token'] for row in r0['perLayer']]}")

            head = nn.Linear(128, 16).to(self.device).train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=2e-3)

            B = 12
            data_x = torch.randn(B, 3, img_size, img_size, device=self.device)
            data_y = torch.randn(B, 16, device=self.device)

            steps = 30
            grow_every = 5

            for t in range(1, steps + 1):
                pred = head(wrapper(data_x))
                loss = F.mse_loss(pred, data_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")

                if (t % grow_every) == 0:
                    wrapper.Update("autogrow")
                    rk = wrapper.Update("ranks")["ranks"]
                    feat_r = rk["sum"]["feat"]
                    patch_r = rk["sum"]["patch"]
                    token_layers = [row["token"] for row in rk["perLayer"]]
                    print(f"[Adaptive@{t}] feat={feat_r}, patch={patch_r}, token per-layer={token_layers}")

                opt.step()

            rk_final = wrapper.Update("ranks")["ranks"]
            feat_r = rk_final["sum"]["feat"]
            patch_r = rk_final["sum"]["patch"]
            token_layers = [row["token"] for row in rk_final["perLayer"]]
            token_sum = sum(token_layers)
            total_rank = feat_r + patch_r + token_sum

            res = wrapper.Update("commit")
            assert res["ok"], "Commit failed."
            stats = res.get("commit_stats", {})
            committed_rank = int(stats.get("committed_rank", 0))
            committed_triples = int(stats.get("committed_triples", 0))
            print(f"[Commit] committed_triples={committed_triples}, committed_rank={committed_rank}")

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 3, img_size, img_size, device=self.device)
            with torch.no_grad():
                y0 = base(x_chk)
                y1 = wrapper(x_chk)
            max_abs = (y0 - y1).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper vs base mismatch after commit: {max_abs:.3e}"

            if total_rank == 0:
                assert committed_rank == 0 and committed_triples == 0, f"Expected no injection, but got committed_rank={committed_rank}, triples={committed_triples}"
                print("[Adaptive] No growth needed. Passed without injection.")
            else:
                def count_lora_params(adapter):
                    total_r, total_p = 0, 0
                    for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
                        total_r += int(A.shape[0])
                        total_p += int(A.numel() + B.numel() + 1)
                    return total_r, total_p

                feat_rank, feat_params = count_lora_params(base.cnn_feat_adapter)
                patch_rank, patch_params = count_lora_params(base.patch_adapter)
                token_layer_ranks, token_params_total = [], 0
                for ta in base.token_adapters:
                    r, p = count_lora_params(ta)
                    token_layer_ranks.append(r)
                    token_params_total += p

                total_injected_rank = feat_rank + patch_rank + sum(token_layer_ranks)
                total_injected_params = feat_params + patch_params + token_params_total
                print(f"[Injected-Adaptive] feat: rank={feat_rank}, params={feat_params}; "
                      f"patch: rank={patch_rank}, params={patch_params}; "
                      f"token per-layer ranks={token_layer_ranks}, token_total_params={token_params_total}; "
                      f"TOTAL rank={total_injected_rank}, TOTAL params={total_injected_params}")

            print("WrapperAdaptiveGrowAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperAdaptiveGrowAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAdaptiveGrowAndCommit error:\n", e)
            return False

    def GradCoverageReport(self, min_ratio: float = 0.60):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            head = nn.Linear(128, 16).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 3, 64, 64, device=self.device)
            y = torch.randn(8, 16, device=self.device)
            pred = head(model(x))
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            named = dict(list(model.named_parameters()) + [('head.'+k, v) for k,v in head.named_parameters()])
            total_trainable = sum(1 for p in named.values() if p.requires_grad)
            total_with_grad = sum(1 for p in named.values() if p.requires_grad and (p.grad is not None))
            ratio = total_with_grad / max(1, total_trainable)

            must_have = [
                "cnn_extractor.conv1.conv.weight",
                "patch_embed.weight",
                "transformer_layers.0.self_atten.out_proj.weight",
                "transformer_layers.0.linear1.weight",
                "mlp.2.weight",  
                "mlp.6.weight",  
                "head.weight", ]
            missing = [n for n in must_have if (n in named) and (named[n].grad is None)]
            assert len(missing) == 0, f"The key layer does not get the gradient: {missing}"
            assert ratio >= min_ratio, f"Gradient coverage is too low: {ratio:.2%} < {min_ratio:.2%}"

            print(f"GradCoverageReport passed. grad_ratio={ratio:.2%}")
            return True
        except AssertionError as e:
            print(f"GradCoverageReport failed: {e}")
            return False
        except Exception as e:
            print(f"GradCoverageReport error: {e}")
            return False

    def LossDecreasesWithHebbToggle(self, steps: int = 80):
        try:
            for flag in (False, True):
                model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=flag).to(self.device)
                head = nn.Linear(128, 16).to(self.device)
                model.train(); head.train()
                opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

                B = 16
                data_x = torch.randn(B, 3, 64, 64, device=self.device)
                data_y = torch.randn(B, 16, device=self.device)

                with torch.no_grad():
                    start = F.mse_loss(head(model(data_x)), data_y).item()

                hist = []
                for _ in range(steps):
                    pred = head(model(data_x))
                    loss = F.mse_loss(pred, data_y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    hist.append(loss.item())

                end = hist[-1]
                tail_mean = sum(hist[-10:]) / min(10, len(hist))
                assert tail_mean <= 0.5 * start, f"Hebbian={flag} insufficient convergence: start={start:.4f}, tail_mean={tail_mean:.4f}"
                print(f"LossDecreasesWithHebbToggle({flag}) passed. start={start:.4f} -> end={end:.4f}")
            return True
        except AssertionError as e:
            print(f"LossDecreasesWithHebbToggle failed: {e}")
            return False
        except Exception as e:
            print(f"LossDecreasesWithHebbToggle error: {e}")
            return False

    def HebbianMemoryLifecycle(self):
        try:
            conv = HebbianConv2d(3, 8, 3, stride=1, padding=1, useHebbian=True).to(self.device)
            x = torch.randn(4, 3, 32, 32, device=self.device)
            n0 = conv.hebb_memory.norm().item()
            for _ in range(3):
                _ = conv(x)
            n1 = conv.hebb_memory.norm().item()
            assert n1 > n0 + 1e-12, f"Conv Hebbian memory no growth: before={n0:.3e}, after={n1:.3e}"
            conv.ResetHebbianMemory()
            n2 = conv.hebb_memory.norm().item()
            assert n2 < 1e-12, f"Conv Hebbian memory unclear zero: now={n2:.3e}"

            lin = HebbianLinear(32, 16, useHebbian=True).to(self.device)
            z = torch.randn(6, 32, device=self.device)
            n0 = lin.hebb_memory.norm().item()
            for _ in range(3):
                _ = lin(z)
            n1 = lin.hebb_memory.norm().item()
            assert n1 > n0 + 1e-12, f"Linear Hebbian memory no growth: before={n0:.3e}, after={n1:.3e}"
            lin.ResetHebbianMemory()
            n2 = lin.hebb_memory.norm().item()
            assert n2 < 1e-12, f"Linear Hebbian memory unclear zero: now={n2:.3e}"

            print("HebbianMemoryLifecycle passed.")
            return True
        except AssertionError as e:
            print(f"HebbianMemoryLifecycle failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianMemoryLifecycle error: {e}")
            return False

    def WrapperKeepsBaseEval(self):
        try:
            base = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()
            assert wrapper.training and (not base.training), "When wrapper.train() is used, base should be eval()"
            print("WrapperKeepsBaseEval passed.")
            return True
        except AssertionError as e:
            print(f"WrapperKeepsBaseEval failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperKeepsBaseEval error: {e}")
            return False

    def SmallBatchSafety(self):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8,numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            head = nn.Linear(128, 16).to(self.device)
            model.eval(); head.train()  
            x = torch.randn(1, 3, 64, 64, device=self.device)
            y = torch.randn(1, 16, device=self.device)
            pred = head(model(x))
            loss = F.mse_loss(pred, y)
            head.zero_grad(set_to_none=True)
            loss.backward()
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head gradient abnormality when batch=1"
            print("SmallBatchSafety passed.")
            return True
        except AssertionError as e:
            print(f"SmallBatchSafety failed: {e}")
            return False
        except Exception as e:
            print(f"SmallBatchSafety error: {e}")
            return False

    def RunAll(self):
        results = {
            "HebbianConv2d": self.TestHebbianConv2d(),
            "HebbianLinear": self.TestHebbianLinear(),
            "PerceiveExtractorForward": self.TestPerceiveExtractor(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "ParamsActuallyChange": self.ParamsActuallyChange(),
            "NormalTrainingConvergence": self.TestNormalTrainingConvergence(),
            "WrapperForwardEqualWhenNoInitRank": self.WrapperForwardEqualWhenNoInitRank(),
            "WrapperAPIBasics": self.WrapperAPIBasics(),
            "WrapperManualGrowTrainAndCommit": self.WrapperManualGrowTrainAndCommit(),
            "WrapperAutoGrowDecreaseRank": self.WrapperAutoGrowDecreaseRank(),
            "WrapperPipelineCompatible": self.WrapperPipelineCompatible(),
            "WrapperAdaptiveGrowAndCommit": self.WrapperAdaptiveGrowAndCommit(),
            "GradCoverageReport": self.GradCoverageReport(),
            "LossDecreasesWithHebbToggle": self.LossDecreasesWithHebbToggle(),
            "HebbianMemoryLifecycle": self.HebbianMemoryLifecycle(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "SmallBatchSafety": self.SmallBatchSafety(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nPerception module tests (with wrapper): {passed}/{len(results)} passed.")
        return results

