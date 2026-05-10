import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, RoPEMultiheadAttention


@dataclass
class TopDownContext:
    GoalBias: Optional[torch.Tensor] = None
    PredictedVisual: Optional[Any] = None
    Precision: Optional[torch.Tensor] = None
    SelfSemantic: Optional[torch.Tensor] = None
    IntentSemantic: Optional[torch.Tensor] = None
    MemoryCue: Optional[torch.Tensor] = None


@dataclass
class VisualState:
    LegacyFeat: torch.Tensor
    GlobalFeat: torch.Tensor
    VentralFeat: torch.Tensor
    DorsalFeat: torch.Tensor
    MotionToken: torch.Tensor
    QualityToken: torch.Tensor
    PredErrorToken: torch.Tensor
    ObjectTokens: torch.Tensor
    PatchTokens: torch.Tensor
    NextState: Dict[str, torch.Tensor] = field(default_factory=dict)


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


def FrobeniusCapPerSample(mem: torch.Tensor, cap: Optional[float]):
    if cap is None:
        return
    with torch.no_grad():
        B = mem.size(0)
        flat = mem.reshape(B, -1)
        n = torch.linalg.vector_norm(flat, ord=2, dim=1)
        scale = (cap / (n + 1e-12)).clamp(max=1.0) 
        mem.mul_(scale.view(B, *([1] * (mem.dim() - 1))))


class GrowableLoRAConv2d(nn.Module):
    def __init__(self, targetConv: nn.Conv2d):
        super().__init__()
        self.target = targetConv 
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

        w = self.target.weight # [cout, cin, kh, kw]
        self.cout, self.cin, self.kh, self.kw = w.shape

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        ksz = self.kh * self.kw
        if init is None: init = {}

        factory = {"device": self.target.weight.device, "dtype": self.target.weight.dtype}

        A = init.get("A", torch.randn(addRank, self.cin * ksz, **factory) * 1e-4) 
        B = init.get("B", torch.zeros(self.cout, addRank, **factory))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.as_tensor(s, **factory))

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
            delta = delta + torch.tanh(s) * GetParametersScale(s) * (B @ A)
        return delta.view(self.cout, self.cin, self.kh, self.kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.target.weight

        if hasattr(self.target, "Preprocess"):
            x = self.target.Preprocess(x)

        delta = self.DeltaWeight()
        if delta is not None:
            w = w + delta
        return F.conv2d(x, w, self.target.bias, stride=self.target.stride,
                        padding=self.target.padding, dilation=self.target.dilation, groups=self.target.groups)


class GrowableConv1x1Adapter(AGICoreModule):
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        factory = {"device": self.device, "dtype": self.dtype}

        A = init.get("A", torch.randn(addRank, self.C, 1, 1, **factory) * 1e-4)
        B = init.get("B", torch.zeros(self.C, addRank, 1, 1, **factory))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.as_tensor(s, **factory))

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
            y = y + torch.tanh(s) * GetParametersScale(s) * z
        return y


class GrowableTokenAdapter(AGICoreModule):
    def __init__(self, dim: int):
        super().__init__()
        self.D = dim
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        factory = {"device": self.device, "dtype": self.dtype}
        
        A = init.get("A", torch.randn(addRank, self.D, **factory) * 1e-4)
        B = init.get("B", torch.zeros(self.D, addRank, **factory))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.as_tensor(s, **factory))

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
            y = y + torch.tanh(s) * GetParametersScale(s) * z
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

    def Shift(self, x: torch.Tensor, dim: int, step: int) -> torch.Tensor:
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
        beta = torch.tanh(self.gauge_beta(x_norm)) * self.gauge_bias_scale
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



class HebbianConv2d(AGICoreModule):
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        hebbRate: float = 1e-3,
        emaMomentum: float = 0.995,
        applyScale: float = 0.25,
        memNormCap: Optional[float] = 1.0,
        bias: bool = False,
        useHebbian: bool = True,):
        super().__init__()
        self.kernel_size = kernelSize if isinstance(kernelSize, tuple) else (kernelSize, kernelSize)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)

        self.conv = nn.Conv2d(
            inChannels, outChannels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bias,)

        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap
        self.use_hebbian = bool(useHebbian)

        self.register_buffer("hebb_memory", torch.empty(0), persistent=True)

    def ResetHebbianMemory(self):
        with torch.no_grad():
            self.hebb_memory = torch.empty(0, device=self.device, dtype=self.dtype)

    def EnsureB(self, B: int, device, dtype):
        w = self.conv.weight
        target_shape = (B, w.size(0), w.size(1), w.size(2), w.size(3)) 
        if (self.hebb_memory.numel() == 0) or (self.hebb_memory.shape != target_shape):
            self.hebb_memory = torch.zeros(*target_shape, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, inC, H, W = x.shape
        w = self.conv.weight
        outC = w.size(0)
        g = self.groups
        in_per_g = inC // g

        if self.use_hebbian:
            self.EnsureB(B, self.device, self.dtype)
            w_eff = w.unsqueeze(0) + self.apply_scale * self.hebb_memory.detach()
        else:
            w_eff = w.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()

        x_big = x.reshape(1, B * inC, H, W)

        w_big = w_eff.reshape(B * outC, w.size(1), w.size(2), w.size(3)).clone()

        groups_total = B * g

        if self.conv.bias is None:
            b_big = None
        else:
            b_big = self.conv.bias.repeat(B)

        out_big = F.conv2d(
            x_big, w_big, b_big,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups_total,)
        if not torch.isfinite(out_big).all():
            out_big = F.conv2d(
                x_big.clone(), w_big.clone(), b_big,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups_total,)
        
        Hout, Wout = out_big.shape[-2], out_big.shape[-1]
        out = out_big.reshape(B, outC, Hout, Wout)

        if self.use_hebbian:
            with torch.no_grad():
                x_unfold = F.unfold(
                    x.detach(),
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,)

                out_unfold = out.detach().reshape(B, outC, -1)
                L = out_unfold.size(-1)
                N = float(L) if L > 0 else 1.0 

                x_unfold_g = x_unfold.reshape(B, g, in_per_g * (self.kernel_size[0] * self.kernel_size[1]), L)
                out_unfold_g = out_unfold.reshape(B, g, outC // g, L)

                xu = x_unfold_g
                yu = out_unfold_g

                hebb_term = torch.einsum("bgol,bgil->bgoi", yu, xu) / N

                y2_mean = yu.square().sum(dim=-1) / N

                mem = self.hebb_memory.reshape(B, g, outC // g, -1)
                decay = y2_mean.unsqueeze(-1) * mem

                delta = self.hebb_rate * (hebb_term - decay) 
                delta = delta.reshape_as(self.hebb_memory)

                self.hebb_memory.mul_(self.ema_alpha).add_(delta, alpha=(1.0 - self.ema_alpha))

                FrobeniusCapPerSample(self.hebb_memory, self.mem_norm_cap)

        return out



class HebbianLinear(AGICoreModule):
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
        useHebbian: bool = True,):
        super().__init__()
        self.inFeatures = int(inFeatures)
        self.outFeatures = int(outFeatures)
        self.use_bias = bool(bias)

        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)
        self.bias = nn.Parameter(torch.zeros(outFeatures))

        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap
        self.normalize = bool(normalize)
        self.weight_constraint = weightConstraint
        self.use_hebbian = bool(useHebbian)

        self.register_buffer("hebb_memory", torch.empty(0), persistent=True)

    def ResetHebbianMemory(self):
        with torch.no_grad():
            self.hebb_memory = torch.empty(0, device=self.device, dtype=self.dtype)

    def EnsureB(self, B: int, device, dtype):
        if (self.hebb_memory.numel() == 0) or (self.hebb_memory.shape != (B, self.outFeatures, self.inFeatures)):
            self.hebb_memory = torch.zeros(B, self.outFeatures, self.inFeatures, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        if self.use_hebbian:
            self.EnsureB(B, self.device, self.dtype)
            w_eff = self.weight.unsqueeze(0) + self.apply_scale * self.hebb_memory.detach() 
        else:
            w_eff = self.weight.unsqueeze(0) 

        x2 = x.reshape(B, -1, self.inFeatures)  
        y2 = torch.einsum("bni,boi->bno", x2, w_eff)
        if self.use_bias:
            y2 = y2 + self.bias.view(1, 1, -1)
        y = y2.view(*x.shape[:-1], self.outFeatures) 

        if self.normalize:
            mean = y.mean(dim=-1, keepdim=True)
            var = y.var(dim=-1, keepdim=True, unbiased=False)
            y_hat = (y - mean) / torch.sqrt(var + 1e-5)
        else:
            y_hat = y

        if self.use_hebbian:
            with torch.no_grad():
                yh2 = y_hat.reshape(B, -1, self.outFeatures) 
                N = float(yh2.size(1)) if yh2.size(1) > 0 else 1.0

                x32 = x2
                y32 = yh2

                hebb_term = torch.einsum("bno,bni->boi", y32, x32) / N

                y_sq_mean = y32.square().mean(dim=1)

                decay = y_sq_mean.unsqueeze(-1) * self.hebb_memory
                delta = self.hebb_rate * (hebb_term - decay)

                self.hebb_memory.mul_(self.ema_alpha).add_(delta, alpha=(1.0 - self.ema_alpha))

                if self.weight_constraint == "clip":
                    self.hebb_memory.clamp_(-1.0, 1.0)
                elif self.weight_constraint == "norm":
                    eps = 1e-8
                    n = self.hebb_memory.norm(dim=-1, keepdim=True).clamp_min(eps)
                    self.hebb_memory.div_(n)

                FrobeniusCapPerSample(self.hebb_memory, self.mem_norm_cap)

        return y_hat



class TransformerEncode(AGICoreModule):
    def __init__(self, modelDim: int, headNum: int, dimFeedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_atten = RoPEMultiheadAttention(
            embedDim=modelDim,
            numHeads=headNum,
            dropout=dropout,)
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
            attnMask=srcMask,
            keyPaddingMask=srcKeyPaddingMask,
            needWeights=False)
        
        src = src + self.dropout1(src2)

        src_norm2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(src2)
        return src
    

class ResidualBlock(AGICoreModule):
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1, useHebbian: bool = False):
        super().__init__()
        self.use_downsample = bool(stride != 1 or inChannels != outChannels)
        self.downsample = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
            Norm2d(outChannels))
            
        self.conv1 = HebbianConv2d(inChannels, outChannels, 3, stride=stride, padding=1,bias=False, useHebbian=useHebbian)
        self.bn1 = Norm2d(outChannels)
        self.conv2 = HebbianConv2d(outChannels, outChannels, 3, stride=1, padding=1,bias=False, useHebbian=useHebbian)
        self.bn2 = Norm2d(outChannels)
        self.relu = nn.SiLU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.use_downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class CNNFeatureExtractor(AGICoreModule):
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


class PerceiveExtractor(AGICoreModule):
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
        posDrop: float = 0.1,
        objectTokenCount: int = 16):
        super().__init__()

        assert embedDim % numHeads == 0, "embed_dim must be divisible by num_heads"

        self.img_size = imgSize
        self.patch_size = patchSize
        self.embed_dim = int(embedDim)
        self.legacy_dim = int(embedDim * 2)
        self.object_token_count = int(objectTokenCount)
        self.use_hebbian = useHebbian
        self.base_channels = baseChannels

        self.cnn_extractor = CNNFeatureExtractor(
            inChannels=3,
            baseChannels=baseChannels,
            useHebbian=useHebbian)

        with torch.no_grad():
            mods = [m for m in self.cnn_extractor.modules() if hasattr(m, "use_hebbian")]
            old = [bool(m.use_hebbian) for m in mods]
            for m in mods: m.use_hebbian = False

            dummy = torch.zeros(1, 3, imgSize, imgSize)
            fmap = self.cnn_extractor(dummy)
            Hf, Wf = fmap.shape[-2], fmap.shape[-1]

            for m, v in zip(mods, old):
                m.use_hebbian = v

        cnn_feat_dim = baseChannels * 16

        self.patch_embed = SheafGaugeConv2d(
            in_channels=cnn_feat_dim,
            out_channels=embedDim,
            kernel_size=patchSize,
            stride=patchSize,
            bias=False,
            device=self.device,
            dtype=self.dtype,
            sheaf_alpha=0.1,
            sheaf_iters=1,
            gauge_groups=1, 
            gauge_scale=0.1,
            gauge_bias_scale=0.1)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedDim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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

        self.ventral_proj = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.dorsal_proj = nn.Sequential(
            nn.LayerNorm(embedDim * 2),
            nn.Linear(embedDim * 2, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.motion_proj = nn.Sequential(
            nn.LayerNorm(embedDim * 2),
            nn.Linear(embedDim * 2, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.quality_proj = nn.Sequential(
            nn.Linear(4, embedDim),
            nn.LayerNorm(embedDim),
            nn.SiLU(),
            nn.Linear(embedDim, embedDim),
            nn.LayerNorm(embedDim))

        self.pred_error_input_dim = self.legacy_dim * 3 + embedDim * 2
        self.pred_error_proj = nn.Sequential(
            nn.LayerNorm(self.pred_error_input_dim),
            nn.Linear(self.pred_error_input_dim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.object_queries = nn.Parameter(torch.randn(self.object_token_count, embedDim) * 0.02)
        self.object_key = nn.Linear(embedDim, embedDim)
        self.object_value = nn.Linear(embedDim, embedDim)
        self.object_post = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.temporal_state = nn.GRUCell(self.legacy_dim, self.legacy_dim)
        self.temporal_norm = nn.LayerNorm(self.legacy_dim)
        self.topdown_gate = nn.Sequential(
            nn.Linear(self.legacy_dim * 3, self.legacy_dim),
            nn.SiLU(),
            nn.Linear(self.legacy_dim, self.legacy_dim),
            nn.Sigmoid())

        self.legacy_fusion = nn.Sequential(
            nn.LayerNorm(self.legacy_dim + embedDim * 5),
            nn.Linear(self.legacy_dim + embedDim * 5, self.legacy_dim),
            nn.GELU(),
            nn.Linear(self.legacy_dim, self.legacy_dim),
            nn.LayerNorm(self.legacy_dim))

        self.motion_decoder = nn.Linear(embedDim, embedDim)
        self.pred_error_decoder = nn.Linear(embedDim, self.pred_error_input_dim)

        self.InitWeights()

    def QualityStats(self, x: torch.Tensor) -> torch.Tensor:
        x_det = x.detach()
        mean = x_det.mean(dim=(1, 2, 3))
        std = x_det.std(dim=(1, 2, 3), unbiased=False)
        if x_det.size(-1) > 1:
            gx = (x_det[..., :, 1:] - x_det[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        else:
            gx = torch.zeros_like(mean)
        if x_det.size(-2) > 1:
            gy = (x_det[..., 1:, :] - x_det[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        else:
            gy = torch.zeros_like(mean)
        grad = 0.5 * (gx + gy)
        clipped = ((x_det <= 0.01) | (x_det >= 0.99)).float().mean(dim=(1, 2, 3))
        return torch.stack([mean, std, grad, clipped], dim=-1)

    def BuildObjectTokens(self, patchTokens: torch.Tensor) -> torch.Tensor:
        B, _, D = patchTokens.shape
        k = self.object_key(patchTokens)
        v = self.object_value(patchTokens)
        q = self.object_queries
        scores = torch.einsum("kd,bnd->bkn", q, k) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        tokens = torch.einsum("bkn,bnd->bkd", weights, v)
        return self.object_post(tokens)

    def ObjectAttentionError(self, currentObjects: torch.Tensor, predictedObjects: torch.Tensor) -> torch.Tensor:
        D = int(currentObjects.size(-1))
        scores = torch.matmul(currentObjects, predictedObjects.transpose(1, 2)) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        aligned_pred = torch.matmul(weights, predictedObjects)
        return (currentObjects - aligned_pred).mean(dim=1)

    def BuildStructuredPredictionError(
        self,
        preliminaryLegacy: torch.Tensor,
        globalFeat: torch.Tensor,
        motionToken: torch.Tensor,
        objectTokens: torch.Tensor,
        predicted: Optional[Dict[str, torch.Tensor]],
        precision: torch.Tensor,) -> torch.Tensor:
        if predicted is None:
            legacy_err = torch.zeros_like(preliminaryLegacy)
            global_err = torch.zeros_like(globalFeat)
            object_err = torch.zeros_like(motionToken)
            motion_err = torch.zeros_like(motionToken)
            basis_err = torch.zeros_like(globalFeat)
        else:
            legacy_err = preliminaryLegacy - predicted["LegacyFeat"].detach()
            global_err = globalFeat - predicted["GlobalFeat"].detach()
            object_err = self.ObjectAttentionError(objectTokens, predicted["ObjectTokens"].detach())
            motion_err = motionToken - predicted["MotionPred"].detach()
            basis_err = globalFeat - predicted["PredErrorBasis"].detach()

        p = precision.view(-1, 1)
        legacy_err = legacy_err * p
        global_err = global_err * p
        object_err = object_err * p
        motion_err = motion_err * p
        basis_err = basis_err * p

        return torch.cat([legacy_err, global_err, object_err, motion_err, basis_err], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        prevVisualState: Optional[VisualState] = None,) -> VisualState:
        # x: [B, 3, H, W]
        frame = x
        feat = self.cnn_extractor(frame)  # [B, C, Hf, Wf]

        feat = self.cnn_feat_adapter(feat) 

        patches = self.patch_adapter(feat)  # [B, embed_dim, Ph, Pw]
        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, embed_dim]
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

        preliminary_legacy = torch.cat([out, global_patch], dim=1)

        ventral_feat = self.ventral_proj(out)

        if prevVisualState is not None:
            prev_ventral = prevVisualState.VentralFeat
            ventral_delta = ventral_feat - prev_ventral.detach()
        else:
            ventral_delta = torch.zeros_like(ventral_feat)

        dorsal_feat = self.dorsal_proj(torch.cat([ventral_feat, ventral_delta], dim=-1)) # [B, embedDim]
        motion_token = self.motion_proj(torch.cat([dorsal_feat, ventral_delta], dim=-1)) # [B, embedDim]

        object_tokens = self.BuildObjectTokens(patch_tokens)

        if (prevVisualState is not None
            and "TemporalState" in prevVisualState.NextState):
            h_prev = prevVisualState.NextState["TemporalState"]
        else:
            h_prev = preliminary_legacy.new_zeros(preliminary_legacy.shape)

        h_next = self.temporal_state(preliminary_legacy, h_prev.detach())
        temporal_feat = self.temporal_norm(h_next)

        topdown_feat = topDownContext.MemoryCue
        td_gate = self.topdown_gate(torch.cat([preliminary_legacy, temporal_feat, topdown_feat], dim=-1))
        global_feat = td_gate * preliminary_legacy + (1.0 - td_gate) * temporal_feat

        quality_token = self.quality_proj(self.QualityStats(frame))

        predicted = topDownContext.PredictedVisual
        pred_error_target = self.BuildStructuredPredictionError(
            preliminary_legacy,
            global_feat,
            motion_token,
            object_tokens,
            predicted,
            topDownContext.Precision)
        pred_error_token = self.pred_error_proj(pred_error_target)

        legacy_in = torch.cat([
            global_feat,
            ventral_feat,
            dorsal_feat,
            motion_token,
            quality_token,
            pred_error_token], dim=-1)
        legacy_feat = self.legacy_fusion(legacy_in)

        return VisualState(
            LegacyFeat=legacy_feat,
            GlobalFeat=global_feat,
            VentralFeat=ventral_feat,
            DorsalFeat=dorsal_feat,
            MotionToken=motion_token,
            QualityToken=quality_token,
            PredErrorToken=pred_error_token,
            ObjectTokens=object_tokens,
            PatchTokens=patch_tokens,
            NextState={
                "TemporalState": h_next.detach(),
                "PredErrorTarget": pred_error_target.detach()},)

    def ComputePerceptionLoss(
        self,
        visualState: VisualState,
        prevVisualState: Optional[VisualState] = None,) -> torch.Tensor:
        loss = visualState.LegacyFeat.new_zeros(())

        obj = visualState.ObjectTokens
        obj_n = F.normalize(obj, dim=-1, eps=1e-6)
        sim = torch.matmul(obj_n, obj_n.transpose(1, 2))
        eye = torch.eye(obj.size(1), device=obj.device, dtype=torch.bool).unsqueeze(0)
        diversity = sim.masked_select(~eye).pow(2).mean()
        loss = loss + diversity

        if prevVisualState is not None:
            motion_target = (visualState.VentralFeat - prevVisualState.VentralFeat.detach()).detach()
            motion_pred = self.motion_decoder(visualState.MotionToken)
            loss = loss + F.smooth_l1_loss(motion_pred, motion_target)

        pred_target = visualState.NextState["PredErrorTarget"]
        pred_out = self.pred_error_decoder(visualState.PredErrorToken)
        loss = loss + F.smooth_l1_loss(pred_out, pred_target)

        return loss

    def InitWeights(self):
        for name, m in self.named_modules():
            if isinstance(m, SheafGaugeConv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif "gauge_" in name:
                if hasattr(m, "weight"): nn.init.zeros_(m.weight)
                if hasattr(m, "bias") and m.bias is not None: nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
                if getattr(m, "affine", True):
                    if hasattr(m, "weight") and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.zeros_(m.bias)

            elif isinstance(m, HebbianLinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def ResetHebbianMemory(self): 
        for m in self.modules(): 
            if m is self:
                continue
            
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

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        prevVisualState: Optional[VisualState] = None,) -> VisualState:
        return super().forward(
            x,
            topDownContext=topDownContext,
            prevVisualState=prevVisualState)

    def ComputePerceptionLoss(
        self,
        visualState: VisualState,
        prevVisualState: Optional[VisualState] = None,) -> torch.Tensor:
        return self.base.ComputePerceptionLoss(
            visualState,
            prevVisualState=prevVisualState)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        C_feat = self.base.cnn_feat_adapter.C
        
        patch_w = self.base.patch_embed.weight
        E_out = patch_w.size(0) 
        C_in = patch_w.size(1)   
        kh, kw = self.base.patch_embed.kernel_size
        ksz = kh * kw
        
        D_model = int(self.base.cls_token.size(-1))
        L_trans = len(self.base.transformer_layers)

        def alloc_feat(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, C_feat, device=device, dtype=dtype) * 1e-4) 
            B = nn.Parameter(torch.zeros(C_feat, addRank, device=device, dtype=dtype)) 
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_feat(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        def alloc_patch(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, C_in * ksz, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(E_out, addRank, device=device, dtype=dtype))
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_patch(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        def alloc_token(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, D_model, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(D_model, addRank, device=device, dtype=dtype))
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_token(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        return {
            "feat": SiteSpec("feat", 1, C_feat, C_feat, self.maxRankFeat, alloc_feat, compose_feat),
            "patch": SiteSpec("patch", 1, C_in * ksz, E_out, self.maxRankPatch, alloc_patch, compose_patch),
            "token": SiteSpec("token", L_trans, D_model, D_model, self.maxRankToken, alloc_token, compose_token),}

    def ForwardWithDeltas(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> VisualState:
        frame = x
        topDownContext = kwargs["topDownContext"]
        prevVisualState = kwargs.get("prevVisualState", None)

        feat = self.base.cnn_extractor(frame)
        
        feat = self.base.cnn_feat_adapter(feat)

        deltaFeat2D = deltasPerLayer[0].get("feat", None)
        if deltaFeat2D is not None:
            C = deltaFeat2D.size(0)
            w1x1 = deltaFeat2D.view(C, C, 1, 1)
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
            W_eff = W_eff + deltaPatch2D.view(E, C_in, kh, kw)

        patches = F.conv2d(
            feat_patch,
            W_eff,
            bias=None, 
            stride=self.base.patch_embed.stride,
            padding=self.base.patch_embed.padding,
            dilation=self.base.patch_embed.dilation,
            groups=self.base.patch_embed.groups,)

        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, "b c h w -> b (h w) c")
        cls_tokens = repeat(self.base.cls_token, "1 1 d -> b 1 d", b=B)
        xTok = torch.cat([cls_tokens, patches], dim=1)
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

        preliminary_legacy = torch.cat([out, global_patch], dim=1)

        ventral_feat = self.base.ventral_proj(out)
        if prevVisualState is not None:
            ventral_delta = ventral_feat - prevVisualState.VentralFeat.detach()
        else:
            ventral_delta = torch.zeros_like(ventral_feat)

        dorsal_feat = self.base.dorsal_proj(torch.cat([ventral_feat, ventral_delta], dim=-1))
        motion_token = self.base.motion_proj(torch.cat([dorsal_feat, ventral_delta], dim=-1))
        object_tokens = self.base.BuildObjectTokens(patch_tokens)

        if (prevVisualState is not None
            and "TemporalState" in prevVisualState.NextState):
            h_prev = prevVisualState.NextState["TemporalState"]
        else:
            h_prev = preliminary_legacy.new_zeros(preliminary_legacy.shape)

        h_next = self.base.temporal_state(preliminary_legacy, h_prev.detach())
        temporal_feat = self.base.temporal_norm(h_next)

        topdown_feat = topDownContext.MemoryCue
        td_gate = self.base.topdown_gate(torch.cat([preliminary_legacy, temporal_feat, topdown_feat], dim=-1))
        global_feat = td_gate * preliminary_legacy + (1.0 - td_gate) * temporal_feat

        quality_token = self.base.quality_proj(self.base.QualityStats(frame))
        pred_error_target = self.base.BuildStructuredPredictionError(
            preliminary_legacy,
            global_feat,
            motion_token,
            object_tokens,
            topDownContext.PredictedVisual,
            topDownContext.Precision)
        pred_error_token = self.base.pred_error_proj(pred_error_target)

        legacy_feat = self.base.legacy_fusion(torch.cat([
            global_feat,
            ventral_feat,
            dorsal_feat,
            motion_token,
            quality_token,
            pred_error_token], dim=-1))

        return VisualState(
            LegacyFeat=legacy_feat,
            GlobalFeat=global_feat,
            VentralFeat=ventral_feat,
            DorsalFeat=dorsal_feat,
            MotionToken=motion_token,
            QualityToken=quality_token,
            PredErrorToken=pred_error_token,
            ObjectTokens=object_tokens,
            PatchTokens=patch_tokens,
            NextState={
                "TemporalState": h_next.detach(),
                "PredErrorTarget": pred_error_target.detach()},)

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

        elif site == "patch":
            if layerIdx != 0:
                return False
            init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}
            self.base.patch_adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)

        elif site == "token":
            init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}
            self.base.token_adapters[layerIdx].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)

        else:
            raise ValueError(f"Unknown site: {site}")
        
        return True
        




class TestPerceptionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def MakeTopDownContext(self, model, B: int, dtype: torch.dtype = torch.float32, predictedVisual: Optional[Dict[str, torch.Tensor]] = None) -> TopDownContext:
        runtime = model.base if hasattr(model, "base") else model
        legacy_dim = int(runtime.legacy_dim)
        return TopDownContext(
            PredictedVisual=predictedVisual,
            Precision=torch.ones(B, device=self.device, dtype=dtype),
            MemoryCue=torch.zeros(B, legacy_dim, device=self.device, dtype=dtype),)

    def PerceptionForward(self, model, x: torch.Tensor, prevVisualState: Optional[VisualState] = None, predictedVisual: Optional[Dict[str, torch.Tensor]] = None) -> VisualState:
        return model(
            x,
            topDownContext=self.MakeTopDownContext(model, int(x.size(0)), x.dtype, predictedVisual),
            prevVisualState=prevVisualState)

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
            scale = torch.tanh(s.detach()) * GetParametersScale(s.detach())
            delta = delta + scale * (B2 @ A2) 
        return delta

    def DeltaFromTokenAdapter(self, adapter) -> torch.Tensor:
        if not hasattr(adapter, "A_list") or len(adapter.A_list) == 0:
            D = adapter.D if hasattr(adapter, "D") else 0
            return torch.zeros(D, D, device=self.device)
        D = adapter.D
        delta = torch.zeros(D, D, device=adapter.A_list[0].device, dtype=adapter.A_list[0].dtype)
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            scale = torch.tanh(s.detach()) * GetParametersScale(s.detach())
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
            out = self.PerceptionForward(model, x)
            expected_dim = 512 * 2
            assert tuple(out.LegacyFeat.shape) == (2, expected_dim), f"Output shape does not match: {out.LegacyFeat.shape}"
            print("PerceiveExtractor forward passed.")
            return True
        except AssertionError as e:
            print(f"PerceiveExtractor test failed: {e}")
            return False
        except Exception as e:
            print(f"PerceiveExtractor test error: {e}")
            return False

    def TestPerceiveExtractorIOShapes(self):
        try:
            batch_size = 2
            img_size = 512
            embed_dim = 512

            model = PerceiveExtractor(
                imgSize=img_size,
                patchSize=1,
                embedDim=embed_dim,
                numHeads=8,
                numLayers=6,
                useHebbian=True).to(self.device)
            x = torch.randn(batch_size, 3, img_size, img_size, device=self.device)

            with torch.no_grad():
                out = self.PerceptionForward(model, x)

            expected_out_shape = (batch_size, embed_dim * 2)
            assert tuple(out.LegacyFeat.shape) == expected_out_shape, f"Output shape does not match: {out.LegacyFeat.shape}"

            print(f"PerceiveExtractor forward input shape: {tuple(x.shape)}")
            print(f"PerceiveExtractor forward output shape: {tuple(out.LegacyFeat.shape)}")
            return True
        except AssertionError as e:
            print(f"TestPerceiveExtractorIOShapes failed: {e}")
            return False
        except Exception as e:
            print(f"TestPerceiveExtractorIOShapes error: {e}")
            return False

    def TestPerceiveExtractorStructuredState(self):
        try:
            B = 2
            model = PerceiveExtractor(
                imgSize=64,
                patchSize=1,
                embedDim=512,
                numHeads=8,
                numLayers=2,
                baseChannels=16,
                useHebbian=True).to(self.device)
            x = torch.randn(B, 3, 64, 64, device=self.device)

            with torch.no_grad():
                state0 = self.PerceptionForward(model, x)
                out = self.PerceptionForward(model, x)
                pred_visual = {
                    "LegacyFeat": torch.randn(B, 1024, device=self.device),
                    "GlobalFeat": torch.randn(B, 1024, device=self.device),
                    "ObjectTokens": torch.randn(B, 16, 512, device=self.device),
                    "MotionPred": torch.randn(B, 512, device=self.device),
                    "PredErrorBasis": torch.randn(B, 1024, device=self.device),}
                ctx = TopDownContext(
                    GoalBias=torch.randn(B, 512, device=self.device),
                    PredictedVisual=pred_visual,
                    Precision=torch.ones(B, device=self.device),
                    MemoryCue=torch.randn(B, 1024, device=self.device))
                state1 = model(x, topDownContext=ctx, prevVisualState=state0)

            assert tuple(out.LegacyFeat.shape) == (B, 1024), f"legacy forward shape mismatch: {out.LegacyFeat.shape}"
            assert tuple(state1.LegacyFeat.shape) == (B, 1024)
            assert tuple(state1.GlobalFeat.shape) == (B, 1024)
            assert tuple(state1.VentralFeat.shape) == (B, 512)
            assert tuple(state1.DorsalFeat.shape) == (B, 512)
            assert tuple(state1.MotionToken.shape) == (B, 512)
            assert tuple(state1.QualityToken.shape) == (B, 512)
            assert tuple(state1.PredErrorToken.shape) == (B, 512)
            assert tuple(state1.ObjectTokens.shape) == (B, 16, 512)
            assert state1.PatchTokens.dim() == 3 and state1.PatchTokens.size(0) == B and state1.PatchTokens.size(-1) == 512
            assert "TemporalState" in state1.NextState
            assert "PredErrorTarget" in state1.NextState
            model.ResetHebbianMemory()
            assert tuple(state1.NextState["TemporalState"].shape) == (B, 1024)
            assert tuple(state1.NextState["PredErrorTarget"].shape) == (B, model.pred_error_input_dim)
            print("PerceiveExtractor structured state passed.")
            return True
        except AssertionError as e:
            print(f"TestPerceiveExtractorStructuredState failed: {e}")
            return False
        except Exception as e:
            print(f"TestPerceiveExtractorStructuredState error: {e}")
            return False

    def TrainStepSmoke(self):
        try:
            model = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16, useHebbian=True).to(self.device)
            model.train()
            head = nn.Linear(64 * 2, 16).to(self.device)
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 3, 64, 64, device=self.device)
            target = torch.randn(8, 16, device=self.device)

            out = self.PerceptionForward(model, x)
            pred = head(out.LegacyFeat)
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
                pred = head(self.PerceptionForward(model, x).LegacyFeat)
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
                pred = head(self.PerceptionForward(model, x).LegacyFeat)
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
                start = F.mse_loss(head(self.PerceptionForward(model, data_x).LegacyFeat), data_y).item()

            for t in range(1, steps + 1):
                pred = head(self.PerceptionForward(model, data_x).LegacyFeat)
                loss = F.mse_loss(pred, data_y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if (t % logEvery) == 0 or t == 1:
                    print(f"[PerceptionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(self.PerceptionForward(model, data_x).LegacyFeat), data_y).item()

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
                y_base = self.PerceptionForward(base, x)
                y_wrap = self.PerceptionForward(wrapper, x)

            max_abs = (y_base.LegacyFeat - y_wrap.LegacyFeat).abs().max().item()
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
                pred = head(self.PerceptionForward(wrapper, x).LegacyFeat)
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
                y0 = self.PerceptionForward(base, x_chk)
                y1 = self.PerceptionForward(wrapper, x_chk)
            assert torch.allclose(y0.LegacyFeat, y1.LegacyFeat, atol=1e-6, rtol=1e-4), "base vs wrapper mismatch after commit."

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

            out = self.PerceptionForward(wrapper, x)
            pred = head(out.LegacyFeat)
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
                pred = head(self.PerceptionForward(wrapper, data_x).LegacyFeat)
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
                y0 = self.PerceptionForward(base, x_chk)
                y1 = self.PerceptionForward(wrapper, x_chk)
            max_abs = (y0.LegacyFeat - y1.LegacyFeat).abs().max().item()
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
            pred = head(self.PerceptionForward(model, x).LegacyFeat)
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
                    start = F.mse_loss(head(self.PerceptionForward(model, data_x).LegacyFeat), data_y).item()

                hist = []
                for _ in range(steps):
                    pred = head(self.PerceptionForward(model, data_x).LegacyFeat)
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
            pred = head(self.PerceptionForward(model, x).LegacyFeat)
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
            "PerceiveExtractorIOShapes": self.TestPerceiveExtractorIOShapes(),
            "PerceiveExtractorStructuredState": self.TestPerceiveExtractorStructuredState(),
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
