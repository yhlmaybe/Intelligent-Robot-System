import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, RoPEMultiheadAttention, HungarianAssignment
from ModuleMessagerManager import ModuleDim


@dataclass
class TopDownContext:
    Precision: torch.Tensor
    MemoryCue: torch.Tensor
    PredictedVisual: Optional[Any] = None


@dataclass
class VisualState:
    IntegratedFeat: torch.Tensor
    GlobalFeat: torch.Tensor
    VentralFeat: torch.Tensor
    DorsalFeat: torch.Tensor
    MotionToken: torch.Tensor
    QualityToken: torch.Tensor
    PredErrorToken: torch.Tensor
    ObjectTokens: torch.Tensor
    PatchTokens: torch.Tensor
    SemanticNodes: Dict[str, torch.Tensor]
    Auxiliary: Dict[str, torch.Tensor] = field(default_factory=dict)


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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        deep = self.relu(self.bn2(self.conv2(layer4)))
        return {
            "Layer1": layer1,
            "Layer2": layer2,
            "Layer3": layer3,
            "Deep": deep,}


class DepthGeometryFusion(AGICoreModule):
    def __init__(
        self,
        featureChannels: int,
        midChannels: int,
        shallowChannels: int,
        fineChannels: int,
        minDepthMeters: float = 0.05,
        maxDepthMeters: float = 20.0,
        sensorDropout: float = 0.1,
        virtualPlanarityWindow: int = 5,
        virtualDisagreeThreshold: float = 0.20,
        virtualContentMargin: float = 0.05,):
        super().__init__()
        self.feature_channels = int(featureChannels)
        self.min_depth_meters = float(minDepthMeters)
        self.max_depth_meters = float(maxDepthMeters)
        self.sensor_dropout = float(sensorDropout)
        window = max(3, int(virtualPlanarityWindow))
        self.virtual_planarity_window = window if window % 2 == 1 else window + 1
        self.virtual_disagree_threshold = float(virtualDisagreeThreshold)
        self.virtual_content_margin = float(virtualContentMargin)
        hidden = max(16, self.feature_channels // 8)
        self.hidden = hidden

        self.depth_deep = nn.Sequential(
            nn.Conv2d(self.feature_channels, hidden, kernel_size=3, padding=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),)
        self.depth_mid = nn.Sequential(
            nn.Conv2d(int(midChannels), hidden, kernel_size=1, bias=False),
            Norm2d(hidden),)
        self.depth_shallow = nn.Sequential(
            nn.Conv2d(int(shallowChannels), hidden, kernel_size=1, bias=False),
            Norm2d(hidden),)
        self.depth_fine = nn.Sequential(
            nn.Conv2d(int(fineChannels), hidden, kernel_size=1, bias=False),
            Norm2d(hidden),)
        self.depth_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                Norm2d(hidden),
                nn.SiLU(),)
            for _ in range(3)])
        self.monocular_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),)

        self.virtual_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),)

        self.sensor_var_head = nn.Sequential(
            nn.Conv2d(hidden + 4, hidden, kernel_size=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),)

        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(8, hidden, kernel_size=3, padding=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.feature_channels, kernel_size=1, bias=True),)
        self.geometry_gate = nn.Sequential(
            nn.Conv2d(self.feature_channels * 2 + 2, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, self.feature_channels, kernel_size=1, bias=True),)

        self.sensor_log_variance = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def SpatialGradient(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
        return grad_x, grad_y

    def LocalStd(self, value: torch.Tensor, window: int) -> torch.Tensor:
        pad = int(window) // 2
        mean = F.avg_pool2d(value, int(window), stride=1, padding=pad)
        second = F.avg_pool2d(value * value, int(window), stride=1, padding=pad)
        return (second - mean * mean).clamp_min(0.0).add(1e-6).sqrt()

    def ResampleSensorDepth(
        self,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        size: Tuple[int, int],) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = depthValid.bool()
        valid_float = valid.to(depth.dtype)
        clean_depth = torch.where(valid, depth, torch.ones_like(depth))
        inverse = clean_depth.reciprocal() * valid_float
        inverse_sum = F.interpolate(inverse, size=size, mode="area")
        valid_weight = F.interpolate(valid_float, size=size, mode="area")
        inverse_resized = inverse_sum / valid_weight.clamp_min(1e-6)
        inverse_resized = inverse_resized * (valid_weight > 1e-6).to(inverse_resized.dtype)
        return inverse_resized, valid_weight.clamp(0.0, 1.0)

    def DecodeMonocularDepth(
        self,
        rgbFeatures: torch.Tensor,
        midFeatures: torch.Tensor,
        shallowFeatures: torch.Tensor,
        fineFeatures: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = self.depth_deep(rgbFeatures)
        decoded = F.interpolate(decoded, size=midFeatures.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.depth_refine[0](decoded + self.depth_mid(midFeatures))
        decoded = F.interpolate(decoded, size=shallowFeatures.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.depth_refine[1](decoded + self.depth_shallow(shallowFeatures))
        decoded = F.interpolate(decoded, size=fineFeatures.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.depth_refine[2](decoded + self.depth_fine(fineFeatures))
        raw_visual = self.monocular_head(decoded)
        log_min_inverse = math.log(1.0 / self.max_depth_meters)
        log_max_inverse = math.log(1.0 / self.min_depth_meters)
        mono_log_inverse = log_min_inverse + (
            log_max_inverse - log_min_inverse) * torch.sigmoid(raw_visual[:, :1])
        mono_inverse = mono_log_inverse.exp()
        mono_log_variance = raw_visual[:, 1:2].clamp(-6.0, 6.0)
        return mono_inverse, mono_log_variance, decoded

    def BackprojectDepth(
        self,
        depth: torch.Tensor,
        cameraIntrinsics: torch.Tensor,
        sourceSize: Tuple[int, int],) -> torch.Tensor:
        B, _, H, W = depth.shape
        fx, fy = cameraIntrinsics[:, 0, 0], cameraIntrinsics[:, 1, 1]
        cx, cy = cameraIntrinsics[:, 0, 2], cameraIntrinsics[:, 1, 2]
        src_h, src_w = sourceSize
        fx = fx * (float(W) / float(src_w))
        fy = fy * (float(H) / float(src_h))
        cx = cx * (float(W) / float(src_w))
        cy = cy * (float(H) / float(src_h))
        yy, xx = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing="ij",)
        z = depth
        x = (xx.view(1, 1, H, W) - cx.view(B, 1, 1, 1)) * z / fx.view(B, 1, 1, 1).clamp_min(1e-6)
        y = (yy.view(1, 1, H, W) - cy.view(B, 1, 1, 1)) * z / fy.view(B, 1, 1, 1).clamp_min(1e-6)
        return torch.cat([x, y, z], dim=1)

    @staticmethod
    def QuaternionRotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        q_vec = quat[:, :3].view(quat.size(0), 3, 1, 1).expand_as(vec)
        q_w = quat[:, 3].view(quat.size(0), 1, 1, 1)
        cross1 = torch.cross(q_vec, vec, dim=1)
        return vec + 2.0 * (q_w * cross1 + torch.cross(q_vec, cross1, dim=1))

    def WarpPrevDepth(
        self,
        curDepth: torch.Tensor,
        prevDepth: torch.Tensor,
        cameraIntrinsics: torch.Tensor,
        sourceSize: Tuple[int, int],
        cameraMotion: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = curDepth.shape
        sx = float(W) / float(sourceSize[1])
        sy = float(H) / float(sourceSize[0])
        fx = (cameraIntrinsics[:, 0, 0] * sx).view(B, 1, 1, 1).clamp_min(1e-6)
        fy = (cameraIntrinsics[:, 1, 1] * sy).view(B, 1, 1, 1).clamp_min(1e-6)
        cx = (cameraIntrinsics[:, 0, 2] * sx).view(B, 1, 1, 1)
        cy = (cameraIntrinsics[:, 1, 2] * sy).view(B, 1, 1, 1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=curDepth.device, dtype=curDepth.dtype),
            torch.arange(W, device=curDepth.device, dtype=curDepth.dtype),
            indexing="ij",)
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        point_cur = torch.cat([
            (xx - cx) * curDepth / fx,
            (yy - cy) * curDepth / fy,
            curDepth], dim=1)

        translation = cameraMotion[:, :3].view(B, 3, 1, 1)
        rotation = cameraMotion[:, 3:7]
        point_prev = self.QuaternionRotate(rotation, point_cur) + translation
        expected_prev = point_prev[:, 2:3]
        inv_z = expected_prev.clamp_min(1e-3).reciprocal()
        grid_x = 2.0 * (fx * point_prev[:, 0:1] * inv_z + cx) / float(max(W - 1, 1)) - 1.0
        grid_y = 2.0 * (fy * point_prev[:, 1:2] * inv_z + cy) / float(max(H - 1, 1)) - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)
        sampled_prev = F.grid_sample(
            prevDepth, grid, mode="bilinear", padding_mode="border", align_corners=True)
        in_bounds = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
        valid = (in_bounds & (expected_prev > 1e-3)).to(curDepth.dtype)
        return expected_prev, sampled_prev, valid

    def forward(
        self,
        rgbFeatures: torch.Tensor,
        midFeatures: torch.Tensor,
        shallowFeatures: torch.Tensor,
        fineFeatures: torch.Tensor,
        depth: torch.Tensor,
        depthValid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, _, H, W = rgbFeatures.shape
        mono_inverse, mono_log_variance, trunk_features = self.DecodeMonocularDepth(
            rgbFeatures, midFeatures, shallowFeatures, fineFeatures)
        visual_precision = torch.exp(-mono_log_variance).clamp_max(1e4)

        sensor_inverse, sensor_valid = self.ResampleSensorDepth(
            depth, depthValid, tuple(mono_inverse.shape[-2:]))
        sensor_observed_valid = sensor_valid
        if self.training and self.sensor_dropout > 0.0:
            keep = (torch.rand(B, 1, 1, 1, device=rgbFeatures.device) >= self.sensor_dropout).to(rgbFeatures.dtype)
            sensor_valid = sensor_valid * keep

        disagreement = ((mono_inverse - sensor_inverse).abs() * sensor_valid) / mono_inverse.abs().clamp_min(1e-6)

        sensor_var_cue = torch.cat([sensor_inverse, sensor_valid, mono_inverse, disagreement], dim=1)
        sensor_log_var_delta = self.sensor_var_head(torch.cat([trunk_features, sensor_var_cue], dim=1))
        sensor_log_var_spatial = (
            self.sensor_log_variance + sensor_log_var_delta).clamp(-8.0, 8.0)
        sensor_precision_scale = torch.exp(-sensor_log_var_spatial)
        sensor_precision = sensor_valid * sensor_precision_scale

        virtual_logits = self.virtual_head(trunk_features)
        p_virtual = torch.sigmoid(virtual_logits)

        mono_precision_physical = visual_precision * (1.0 - p_virtual)
        total_precision_physical = mono_precision_physical + sensor_precision
        fused_inverse = (
            mono_precision_physical * mono_inverse
            + sensor_precision * sensor_inverse) / total_precision_physical.clamp_min(1e-6)
        physical_depth = fused_inverse.clamp_min(1.0 / self.max_depth_meters).reciprocal()
        physical_log_variance = -torch.log(total_precision_physical.clamp_min(1e-6))
        sensor_reliability = sensor_precision / total_precision_physical.clamp_min(1e-6)
        content_depth = mono_inverse.clamp_min(1.0 / self.max_depth_meters).reciprocal()

        log_content = content_depth.clamp_min(self.min_depth_meters).log()
        log_physical = physical_depth.clamp_min(self.min_depth_meters).log()
        d_grad_x, d_grad_y = self.SpatialGradient(log_physical)
        confidence = torch.sigmoid(-physical_log_variance)
        geometry_cues = torch.cat([
            log_physical, d_grad_x, d_grad_y, confidence,
            sensor_reliability, sensor_valid,
            p_virtual, log_content,], dim=1)
        geometry_cues_full = F.interpolate(geometry_cues, size=(H, W), mode="bilinear", align_corners=False)
        geometry_features = self.geometry_encoder(geometry_cues_full)
        gate = torch.sigmoid(self.geometry_gate(torch.cat([
            rgbFeatures,
            geometry_features,
            geometry_cues_full[:, 4:5],
            geometry_cues_full[:, 6:7],], dim=1)))
        fused_features = rgbFeatures + gate * geometry_features

        depth_state = {
            "MonocularDepth": content_depth,
            "MonocularDepthLogVariance": mono_log_variance,
            "MetricDepth": physical_depth,
            "MetricDepthLogVariance": physical_log_variance,
            "SensorDepthReliability": sensor_reliability,
            "SensorDepthValid": sensor_observed_valid,
            "SensorDepthUsed": sensor_valid,
            "ContentDepth": content_depth,
            "VirtualMask": p_virtual,
            "VirtualMaskLogits": virtual_logits,
            "SensorLogVarianceSpatial": sensor_log_var_spatial,}

        if self.training:
            sensor_local_std = self.LocalStd(sensor_inverse * sensor_valid, self.virtual_planarity_window)
            mono_local_std = self.LocalStd(mono_inverse, self.virtual_planarity_window)
            depth_state["VirtualTarget"] = (
                (sensor_valid > 0.5)
                & (disagreement > self.virtual_disagree_threshold)
                & (mono_local_std > sensor_local_std + self.virtual_content_margin)).to(p_virtual.dtype)
            feat_grad_x, feat_grad_y = self.SpatialGradient(fineFeatures.mean(dim=1, keepdim=True))
            edge_w_x = (-feat_grad_x.abs() * 5.0).exp()
            edge_w_y = (-feat_grad_y.abs() * 5.0).exp()
            depth_state["EdgeAwareSmoothness"] = (
                (d_grad_x.abs() * edge_w_x).mean() + (d_grad_y.abs() * edge_w_y).mean())

        return fused_features, depth_state


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
        objectTokenCount: int = ModuleDim.PstObservedSlots,
        enableRecallAuxiliary: bool = False,
        recallKwargs: Optional[Dict[str, Any]] = None):
        super().__init__()

        assert embedDim % numHeads == 0, "embed_dim must be divisible by num_heads"

        self.img_size = imgSize
        self.patch_size = patchSize
        self.embed_dim = int(embedDim)
        self.integrated_dim = int(embedDim * 2)
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
            fmap = self.cnn_extractor(dummy)["Deep"]
            Hf, Wf = fmap.shape[-2], fmap.shape[-1]

            for m, v in zip(mods, old):
                m.use_hebbian = v

        cnn_feat_dim = baseChannels * 16

        self.depth_fusion = DepthGeometryFusion(
            featureChannels=cnn_feat_dim,
            midChannels=baseChannels * 4,
            shallowChannels=baseChannels * 2,
            fineChannels=baseChannels)
        self.depth_attention_strength = nn.Parameter(torch.tensor(-4.0))

        default_intrinsics = torch.eye(3)
        default_intrinsics[0, 0] = float(imgSize)
        default_intrinsics[1, 1] = float(imgSize)
        default_intrinsics[0, 2] = float(imgSize) * 0.5
        default_intrinsics[1, 2] = float(imgSize) * 0.5
        self.register_buffer("camera_intrinsics", default_intrinsics, persistent=False)

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

        self.cortical_proj = nn.Sequential(
            nn.LayerNorm(self.integrated_dim),
            nn.Linear(self.integrated_dim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.ventral_proj = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.magno_proj = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.geometry_summary_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.dorsal_proj = nn.Sequential(
            nn.LayerNorm(embedDim * 3),
            nn.Linear(embedDim * 3, embedDim),
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

        self.precision_head = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, 5),
            nn.Softplus())

        self.pred_error_input_dim = self.integrated_dim * 3 + embedDim * 2
        self.pred_error_proj = nn.Sequential(
            nn.LayerNorm(self.pred_error_input_dim),
            nn.Linear(self.pred_error_input_dim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.error_to_state = nn.Linear(self.integrated_dim, self.integrated_dim)
        self.correction_gain = nn.Parameter(torch.tensor(0.0))

        self.object_queries = nn.Parameter(torch.randn(self.object_token_count, embedDim) * 0.02)
        self.object_key = nn.Linear(embedDim, embedDim)
        self.object_value = nn.Linear(embedDim, embedDim)
        self.object_post = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim))

        self.object_geometry_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, embedDim))

        self.temporal_state = nn.GRUCell(self.integrated_dim, self.integrated_dim)
        self.temporal_norm = nn.LayerNorm(self.integrated_dim)
        self.topdown_gate = nn.Sequential(
            nn.Linear(self.integrated_dim * 3, self.integrated_dim),
            nn.SiLU(),
            nn.Linear(self.integrated_dim, self.integrated_dim),
            nn.Sigmoid())

        self.integrated_fusion = nn.Sequential(
            nn.LayerNorm(self.integrated_dim + embedDim * 5),
            nn.Linear(self.integrated_dim + embedDim * 5, self.integrated_dim),
            nn.GELU(),
            nn.Linear(self.integrated_dim, self.integrated_dim),
            nn.LayerNorm(self.integrated_dim))

        self.motion_decoder = nn.Linear(embedDim, embedDim)

        recall_kwargs = {} if recallKwargs is None else dict(recallKwargs)
        recall_kwargs.setdefault("embedDim", self.embed_dim)
        recall_kwargs.setdefault("integratedDim", self.integrated_dim)
        recall_kwargs["enableAuxiliary"] = bool(enableRecallAuxiliary)
        self.recall_heads = PerceptionRecallHeads(**recall_kwargs)

        self.InitWeights()
        nn.init.zeros_(self.object_geometry_proj[-1].weight)
        nn.init.zeros_(self.object_geometry_proj[-1].bias)
        nn.init.zeros_(self.depth_fusion.geometry_encoder[-1].weight)
        nn.init.zeros_(self.depth_fusion.geometry_encoder[-1].bias)
        nn.init.zeros_(self.depth_fusion.virtual_head[-1].weight)
        nn.init.constant_(self.depth_fusion.virtual_head[-1].bias, -5.0)
        nn.init.zeros_(self.depth_fusion.sensor_var_head[-1].weight)
        nn.init.zeros_(self.depth_fusion.sensor_var_head[-1].bias)
        nn.init.zeros_(self.error_to_state.weight)
        nn.init.zeros_(self.error_to_state.bias)
        nn.init.zeros_(self.precision_head[1].weight)
        nn.init.constant_(self.precision_head[1].bias, 0.5413)

    def QualityStats(self, x: torch.Tensor) -> torch.Tensor:
        x_det = x.detach()
        mean = x_det.mean(dim=(1, 2, 3))
        std = x_det.std(dim=(1, 2, 3), unbiased=False)
        gx = (x_det[..., :, 1:] - x_det[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        gy = (x_det[..., 1:, :] - x_det[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        grad = 0.5 * (gx + gy)
        clipped = ((x_det <= 0.01) | (x_det >= 0.99)).float().mean(dim=(1, 2, 3))
        return torch.stack([mean, std, grad, clipped], dim=-1)

    def BuildDepthAttentionBias(
        self,
        depthState: Dict[str, torch.Tensor],
        patchHeight: int,
        patchWidth: int,
        cameraIntrinsics: torch.Tensor,
        frameSize: Tuple[int, int]) -> torch.Tensor:
        metric_depth = F.interpolate(
            depthState["MetricDepth"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)
        log_var = F.interpolate(
            depthState["MetricDepthLogVariance"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False).flatten(2).transpose(1, 2)
        reliability = torch.sigmoid(-log_var)
        pair_reliability = torch.sqrt(
            reliability * reliability.transpose(1, 2)).clamp(0.0, 1.0)
        points = self.depth_fusion.BackprojectDepth(
            metric_depth,
            cameraIntrinsics,
            frameSize).flatten(2).transpose(1, 2)
        separation = torch.cdist(points, points, p=2)
        strength = F.softplus(self.depth_attention_strength)
        patch_bias = -(strength * pair_reliability * separation)
        return F.pad(patch_bias, (1, 0, 1, 0), value=0.0)

    @staticmethod
    def MaskedMean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        safe_value = torch.where(weight > 0.0, value, torch.zeros_like(value))
        return (safe_value * weight).sum() / weight.sum().clamp_min(1.0)

    def ComputeDepthGeometryLoss(
        self,
        visualState: VisualState,
        depthTarget: torch.Tensor,
        depthTargetValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,
        prevVisualValid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        target_inverse, target_weight = self.depth_fusion.ResampleSensorDepth(
            depthTarget,
            depthTargetValid,
            tuple(visualState.Auxiliary["MonocularDepth"].shape[-2:]),)
        target_depth = target_inverse.clamp_min(1.0 / self.depth_fusion.max_depth_meters).reciprocal()
        valid = (target_weight > 1e-6).to(target_depth.dtype)
        mono_depth = visualState.Auxiliary["MonocularDepth"]
        residual = mono_depth.clamp_min(1e-6).log() - target_depth.clamp_min(1e-6).log()
        loss_mono = self.MaskedMean(F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"), valid)

        mono_log_var = visualState.Auxiliary["MonocularDepthLogVariance"]
        nll = 0.5 * torch.exp(-mono_log_var) * residual.square() + 0.5 * mono_log_var
        loss_uncertainty = self.MaskedMean(nll, valid)

        pred_gx, pred_gy = self.depth_fusion.SpatialGradient(mono_depth.clamp_min(1e-6).log())
        tgt_gx, tgt_gy = self.depth_fusion.SpatialGradient(target_depth.clamp_min(1e-6).log())
        valid_gx, valid_gy = self.depth_fusion.SpatialGradient(valid)
        valid_gx = (valid_gx.abs() < 0.5).to(valid.dtype) * valid
        valid_gy = (valid_gy.abs() < 0.5).to(valid.dtype) * valid
        loss_gradient = (
            self.MaskedMean((pred_gx - tgt_gx).abs(), valid_gx)
            + self.MaskedMean((pred_gy - tgt_gy).abs(), valid_gy))

        losses = {
            "loss_depth_mono": loss_mono,
            "loss_depth_uncertainty": loss_uncertainty,
            "loss_depth_gradient": loss_gradient,}
        total = loss_mono + 0.05 * loss_uncertainty + 0.25 * loss_gradient

        fused_residual = (
            visualState.Auxiliary["MetricDepth"].clamp_min(1e-6).log()
            - target_depth.clamp_min(1e-6).log())
        loss_fused = self.MaskedMean(
            F.smooth_l1_loss(fused_residual, torch.zeros_like(fused_residual), reduction="none"),
            valid)
        losses["loss_depth_fused"] = loss_fused
        total = total + 0.25 * loss_fused

        edge_smoothness = visualState.Auxiliary["EdgeAwareSmoothness"]
        losses["loss_depth_smoothness"] = edge_smoothness
        total = total + 0.05 * edge_smoothness

        virtual_logits = visualState.Auxiliary["VirtualMaskLogits"]
        virtual_target = visualState.Auxiliary["VirtualTarget"]
        bce_weight = visualState.Auxiliary["SensorDepthUsed"]
        bce_raw = F.binary_cross_entropy_with_logits(virtual_logits, virtual_target, reduction="none")
        loss_virtual = (bce_raw * bce_weight).sum() / bce_weight.sum().clamp_min(1.0)
        sparsity = torch.sigmoid(virtual_logits).mean()
        losses["loss_depth_virtual"] = loss_virtual
        losses["loss_depth_virtual_sparsity"] = sparsity
        total = total + 0.1 * loss_virtual + 0.005 * sparsity

        if prevVisualState is not None and cameraMotion is not None:
            prev_depth = prevVisualState.Auxiliary["MetricDepth"].detach()
            cur_depth = visualState.Auxiliary["MetricDepth"]
            camera_intrinsics = self.CameraIntrinsicsBatch(cur_depth.size(0))
            expected_prev, sampled_prev, warp_valid = self.depth_fusion.WarpPrevDepth(
                cur_depth, prev_depth, camera_intrinsics,
                (self.img_size, self.img_size), cameraMotion)
            if prevVisualValid is not None:
                temporal_valid = prevVisualValid.to(
                    device=warp_valid.device,
                    dtype=torch.bool).view(-1, 1, 1, 1)
                warp_valid = torch.where(
                    temporal_valid, warp_valid, torch.zeros_like(warp_valid))
            residual = (expected_prev.clamp_min(1e-6).log() - sampled_prev.clamp_min(1e-6).log()).abs()
            cur_gx, cur_gy = self.depth_fusion.SpatialGradient(cur_depth.clamp_min(1e-6).log())
            reliability = (-(cur_gx.abs() + cur_gy.abs()) * 5.0).exp()
            temporal = self.MaskedMean(residual, warp_valid * reliability)
            losses["loss_depth_temporal"] = temporal
            total = total + 0.02 * temporal

        losses["loss"] = total
        return losses

    def BuildPatchGeometry(
        self,
        depthState: Dict[str, torch.Tensor],
        patchHeight: int,
        patchWidth: int,
        cameraIntrinsics: torch.Tensor,
        frameSize: Tuple[int, int],) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = F.interpolate(
            depthState["MetricDepth"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)
        confidence = torch.sigmoid(-F.interpolate(
            depthState["MetricDepthLogVariance"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False))
        sensor_reliability = F.interpolate(
            depthState["SensorDepthReliability"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)
        virtual_mask = F.interpolate(
            depthState["VirtualMask"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)
        xyz = self.depth_fusion.BackprojectDepth(
            depth,
            cameraIntrinsics,
            frameSize)
        coordinate_valid = confidence
        evidence = torch.cat([xyz, confidence, sensor_reliability, virtual_mask], dim=1)
        evidence = rearrange(evidence, "b c h w -> b (h w) c")
        return evidence, rearrange(coordinate_valid, "b c h w -> b (h w) c")

    def BuildObjectTokens(
        self,
        patchTokens: torch.Tensor,
        patchGeometry: torch.Tensor,
        patchCoordinateValid: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, D = patchTokens.shape
        k = self.object_key(patchTokens)
        v = self.object_value(patchTokens)
        q = self.object_queries
        scores = torch.einsum("kd,bnd->bkn", q, k) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        tokens = torch.einsum("bkn,bnd->bkd", weights, v)
        object_geometry = torch.einsum("bkn,bnd->bkd", weights, patchGeometry)
        tokens = tokens + self.object_geometry_proj(object_geometry)
        object_valid = torch.einsum("bkn,bnd->bkd", weights, patchCoordinateValid)
        return self.object_post(tokens), object_geometry, object_valid, weights

    def BuildMotionSummary(
        self,
        patchMotion: torch.Tensor,
        patchWeights: torch.Tensor,
        patchReliability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = int(patchMotion.size(-1))
        motion_tokens = self.magno_proj(F.layer_norm(patchMotion, (D,)))
        motion_weight = patchWeights * patchReliability.squeeze(-1).detach()
        motion_weight = motion_weight / motion_weight.sum(dim=1, keepdim=True)
        motion_summary = (motion_tokens * motion_weight.unsqueeze(-1)).sum(dim=1)
        return motion_summary, motion_tokens, motion_weight

    def WarpPrevPatchTokens(
        self,
        prevVisualState: VisualState,
        depthState: Dict[str, torch.Tensor],
        patchHeight: int,
        patchWidth: int,
        cameraIntrinsics: torch.Tensor,
        frameSize: Tuple[int, int],
        cameraMotion: torch.Tensor,
        currentPatchTokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = currentPatchTokens.size(0)
        cur_depth = F.interpolate(
            depthState["MetricDepth"],
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)
        prev_depth = F.interpolate(
            prevVisualState.Auxiliary["MetricDepth"].detach(),
            size=(patchHeight, patchWidth),
            mode="bilinear",
            align_corners=False)

        sx = float(patchWidth) / float(frameSize[1])
        sy = float(patchHeight) / float(frameSize[0])
        fx = (cameraIntrinsics[:, 0, 0] * sx).view(B, 1, 1, 1)
        fy = (cameraIntrinsics[:, 1, 1] * sy).view(B, 1, 1, 1)
        cx = (cameraIntrinsics[:, 0, 2] * sx).view(B, 1, 1, 1)
        cy = (cameraIntrinsics[:, 1, 2] * sy).view(B, 1, 1, 1)

        yy, xx = torch.meshgrid(
            torch.arange(patchHeight, device=cur_depth.device, dtype=cur_depth.dtype),
            torch.arange(patchWidth, device=cur_depth.device, dtype=cur_depth.dtype),
            indexing="ij",)
        xx = xx.view(1, 1, patchHeight, patchWidth)
        yy = yy.view(1, 1, patchHeight, patchWidth)
        point_cur = torch.cat([
            (xx - cx) * cur_depth / fx,
            (yy - cy) * cur_depth / fy,
            cur_depth], dim=1)

        motion = cameraMotion.detach()
        rotation = motion[:, 3:7]
        point_prev = self.depth_fusion.QuaternionRotate(rotation, point_cur) + motion[:, :3].view(B, 3, 1, 1)
        expected_prev = point_prev[:, 2:3]
        inv_z = expected_prev.clamp_min(1e-3).reciprocal()
        grid_x = 2.0 * (fx * point_prev[:, 0:1] * inv_z + cx) / float(patchWidth - 1) - 1.0
        grid_y = 2.0 * (fy * point_prev[:, 1:2] * inv_z + cy) / float(patchHeight - 1) - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)

        prev_tokens = rearrange(prevVisualState.PatchTokens.detach(), "b (h w) d -> b d h w", h=patchHeight, w=patchWidth)
        warped_tokens = F.grid_sample(
            prev_tokens,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True)
        sampled_prev_depth = F.grid_sample(
            prev_depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True)
        in_bounds = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
        depth_residual = (
            expected_prev.clamp_min(1e-6).log()
            - sampled_prev_depth.clamp_min(1e-6).log()).abs()
        valid = (
            in_bounds
            & (expected_prev > 1e-3)
            & (sampled_prev_depth > 1e-3))
        valid = valid * (-depth_residual * 3.0).exp()
        return (
            rearrange(warped_tokens, "b d h w -> b (h w) d"),
            rearrange(valid, "b c h w -> b (h w) c"),
            rearrange(depth_residual, "b c h w -> b (h w) c"),)

    def ObjectAttentionError(self, currentObjects: torch.Tensor, predictedObjects: torch.Tensor) -> torch.Tensor:
        D = int(currentObjects.size(-1))
        scores = torch.matmul(currentObjects, predictedObjects.transpose(1, 2)) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        aligned_pred = torch.matmul(weights, predictedObjects)
        return (currentObjects - aligned_pred).mean(dim=1)

    def BuildStructuredPredictionError(
        self,
        integratedState: torch.Tensor,
        globalFeat: torch.Tensor,
        motionToken: torch.Tensor,
        objectTokens: torch.Tensor,
        predicted: Optional[Dict[str, torch.Tensor]],
        precisionStreams: torch.Tensor,) -> torch.Tensor:
        if predicted is None:
            integrated_err = torch.zeros_like(integratedState)
            global_err = torch.zeros_like(globalFeat)
            object_err = torch.zeros_like(motionToken)
            motion_err = torch.zeros_like(motionToken)
            basis_err = torch.zeros_like(globalFeat)
        else:
            integrated_err = integratedState - predicted["IntegratedFeat"].detach()
            global_err = globalFeat - predicted["GlobalFeat"].detach()
            object_err = self.ObjectAttentionError(objectTokens, predicted["ObjectTokens"].detach())
            motion_err = motionToken - predicted["MotionPred"].detach()
            basis_err = globalFeat - predicted["PredErrorBasis"].detach()

        integrated_err = integrated_err * precisionStreams[:, 0:1]
        global_err = global_err * precisionStreams[:, 1:2]
        object_err = object_err * precisionStreams[:, 2:3]
        motion_err = motion_err * precisionStreams[:, 3:4]
        basis_err = basis_err * precisionStreams[:, 4:5]

        return torch.cat([integrated_err, global_err, object_err, motion_err, basis_err], dim=-1)

    @torch.no_grad()
    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        k = torch.as_tensor(intrinsics)
        if k.dim() == 3:
            k = k[0]
        if sourceSize is not None:
            sx = float(self.img_size) / float(sourceSize[1])
            sy = float(self.img_size) / float(sourceSize[0])
            k = k.clone()
            k[0, 0] *= sx
            k[1, 1] *= sy
            k[0, 2] *= sx
            k[1, 2] *= sy
        self.camera_intrinsics.copy_(k)

    def CameraIntrinsicsBatch(self, batchSize: int) -> torch.Tensor:
        return self.camera_intrinsics.unsqueeze(0).expand(int(batchSize), -1, -1)

    def AssembleVisualState(
        self,
        frame: torch.Tensor,
        tokens: torch.Tensor,
        depthState: Dict[str, torch.Tensor],
        cameraIntrinsics: torch.Tensor,
        patchHeight: int,
        patchWidth: int,
        topDownContext: TopDownContext,
        prevVisualState: Optional[VisualState],
        cameraMotion: Optional[torch.Tensor],
        prevVisualValid: Optional[torch.Tensor] = None,) -> VisualState:
        x = self.encoder_norm(tokens)
        cls_rep = x[:, 0, :]
        mlp_out = self.mlp(cls_rep)
        gate = self.adaptive_gate(mlp_out)
        out = gate * mlp_out + (1 - gate) * cls_rep
        out = self.output_norm(out)

        patch_tokens = x[:, 1:, :]
        patch_weights = F.softmax(self.patch_aggregator(patch_tokens).squeeze(-1), dim=1)
        global_patch = (patch_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)
        preliminary_integrated = torch.cat([out, global_patch], dim=1)

        precision_streams = topDownContext.Precision.view(-1, 1) * self.precision_head(out)

        predicted = topDownContext.PredictedVisual
        if predicted is not None:
            precision_streams = precision_streams * predicted["PriorConfidence"].detach().view(-1, 1)
            integrated_err = (preliminary_integrated - predicted["IntegratedFeat"].detach()) * precision_streams[:, 0:1]
            corrected_integrated = preliminary_integrated - torch.sigmoid(self.correction_gain) * self.error_to_state(integrated_err)
        else:
            corrected_integrated = preliminary_integrated

        patch_geometry, patch_coordinate_valid = self.BuildPatchGeometry(
            depthState, patchHeight, patchWidth, cameraIntrinsics=cameraIntrinsics, frameSize=tuple(frame.shape[-2:]))
        object_tokens, object_geometry, object_coordinate_valid, object_patch_weights = self.BuildObjectTokens(
            patch_tokens, patchGeometry=patch_geometry, patchCoordinateValid=patch_coordinate_valid)

        geometry_reliability = patch_coordinate_valid.detach()
        geometry_weight = patch_weights * geometry_reliability.squeeze(-1)
        geometry_weight = geometry_weight / geometry_weight.sum(dim=1, keepdim=True)
        geometry_summary = self.geometry_summary_proj((patch_geometry * geometry_weight.unsqueeze(-1)).sum(dim=1))
        shared = self.cortical_proj(corrected_integrated)
        ventral_feat = self.ventral_proj(shared)

        if prevVisualState is not None:
            previous_valid = (
                torch.ones(patch_tokens.size(0), device=patch_tokens.device, dtype=torch.bool)
                if prevVisualValid is None
                else prevVisualValid.to(device=patch_tokens.device, dtype=torch.bool).view(-1))
            previous_mask = previous_valid.view(-1, 1, 1)
            if cameraMotion is not None:
                camera_motion_from_prev = cameraMotion
                warped_prev_tokens, warp_valid, warp_depth_residual = self.WarpPrevPatchTokens(
                    prevVisualState,
                    depthState,
                    patchHeight,
                    patchWidth,
                    cameraIntrinsics,
                    tuple(frame.shape[-2:]),
                    camera_motion_from_prev,
                    patch_tokens)
            else:
                camera_motion_from_prev = patch_tokens.new_zeros(patch_tokens.size(0), ModuleDim.PstPoseDim)
                camera_motion_from_prev[:, 6] = 1.0
                warped_prev_tokens = prevVisualState.PatchTokens.detach()
                warp_valid = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
                warp_depth_residual = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
            warped_prev_tokens = torch.where(
                previous_mask,
                warped_prev_tokens,
                patch_tokens.detach())
            warp_valid = torch.where(
                previous_mask, warp_valid, torch.zeros_like(warp_valid))
            warp_depth_residual = torch.where(
                previous_mask,
                warp_depth_residual,
                torch.zeros_like(warp_depth_residual))
            identity_motion = camera_motion_from_prev.new_zeros(camera_motion_from_prev.shape)
            identity_motion[:, 6] = 1.0
            camera_motion_from_prev = torch.where(
                previous_valid.view(-1, 1),
                camera_motion_from_prev,
                identity_motion)
            patch_motion = patch_tokens - warped_prev_tokens
        else:
            camera_motion_from_prev = patch_tokens.new_zeros(patch_tokens.size(0), ModuleDim.PstPoseDim)
            camera_motion_from_prev[:, 6] = 1.0
            warped_prev_tokens = torch.zeros_like(patch_tokens)
            warp_valid = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
            warp_depth_residual = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
            patch_motion = torch.zeros_like(patch_tokens)

        magno_summary, patch_motion_tokens, motion_weights = self.BuildMotionSummary(
            patch_motion,
            patch_weights,
            geometry_reliability)
        object_motion = torch.einsum("bkn,bnd->bkd", object_patch_weights, patch_motion_tokens)
        dorsal_candidate = self.dorsal_proj(torch.cat([shared, geometry_summary, magno_summary], dim=-1))
        geometry_confidence = (patch_weights * geometry_reliability.squeeze(-1)).sum(dim=1, keepdim=True)
        dorsal_feat = geometry_confidence * dorsal_candidate + (1.0 - geometry_confidence) * shared
        motion_token = self.motion_proj(torch.cat([magno_summary, dorsal_feat], dim=-1))

        if prevVisualState is not None:
            h_prev = torch.where(
                previous_valid.view(-1, 1),
                prevVisualState.Auxiliary["TemporalState"],
                torch.zeros_like(prevVisualState.Auxiliary["TemporalState"]))
        else:
            h_prev = corrected_integrated.new_zeros(corrected_integrated.shape)
        h_next = self.temporal_state(corrected_integrated, h_prev.detach())
        temporal_feat = self.temporal_norm(h_next)
        td_gate = self.topdown_gate(torch.cat([corrected_integrated, temporal_feat, topDownContext.MemoryCue], dim=-1))
        global_feat = td_gate * corrected_integrated + (1.0 - td_gate) * temporal_feat

        quality_token = self.quality_proj(self.QualityStats(frame))

        pred_error_target = self.BuildStructuredPredictionError(
            corrected_integrated, global_feat, motion_token, object_tokens, predicted, precision_streams)
        pred_error_token = self.pred_error_proj(pred_error_target)

        integrated_feat = self.integrated_fusion(torch.cat([
            global_feat, ventral_feat, dorsal_feat, motion_token, quality_token, pred_error_token], dim=-1))

        semantic_nodes = {
            **self.recall_heads.ForwardNodes(object_tokens),
            **self.recall_heads.ForwardScene(integrated_feat, ventral_feat, dorsal_feat)}
        return VisualState(
            IntegratedFeat=integrated_feat,
            GlobalFeat=global_feat,
            VentralFeat=ventral_feat,
            DorsalFeat=dorsal_feat,
            MotionToken=motion_token,
            QualityToken=quality_token,
            PredErrorToken=pred_error_token,
            ObjectTokens=object_tokens,
            PatchTokens=patch_tokens,
            SemanticNodes=semantic_nodes,
            Auxiliary={
                "TemporalState": h_next.detach(),
                "PredErrorTarget": pred_error_target.detach(),
                **depthState,
                "ObjectMotion": object_motion,
                "ObjectGeometry": object_geometry,
                "ObjectGeometryValid": object_coordinate_valid,
                "PatchMotionTokens": patch_motion_tokens.detach(),
                "PatchMotionReliability": geometry_reliability.detach(),
                "PatchMotionWeights": motion_weights.detach(),
                "PatchMotionDepthResidual": warp_depth_residual.detach(),
                "WarpedPrevPatchTokens": warped_prev_tokens.detach(),
                "WarpPrevPatchValid": warp_valid.detach(),
                "CameraMotionFromPrev": camera_motion_from_prev.detach(),
                "DorsalReliabilityGate": geometry_confidence.detach()},)

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,
        prevVisualValid: Optional[torch.Tensor] = None,) -> VisualState:
        # x: [B, 3, H, W]
        frame = x
        pyramid = self.cnn_extractor(frame)
        feat, depth_state = self.depth_fusion(
            pyramid["Deep"],
            pyramid["Layer3"],
            pyramid["Layer2"],
            pyramid["Layer1"],
            depth=depth,
            depthValid=depthValid)

        feat = self.cnn_feat_adapter(feat)

        patches = self.patch_adapter(feat)  # [B, embed_dim, Ph, Pw]
        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, embed_dim]
        x = self.pos_drop(x)
        camera_intrinsics = self.CameraIntrinsicsBatch(B)
        depth_attention_bias = self.BuildDepthAttentionBias(
            depth_state,
            Ph,
            Pw,
            cameraIntrinsics=camera_intrinsics,
            frameSize=tuple(frame.shape[-2:]))

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, srcMask=depth_attention_bias)
            x = self.token_adapters[i](x)

        return self.AssembleVisualState(
            frame, x, depth_state, camera_intrinsics, Ph, Pw, topDownContext,
            prevVisualState, cameraMotion, prevVisualValid)

    def ComputePerceptionLoss(
        self,
        visualState: VisualState,
        depthTarget: torch.Tensor,
        depthTargetValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,
        prevVisualValid: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        loss = visualState.IntegratedFeat.new_zeros(())

        obj = visualState.ObjectTokens
        obj_n = F.normalize(obj, dim=-1, eps=1e-6)
        sim = torch.matmul(obj_n, obj_n.transpose(1, 2))
        eye = torch.eye(obj.size(1), device=obj.device, dtype=torch.bool).unsqueeze(0)
        diversity = sim.masked_select(~eye).pow(2).mean()
        loss = loss + 0.05 * diversity

        if prevVisualState is not None:
            motion_target = (visualState.VentralFeat - prevVisualState.VentralFeat.detach()).detach()
            motion_pred = self.motion_decoder(visualState.MotionToken)
            motion_loss = F.smooth_l1_loss(
                motion_pred, motion_target, reduction="none").flatten(1).mean(dim=1)
            motion_valid = (
                torch.ones_like(motion_loss)
                if prevVisualValid is None
                else prevVisualValid.to(device=motion_loss.device, dtype=motion_loss.dtype).view(-1))
            loss = loss + 0.05 * (
                motion_loss * motion_valid).sum() / motion_valid.sum().clamp_min(1.0)

        depth_losses = self.ComputeDepthGeometryLoss(
            visualState,
            depthTarget=depthTarget,
            depthTargetValid=depthTargetValid,
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion,
            prevVisualValid=prevVisualValid)
        loss = loss + depth_losses["loss"]

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

    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        self.base.SetCameraIntrinsics(intrinsics, sourceSize=sourceSize)

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,
        prevVisualValid: Optional[torch.Tensor] = None,) -> VisualState:
        return super().forward(
            x,
            topDownContext=topDownContext,
            prevVisualState=prevVisualState,
            prevVisualValid=prevVisualValid,
            depth=depth,
            depthValid=depthValid,
            cameraMotion=cameraMotion)

    def ComputePerceptionLoss(
        self,
        visualState: VisualState,
        depthTarget: torch.Tensor,
        depthTargetValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,
        prevVisualValid: Optional[torch.Tensor] = None,) -> torch.Tensor:
        return self.base.ComputePerceptionLoss(
            visualState,
            depthTarget=depthTarget,
            depthTargetValid=depthTargetValid,
            prevVisualState=prevVisualState,
            prevVisualValid=prevVisualValid,
            cameraMotion=cameraMotion,)

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
        prevVisualValid = kwargs.get("prevVisualValid", None)
        cameraMotion = kwargs.get("cameraMotion", None)
        depth = kwargs["depth"]
        depth_valid = kwargs["depthValid"]
        camera_intrinsics = self.base.CameraIntrinsicsBatch(int(frame.size(0)))

        pyramid = self.base.cnn_extractor(frame)
        feat, depth_state = self.base.depth_fusion(
            pyramid["Deep"],
            pyramid["Layer3"],
            pyramid["Layer2"],
            pyramid["Layer1"],
            depth=depth,
            depthValid=depth_valid)
        
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
        depth_attention_bias = self.base.BuildDepthAttentionBias(
            depth_state,
            Ph,
            Pw,
            cameraIntrinsics=camera_intrinsics,
            frameSize=tuple(frame.shape[-2:]))

        for i, layer in enumerate(self.base.transformer_layers):
            xTok = layer(xTok, srcMask=depth_attention_bias)
            
            xTok = self.base.token_adapters[i](xTok)
            
            deltaTok2D = deltasPerLayer[i].get("token", None)
            if deltaTok2D is not None:
                xTok = xTok + (xTok @ deltaTok2D.t())

        return self.base.AssembleVisualState(
            frame, xTok, depth_state, camera_intrinsics, Ph, Pw, topDownContext,
            prevVisualState, cameraMotion, prevVisualValid)

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
        




class PerceptionRecallHeads(nn.Module):
    def __init__(
        self,
        embedDim: int = 512,
        integratedDim: int = 1024,
        numObjectClasses: int = ModuleDim.PstObjectClasses,
        numPartClasses: int = ModuleDim.PstPartClasses,
        numSemanticClasses: int = ModuleDim.PstObjectClasses,
        numSceneClasses: int = ModuleDim.PstSceneClasses,
        numGlobalLabels: int = ModuleDim.PstGlobalLabels,
        numSymbols: int = ModuleDim.PstSymbolClasses,
        identityDim: int = ModuleDim.PstIdentityDim,
        textDim: int = ModuleDim.PstTextDim,
        reconSize: int = 32,
        enableAuxiliary: bool = False,
        hiddenDim: Optional[int] = None,):
        super().__init__()
        self.embed_dim = int(embedDim)
        self.integrated_dim = int(integratedDim)
        self.num_object_classes = int(numObjectClasses)
        self.num_part_classes = int(numPartClasses)
        self.num_semantic_classes = int(numSemanticClasses)
        self.num_scene_classes = int(numSceneClasses)
        self.num_global_labels = int(numGlobalLabels)
        self.num_symbols = int(numSymbols)
        self.identity_dim = int(identityDim)
        self.text_dim = int(textDim)
        self.recon_size = int(reconSize)
        self.enable_auxiliary = bool(enableAuxiliary)
        hidden = int(hiddenDim if hiddenDim is not None else embedDim)

        self.node_trunk = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),)

        self.node_logits = nn.Linear(hidden, 2)
        self.level_logits = nn.Linear(hidden, 3)
        self.object_class_logits = nn.Linear(hidden, self.num_object_classes)
        self.part_class_logits = nn.Linear(hidden, self.num_part_classes)
        self.pose_camera_head = nn.Linear(hidden, ModuleDim.PstPoseDim)
        self.size_3d_head = nn.Linear(hidden, 3)
        self.bbox_2d_head = nn.Linear(hidden, 4)
        self.visible_ratio_head = nn.Linear(hidden, 1)
        self.occlusion_ratio_head = nn.Linear(hidden, 1)
        self.has_text_logits = nn.Linear(hidden, 2)
        self.text_embed_head = nn.Linear(hidden, self.text_dim)
        self.symbol_logits = nn.Linear(hidden, self.num_symbols)
        self.identity_head = nn.Linear(hidden, self.identity_dim)
        self.parent_q = nn.Linear(hidden, hidden)
        self.parent_k = nn.Linear(hidden, hidden)
        self.parent_scale = hidden ** -0.5

        global_in = self.integrated_dim + self.embed_dim * 2
        self.global_trunk = nn.Sequential(
            nn.LayerNorm(global_in),
            nn.Linear(global_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),)
        self.scene_logits = nn.Linear(hidden, self.num_scene_classes)
        self.global_label_logits = nn.Linear(hidden, self.num_global_labels)

        if self.enable_auxiliary:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * self.recon_size * self.recon_size),
                nn.Sigmoid(),)

            self.patch_trunk = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),)
            self.patch_class_logits = nn.Linear(hidden, self.num_semantic_classes)
            self.patch_depth = nn.Linear(hidden, 1)
            self.patch_normal = nn.Linear(hidden, 3)

    def Pose(self, rawPose: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(rawPose[..., 3:7].float(), dim=-1, eps=1e-6).to(rawPose.dtype)
        return torch.cat([rawPose[..., :3], quat], dim=-1)

    def ForwardNodes(self, objectTokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        node_h = self.node_trunk(objectTokens)
        parent_logits = torch.matmul(
            self.parent_q(node_h),
            self.parent_k(node_h).transpose(1, 2)) * self.parent_scale
        eye = torch.eye(parent_logits.size(1), device=parent_logits.device, dtype=torch.bool)
        parent_logits = parent_logits.masked_fill(eye.unsqueeze(0), torch.finfo(parent_logits.dtype).min)
        bbox_raw = torch.sigmoid(self.bbox_2d_head(node_h))
        bbox_min = torch.minimum(bbox_raw[..., :2], bbox_raw[..., 2:4])
        bbox_max = torch.maximum(bbox_raw[..., :2], bbox_raw[..., 2:4])
        return {
            "node_logits": self.node_logits(node_h),
            "level_logits": self.level_logits(node_h),
            "object_class_logits": self.object_class_logits(node_h),
            "part_class_logits": self.part_class_logits(node_h),
            "parent_logits": parent_logits,
            "pose_camera": self.Pose(self.pose_camera_head(node_h)),
            "size_3d": F.softplus(self.size_3d_head(node_h)),
            "bbox_2d": torch.cat([bbox_min, bbox_max], dim=-1),
            "visible_ratio": torch.sigmoid(self.visible_ratio_head(node_h).squeeze(-1)),
            "occlusion_ratio": torch.sigmoid(self.occlusion_ratio_head(node_h).squeeze(-1)),
            "has_text_logits": self.has_text_logits(node_h),
            "text_embed": F.normalize(self.text_embed_head(node_h), dim=-1, eps=1e-6),
            "symbol_logits": self.symbol_logits(node_h),
            "identity_embed": F.normalize(self.identity_head(node_h), dim=-1, eps=1e-6)}

    def ForwardScene(
        self,
        integratedFeat: torch.Tensor,
        ventralFeat: torch.Tensor,
        dorsalFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_h = self.global_trunk(torch.cat([integratedFeat, ventralFeat, dorsalFeat], dim=-1))
        return {
            "scene_logits": self.scene_logits(global_h),
            "global_label_logits": self.global_label_logits(global_h)}

    def forward(self, visualState: VisualState) -> Dict[str, torch.Tensor]:
        assert self.enable_auxiliary
        node_out = visualState.SemanticNodes
        global_in = torch.cat([
            visualState.IntegratedFeat,
            visualState.VentralFeat,
            visualState.DorsalFeat,], dim=-1)
        global_h = self.global_trunk(global_in)
        patch_h = self.patch_trunk(visualState.PatchTokens)
        node_h = self.node_trunk(visualState.ObjectTokens)
        B = visualState.IntegratedFeat.size(0)

        return {
            **node_out,
            "reconstruction": self.reconstruction_head(global_h).view(B, 3, self.recon_size, self.recon_size),
            "patch_class_logits": self.patch_class_logits(patch_h),
            "patch_depth": F.softplus(self.patch_depth(patch_h).squeeze(-1)),
            "patch_normal": F.normalize(self.patch_normal(patch_h), dim=-1, eps=1e-6),
            "node_mask_logits": torch.einsum("bkh,bnh->bkn", node_h, patch_h) / math.sqrt(node_h.size(-1))}


class PerceptionRecallLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        noObjectWeight: float = 0.1,
        identityBankSize: int = 2048,
        identityDim: int = ModuleDim.PstIdentityDim,
        identityTemperature: float = 0.07):
        super().__init__()
        self.weights = {
            "node": 1.0,
            "level": 1.0,
            "object_class": 1.0,
            "part_class": 1.0,
            "parent": 1.0,
            "pose_camera": 3.0,
            "size_3d": 1.0,
            "bbox_2d": 1.0,
            "visibility": 0.5,
            "occlusion": 0.5,
            "has_text": 0.5,
            "text_embed": 0.5,
            "symbol": 0.5,
            "identity": 0.25,
            "node_mask": 1.0,
            "scene": 1.0,
            "global_labels": 1.0,
            "reconstruction": 0.2,
            "patch_semantic": 1.0,
            "patch_depth": 1.0,
            "patch_normal": 1.0}
        if weights is not None:
            self.weights.update(weights)
        self.no_object_weight = float(noObjectWeight)
        self.identity_temperature = float(identityTemperature)
        self.register_buffer("identity_bank_embed", torch.zeros(int(identityBankSize), int(identityDim)))
        self.register_buffer("identity_bank_track", torch.full((int(identityBankSize),), -1, dtype=torch.long))
        self.register_buffer("identity_bank_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("identity_bank_count", torch.zeros((), dtype=torch.long))

    @staticmethod
    def QuaternionAngle(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        q_pred = pred[..., 3:7]
        q_tgt = tgt[..., 3:7]
        dot = (q_pred * q_tgt).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.atan2(torch.sqrt((1.0 - dot * dot).clamp_min(0.0)), dot.clamp_min(1e-6))

    def PoseCost(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        trans = torch.cdist(pred[..., :3], tgt[..., :3], p=1)
        q_pred = pred[..., 3:7]
        q_tgt = tgt[..., 3:7]
        dot = torch.matmul(q_pred, q_tgt.t()).abs().clamp(0.0, 1.0)
        angle = 2.0 * torch.atan2(torch.sqrt((1.0 - dot * dot).clamp_min(0.0)), dot.clamp_min(1e-6))
        return trans + angle

    def PoseLoss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        trans = F.smooth_l1_loss(pred[..., :3], tgt[..., :3])
        return trans + self.QuaternionAngle(pred, tgt).mean()

    @torch.no_grad()
    def ResetIdentityBank(self) -> None:
        self.identity_bank_track.fill_(-1)
        self.identity_bank_ptr.zero_()
        self.identity_bank_count.zero_()

    @torch.no_grad()
    def EnqueueIdentity(self, embedding: torch.Tensor, trackId: torch.Tensor) -> None:
        n = int(embedding.size(0))
        if n == 0:
            return
        size = int(self.identity_bank_embed.size(0))
        if n > size:
            embedding = embedding[-size:]
            trackId = trackId[-size:]
            n = size
        embedding = embedding.to(self.identity_bank_embed.dtype)
        trackId = trackId.to(self.identity_bank_track.dtype)
        ptr = int(self.identity_bank_ptr.item())
        end = ptr + n
        if end <= size:
            self.identity_bank_embed[ptr:end] = embedding
            self.identity_bank_track[ptr:end] = trackId
        else:
            first = size - ptr
            self.identity_bank_embed[ptr:] = embedding[:first]
            self.identity_bank_track[ptr:] = trackId[:first]
            self.identity_bank_embed[:end - size] = embedding[first:]
            self.identity_bank_track[:end - size] = trackId[first:]
        self.identity_bank_ptr.fill_(end % size)
        self.identity_bank_count.fill_(min(size, int(self.identity_bank_count.item()) + n))

    def IdentityContrastive(self, embedding: torch.Tensor, trackId: torch.Tensor) -> torch.Tensor:
        n = int(embedding.size(0))
        trackId = trackId.to(torch.long)
        count = int(self.identity_bank_count.item())
        keys = torch.cat([embedding.detach(), self.identity_bank_embed[:count].to(embedding.dtype)], dim=0)
        key_track = torch.cat([trackId, self.identity_bank_track[:count]], dim=0)
        logits = torch.matmul(embedding, keys.t()) / self.identity_temperature
        self_mask = torch.zeros_like(logits, dtype=torch.bool)
        self_mask[:, :n] = torch.eye(n, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
        positives = trackId.unsqueeze(1).eq(key_track.unsqueeze(0)) & ~self_mask
        valid = positives.any(dim=-1)
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        positive_log_prob = (
            log_prob.masked_fill(~positives, 0.0).sum(dim=-1)
            / positives.sum(dim=-1).clamp_min(1))
        loss = -positive_log_prob[valid].mean() if bool(valid.any()) else embedding.new_zeros(())
        self.EnqueueIdentity(embedding.detach(), trackId)
        return loss

    def ReconstructionTarget(self, targets: Dict[str, torch.Tensor], size: int) -> torch.Tensor:
        return F.interpolate(
            targets["rgb"],
            size=(size, size),
            mode="bilinear",
            align_corners=False).clamp(0.0, 1.0)

    def SemanticTarget(
        self,
        targets: Dict[str, torch.Tensor],
        numPatches: int) -> torch.Tensor:
        tensor = targets["semantic_segmentation"]
        grid = int(math.sqrt(numPatches))
        down = F.interpolate(tensor.unsqueeze(1).float(), size=(grid, grid), mode="nearest")
        return down[:, 0].reshape(tensor.size(0), numPatches).long()

    def DepthTarget(
        self,
        targets: Dict[str, torch.Tensor],
        numPatches: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = targets["depth"]
        valid = targets["depth_valid"].to(tensor.dtype)
        grid = int(math.sqrt(numPatches))
        inverse = torch.where(valid > 0.0, tensor.clamp_min(1e-6).reciprocal(), torch.zeros_like(tensor))
        weight = F.adaptive_avg_pool2d(valid, (grid, grid))
        pooled_inverse = F.adaptive_avg_pool2d(inverse, (grid, grid)) / weight.clamp_min(1e-6)
        target = pooled_inverse.clamp_min(1e-6).reciprocal()
        target_valid = weight > 0.0
        target = target[:, 0].reshape(tensor.size(0), numPatches)
        target_valid = target_valid[:, 0].reshape(tensor.size(0), numPatches)
        return target, target_valid

    def NormalTarget(self, targets: Dict[str, torch.Tensor], numPatches: int) -> torch.Tensor:
        tensor = targets["normal"]
        grid = int(math.sqrt(numPatches))
        normal = F.interpolate(tensor, size=(grid, grid), mode="bilinear", align_corners=False)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        return rearrange(normal, "b c h w -> b (h w) c")

    def NodeMaskTarget(
        self,
        nodeMasks: torch.Tensor,
        gtIndex: torch.Tensor,
        numPatches: int) -> torch.Tensor:
        grid = int(math.sqrt(numPatches))
        masks = nodeMasks[gtIndex]
        return F.adaptive_max_pool2d(masks.unsqueeze(1).float(), (grid, grid)).flatten(1)

    def forward(self, recallOut: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, K, _ = recallOut["node_logits"].shape
        device = recallOut["node_logits"].device
        losses: Dict[str, torch.Tensor] = {}
        total = recallOut["node_logits"].new_zeros(())

        def add(name: str, value: torch.Tensor) -> None:
            nonlocal total
            losses[f"loss_{name}"] = value
            total = total + float(self.weights.get(name, 1.0)) * value

        node_terms: List[torch.Tensor] = []
        level_terms: List[torch.Tensor] = []
        object_class_terms: List[torch.Tensor] = []
        part_class_terms: List[torch.Tensor] = []
        parent_terms: List[torch.Tensor] = []
        pose_terms: List[torch.Tensor] = []
        size_terms: List[torch.Tensor] = []
        bbox_terms: List[torch.Tensor] = []
        visibility_terms: List[torch.Tensor] = []
        occlusion_terms: List[torch.Tensor] = []
        has_text_terms: List[torch.Tensor] = []
        text_terms: List[torch.Tensor] = []
        symbol_terms: List[torch.Tensor] = []
        mask_terms: List[torch.Tensor] = []
        identity_embed_chunks: List[torch.Tensor] = []
        identity_track_chunks: List[torch.Tensor] = []

        for b in range(B):
            gt_idx = torch.nonzero(targets["node_valid"][b], as_tuple=False).flatten()
            if gt_idx.numel() == 0:
                node_target = torch.zeros(K, device=device, dtype=torch.long)
                weight = recallOut["node_logits"].new_tensor([self.no_object_weight, 1.0])
                node_terms.append(F.cross_entropy(recallOut["node_logits"][b], node_target, weight=weight))
                continue

            with torch.no_grad():
                node_prob = F.softmax(recallOut["node_logits"][b], dim=-1)[:, 1]
                target_levels = targets["node_level"][b, gt_idx]
                target_classes = targets["object_classes"][b, gt_idx]
                target_parts = targets["part_classes"][b, gt_idx]
                target_poses = targets["pose_camera"][b, gt_idx]
                cost = -node_prob[:, None]
                cost = cost - F.softmax(recallOut["level_logits"][b], dim=-1)[:, target_levels]
                class_cost = cost.new_zeros(K, gt_idx.numel())
                object_match = target_levels == 0
                part_match = target_levels > 0
                if object_match.any():
                    class_cost[:, object_match] = F.softmax(
                        recallOut["object_class_logits"][b], dim=-1)[:, target_classes[object_match]]
                if part_match.any():
                    class_cost[:, part_match] = F.softmax(
                        recallOut["part_class_logits"][b], dim=-1)[:, target_parts[part_match]]
                cost = cost - class_cost
                cost = cost + 0.25 * self.PoseCost(recallOut["pose_camera"][b], target_poses)
                pred_idx, local_idx = HungarianAssignment(cost)
            matched_gt = gt_idx[local_idx]
            node_target = torch.zeros(K, device=device, dtype=torch.long)
            node_target[pred_idx] = 1
            weight = recallOut["node_logits"].new_tensor([self.no_object_weight, 1.0])
            node_terms.append(F.cross_entropy(recallOut["node_logits"][b], node_target, weight=weight))
            levels = targets["node_level"][b, matched_gt]
            level_terms.append(F.cross_entropy(recallOut["level_logits"][b, pred_idx], levels))
            object_select = levels == 0
            part_select = levels > 0
            if object_select.any():
                object_class_terms.append(F.cross_entropy(
                    recallOut["object_class_logits"][b, pred_idx[object_select]],
                    targets["object_classes"][b, matched_gt[object_select]]))
            if part_select.any():
                part_class_terms.append(F.cross_entropy(
                    recallOut["part_class_logits"][b, pred_idx[part_select]],
                    targets["part_classes"][b, matched_gt[part_select]]))
            pose_terms.append(self.PoseLoss(recallOut["pose_camera"][b, pred_idx], targets["pose_camera"][b, matched_gt]))
            size_terms.append(F.smooth_l1_loss(recallOut["size_3d"][b, pred_idx], targets["size_3d"][b, matched_gt]))
            bbox_terms.append(F.smooth_l1_loss(recallOut["bbox_2d"][b, pred_idx], targets["bbox_2d"][b, matched_gt]))
            visibility_terms.append(F.smooth_l1_loss(recallOut["visible_ratio"][b, pred_idx], targets["visible_ratio"][b, matched_gt]))
            occlusion_terms.append(F.smooth_l1_loss(recallOut["occlusion_ratio"][b, pred_idx], targets["occlusion_ratio"][b, matched_gt]))
            has_text_terms.append(F.cross_entropy(recallOut["has_text_logits"][b, pred_idx], targets["has_text"][b, matched_gt]))
            with_text = targets["has_text"][b, matched_gt].bool()
            if with_text.any():
                target_text = F.normalize(targets["text_embed"][b, matched_gt[with_text]], dim=-1, eps=1e-6)
                text_terms.append((1.0 - (
                    recallOut["text_embed"][b, pred_idx[with_text]] * target_text).sum(dim=-1)).mean())
                symbol_terms.append(F.cross_entropy(
                    recallOut["symbol_logits"][b, pred_idx[with_text]],
                    targets["symbol_type"][b, matched_gt[with_text]]))
            if part_select.any():
                gt_to_pred = torch.full((targets["node_valid"].size(1),), -1, device=device, dtype=torch.long)
                gt_to_pred[matched_gt] = pred_idx
                parent_gt = targets["parent_index"][b, matched_gt[part_select]]
                part_pred = pred_idx[part_select]
                parent_pred = torch.where(parent_gt >= 0, gt_to_pred[parent_gt.clamp_min(0)], parent_gt.new_full((), -1))
                keep = parent_pred >= 0
                if keep.any():
                    parent_terms.append(F.cross_entropy(
                        recallOut["parent_logits"][b, part_pred[keep]],
                        parent_pred[keep]))
            mask_target = self.NodeMaskTarget(
                targets["node_instance_masks"][b],
                matched_gt,
                recallOut["node_mask_logits"].size(-1))
            mask_terms.append(F.binary_cross_entropy_with_logits(
                recallOut["node_mask_logits"][b, pred_idx],
                mask_target))
            identity_embed_chunks.append(recallOut["identity_embed"][b, pred_idx])
            identity_track_chunks.append(targets["track_id"][b, matched_gt])

        if node_terms:
            add("node", torch.stack(node_terms).mean())
        if level_terms:
            add("level", torch.stack(level_terms).mean())
        if object_class_terms:
            add("object_class", torch.stack(object_class_terms).mean())
        if part_class_terms:
            add("part_class", torch.stack(part_class_terms).mean())
        if parent_terms:
            add("parent", torch.stack(parent_terms).mean())
        if pose_terms:
            add("pose_camera", torch.stack(pose_terms).mean())
            add("size_3d", torch.stack(size_terms).mean())
            add("bbox_2d", torch.stack(bbox_terms).mean())
            add("visibility", torch.stack(visibility_terms).mean())
            add("occlusion", torch.stack(occlusion_terms).mean())
            add("has_text", torch.stack(has_text_terms).mean())
            add("node_mask", torch.stack(mask_terms).mean())
        if text_terms:
            add("text_embed", torch.stack(text_terms).mean())
            add("symbol", torch.stack(symbol_terms).mean())
        if identity_embed_chunks:
            add("identity", self.IdentityContrastive(
                torch.cat(identity_embed_chunks, dim=0),
                torch.cat(identity_track_chunks, dim=0)))
        add("scene", F.cross_entropy(recallOut["scene_logits"], targets["scene_class"]))
        add("global_labels", F.binary_cross_entropy_with_logits(
            recallOut["global_label_logits"], targets["global_labels"].to(recallOut["global_label_logits"].dtype)))
        image = self.ReconstructionTarget(targets, recallOut["reconstruction"].size(-1))
        add("reconstruction", F.l1_loss(recallOut["reconstruction"], image))
        patch_sem = self.SemanticTarget(targets, recallOut["patch_class_logits"].size(1))
        logits = recallOut["patch_class_logits"].reshape(-1, recallOut["patch_class_logits"].size(-1))
        add("patch_semantic", F.cross_entropy(logits, patch_sem.reshape(-1)))
        patch_depth, patch_depth_valid = self.DepthTarget(targets, recallOut["patch_depth"].size(1))
        error = F.smooth_l1_loss(recallOut["patch_depth"], patch_depth, reduction="none")
        valid = patch_depth_valid.to(error.dtype)
        add("patch_depth", (error * valid).sum() / valid.sum().clamp_min(1.0))
        patch_normal = self.NormalTarget(targets, recallOut["patch_normal"].size(1))
        normal_error = 1.0 - (recallOut["patch_normal"] * patch_normal).sum(dim=-1).clamp(-1.0, 1.0)
        add("patch_normal", (normal_error * valid).sum() / valid.sum().clamp_min(1.0))
        losses["loss"] = total
        return losses


class PerceptionTrainer(nn.Module):
    def __init__(
        self,
        recallLossKwargs: Optional[Dict[str, Any]] = None,
        **extractorKwargs: Any,):
        super().__init__()
        extractorKwargs = dict(extractorKwargs)
        extractorKwargs["enableRecallAuxiliary"] = True
        self.extractor = PerceiveExtractor(**extractorKwargs)
        self.recall_heads = self.extractor.recall_heads
        recallLossKwargs = {} if recallLossKwargs is None else dict(recallLossKwargs)
        self.recall_loss = PerceptionRecallLoss(**recallLossKwargs)

    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        self.extractor.SetCameraIntrinsics(intrinsics, sourceSize=sourceSize)

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        targets: Dict[str, torch.Tensor],
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None,) -> Dict[str, Any]:
        visual_state = self.extractor(
            x,
            topDownContext=topDownContext,
            prevVisualState=prevVisualState,
            depth=depth,
            depthValid=depthValid,
            cameraMotion=cameraMotion)
        recall_out = self.recall_heads(visual_state)
        loss_self = self.extractor.ComputePerceptionLoss(
            visual_state,
            depthTarget=targets["depth"],
            depthTargetValid=targets["depth_valid"],
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion)
        recall_losses = self.recall_loss(recall_out, targets)
        return {
            "visual_state": visual_state,
            "recall_out": recall_out,
            "loss_self_supervised": loss_self,
            "loss_recall": recall_losses["loss"],
            "loss_total": loss_self + recall_losses["loss"],
            **{name: value for name, value in recall_losses.items() if name != "loss"}}


class TestPerceptionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def MakeTopDownContext(self, model, B: int, dtype: torch.dtype = torch.float32, predictedVisual: Optional[Dict[str, torch.Tensor]] = None) -> TopDownContext:
        runtime = model.base if hasattr(model, "base") else model
        integrated_dim = int(runtime.integrated_dim)
        return TopDownContext(
            PredictedVisual=predictedVisual,
            Precision=torch.ones(B, device=self.device, dtype=dtype),
            MemoryCue=torch.zeros(B, integrated_dim, device=self.device, dtype=dtype),)

    def PerceptionForward(self, model, x: torch.Tensor, prevVisualState: Optional[VisualState] = None, predictedVisual: Optional[Dict[str, torch.Tensor]] = None) -> VisualState:
        B, _, H, W = x.shape
        depth = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        return model(
            x,
            topDownContext=self.MakeTopDownContext(model, int(x.size(0)), x.dtype, predictedVisual),
            depth=depth,
            depthValid=torch.ones_like(depth, dtype=torch.bool),
            prevVisualState=prevVisualState)

    def MakeSyntheticTargets(
        self,
        frames: torch.Tensor,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        normal: torch.Tensor,
        semantic: torch.Tensor,
        nodes: int = 2) -> Dict[str, torch.Tensor]:
        B, _, H, W = frames.shape
        pose = torch.zeros(B, nodes, 7, device=self.device)
        pose[..., 2] = 1.0
        pose[..., 6] = 1.0
        camera_pose_world = torch.zeros(B, 7, device=self.device)
        camera_pose_world[:, 6] = 1.0
        valid = torch.ones(B, nodes, device=self.device, dtype=torch.bool)
        level = torch.zeros(B, nodes, device=self.device, dtype=torch.long)
        parent = torch.full((B, nodes), -1, device=self.device, dtype=torch.long)
        object_class = torch.ones(B, nodes, device=self.device, dtype=torch.long)
        part_class = torch.zeros(B, nodes, device=self.device, dtype=torch.long)
        if nodes > 1:
            level[:, 1:] = 1
            parent[:, 1:] = 0
            object_class[:, 1:] = 0
            part_class[:, 1:] = 1
        masks = torch.zeros(B, nodes, H, W, device=self.device, dtype=torch.bool)
        masks[:, 0, : H // 2, : W // 2] = True
        if nodes > 1:
            masks[:, 1:, H // 2:, W // 2:] = True
        relation = torch.zeros(B, nodes, nodes, device=self.device, dtype=torch.long)
        relation_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        relation_valid = relation_valid & ~torch.eye(nodes, device=self.device, dtype=torch.bool).unsqueeze(0)
        if nodes > 1:
            relation[:, 1:, 0] = 1
        return {
            "rgb": frames,
            "depth": depth,
            "depth_valid": depthValid,
            "normal": normal,
            "semantic_segmentation": semantic,
            "scene_class": torch.ones(B, device=self.device, dtype=torch.long),
            "global_labels": torch.ones(B, ModuleDim.PstGlobalLabels, device=self.device),
            "node_valid": valid,
            "node_level": level,
            "parent_index": parent,
            "object_classes": object_class,
            "part_classes": part_class,
            "track_id": torch.arange(nodes, device=self.device).unsqueeze(0).expand(B, -1),
            "pose_camera": pose,
            "pose_world": pose,
            "size_3d": torch.ones(B, nodes, 3, device=self.device) * 0.1,
            "bbox_2d": torch.ones(B, nodes, 4, device=self.device) * 0.25,
            "node_instance_masks": masks,
            "visible_ratio": torch.ones(B, nodes, device=self.device),
            "occlusion_ratio": torch.zeros(B, nodes, device=self.device),
            "has_text": torch.zeros(B, nodes, device=self.device, dtype=torch.long),
            "text_embed": torch.zeros(B, nodes, ModuleDim.PstTextDim, device=self.device),
            "symbol_type": torch.zeros(B, nodes, device=self.device, dtype=torch.long),
            "node_state": torch.zeros(B, nodes, ModuleDim.PstStateDim, device=self.device),
            "node_state_valid": valid,
            "node_attributes": torch.zeros(B, nodes, ModuleDim.PstAttrDim, device=self.device),
            "node_attributes_valid": level == 0,
            "relation_type": relation,
            "relation_valid": relation_valid,
            "external_relation": torch.zeros(B, nodes, ModuleDim.PstRelationClasses, device=self.device),
            "external_relation_valid": valid,
            "motion": pose,
            "motion_valid": valid,
            "is_moving": torch.zeros(B, nodes, device=self.device),
            "affordance": torch.zeros(B, nodes, ModuleDim.PstAffordanceDim, device=self.device),
            "affordance_valid": level == 0,
            "contact": torch.zeros(B, nodes, device=self.device),
            "contact_valid": valid,
            "contact_force": torch.zeros(B, nodes, 2, device=self.device),
            "contact_point_camera": torch.zeros(B, nodes, 3, device=self.device),
            "camera_pose_world": camera_pose_world}

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
            assert tuple(out.IntegratedFeat.shape) == (2, expected_dim), f"Output shape does not match: {out.IntegratedFeat.shape}"
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
            assert tuple(out.IntegratedFeat.shape) == expected_out_shape, f"Output shape does not match: {out.IntegratedFeat.shape}"

            print(f"PerceiveExtractor forward input shape: {tuple(x.shape)}")
            print(f"PerceiveExtractor forward output shape: {tuple(out.IntegratedFeat.shape)}")
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
                    "IntegratedFeat": torch.randn(B, 1024, device=self.device),
                    "GlobalFeat": torch.randn(B, 1024, device=self.device),
                    "ObjectTokens": torch.randn(B, model.object_token_count, 512, device=self.device),
                    "MotionPred": torch.randn(B, 512, device=self.device),
                    "PriorConfidence": torch.ones(B, device=self.device),
                    "PredErrorBasis": torch.randn(B, 1024, device=self.device),}
                ctx = TopDownContext(
                    PredictedVisual=pred_visual,
                    Precision=torch.ones(B, device=self.device),
                    MemoryCue=torch.randn(B, 1024, device=self.device))
                depth = torch.ones(B, 1, 64, 64, device=self.device)
                intrinsics = torch.eye(3, device=self.device)
                intrinsics[0, 0] = intrinsics[1, 1] = 50.0
                intrinsics[0, 2] = intrinsics[1, 2] = 32.0
                model.SetCameraIntrinsics(intrinsics)
                state1 = model(
                    x,
                    topDownContext=ctx,
                    depth=depth,
                    depthValid=torch.ones_like(depth, dtype=torch.bool),
                    prevVisualState=state0)

            assert tuple(out.IntegratedFeat.shape) == (B, 1024), f"integrated forward shape mismatch: {out.IntegratedFeat.shape}"
            assert tuple(state1.IntegratedFeat.shape) == (B, 1024)
            assert tuple(state1.GlobalFeat.shape) == (B, 1024)
            assert tuple(state1.VentralFeat.shape) == (B, 512)
            assert tuple(state1.DorsalFeat.shape) == (B, 512)
            assert tuple(state1.MotionToken.shape) == (B, 512)
            assert tuple(state1.QualityToken.shape) == (B, 512)
            assert tuple(state1.PredErrorToken.shape) == (B, 512)
            assert tuple(state1.ObjectTokens.shape) == (B, model.object_token_count, 512)
            assert state1.PatchTokens.dim() == 3 and state1.PatchTokens.size(0) == B and state1.PatchTokens.size(-1) == 512
            assert tuple(state1.SemanticNodes["node_logits"].shape) == (B, model.object_token_count, 2)
            assert tuple(state1.SemanticNodes["pose_camera"].shape) == (B, model.object_token_count, 7)
            assert tuple(state1.SemanticNodes["parent_logits"].shape) == (B, model.object_token_count, model.object_token_count)
            bbox = state1.SemanticNodes["bbox_2d"]
            assert bool((bbox[..., :2] <= bbox[..., 2:4]).all().item())
            assert tuple(state1.SemanticNodes["scene_logits"].shape) == (B, ModuleDim.PstSceneClasses)
            assert not model.recall_heads.enable_auxiliary and not hasattr(model.recall_heads, "reconstruction_head")
            assert "TemporalState" in state1.Auxiliary
            assert "PredErrorTarget" in state1.Auxiliary
            model.ResetHebbianMemory()
            assert tuple(state1.Auxiliary["TemporalState"].shape) == (B, 1024)
            assert tuple(state1.Auxiliary["PredErrorTarget"].shape) == (B, model.pred_error_input_dim)
            print("PerceiveExtractor structured state passed.")
            return True
        except AssertionError as e:
            print(f"TestPerceiveExtractorStructuredState failed: {e}")
            return False
        except Exception as e:
            print(f"TestPerceiveExtractorStructuredState error: {e}")
            return False

    def TestRGBDGeometryAndSupervision(self):
        try:
            B = 2
            model = PerceiveExtractor(
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16,
                useHebbian=False).to(self.device)
            model.train()
            frames = torch.rand(B, 3, 64, 64, device=self.device)
            sensor_depth = torch.full((B, 1, 64, 64), 1.5, device=self.device)
            sensor_valid = torch.ones_like(sensor_depth, dtype=torch.bool)
            sensor_valid[..., ::4] = False
            sensor_depth = torch.where(sensor_valid, sensor_depth, torch.zeros_like(sensor_depth))
            target_depth = torch.full_like(sensor_depth, 1.25)
            target_valid = torch.ones_like(sensor_depth, dtype=torch.bool)
            target_normal = torch.zeros(B, 3, 64, 64, device=self.device)
            target_normal[:, 2] = 1.0
            target_semantic = torch.zeros(B, 64, 64, device=self.device, dtype=torch.long)
            intrinsics = torch.eye(3, device=self.device)
            intrinsics[0, 0] = intrinsics[1, 1] = 50.0
            intrinsics[0, 2] = intrinsics[1, 2] = 32.0
            model.SetCameraIntrinsics(intrinsics)
            context = self.MakeTopDownContext(model, B, frames.dtype)

            visual_state = model(
                frames,
                topDownContext=context,
                depth=sensor_depth,
                depthValid=sensor_valid)
            assert tuple(visual_state.Auxiliary["MetricDepth"].shape) == (B, 1, 16, 16)
            assert tuple(visual_state.Auxiliary["ObjectGeometry"].shape) == (B, 16, 6)
            assert tuple(visual_state.Auxiliary["ObjectMotion"].shape) == (B, 16, model.embed_dim)
            assert tuple(visual_state.Auxiliary["VirtualMask"].shape) == (B, 1, 16, 16)
            assert tuple(visual_state.Auxiliary["ContentDepth"].shape) == (B, 1, 16, 16)
            assert tuple(visual_state.Auxiliary["SensorLogVarianceSpatial"].shape) == (B, 1, 16, 16)
            assert visual_state.Auxiliary["EdgeAwareSmoothness"].dim() == 0
            assert float(visual_state.Auxiliary["VirtualMask"].mean().item()) < 0.5
            assert bool((visual_state.Auxiliary["ObjectGeometryValid"] > 0).any().item())

            depth_losses = model.ComputeDepthGeometryLoss(
                visual_state,
                depthTarget=target_depth,
                depthTargetValid=target_valid)
            model.zero_grad(set_to_none=True)
            depth_losses["loss"].backward()
            depth_grad = model.depth_fusion.monocular_head[-1].weight.grad
            assert depth_grad is not None and bool(torch.isfinite(depth_grad).all().item())

            trainer = PerceptionTrainer(
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16,
                useHebbian=False).to(self.device)
            assert trainer.recall_heads.enable_auxiliary and hasattr(trainer.recall_heads, "global_trunk")
            trainer.SetCameraIntrinsics(intrinsics)
            targets = self.MakeSyntheticTargets(
                frames, target_depth, target_valid, target_normal, target_semantic)
            assert "camera_pose_world" in targets and "camera_motion" not in targets
            train_out = trainer(
                frames,
                topDownContext=self.MakeTopDownContext(trainer.extractor, B, frames.dtype),
                depth=sensor_depth,
                depthValid=sensor_valid,
                targets=targets)
            assert bool((train_out["visual_state"].Auxiliary["ObjectGeometryValid"] > 0).any().item())
            assert tuple(train_out["visual_state"].Auxiliary["ObjectMotion"].shape) == (B, 16, trainer.extractor.embed_dim)
            assert train_out["recall_out"]["node_logits"] is train_out["visual_state"].SemanticNodes["node_logits"]
            assert "loss_node" in train_out and "loss_patch_normal" in train_out
            runtime_model = PerceiveExtractor(
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16,
                useHebbian=False).to(self.device)
            load_result = runtime_model.load_state_dict(trainer.extractor.state_dict(), strict=False)
            assert any(key.startswith("recall_heads.reconstruction_head") for key in load_result.unexpected_keys)
            assert not any(key.startswith("recall_heads.global_trunk") for key in load_result.unexpected_keys)
            assert not any(key.startswith("recall_heads.node_trunk") for key in load_result.unexpected_keys)
            assert bool(torch.isfinite(train_out["loss_total"]).item())
            print("RGBD geometry and supervision passed.")
            return True
        except AssertionError as e:
            print(f"TestRGBDGeometryAndSupervision failed: {e}")
            return False
        except Exception as e:
            print(f"TestRGBDGeometryAndSupervision error: {e}")
            return False

    def TestRecallLossDecreases(self):
        try:
            B, K, D, P, C = 1, 4, 16, 4, 4
            heads = PerceptionRecallHeads(
                embedDim=D,
                integratedDim=32,
                numObjectClasses=C,
                reconSize=4,
                enableAuxiliary=True,
                hiddenDim=16).to(self.device)
            loss_fn = PerceptionRecallLoss()
            optimizer = torch.optim.Adam(heads.parameters(), lr=2e-2)
            objects = torch.randn(B, K, D, device=self.device)
            patches = torch.randn(B, P, D, device=self.device)
            integrated = torch.randn(B, 32, device=self.device)
            global_feat = torch.randn(B, 32, device=self.device)
            ventral = torch.randn(B, D, device=self.device)
            dorsal = torch.randn(B, D, device=self.device)
            token = torch.randn(B, D, device=self.device)
            normal = torch.zeros(B, 3, 4, 4, device=self.device)
            normal[:, 2] = 1.0
            target_rgb = torch.rand(B, 3, 4, 4, device=self.device)
            targets = self.MakeSyntheticTargets(
                target_rgb,
                torch.ones(B, 1, 4, 4, device=self.device),
                torch.ones(B, 1, 4, 4, device=self.device, dtype=torch.bool),
                normal,
                torch.ones(B, 4, 4, device=self.device, dtype=torch.long),
                nodes=1)

            def loss_value() -> torch.Tensor:
                state = VisualState(
                    IntegratedFeat=integrated,
                    GlobalFeat=global_feat,
                    VentralFeat=ventral,
                    DorsalFeat=dorsal,
                    MotionToken=token,
                    QualityToken=token,
                    PredErrorToken=token,
                    ObjectTokens=objects,
                    PatchTokens=patches,
                    SemanticNodes={
                        **heads.ForwardNodes(objects),
                        **heads.ForwardScene(integrated, ventral, dorsal)})
                return loss_fn(heads(state), targets)["loss"]

            initial = float(loss_value().detach())
            for _ in range(40):
                loss = loss_value()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            final = float(loss_value().detach())
            assert final < initial, f"recall loss did not decrease: {initial:.4f} -> {final:.4f}"
            print(f"Recall loss decreases passed. {initial:.4f} -> {final:.4f}")
            return True
        except AssertionError as e:
            print(f"TestRecallLossDecreases failed: {e}")
            return False
        except Exception as e:
            print(f"TestRecallLossDecreases error: {e}")
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
            pred = head(out.IntegratedFeat)
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
                pred = head(self.PerceptionForward(model, x).IntegratedFeat)
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
                pred = head(self.PerceptionForward(model, x).IntegratedFeat)
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
                start = F.mse_loss(head(self.PerceptionForward(model, data_x).IntegratedFeat), data_y).item()

            for t in range(1, steps + 1):
                pred = head(self.PerceptionForward(model, data_x).IntegratedFeat)
                loss = F.mse_loss(pred, data_y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if (t % logEvery) == 0 or t == 1:
                    print(f"[PerceptionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(self.PerceptionForward(model, data_x).IntegratedFeat), data_y).item()

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

            max_abs = (y_base.IntegratedFeat - y_wrap.IntegratedFeat).abs().max().item()
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
                pred = head(self.PerceptionForward(wrapper, x).IntegratedFeat)
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
            assert torch.allclose(y0.IntegratedFeat, y1.IntegratedFeat, atol=1e-6, rtol=1e-4), "base vs wrapper mismatch after commit."

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
            pred = head(out.IntegratedFeat)
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
                pred = head(self.PerceptionForward(wrapper, data_x).IntegratedFeat)
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
            max_abs = (y0.IntegratedFeat - y1.IntegratedFeat).abs().max().item()
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
            pred = head(self.PerceptionForward(model, x).IntegratedFeat)
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
                    start = F.mse_loss(head(self.PerceptionForward(model, data_x).IntegratedFeat), data_y).item()

                hist = []
                for _ in range(steps):
                    pred = head(self.PerceptionForward(model, data_x).IntegratedFeat)
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
            pred = head(self.PerceptionForward(model, x).IntegratedFeat)
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

    def PartialPreviousVisualMask(self):
        try:
            B, H = 2, 32
            model = PerceiveExtractor(
                imgSize=H,
                patchSize=1,
                embedDim=32,
                numHeads=4,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=4,
                useHebbian=False).to(self.device).eval()
            model.SetCameraIntrinsics(torch.tensor(
                [[24.0, 0.0, 15.5], [0.0, 24.0, 15.5], [0.0, 0.0, 1.0]],
                device=self.device))
            depth = torch.ones(B, 1, H, H, device=self.device)
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
            top_down = self.MakeTopDownContext(model, B)
            previous = model(
                torch.rand(B, 3, H, H, device=self.device),
                topDownContext=top_down,
                depth=depth,
                depthValid=depth_valid)
            camera_motion = torch.zeros(B, 7, device=self.device)
            camera_motion[:, 6] = 1.0
            current = model(
                torch.rand(B, 3, H, H, device=self.device),
                topDownContext=top_down,
                depth=depth,
                depthValid=depth_valid,
                prevVisualState=previous,
                cameraMotion=camera_motion,
                prevVisualValid=torch.tensor([False, True], device=self.device))
            expected_identity = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                device=self.device)
            ok = (
                torch.equal(
                    current.Auxiliary["WarpedPrevPatchTokens"][0],
                    current.PatchTokens[0].detach())
                and bool(torch.isfinite(current.Auxiliary["WarpPrevPatchValid"][0]).all().item())
                and float(current.Auxiliary["WarpPrevPatchValid"][0].abs().sum().item()) == 0.0
                and float(current.Auxiliary["PatchMotionDepthResidual"][0].abs().sum().item()) == 0.0
                and torch.equal(current.Auxiliary["CameraMotionFromPrev"][0], expected_identity))
            print(f"PartialPreviousVisualMask {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"PartialPreviousVisualMask error: {e}")
            return False

    def RunAll(self):
        results = {
            "HebbianConv2d": self.TestHebbianConv2d(),
            "HebbianLinear": self.TestHebbianLinear(),
            "PerceiveExtractorForward": self.TestPerceiveExtractor(),
            "PerceiveExtractorIOShapes": self.TestPerceiveExtractorIOShapes(),
            "PerceiveExtractorStructuredState": self.TestPerceiveExtractorStructuredState(),
            "RGBDGeometryAndSupervision": self.TestRGBDGeometryAndSupervision(),
            "RecallLossDecreases": self.TestRecallLossDecreases(),
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
            "SmallBatchSafety": self.SmallBatchSafety(),
            "PartialPreviousVisualMask": self.PartialPreviousVisualMask(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nPerception module tests (with wrapper): {passed}/{len(results)} passed.")
        return results
