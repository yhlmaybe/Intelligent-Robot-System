import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
from FunctionTools import DynamicAdapterTopologyMixin, GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, RoPEMultiheadAttention, HungarianAssignment, SynchronizeDynamicAdapterTopologiesForFullLoad
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


def FrobeniusCapPerSample(mem: torch.Tensor):
    with torch.no_grad():
        B = mem.size(0)
        flat = mem.reshape(B, -1)
        n = torch.linalg.vector_norm(flat, ord=2, dim=1)
        scale = (1.0 / (n + 1e-12)).clamp(max=1.0)
        mem.mul_(scale.view(B, *([1] * (mem.dim() - 1))))


class GrowableLoRAConv2d(DynamicAdapterTopologyMixin, nn.Module):
    def __init__(self, targetConv: nn.Conv2d):
        super().__init__()
        object.__setattr__(self, "target", targetConv)
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()
        self.register_buffer(
            "topology_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True)

        w = self.target.weight # [cout, cin, kh, kw]
        self.cout, self.cin, self.kh, self.kw = w.shape

    def ValidateDynamicAdapterEntry(
        self,
        aValue: torch.Tensor,
        bValue: torch.Tensor,
        scale: torch.Tensor,) -> bool:
        return (
            aValue.dim() == 2
            and tuple(aValue.shape[1:]) == (self.cin * self.kh * self.kw,)
            and tuple(bValue.shape) == (self.cout, int(aValue.size(0)))
            and scale.numel() == 1)

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
        self.topology_count.fill_(len(self.A_list))

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


class GrowableConv1x1Adapter(DynamicAdapterTopologyMixin, AGICoreModule):
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()
        self.register_buffer(
            "topology_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True)

    def ValidateDynamicAdapterEntry(
        self,
        aValue: torch.Tensor,
        bValue: torch.Tensor,
        scale: torch.Tensor,) -> bool:
        return (
            aValue.dim() == 4
            and tuple(aValue.shape[1:]) == (self.C, 1, 1)
            and tuple(bValue.shape) == (self.C, int(aValue.size(0)), 1, 1)
            and scale.numel() == 1)

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
        self.topology_count.fill_(len(self.A_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.A_list) == 0:
            return x
        y = x
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            z = F.conv2d(x, A, bias=None, stride=1, padding=0)
            z = F.conv2d(z, B, bias=None, stride=1, padding=0)
            y = y + torch.tanh(s) * GetParametersScale(s) * z
        return y


class GrowableTokenAdapter(DynamicAdapterTopologyMixin, AGICoreModule):
    def __init__(self, dim: int):
        super().__init__()
        self.D = dim
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()
        self.register_buffer(
            "topology_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True)

    def ValidateDynamicAdapterEntry(
        self,
        aValue: torch.Tensor,
        bValue: torch.Tensor,
        scale: torch.Tensor,) -> bool:
        return (
            aValue.dim() == 2
            and tuple(aValue.shape[1:]) == (self.D,)
            and tuple(bValue.shape) == (self.D, int(aValue.size(0)))
            and scale.numel() == 1)

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
        self.topology_count.fill_(len(self.A_list))

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

        assert in_channels % gauge_groups == 0, (
            "in_channels must be divisible by gauge_groups")
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
        g_h = F.softplus(self.sheaf_gain_h).view(1, -1, 1, 1)
        g_v = F.softplus(self.sheaf_gain_v).view(1, -1, 1, 1)
        gain_sum = (g_h + g_v).clamp_min(self.eps)
        gain_scale = gain_sum.clamp_min(1.0)
        g_h = g_h / gain_scale
        g_v = g_v / gain_scale
        alpha = 0.24 * torch.sigmoid(self.sheaf_alpha)

        x_cur = x
        for _ in range(self.sheaf_iters):
            left = self.Shift(x_cur, dim=-1, step=-1)
            right = self.Shift(x_cur, dim=-1, step=+1)
            up = self.Shift(x_cur, dim=-2, step=-1)
            down = self.Shift(x_cur, dim=-2, step=+1)

            h_mean = 0.5 * (left + right)
            v_mean = 0.5 * (up + down)

            msg = g_h * (h_mean - x_cur) + g_v * (v_mean - x_cur)
            x_cur = x_cur + alpha * msg
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
        groups: int = 1,):
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
            bias=False,)

        self.register_buffer("hebb_memory", torch.empty(0), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,):
        self.ResetHebbianMemory()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if doneMask is not None and self.hebb_memory.numel() > 0:
                mask = doneMask.view(-1)
                if int(mask.numel()) != int(self.hebb_memory.size(0)):
                    raise ValueError("doneMask batch size must match Hebbian convolution memory")
                self.hebb_memory[mask] = 0
                return
            self.hebb_memory.zero_()

    def EnsureB(self, B: int):
        w = self.conv.weight
        if self.hebb_memory.size(0) != B:
            self.hebb_memory = w.new_zeros(B, *w.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, inC, H, W = x.shape
        w = self.conv.weight
        outC = w.size(0)
        g = self.groups
        in_per_g = inC // g

        w_eff = w.unsqueeze(0) + 0.25 * self.hebb_memory.detach()
        x_big = x.reshape(1, B * inC, H, W)
        w_big = w_eff.reshape(
            B * outC,
            w.size(1),
            w.size(2),
            w.size(3))

        groups_total = B * g

        out_big = F.conv2d(
            x_big,
            w_big,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups_total,)

        Hout, Wout = out_big.shape[-2], out_big.shape[-1]
        out = out_big.reshape(B, outC, Hout, Wout)
        with torch.no_grad():
            x_unfold = F.unfold(
                x.detach(),
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,)

            out_unfold = out.detach().reshape(B, outC, -1)
            L = out_unfold.size(-1)
            N = float(L)

            x_unfold_g = x_unfold.reshape(
                B,
                g,
                in_per_g * (self.kernel_size[0] * self.kernel_size[1]),
                L)

            out_unfold_g = out_unfold.reshape(B, g, outC // g, L)

            hebb_term = torch.einsum(
                "bgol,bgil->bgoi",
                out_unfold_g,
                x_unfold_g) / N

            y2_mean = out_unfold_g.square().sum(dim=-1) / N
            mem = self.hebb_memory.reshape(B, g, outC // g, -1)
            decay_scale = 0.995 * torch.exp(
                -(5e-6 / 0.995) * y2_mean.unsqueeze(-1))
            mem.mul_(decay_scale)
            mem.add_(hebb_term, alpha=5e-6)

            FrobeniusCapPerSample(self.hebb_memory)

        return out



class HebbianLinear(AGICoreModule):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,):
        super().__init__()
        self.inFeatures = int(inFeatures)
        self.outFeatures = int(outFeatures)

        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)
        self.bias = nn.Parameter(torch.zeros(outFeatures))

        self.register_buffer("hebb_memory", torch.empty(0), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,):
        self.ResetHebbianMemory()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if doneMask is not None and self.hebb_memory.numel() > 0:
                mask = doneMask.view(-1)
                if int(mask.numel()) != int(self.hebb_memory.size(0)):
                    raise ValueError("doneMask batch size must match Hebbian linear memory")
                self.hebb_memory[mask] = 0
                return
            self.hebb_memory.zero_()

    def EnsureB(self, B: int):
        if self.hebb_memory.size(0) != B:
            self.hebb_memory = self.weight.new_zeros(
                B, self.outFeatures, self.inFeatures)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x2 = x.reshape(B, -1, self.inFeatures)
        w_eff = (
            self.weight.unsqueeze(0)
            + 0.2 * self.hebb_memory.detach())
        y2 = torch.einsum("bni,boi->bno", x2, w_eff)
        y2 = y2 + self.bias.view(1, 1, -1)
        y = y2.view(*x.shape[:-1], self.outFeatures)

        with torch.no_grad():
            N = float(y2.size(1))
            hebb_term = torch.einsum("bno,bni->boi", y2, x2) / N
            y_sq_mean = y2.square().mean(dim=1)
            decay_scale = 0.995 * torch.exp(
                -(5e-5 / 0.995) * y_sq_mean.unsqueeze(-1))
            self.hebb_memory.mul_(decay_scale)
            self.hebb_memory.add_(hebb_term, alpha=5e-5)
            FrobeniusCapPerSample(self.hebb_memory)

        return y



class PerceptionRoPEMultiheadAttention(RoPEMultiheadAttention):
    def Apply2DRotary(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        ) -> torch.Tensor:
        rotary_dim = int(self.rope.dim)
        if rotary_dim <= 0:
            return value
        row_dim = 2 * ((rotary_dim // 2) // 2)
        column_dim = rotary_dim - row_dim
        angle_dtype = torch.float32

        def axis_angle(
            dim: int,
            coordinate: torch.Tensor,
            ) -> torch.Tensor:
            if dim <= 0:
                return torch.empty(
                    int(value.size(-2)),
                    0,
                    device=value.device,
                    dtype=angle_dtype)

            inverse_frequency = 1.0 / (
                self.rope.base ** (
                    torch.arange(
                        0,
                        dim,
                        2,
                        device=value.device,
                        dtype=angle_dtype) / float(dim)))

            angle = coordinate.to(
                device=value.device,
                dtype=angle_dtype).unsqueeze(1) * inverse_frequency.unsqueeze(0)

            return torch.repeat_interleave(angle, repeats=2, dim=-1)

        spatial_angle = torch.cat([
            axis_angle(row_dim, positions[:, 0]),
            axis_angle(column_dim, positions[:, 1]),], dim=-1)

        cosine = spatial_angle.cos().to(value.dtype).view(1, 1, -1, rotary_dim)
        sine = spatial_angle.sin().to(value.dtype).view(1, 1, -1, rotary_dim)
        rotary = value[..., :rotary_dim]
        rotated = rotary * cosine + self.rope.RotateHalf(rotary) * sine
        passthrough = value[..., rotary_dim:]

        return (
            rotated
            if passthrough.numel() == 0
            else torch.cat([rotated, passthrough], dim=-1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotaryPositions2D: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        needWeights: bool = True,
        attnMask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, query_length, _ = query.shape
        key_length = int(key.size(1))

        q_raw = self.ReshapeHeads(self.q_proj(query))
        k_raw = self.ReshapeHeads(self.k_proj(key))
        v = self.ReshapeHeads(self.v_proj(value))

        q = self.Apply2DRotary(
            q_raw,
            rotaryPositions2D)

        k = self.Apply2DRotary(
            k_raw,
            rotaryPositions2D)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attnMask is not None:
            mask = self.PrepareMask(
                attnMask,
                batch,
                query_length,
                key_length)
            scores = (
                scores.masked_fill(mask, -torch.inf)
                if mask.dtype == torch.bool
                else scores + mask)
        if keyPaddingMask is not None:
            padding = keyPaddingMask.reshape(batch, 1, 1, key_length)
            scores = scores.masked_fill(padding, -torch.inf)

        attention_probability = F.softmax(scores, dim=-1)
        attention_probability = torch.nan_to_num(
            attention_probability,
            nan=0.0,
            posinf=0.0,
            neginf=0.0)
        attention = self.attn_drop(attention_probability)
        output = self.out_proj(self.MergeHeads(torch.matmul(attention, v)))
        query_has_key = attention_probability.sum(dim=-1).gt(0).any(dim=1)
        output = output.masked_fill(~query_has_key.unsqueeze(-1), 0.0)
        weights = attention_probability.mean(dim=1) if needWeights else None
        return output, weights


class TransformerEncode(AGICoreModule):
    def __init__(self, modelDim: int, headNum: int, dimFeedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_atten = PerceptionRoPEMultiheadAttention(
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

    def forward(
        self,
        src: torch.Tensor,
        rotaryPositions2D: torch.Tensor,
        srcMask: Optional[torch.Tensor] = None,
        srcKeyPaddingMask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        src_norm1 = self.norm1(src)
        src2, _ = self.self_atten(
            src_norm1, src_norm1, src_norm1,
            attnMask=srcMask,
            keyPaddingMask=srcKeyPaddingMask,
            rotaryPositions2D=rotaryPositions2D,
            needWeights=False)
        
        src = src + self.dropout1(src2)

        src_norm2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(src2)
        return src
    

class ResidualBlock(AGICoreModule):
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()
        self.use_downsample = bool(stride != 1 or inChannels != outChannels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(
                    inChannels,
                    outChannels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                Norm2d(outChannels))
            if self.use_downsample
            else nn.Identity())
            
        self.conv1 = HebbianConv2d(
            inChannels, outChannels, 3, stride=stride, padding=1)
        self.bn1 = Norm2d(outChannels)
        self.conv2 = HebbianConv2d(
            outChannels, outChannels, 3, stride=1, padding=1)
        self.bn2 = Norm2d(outChannels)
        self.relu = nn.SiLU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class StableAnisotropicDiffusion(AGICoreModule):
    def __init__(self, iterations: int = 1, eps: float = 1e-6):
        super().__init__()
        self.iterations = max(0, int(iterations))
        self.eps = float(eps)
        self.raw_step = nn.Parameter(torch.tensor(0.0))
        self.raw_kappa = nn.Parameter(torch.tensor(0.0))

    def StepSize(self) -> torch.Tensor:
        return 0.24 * torch.sigmoid(self.raw_step)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.iterations == 0:
            return value
        step = self.StepSize().to(value.dtype)
        kappa = (F.softplus(self.raw_kappa) + self.eps).to(value.dtype)
        current = value
        for _ in range(self.iterations):
            value_pad = F.pad(current, (1, 1, 1, 1), mode="replicate")
            guidance = current.mean(dim=1, keepdim=True)
            guidance_pad = F.pad(guidance, (1, 1, 1, 1), mode="replicate")
            neighbors = (
                value_pad[..., 1:-1, :-2],
                value_pad[..., 1:-1, 2:],
                value_pad[..., :-2, 1:-1],
                value_pad[..., 2:, 1:-1],)
            guidance_neighbors = (
                guidance_pad[..., 1:-1, :-2],
                guidance_pad[..., 1:-1, 2:],
                guidance_pad[..., :-2, 1:-1],
                guidance_pad[..., 2:, 1:-1],)
            update = torch.zeros_like(current)
            for neighbor, neighbor_guidance in zip(neighbors, guidance_neighbors):
                gradient = neighbor_guidance - guidance
                conductance = torch.exp(-torch.square(gradient / kappa))
                update = update + conductance * (neighbor - current)
            current = current + step * update
        return current


class CorticalEarlyVision(AGICoreModule):
    def __init__(
        self,
        outChannels: int,
        orientations: int = 6,
        kernelSize: int = 13,
        wavelength: float = 4.0,
        sigma: float = 2.0,
        aspectRatio: float = 0.7,):
        super().__init__()
        self.orientations = int(orientations)
        self.kernel_size = int(kernelSize)
        self.eps = 1e-6
        # The same quadrature bank is evaluated as a stationary dyadic filter
        # bank on the half-resolution cortical lattice (effective RGB
        # wavelengths are wavelength * 2 * scale).  No scale is decimated, so
        # phase comparisons refer to the same retinal sample at every scale.
        self.frequency_scales = (1.0, 2.0, 4.0)

        even, odd = self.BuildGaborBank(
            self.orientations,
            self.kernel_size,
            float(wavelength),
            float(sigma),
            float(aspectRatio))
        self.register_buffer("gabor_even", even, persistent=True)
        self.register_buffer("gabor_odd", odd, persistent=True)
        self.register_buffer(
            "gabor_quadrature",
            torch.cat([even, odd], dim=0),
            persistent=False)
        self.register_buffer(
            "anti_alias_kernel",
            self.BinomialKernel(),
            persistent=True)
        self.register_buffer(
            "scale_response_calibration",
            self.BuildScaleResponseCalibration(
                self.orientations,
                self.frequency_scales,
                float(wavelength)),
            persistent=False)
        orientation_phase = (
            2.0 * math.pi
            * torch.arange(self.orientations, dtype=torch.float32)
            / float(self.orientations))
        self.register_buffer(
            "orientation_cosine",
            orientation_phase.cos().view(1, self.orientations, 1, 1),
            persistent=False)
        self.register_buffer(
            "orientation_sine",
            orientation_phase.sin().view(1, self.orientations, 1, 1),
            persistent=False)
        collinear, surround = self.BuildContourKernels(
            self.orientations,
            kernelSize=7)
        self.register_buffer("collinear_kernel", collinear, persistent=True)
        self.register_buffer("surround_kernel", surround, persistent=True)

        self.divisive_bias_raw = nn.Parameter(torch.tensor(-2.0))
        self.collinear_gain_raw = nn.Parameter(torch.tensor(-1.0))
        self.surround_gain_raw = nn.Parameter(torch.tensor(-1.0))
        self.spectral_scale_logits = nn.Parameter(torch.zeros(
            len(self.frequency_scales)))
        self.phase_congruency_gain_raw = nn.Parameter(torch.tensor(-2.0))
        self.fast_decay_raw = nn.Parameter(torch.tensor(0.0))
        self.slow_gap_raw = nn.Parameter(torch.tensor(math.log(4.0)))
        self.diffusion = StableAnisotropicDiffusion(iterations=1)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.orientations * 2, int(outChannels), kernel_size=1, bias=False),
            Norm2d(int(outChannels)),
            nn.SiLU())
        self.register_load_state_dict_post_hook(
            self.RefreshGaborQuadratureAfterLoad)

    @torch.no_grad()
    def RefreshGaborQuadratureAfterLoad(
        self,
        module: nn.Module,
        incompatibleKeys: Any,
        ) -> None:
        del module, incompatibleKeys
        combined = torch.cat([self.gabor_even, self.gabor_odd], dim=0)
        self.gabor_quadrature.copy_(combined)

    @staticmethod
    def BinomialKernel() -> torch.Tensor:
        vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(vector, vector)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 5, 5)

    @staticmethod
    def BuildGaborBank(
        orientations: int,
        kernelSize: int,
        wavelength: float,
        sigma: float,
        aspectRatio: float,) -> Tuple[torch.Tensor, torch.Tensor]:
        radius = kernelSize // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij")
        even_bank = []
        odd_bank = []
        for index in range(orientations):
            theta = math.pi * float(index) / float(orientations)
            x_theta = xx * math.cos(theta) + yy * math.sin(theta)
            y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
            envelope = torch.exp(-(
                x_theta.square() + (aspectRatio * y_theta).square()
            ) / (2.0 * sigma * sigma))
            phase = 2.0 * math.pi * x_theta / wavelength
            even = envelope * torch.cos(phase)
            odd = envelope * torch.sin(phase)
            even = even - even.mean()
            odd = odd - odd.mean()
            even_bank.append(even / even.norm().clamp_min(1e-8))
            odd_bank.append(odd / odd.norm().clamp_min(1e-8))
        return (
            torch.stack(even_bank).unsqueeze(1),
            torch.stack(odd_bank).unsqueeze(1))

    @staticmethod
    def BuildContourKernels(
        orientations: int,
        kernelSize: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        radius = kernelSize // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij")
        collinear_bank = []
        surround_bank = []
        for index in range(orientations):
            theta = math.pi * float(index) / float(orientations)
            x_theta = xx * math.cos(theta) + yy * math.sin(theta)
            y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
            # The carrier varies along x_theta, so the perceived contour is
            # tangent to y_theta.  Facilitation follows that tangent; the
            # orthogonal elongated field supplies cross-contour inhibition.
            collinear = torch.exp(-0.5 * (
                x_theta.square() / 0.64 + y_theta.square() / 6.25))
            surround = torch.exp(-0.5 * (
                x_theta.square() / 6.25 + y_theta.square() / 0.64))
            collinear = collinear / collinear.sum().clamp_min(1e-8)
            surround = surround / surround.sum().clamp_min(1e-8)
            collinear_bank.append(collinear)
            surround_bank.append(surround)
        return (
            torch.stack(collinear_bank).unsqueeze(1),
            torch.stack(surround_bank).unsqueeze(1))

    def AntiAliasDownsample(self, value: torch.Tensor, groups: int = 1) -> torch.Tensor:
        kernel = self.anti_alias_kernel.expand(groups, 1, -1, -1)
        value = F.pad(value, (2, 2, 2, 2), mode="replicate")
        return F.conv2d(value, kernel, stride=2, groups=groups)

    @staticmethod
    def BuildScaleResponseCalibration(
        orientations: int,
        frequencyScales: Tuple[float, ...],
        wavelength: float,
        ) -> torch.Tensor:
        """Equalize the preferred-frequency gain of the stationary scales."""
        calibration = []
        for scale_value in frequencyScales:
            scale = int(scale_value)
            angular_frequency = 2.0 * math.pi / (float(wavelength) * scale)
            orientation_gain = []
            for index in range(int(orientations)):
                theta = math.pi * float(index) / float(orientations)
                smoothing_gain = 1.0
                dilation = 1
                while dilation < scale:
                    # The centred five-tap binomial response is cos(w/2)^4.
                    frequency_x = (
                        angular_frequency * math.cos(theta) * dilation)
                    frequency_y = (
                        angular_frequency * math.sin(theta) * dilation)
                    smoothing_gain *= math.cos(0.5 * frequency_x) ** 4
                    smoothing_gain *= math.cos(0.5 * frequency_y) ** 4
                    dilation *= 2
                orientation_gain.append(1.0 / smoothing_gain)
            calibration.append(orientation_gain + orientation_gain)
        return torch.tensor(calibration).view(
            1,
            len(frequencyScales),
            2 * int(orientations),
            1,
            1)

    def MultiscaleQuadrature(
        self,
        luminance: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # A stationary (undecimated) binomial pyramid preserves translation
        # phase while progressively removing frequencies that would alias in
        # the coarser dilated Gabor filters.
        levels = [luminance]
        smoothed = luminance
        smoothing_dilation = 1
        for _ in range(1, len(self.frequency_scales)):
            smoothing_padding = 2 * smoothing_dilation
            smoothed = F.conv2d(
                F.pad(
                    smoothed,
                    (
                        smoothing_padding,
                        smoothing_padding,
                        smoothing_padding,
                        smoothing_padding),
                    mode="replicate"),
                self.anti_alias_kernel,
                dilation=smoothing_dilation)
            levels.append(smoothed)
            smoothing_dilation *= 2

        responses = []
        for level, scale in zip(levels, self.frequency_scales):
            dilation = int(scale)
            padding = (self.kernel_size // 2) * dilation
            response = F.conv2d(
                F.pad(
                    level,
                    (padding, padding, padding, padding),
                    mode="replicate"),
                self.gabor_quadrature,
                dilation=dilation)
            responses.append(response)
        response_stack = (
            torch.stack(responses, dim=1)
            * self.scale_response_calibration)
        return response_stack.chunk(2, dim=2)

    def QuadratureAmplitude(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
        ) -> torch.Tensor:
        # Construct and square-root epsilon in the response dtype.  Subtracting
        # Python sqrt(eps) leaves a false non-zero floor in pure FP16 because
        # eps itself is quantized before the tensor square root.
        epsilon = real.new_full((), self.eps)
        if real.dtype in (torch.float16, torch.bfloat16):
            real_work = real.float()
            imaginary_work = imaginary.float()
            epsilon_work = epsilon.float()
            return (
                torch.sqrt(
                    real_work.square()
                    + imaginary_work.square()
                    + epsilon_work)
                - torch.sqrt(epsilon_work)
            ).clamp_min(0.0).to(dtype=real.dtype)
        return (
            torch.sqrt(
                real.square() + imaginary.square() + epsilon)
            - torch.sqrt(epsilon)
        ).clamp_min(0.0)

    def StablePositiveRatio(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        ) -> torch.Tensor:
        # FP16 division backward overflows/underflows around 1e-6, while BF16
        # loses the unit upper bound through coarse rounding.  Use a
        # representable noise floor and perform only this quotient in FP32;
        # gradients stay bounded and the result returns to the model dtype.
        denominator_floor = max(
            self.eps,
            float(torch.finfo(denominator.dtype).tiny))
        if denominator.dtype in (torch.float16, torch.bfloat16):
            # The entropy derivative contributes |log(eps)| on top of the
            # reciprocal.  sqrt(eps) is also the quadrature amplitude's own
            # noise floor and keeps that combined gradient representable.
            denominator_floor = max(
                denominator_floor,
                math.sqrt(self.eps))
            ratio = (
                numerator.float()
                / denominator.float().clamp_min(denominator_floor)
            ).to(dtype=numerator.dtype)
        else:
            ratio = numerator / denominator.clamp_min(denominator_floor)
        # Every caller divides one non-negative component/magnitude by its
        # corresponding total.  Unit range is part of that statistic's
        # definition; low-precision rounding can otherwise yield > 1.
        return ratio.clamp(max=1.0)

    def MultiscalePhaseStatistics(
        self,
        even: torch.Tensor,
        odd: torch.Tensor,
        energy: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase agreement must remain genuinely multi-scale.  Learned scale
        # weights are used for feature energy below, but not here: if a
        # softmax collapses to one scale, every non-zero signal has a false
        # phase-congruency score of exactly one.
        phase_real = even.mean(dim=1)
        phase_imag = odd.mean(dim=1)
        phase_denominator = energy.mean(dim=1)
        phase_magnitude = self.QuadratureAmplitude(
            phase_real,
            phase_imag)
        phase_congruency = self.StablePositiveRatio(
            phase_magnitude,
            phase_denominator)

        # Entropy across the three scale energies measures local frequency
        # spread.  It is distinct from the frame-level FFT statistic used by
        # QualityToken.
        spectral_probability = self.StablePositiveRatio(
            energy,
            energy.sum(dim=1, keepdim=True))
        scale_entropy = -(
            spectral_probability
            * spectral_probability.clamp_min(self.eps).log()
        ).sum(dim=1) / math.log(float(len(self.frequency_scales)))
        return phase_congruency, scale_entropy

    def OrientationCoherence(self, energy: torch.Tensor) -> torch.Tensor:
        # Gabor orientations are pi-periodic, hence the doubled-angle circular
        # mean.  Coherent contours approach one; isotropic noise approaches
        # zero.  This is a reliability term, not an object/existence mask.
        orientation_energy = energy.mean(dim=1)
        phase_real = (
            orientation_energy * self.orientation_cosine).sum(
                dim=1,
                keepdim=True)
        phase_imag = (
            orientation_energy * self.orientation_sine).sum(
                dim=1,
                keepdim=True)
        phase_magnitude = self.QuadratureAmplitude(
            phase_real,
            phase_imag)
        total_energy = orientation_energy.sum(dim=1, keepdim=True)
        return self.StablePositiveRatio(
            phase_magnitude,
            total_energy)

    def FastDecay(self) -> torch.Tensor:
        return torch.sigmoid(self.fast_decay_raw)

    def SlowDecay(self) -> torch.Tensor:
        fast = self.FastDecay()
        return fast + (1.0 - fast) * torch.sigmoid(self.slow_gap_raw)

    def forward(
        self,
        frame: torch.Tensor,
        previousFast: Optional[torch.Tensor],
        previousSlow: Optional[torch.Tensor],
        previousValid: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        luminance = (
            0.2126 * frame[:, 0:1]
            + 0.7152 * frame[:, 1:2]
            + 0.0722 * frame[:, 2:3])
        luminance = self.AntiAliasDownsample(luminance)
        even, odd = self.MultiscaleQuadrature(luminance)
        energy = self.QuadratureAmplitude(even, odd)

        scale_prior = F.softmax(
            self.spectral_scale_logits,
            dim=0).view(1, -1, 1, 1, 1)
        orientation_energy = (scale_prior * energy).sum(dim=1)
        cortical_energy = orientation_energy
        phase_congruency, scale_entropy = self.MultiscalePhaseStatistics(
            even,
            odd,
            energy)
        orientation_coherence = self.OrientationCoherence(energy)
        spectral_structure = (
            phase_congruency
            * scale_entropy
            * orientation_coherence)
        orientation_energy = orientation_energy * (
            1.0
            + torch.sigmoid(self.phase_congruency_gain_raw)
            * spectral_structure)

        orientation_pool = F.avg_pool2d(
            F.pad(
                orientation_energy.mean(dim=1, keepdim=True),
                (2, 2, 2, 2),
                mode="replicate"),
            kernel_size=5,
            stride=1)
        divisive_bias = F.softplus(self.divisive_bias_raw).to(frame.dtype) + self.eps
        normalized = orientation_energy / (divisive_bias + orientation_pool)
        normalized = self.diffusion(normalized)

        normalized_pad = F.pad(normalized, (3, 3, 3, 3), mode="replicate")
        collinear = F.conv2d(
            normalized_pad,
            self.collinear_kernel.to(frame.dtype),
            groups=self.orientations)
        surround = F.conv2d(
            normalized_pad,
            self.surround_kernel.to(frame.dtype),
            groups=self.orientations)
        v2_response = F.silu(
            normalized
            + torch.sigmoid(self.collinear_gain_raw).to(frame.dtype) * collinear
            - torch.sigmoid(self.surround_gain_raw).to(frame.dtype) * surround)
        current = self.AntiAliasDownsample(
            v2_response,
            groups=self.orientations)

        state_compatible = (
            previousFast is not None
            and previousSlow is not None
            and tuple(previousFast.shape) == tuple(current.shape)
            and tuple(previousSlow.shape) == tuple(current.shape))
        if state_compatible:
            if previousFast.device != current.device or previousSlow.device != current.device:
                raise ValueError(
                    "cortical temporal states must be on the current frame device")
            if previousValid is None:
                raise ValueError(
                    "previousValid is required with cortical temporal states")
            previous_fast = previousFast.detach().to(dtype=current.dtype)
            previous_slow = previousSlow.detach().to(dtype=current.dtype)
            valid = previousValid
            valid = valid.view(-1, 1, 1, 1)
            fast_decay = self.FastDecay()
            slow_decay = self.SlowDecay()
            fast = torch.where(
                valid,
                fast_decay * previous_fast + (1.0 - fast_decay) * current,
                current)
            slow = torch.where(
                valid,
                slow_decay * previous_slow + (1.0 - slow_decay) * current,
                current)
        else:
            fast = current
            slow = current
        temporal_response = fast - slow
        feature = self.feature_projection(torch.cat([current, temporal_response], dim=1))
        raw_energy = self.AntiAliasDownsample(
            cortical_energy.mean(dim=1, keepdim=True))
        return feature, {
            "CorticalFastState": fast.detach(),
            "CorticalSlowState": slow.detach(),
            "CorticalEnergy": raw_energy.detach(),
            "CorticalContextResponse": current.mean(
                dim=1,
                keepdim=True).detach(),
            "CorticalTemporalResponse": temporal_response.mean(
                dim=1,
                keepdim=True).detach()}


class PerceptionEnhancementBlock(AGICoreModule):
    def __init__(self, stemChannels: int):
        super().__init__()
        self.early_vision = CorticalEarlyVision(outChannels=int(stemChannels))
        self.residual_gain = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        frame: torch.Tensor,
        previousVisualState: Optional[VisualState],
        previousValid: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        previous_fast = None
        previous_slow = None
        if previousVisualState is not None:
            previous_fast = previousVisualState.Auxiliary.get("CorticalFastState")
            previous_slow = previousVisualState.Auxiliary.get("CorticalSlowState")
        feature, auxiliary = self.early_vision(
            frame,
            previous_fast,
            previous_slow,
            previousValid)
        return torch.tanh(self.residual_gain) * feature, auxiliary


class SPPContextAdapter(AGICoreModule):
    def __init__(
        self,
        inChannels: int,
        embedDim: int,
        reducedChannels: int = 32,
        bins: Tuple[int, ...] = (1, 2, 4),):
        super().__init__()
        self.bins = tuple(int(value) for value in bins)
        reduced = int(reducedChannels)
        pooled_dim = reduced * sum(value * value for value in self.bins)
        self.reduce = nn.Sequential(
            nn.Conv2d(int(inChannels), reduced, kernel_size=1, bias=False),
            Norm2d(reduced),
            nn.SiLU())
        self.project = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(embedDim)))
        self.residual_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(feature)
        height, width = reduced.shape[-2:]
        if (
            self.bins == (1, 2, 4)
            and height >= 4
            and width >= 4
            and height % 4 == 0
            and width % 4 == 0
        ):
            fine = F.adaptive_avg_pool2d(reduced, (4, 4))
            pooled_maps = {
                1: F.avg_pool2d(fine, kernel_size=4),
                2: F.avg_pool2d(fine, kernel_size=2, stride=2),
                4: fine,}
            pooled = [pooled_maps[size].flatten(1) for size in self.bins]
        else:
            pooled = [
                F.adaptive_avg_pool2d(reduced, (size, size)).flatten(1)
                for size in self.bins]
        context = self.project(torch.cat(pooled, dim=1))
        return torch.tanh(self.residual_gain) * context


class AxialPositionEncoding2D(nn.Module):
    def __init__(self, embedDim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = int(embedDim)
        self.temperature = float(temperature)

    def AxisEncoding(
        self,
        length: int,
        dim: int,
        device: torch.device,) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(int(length), 0, device=device)
        frequency_count = max(1, (dim + 1) // 2)
        exponent = torch.arange(
            frequency_count,
            device=device,
            dtype=torch.float32) / max(frequency_count - 1, 1)
        inverse_frequency = torch.exp(-math.log(self.temperature) * exponent)
        position = torch.arange(int(length), device=device, dtype=torch.float32)
        angle = position.unsqueeze(1) * inverse_frequency.unsqueeze(0)
        encoding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        return encoding[:, :dim]

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,) -> torch.Tensor:
        row_dim = self.embed_dim // 2
        column_dim = self.embed_dim - row_dim
        row = self.AxisEncoding(height, row_dim, device)
        column = self.AxisEncoding(width, column_dim, device)
        row = row[:, None, :].expand(height, width, row_dim)
        column = column[None, :, :].expand(height, width, column_dim)
        return torch.cat([row, column], dim=-1).reshape(
            1,
            height * width,
            self.embed_dim).to(dtype=dtype)


class DenseDepthRefiner(AGICoreModule):
    def __init__(
        self,
        hiddenChannels: int = 16,
        minDepthMeters: float = 0.05,
        maxDepthMeters: float = 20.0,):
        super().__init__()
        hidden = int(hiddenChannels)
        self.min_depth = float(minDepthMeters)
        self.max_depth = float(maxDepthMeters)
        self.trunk = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=3, padding=1, bias=False),
            Norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.SiLU())
        self.output = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)

    def forward(
        self,
        frame: torch.Tensor,
        metricDepth: torch.Tensor,
        metricLogVariance: torch.Tensor,
        applyCorrection: bool = True,
        ) -> Dict[str, torch.Tensor]:
        size = tuple(frame.shape[-2:])
        inverse = F.interpolate(
            metricDepth.clamp_min(self.min_depth).reciprocal(),
            size=size,
            mode="bilinear",
            align_corners=False)
        log_variance = F.interpolate(
            metricLogVariance,
            size=size,
            mode="bilinear",
            align_corners=False)
        if not applyCorrection:
            return {
                "MetricDepthFullRes": inverse.reciprocal(),
                "MetricDepthFullResLogVariance": log_variance.clamp(-8.0, 8.0)}
        correction_size = (
            max(1, int(size[0]) // 2),
            max(1, int(size[1]) // 2))
        if correction_size == size:
            correction_input = torch.cat([frame, inverse], dim=1)
        else:
            correction_input = torch.cat([
                F.interpolate(frame, size=correction_size, mode="area"),
                F.interpolate(inverse, size=correction_size, mode="area"),
            ], dim=1)
        raw = self.output(self.trunk(correction_input))
        if tuple(raw.shape[-2:]) != size:
            raw = F.interpolate(
                raw,
                size=size,
                mode="bilinear",
                align_corners=False)
        correction = 0.1 * inverse * torch.tanh(raw[:, :1])
        refined_inverse = (inverse + correction).clamp(
            1.0 / self.max_depth,
            1.0 / self.min_depth)
        return {
            "MetricDepthFullRes": refined_inverse.reciprocal(),
            "MetricDepthFullResLogVariance": (
                log_variance + raw[:, 1:2].clamp(-2.0, 2.0)).clamp(-8.0, 8.0)}


class ProjectiveTopologyDiagnostics(AGICoreModule):
    @staticmethod
    def DifferenceX(value: torch.Tensor, identityValue: float) -> torch.Tensor:
        if value.size(-1) == 1:
            return torch.full_like(value, float(identityValue))
        difference = value[..., 1:] - value[..., :-1]
        return torch.cat([difference, difference[..., -1:]], dim=-1)

    @staticmethod
    def DifferenceY(value: torch.Tensor, identityValue: float) -> torch.Tensor:
        if value.size(-2) == 1:
            return torch.full_like(value, float(identityValue))
        difference = value[..., 1:, :] - value[..., :-1, :]
        return torch.cat([difference, difference[..., -1:, :]], dim=-2)

    def forward(
        self,
        grid: torch.Tensor,
        domainMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        batch, height, width, _ = grid.shape
        # Determinants and singular values are especially cancellation-prone in
        # AMP.  Diagnostics and the topology loss therefore stay in FP32.
        grid32 = grid.float()
        pixel_x = ((grid32[..., 0] + 1.0) * float(width) - 1.0) * 0.5
        pixel_y = ((grid32[..., 1] + 1.0) * float(height) - 1.0) * 0.5
        dxx = self.DifferenceX(pixel_x, 1.0)
        dxy = self.DifferenceY(pixel_x, 0.0)
        dyx = self.DifferenceX(pixel_y, 0.0)
        dyy = self.DifferenceY(pixel_y, 1.0)
        determinant = dxx * dyy - dxy * dyx
        frobenius_squared = dxx.square() + dxy.square() + dyx.square() + dyy.square()
        discriminant = (
            frobenius_squared.square() - 4.0 * determinant.square()).clamp_min(0.0).sqrt()
        sigma_max = torch.sqrt(
            0.5 * (frobenius_squared + discriminant).clamp_min(0.0))
        # |det(J)| = sigma_min * sigma_max avoids subtractive cancellation in
        # the smaller singular value.
        sigma_min = determinant.abs() / sigma_max.clamp_min(1e-12)
        finite = (
            torch.isfinite(pixel_x)
            & torch.isfinite(pixel_y)
            & torch.isfinite(determinant)
            & torch.isfinite(sigma_min)
            & torch.isfinite(sigma_max))
        if domainMask is None:
            domain = torch.ones(
                batch,
                height,
                width,
                device=grid.device,
                dtype=torch.bool)
        else:
            domain = domainMask.to(device=grid.device, dtype=torch.bool)
            if domain.dim() == 4 and domain.size(1) == 1:
                domain = domain[:, 0]
            if tuple(domain.shape) != (batch, height, width):
                raise ValueError(
                    "domainMask must have shape [B,H,W] or [B,1,H,W]")
        # A singleton spatial axis has no observable derivative; do not invent
        # an identity Jacobian and label it topology-preserving.
        if height <= 1 or width <= 1:
            domain = torch.zeros_like(domain)
        valid_domain = domain & finite
        topology_valid = (
            valid_domain
            & (determinant > 1e-3)
            & (sigma_min > 1e-3)
            & (sigma_max < 100.0))
        safe_determinant = torch.where(finite, determinant, torch.zeros_like(determinant))
        safe_sigma_min = torch.where(finite, sigma_min, torch.zeros_like(sigma_min))
        safe_sigma_max = torch.where(finite, sigma_max, torch.zeros_like(sigma_max))
        yy, xx = torch.meshgrid(
            torch.arange(height, device=grid.device, dtype=torch.float32),
            torch.arange(width, device=grid.device, dtype=torch.float32),
            indexing="ij")
        safe_pixel_x = torch.where(
            torch.isfinite(pixel_x),
            pixel_x,
            xx.view(1, height, width))
        safe_pixel_y = torch.where(
            torch.isfinite(pixel_y),
            pixel_y,
            yy.view(1, height, width))
        rigid_flow = torch.stack([
            safe_pixel_x - xx.view(1, height, width),
            safe_pixel_y - yy.view(1, height, width)], dim=1)
        fold_error = F.relu(1e-3 - safe_determinant).square()
        fold_penalty = torch.where(
            valid_domain,
            fold_error,
            torch.zeros_like(fold_error)).sum() / valid_domain.sum().clamp_min(1)
        return {
            "RigidPatchFlow": rigid_flow,
            "WarpJacobianDet": safe_determinant.unsqueeze(1),
            "WarpJacobianSigmaMin": safe_sigma_min.unsqueeze(1),
            "WarpJacobianSigmaMax": safe_sigma_max.unsqueeze(1),
            "WarpTopologyValid": topology_valid.unsqueeze(1).to(torch.float32),
            "WarpFoldPenalty": fold_penalty}


class CNNFeatureExtractor(AGICoreModule):
    def __init__(self, inChannels: int = 3, baseChannels: int = 64):
        super().__init__()
        self.conv1 = HebbianConv2d(
            inChannels, baseChannels, 7, stride=2, padding=3)
        
        self.bn1 = Norm2d(baseChannels)
        self.relu = nn.SiLU() 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(
            baseChannels, baseChannels, blocks=2, stride=1)
        self.layer2 = self.make_layer(
            baseChannels, baseChannels*2, blocks=2, stride=2)
        self.layer3 = self.make_layer(
            baseChannels*2, baseChannels*4, blocks=2, stride=2)
        self.layer4 = self.make_layer(
            baseChannels*4, baseChannels*8, blocks=2, stride=2)

        self.conv2 = HebbianConv2d(
            baseChannels*8, baseChannels*16, 3, stride=1, padding=1)
        self.bn2 = Norm2d(baseChannels*16)

    def make_layer(self, inC, outC, blocks, stride):
        layers = [ResidualBlock(inC, outC, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(outC, outC, stride=1))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        stemResidual: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        if stemResidual is not None:
            if tuple(stemResidual.shape[-2:]) != tuple(x.shape[-2:]):
                stemResidual = F.interpolate(
                    stemResidual,
                    size=tuple(x.shape[-2:]),
                    mode="bilinear",
                    align_corners=False)
            x = x + stemResidual.to(x)
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

    @staticmethod
    def ScaleIntrinsics(
        cameraIntrinsics: torch.Tensor,
        sourceSize: Tuple[int, int],
        targetSize: Tuple[int, int],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale a pixel-center camera lattice consistently with align_corners=False."""
        source_h, source_w = sourceSize
        target_h, target_w = targetSize
        sx = float(target_w) / float(source_w)
        sy = float(target_h) / float(source_h)
        fx = cameraIntrinsics[:, 0, 0] * sx
        fy = cameraIntrinsics[:, 1, 1] * sy
        skew = cameraIntrinsics[:, 0, 1] * sx
        cx = (cameraIntrinsics[:, 0, 2] + 0.5) * sx - 0.5
        cy = (cameraIntrinsics[:, 1, 2] + 0.5) * sy - 0.5
        return fx, fy, skew, cx, cy

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
        valid = depthValid.bool() & torch.isfinite(depth) & (depth > 0.0)
        valid_float = valid.to(depth.dtype)
        clean_depth = torch.where(valid, depth, torch.ones_like(depth))
        inverse = clean_depth.reciprocal() * valid_float
        inverse_sum = F.interpolate(inverse, size=size, mode="area")
        valid_weight = F.interpolate(valid_float, size=size, mode="area")
        inverse_resized = inverse_sum / valid_weight.clamp_min(1e-6)
        inverse_resized = inverse_resized * (valid_weight > 1e-6).to(inverse_resized.dtype)
        return inverse_resized, valid_weight.clamp(0.0, 1.0)

    def ResampleSensorLogDepth(
        self,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        size: Tuple[int, int],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Area-resample sufficient statistics in the fusion's log-depth domain."""
        valid = depthValid.bool() & torch.isfinite(depth) & (depth > 0.0)
        valid_float = valid.to(depth.dtype)
        clean_depth = torch.where(valid, depth, torch.ones_like(depth)).clamp(
            self.min_depth_meters,
            self.max_depth_meters)
        log_sum = F.interpolate(
            clean_depth.log() * valid_float,
            size=size,
            mode="area")
        valid_weight = F.interpolate(valid_float, size=size, mode="area")
        log_depth = log_sum / valid_weight.clamp_min(1e-6)
        log_depth = torch.where(
            valid_weight > 1e-6,
            log_depth,
            torch.zeros_like(log_depth))
        return log_depth, valid_weight.clamp(0.0, 1.0)

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
        fx, fy, skew, cx, cy = self.ScaleIntrinsics(
            cameraIntrinsics,
            sourceSize,
            (H, W))
        yy, xx = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing="ij",)
        z = depth
        normalized_y = (
            yy.view(1, 1, H, W) - cy.view(B, 1, 1, 1)
        ) / fy.view(B, 1, 1, 1)
        x = (
            xx.view(1, 1, H, W)
            - cx.view(B, 1, 1, 1)
            - skew.view(B, 1, 1, 1) * normalized_y
        ) * z / fx.view(B, 1, 1, 1)
        y = normalized_y * z
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
        fx, fy, skew, cx, cy = self.ScaleIntrinsics(
            cameraIntrinsics,
            sourceSize,
            (H, W))
        fx = fx.view(B, 1, 1, 1)
        fy = fy.view(B, 1, 1, 1)
        skew = skew.view(B, 1, 1, 1)
        cx = cx.view(B, 1, 1, 1)
        cy = cy.view(B, 1, 1, 1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=curDepth.device, dtype=curDepth.dtype),
            torch.arange(W, device=curDepth.device, dtype=curDepth.dtype),
            indexing="ij",)
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        normalized_y = (yy - cy) / fy
        point_cur = torch.cat([
            (xx - cx - skew * normalized_y) * curDepth / fx,
            normalized_y * curDepth,
            curDepth], dim=1)

        point_prev = self.QuaternionRotate(cameraMotion, point_cur)
        expected_prev = point_prev[:, 2:3]
        inv_z = expected_prev.clamp_min(1e-3).reciprocal()
        projected_x = (
            fx * point_prev[:, 0:1] * inv_z
            + skew * point_prev[:, 1:2] * inv_z
            + cx)
        projected_y = fy * point_prev[:, 1:2] * inv_z + cy
        grid_x = 2.0 * (projected_x + 0.5) / float(W) - 1.0
        grid_y = 2.0 * (projected_y + 0.5) / float(H) - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)
        sampled_prev = F.grid_sample(
            prevDepth, grid, mode="bilinear", padding_mode="border", align_corners=False)
        in_bounds = (
            (projected_x >= 0.0)
            & (projected_x <= float(W - 1))
            & (projected_y >= 0.0)
            & (projected_y <= float(H - 1)))
        valid = in_bounds & (expected_prev > 1e-3)
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
        mono_log_depth = -mono_inverse.clamp_min(1.0 / self.max_depth_meters).log()

        sensor_inverse, inverse_valid = self.ResampleSensorDepth(
            depth, depthValid, tuple(mono_inverse.shape[-2:]))
        sensor_log_depth, sensor_valid = self.ResampleSensorLogDepth(
            depth, depthValid, tuple(mono_inverse.shape[-2:]))
        # Both sufficient statistics use the same validity rule.  Keep the
        # conservative intersection if a backend ever rounds their area pools
        # differently.
        sensor_valid = torch.minimum(sensor_valid, inverse_valid)
        sensor_observed_valid = sensor_valid
        if self.training and self.sensor_dropout > 0.0:
            keep = (torch.rand(B, 1, 1, 1, device=rgbFeatures.device) >= self.sensor_dropout).to(rgbFeatures.dtype)
            sensor_valid = sensor_valid * keep

        disagreement = ((mono_inverse - sensor_inverse).abs() * sensor_valid) / mono_inverse.abs().clamp_min(1e-6)

        sensor_var_cue = torch.cat([sensor_inverse, sensor_valid, mono_inverse, disagreement], dim=1)
        sensor_log_var_delta = self.sensor_var_head(torch.cat([trunk_features, sensor_var_cue], dim=1))
        sensor_log_var_spatial = (
            self.sensor_log_variance + sensor_log_var_delta).clamp(-8.0, 8.0)
        virtual_logits = self.virtual_head(trunk_features)
        p_virtual = torch.sigmoid(virtual_logits)

        # Valid external depth is already calibrated metric Z and is the
        # authoritative geometry. Monocular depth only fills missing pixels.
        content_depth = mono_log_depth.exp()
        sensor_depth = sensor_log_depth.exp()
        sensor_mask = sensor_valid > 1e-6
        physical_depth = torch.where(sensor_mask, sensor_depth, content_depth)
        fused_inverse = physical_depth.reciprocal()
        physical_log_variance = torch.where(
            sensor_mask,
            sensor_log_var_spatial,
            mono_log_variance)
        sensor_reliability = sensor_valid

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
            "MetricInverseDepth": fused_inverse,
            "SensorDepthReliability": sensor_reliability,
            "SensorDepthValid": sensor_observed_valid,
            "SensorDepthValidMask": sensor_observed_valid > 1e-6,
            "SensorDepthUsed": sensor_valid,
            "ContentDepth": content_depth,
            "VirtualMask": p_virtual,
            "VirtualMaskLogits": virtual_logits,
            "SensorLogVarianceSpatial": sensor_log_var_spatial,}

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
    """RGB-D representation with one fixed, non-learned camera matrix.

    ``cameraIntrinsics`` describes the externally rectified model pixel grid.
    It is deterministic geometry configuration, never a per-frame model input.
    """
    def __init__(
        self,
        cameraIntrinsics: torch.Tensor,
        imgSize: int = 512,
        patchSize: int = 1,
        embedDim: int = 512,
        numHeads: int = 8,
        numLayers: int = 6,
        baseChannels: int = 64,
        dropout: float = 0.1,
        posDrop: float = 0.1,
        objectTokenCount: int = ModuleDim.PstObservedSlots,
        enableRecallAuxiliary: bool = False,
        recallKwargs: Optional[Dict[str, Any]] = None):
        super().__init__()

        assert embedDim % numHeads == 0, "embed_dim must be divisible by num_heads"

        self.img_size = imgSize
        self.intrinsics_reference_size = (int(imgSize), int(imgSize))
        self.patch_size = patchSize
        self.embed_dim = int(embedDim)
        self.integrated_dim = int(embedDim * 2)
        self.object_token_count = int(objectTokenCount)
        self.base_channels = baseChannels

        self.cnn_extractor = CNNFeatureExtractor(
            inChannels=3,
            baseChannels=baseChannels)
        self.perception_enhancement = PerceptionEnhancementBlock(
            stemChannels=baseChannels)

        cnn_feat_dim = baseChannels * 16

        self.depth_fusion = DepthGeometryFusion(
            featureChannels=cnn_feat_dim,
            midChannels=baseChannels * 4,
            shallowChannels=baseChannels * 2,
            fineChannels=baseChannels)
        self.dense_depth_refiner = DenseDepthRefiner(
            minDepthMeters=self.depth_fusion.min_depth_meters,
            maxDepthMeters=self.depth_fusion.max_depth_meters)
        self.depth_attention_strength = nn.Parameter(torch.tensor(-4.0))

        self.register_buffer(
            "camera_intrinsics",
            cameraIntrinsics.detach().clone(),
            persistent=False)

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
        self.patch_content_projection = nn.Conv2d(
            cnn_feat_dim,
            embedDim,
            kernel_size=patchSize,
            stride=patchSize,
            bias=False)
        # The additive branch starts as an exact no-op until trained.
        self.patch_content_gain = nn.Parameter(torch.tensor(0.0))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedDim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(p=posDrop)
        self.spp_context = SPPContextAdapter(
            inChannels=baseChannels,
            embedDim=embedDim,
            reducedChannels=min(32, max(8, baseChannels)))
        self.axial_position = AxialPositionEncoding2D(embedDim)
        self.axial_position_gain = nn.Parameter(torch.tensor(0.0))
        self.projective_topology = ProjectiveTopologyDiagnostics()

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
        layers.append(HebbianLinear(hidden_dim, hidden_dim))
        layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, embedDim, bias=True))
        layers.append(nn.GELU())
        layers.append(HebbianLinear(embedDim, embedDim))
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
            nn.Linear(embedDim // 4, 1, bias=False))

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
            nn.Linear(5, embedDim),
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
        self.object_geometry_key = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, embedDim, bias=False))
        self.object_geometry_assignment_gain = nn.Parameter(
            torch.tensor(0.1))
        self.object_competition_raw = nn.Parameter(
            torch.tensor(-math.log(4.0)))
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
        self.object_relation_norm1 = nn.LayerNorm(embedDim)
        self.object_relation_attention = nn.MultiheadAttention(
            embedDim,
            numHeads,
            dropout=dropout,
            batch_first=True)
        self.object_relation_norm2 = nn.LayerNorm(embedDim)
        self.object_relation_ff = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedDim * 2, embedDim))
        self.object_relation_output_norm = nn.LayerNorm(embedDim)
        self.object_relation_gain = nn.Parameter(torch.tensor(0.05))

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
        nn.init.zeros_(
            self.recall_heads.position_residual_camera_head.weight)
        nn.init.zeros_(
            self.recall_heads.position_residual_camera_head.bias)
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
        nn.init.zeros_(self.dense_depth_refiner.output.weight)
        nn.init.zeros_(self.dense_depth_refiner.output.bias)
        self._cortical_eval_active = False
        self._patch_content_eval_active = False
        self._spp_eval_active = False
        self._dense_refiner_eval_active = False
        self.RefreshInferenceExecutionFlags()
        self.register_load_state_dict_post_hook(
            self.RefreshInferenceExecutionFlagsAfterLoad)

    @torch.no_grad()
    def RefreshInferenceExecutionFlags(self) -> None:
        self._cortical_eval_active = bool(
            torch.count_nonzero(
                self.perception_enhancement.residual_gain.detach()).item())
        self._patch_content_eval_active = bool(
            torch.count_nonzero(self.patch_content_gain.detach()).item())
        self._spp_eval_active = bool(
            torch.count_nonzero(self.spp_context.residual_gain.detach()).item())
        self._dense_refiner_eval_active = bool(
            torch.count_nonzero(self.dense_depth_refiner.output.weight.detach()).item()
            or torch.count_nonzero(self.dense_depth_refiner.output.bias.detach()).item())
    def RefreshInferenceExecutionFlagsAfterLoad(
        self,
        module: nn.Module,
        incompatibleKeys: Any,
        ) -> None:
        del module, incompatibleKeys
        self.RefreshInferenceExecutionFlags()

    def train(self, mode: bool = True):
        result = super().train(mode)
        if not mode:
            self.RefreshInferenceExecutionFlags()
        return result

    def EnsureB(self, B: int) -> None:
        for module in self.modules():
            if isinstance(module, (HebbianConv2d, HebbianLinear)):
                module.EnsureB(B)

    def BuildAugmentedPyramid(
        self,
        frame: torch.Tensor,
        prevVisualState: Optional[VisualState],
        prevVisualValid: Optional[torch.Tensor],
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.training and not self._cortical_eval_active:
            return self.cnn_extractor(frame), {}
        stem_residual, enhancement_auxiliary = self.perception_enhancement(
            frame,
            prevVisualState,
            prevVisualValid)
        return (
            self.cnn_extractor(frame, stemResidual=stem_residual),
            enhancement_auxiliary)

    def EnhanceDepthState(
        self,
        frame: torch.Tensor,
        depthState: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        dense_state = self.dense_depth_refiner(
            frame,
            depthState["MetricDepth"],
            depthState["MetricDepthLogVariance"],
            applyCorrection=(
                self.training
                or self._dense_refiner_eval_active))
        depthState.update(dense_state)
        return depthState

    def BuildTransformerInput(
        self,
        pyramid: Dict[str, torch.Tensor],
        patchMap: torch.Tensor,
        ) -> Tuple[torch.Tensor, int, int]:
        batch, _, patch_height, patch_width = patchMap.shape
        patch_tokens = rearrange(patchMap, "b c h w -> b (h w) c")
        position = self.axial_position(
            patch_height,
            patch_width,
            patchMap.device,
            patchMap.dtype)
        patch_tokens = patch_tokens + torch.tanh(
            self.axial_position_gain).to(patchMap.dtype) * position
        class_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch)
        if self.training or self._spp_eval_active:
            class_tokens = class_tokens + self.spp_context(
                pyramid["Layer1"]).unsqueeze(1)
        tokens = self.pos_drop(torch.cat([class_tokens, patch_tokens], dim=1))
        return tokens, int(patch_height), int(patch_width)

    @staticmethod
    def BuildRotaryPositions2D(
        patchHeight: int,
        patchWidth: int,
        device: torch.device,
        ) -> torch.Tensor:
        rows, columns = torch.meshgrid(
            torch.arange(patchHeight, device=device, dtype=torch.float32) + 1.0,
            torch.arange(patchWidth, device=device, dtype=torch.float32) + 1.0,
            indexing="ij")
        patches = torch.stack([rows, columns], dim=-1).reshape(-1, 2)
        # CLS is invariant under both rotations.
        return torch.cat([torch.zeros(1, 2, device=device), patches], dim=0)

    def AddPatchContentProjection(
        self,
        feature: torch.Tensor,
        patchMap: torch.Tensor,) -> torch.Tensor:
        if not self.training and not self._patch_content_eval_active:
            return patchMap
        content = self.patch_content_projection(feature)
        return patchMap + torch.tanh(
            self.patch_content_gain).to(patchMap.dtype) * content

    @staticmethod
    def SpatialFrequencyEntropy(
        x: torch.Tensor,
        bandCount: int = 8,
        maxResolution: int = 64,
        ) -> torch.Tensor:
        if int(bandCount) <= 1:
            return x.new_zeros(int(x.size(0)))
        luminance = (
            0.2126 * x[:, 0:1]
            + 0.7152 * x[:, 1:2]
            + 0.0722 * x[:, 2:3])
        target_size = (
            min(int(luminance.size(-2)), int(maxResolution)),
            min(int(luminance.size(-1)), int(maxResolution)))
        if tuple(luminance.shape[-2:]) != target_size:
            luminance = F.adaptive_avg_pool2d(luminance, target_size)

        signal = luminance.float()
        signal = signal - signal.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.rfft2(signal, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        height = int(signal.size(-2))
        width = int(signal.size(-1))
        if height == 1 and width == 1:
            return x.new_zeros(int(signal.size(0)))
        fy = torch.fft.fftfreq(
            height,
            device=signal.device,
            dtype=signal.dtype)
        fx = torch.fft.rfftfreq(
            width,
            device=signal.device,
            dtype=signal.dtype)
        radius = torch.sqrt(
            fy.view(-1, 1).square() + fx.view(1, -1).square())
        frequency_mask = radius > 0.0
        radius_values = radius[frequency_mask]
        min_radius = radius_values.amin()
        max_radius = radius_values.amax()
        log_radius = torch.log(radius_values / min_radius)
        log_span = torch.log(max_radius / min_radius).clamp_min(1e-6)
        band_index = torch.floor(
            log_radius / log_span * float(bandCount)
        ).to(torch.long).clamp(max=int(bandCount) - 1)

        batch = int(signal.size(0))
        selected_power = power[:, 0, frequency_mask]
        expanded_index = band_index.view(1, -1).expand(batch, -1)
        band_power = signal.new_zeros(batch, int(bandCount))
        band_power.scatter_add_(1, expanded_index, selected_power)
        band_count = signal.new_zeros(int(bandCount))
        band_count.scatter_add_(
            0,
            band_index,
            torch.ones_like(radius_values))
        mean_band_power = band_power / band_count.clamp_min(1.0).view(1, -1)
        total_power = mean_band_power.sum(dim=-1, keepdim=True)
        probability = mean_band_power / total_power.clamp_min(1e-12)
        entropy = -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(float(bandCount))
        entropy = torch.where(
            total_power.squeeze(-1) > 1e-12,
            entropy,
            torch.zeros_like(entropy))
        return entropy.to(dtype=x.dtype)

    def QualityStats(self, x: torch.Tensor) -> torch.Tensor:
        x_det = x.detach()
        mean = x_det.mean(dim=(1, 2, 3))
        std = x_det.std(dim=(1, 2, 3), unbiased=False)
        gx = (x_det[..., :, 1:] - x_det[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        gy = (x_det[..., 1:, :] - x_det[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        grad = 0.5 * (gx + gy)
        clipped = ((x_det <= 0.01) | (x_det >= 0.99)).to(x_det.dtype).mean(dim=(1, 2, 3))
        spectral_entropy = self.SpatialFrequencyEntropy(x_det)
        return torch.stack([
            mean,
            std,
            grad,
            clipped,
            spectral_entropy], dim=-1)

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
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None) -> Dict[str, torch.Tensor]:
        target_log_depth, target_weight = self.depth_fusion.ResampleSensorLogDepth(
            depthTarget,
            depthTargetValid,
            tuple(visualState.Auxiliary["MonocularDepth"].shape[-2:]),)
        target_depth = target_log_depth.exp().clamp(
            self.depth_fusion.min_depth_meters,
            self.depth_fusion.max_depth_meters)
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

        if "MetricDepthFullRes" in visualState.Auxiliary:
            full_log_depth, full_weight = self.depth_fusion.ResampleSensorLogDepth(
                depthTarget,
                depthTargetValid,
                tuple(visualState.Auxiliary["MetricDepthFullRes"].shape[-2:]))
            full_target = full_log_depth.exp().clamp(
                self.depth_fusion.min_depth_meters,
                self.depth_fusion.max_depth_meters)
            full_valid = (full_weight > 1e-6).to(full_target.dtype)
            full_residual = (
                visualState.Auxiliary["MetricDepthFullRes"].clamp_min(1e-6).log()
                - full_target.clamp_min(1e-6).log())
            loss_full = self.MaskedMean(
                F.smooth_l1_loss(
                    full_residual,
                    torch.zeros_like(full_residual),
                    reduction="none"),
                full_valid)
            losses["loss_depth_full_res"] = loss_full
            total = total + 0.1 * loss_full
            full_log_variance = visualState.Auxiliary[
                "MetricDepthFullResLogVariance"]
            full_nll = (
                0.5 * torch.exp(-full_log_variance) * full_residual.square()
                + 0.5 * full_log_variance)
            loss_full_uncertainty = self.MaskedMean(full_nll, full_valid)
            losses["loss_depth_full_res_uncertainty"] = loss_full_uncertainty
            total = total + 0.02 * loss_full_uncertainty

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

        if prevVisualState is not None:
            temporal = total.new_zeros(())
            prev_depth = prevVisualState.Auxiliary["MetricDepth"].detach()
            cur_depth = visualState.Auxiliary["MetricDepth"]
            camera_intrinsics = self.CameraIntrinsicsBatch(cur_depth.size(0))
            expected_prev, sampled_prev, warp_valid = self.depth_fusion.WarpPrevDepth(
                cur_depth,
                prev_depth,
                camera_intrinsics,
                self.CameraIntrinsicsReferenceSize(),
                cameraMotion)
            temporal_valid = prevVisualValid.view(-1, 1, 1, 1)
            warp_valid = warp_valid & temporal_valid
            occlusion_margin = 0.02 + 0.02 * expected_prev
            warp_valid = warp_valid & (
                sampled_prev >= expected_prev - occlusion_margin)
            residual = (
                expected_prev.clamp_min(1e-6).log()
                - sampled_prev.clamp_min(1e-6).log()).abs()
            cur_gx, cur_gy = self.depth_fusion.SpatialGradient(
                cur_depth.clamp_min(1e-6).log())
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
        _, _, D = patchTokens.shape
        k = F.normalize(
            self.object_key(patchTokens),
            dim=-1,
            eps=1e-6)
        v = self.object_value(patchTokens)
        q = F.normalize(self.object_queries, dim=-1, eps=1e-6)
        score_scale = max(float(D) ** 0.5, 1.0)
        scores = torch.einsum("kd,bnd->bkn", q, k) * score_scale

        geometry_reliability = patchCoordinateValid.detach()
        geometry_center = (
            patchGeometry[..., :3] * geometry_reliability
        ).sum(dim=1, keepdim=True) / geometry_reliability.sum(
            dim=1,
            keepdim=True).clamp_min(1e-6)
        assignment_geometry = torch.cat([
            patchGeometry[..., :3] - geometry_center,
            patchGeometry[..., 3:],
        ], dim=-1)
        geometry_key = F.normalize(
            self.object_geometry_key(assignment_geometry),
            dim=-1,
            eps=1e-6)
        geometry_scores = torch.einsum(
            "kd,bnd->bkn",
            q,
            geometry_key) * score_scale
        scores = scores + torch.tanh(
            self.object_geometry_assignment_gain
        ) * geometry_scores * geometry_reliability.squeeze(-1).unsqueeze(1)

        content_weights = F.softmax(scores, dim=-1)
        slot_competition = F.softmax(scores, dim=1) * float(
            self.object_token_count)
        competition_gain = torch.sigmoid(self.object_competition_raw)
        weights = content_weights * (
            (1.0 - competition_gain)
            + competition_gain * slot_competition)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        tokens = torch.einsum("bkn,bnd->bkd", weights, v)
        object_geometry = torch.einsum("bkn,bnd->bkd", weights, patchGeometry)
        object_valid = torch.einsum("bkn,bnd->bkd", weights, patchCoordinateValid)
        tokens = tokens + object_valid.detach() * self.object_geometry_proj(
            object_geometry)
        tokens = self.object_post(tokens)

        # object_valid measures coordinate observability, not object existence;
        # appearance-only slots must remain able to exchange relation messages.
        relation_input = self.object_relation_norm1(tokens)
        relation, _ = self.object_relation_attention(
            relation_input,
            relation_input,
            relation_input,
            need_weights=False)
        relation_gain = torch.tanh(self.object_relation_gain)
        tokens = tokens + relation_gain * relation
        tokens = tokens + relation_gain * self.object_relation_ff(
            self.object_relation_norm2(tokens))
        tokens = self.object_relation_output_norm(tokens)
        return tokens, object_geometry, object_valid, weights

    def BuildMotionSummary(
        self,
        patchMotion: torch.Tensor,
        patchWeights: torch.Tensor,
        patchReliability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = int(patchMotion.size(-1))
        motion_tokens = self.magno_proj(F.layer_norm(patchMotion, (D,)))
        motion_weight = patchWeights * patchReliability.squeeze(-1).detach()
        motion_weight = motion_weight / motion_weight.sum(
            dim=1,
            keepdim=True).clamp_min(1e-6)
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
        currentPatchTokens: torch.Tensor) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor]:
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

        fx, fy, skew, cx, cy = self.depth_fusion.ScaleIntrinsics(
            cameraIntrinsics,
            frameSize,
            (patchHeight, patchWidth))
        fx = fx.view(B, 1, 1, 1)
        fy = fy.view(B, 1, 1, 1)
        skew = skew.view(B, 1, 1, 1)
        cx = cx.view(B, 1, 1, 1)
        cy = cy.view(B, 1, 1, 1)

        yy, xx = torch.meshgrid(
            torch.arange(patchHeight, device=cur_depth.device, dtype=cur_depth.dtype),
            torch.arange(patchWidth, device=cur_depth.device, dtype=cur_depth.dtype),
            indexing="ij",)
        xx = xx.view(1, 1, patchHeight, patchWidth)
        yy = yy.view(1, 1, patchHeight, patchWidth)
        normalized_y = (yy - cy) / fy
        point_cur = torch.cat([
            (xx - cx - skew * normalized_y) * cur_depth / fx,
            normalized_y * cur_depth,
            cur_depth], dim=1)

        point_prev = self.depth_fusion.QuaternionRotate(cameraMotion, point_cur)
        expected_prev = point_prev[:, 2:3]
        inv_z = expected_prev.clamp_min(1e-3).reciprocal()
        projected_x = (
            fx * point_prev[:, 0:1] * inv_z
            + skew * point_prev[:, 1:2] * inv_z
            + cx)
        projected_y = fy * point_prev[:, 1:2] * inv_z + cy
        grid_x = 2.0 * (projected_x + 0.5) / float(patchWidth) - 1.0
        grid_y = 2.0 * (projected_y + 0.5) / float(patchHeight) - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)

        prev_tokens = rearrange(prevVisualState.PatchTokens.detach(), "b (h w) d -> b d h w", h=patchHeight, w=patchWidth)
        warped_tokens = F.grid_sample(
            prev_tokens,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False)
        sampled_prev_depth = F.grid_sample(
            prev_depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False)
        in_bounds = (
            (projected_x >= 0.0)
            & (projected_x <= float(patchWidth - 1))
            & (projected_y >= 0.0)
            & (projected_y <= float(patchHeight - 1)))
        depth_residual = (
            expected_prev.clamp_min(1e-6).log()
            - sampled_prev_depth.clamp_min(1e-6).log()).abs()
        valid_support = (
            in_bounds
            & (expected_prev > 1e-3)
            & (sampled_prev_depth > 1e-3)
            & torch.isfinite(depth_residual))
        current_token_map = rearrange(
            currentPatchTokens.detach(),
            "b (h w) d -> b d h w",
            h=patchHeight,
            w=patchWidth)
        warped_tokens = torch.where(
            valid_support.expand(-1, warped_tokens.size(1), -1, -1),
            warped_tokens,
            current_token_map)
        depth_residual = torch.where(
            torch.isfinite(depth_residual),
            depth_residual,
            torch.zeros_like(depth_residual))
        valid = torch.where(
            valid_support,
            (-depth_residual * 3.0).exp(),
            torch.zeros_like(depth_residual))
        return (
            rearrange(warped_tokens, "b d h w -> b (h w) d"),
            rearrange(valid, "b c h w -> b (h w) c"),
            rearrange(depth_residual, "b c h w -> b (h w) c"),
            grid,)

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

    def CameraIntrinsicsBatch(self, batchSize: int) -> torch.Tensor:
        return self.camera_intrinsics.unsqueeze(0).expand(int(batchSize), -1, -1)

    def CameraIntrinsicsReferenceSize(self) -> Tuple[int, int]:
        return self.intrinsics_reference_size

    @staticmethod
    def ValidatePreviousVisualMask(
        frame: torch.Tensor,
        prevVisualValid: torch.Tensor,
        ) -> None:
        batch_size = int(frame.size(0))
        if prevVisualValid.dtype != torch.bool:
            raise TypeError(
                f"prevVisualValid must be bool, got {prevVisualValid.dtype}")
        if tuple(prevVisualValid.shape) != (batch_size,):
            raise ValueError(
                f"prevVisualValid must have shape ({batch_size},), "
                f"got {tuple(prevVisualValid.shape)}")
        if prevVisualValid.device != frame.device:
            raise ValueError(
                f"prevVisualValid must be on {frame.device}, "
                f"got {prevVisualValid.device}")

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
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        enhancementAuxiliary: Optional[Dict[str, torch.Tensor]] = None,) -> VisualState:
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
            depthState,
            patchHeight,
            patchWidth,
            cameraIntrinsics=cameraIntrinsics,
            frameSize=self.CameraIntrinsicsReferenceSize())
        object_tokens, object_geometry, object_coordinate_valid, object_patch_weights = self.BuildObjectTokens(
            patch_tokens, patchGeometry=patch_geometry, patchCoordinateValid=patch_coordinate_valid)

        geometry_reliability = patch_coordinate_valid.detach()
        geometry_weight = patch_weights * geometry_reliability.squeeze(-1)
        geometry_weight = geometry_weight / geometry_weight.sum(dim=1, keepdim=True)
        geometry_summary = self.geometry_summary_proj((patch_geometry * geometry_weight.unsqueeze(-1)).sum(dim=1))
        shared = self.cortical_proj(corrected_integrated)
        ventral_feat = self.ventral_proj(shared)

        diagnostic_factory = {
            "device": patch_tokens.device,
            "dtype": torch.float32}
        topology_auxiliary = {
            "RigidPatchFlow": torch.zeros(
                patch_tokens.size(0), 2, patchHeight, patchWidth,
                **diagnostic_factory),
            "WarpJacobianDet": torch.ones(
                patch_tokens.size(0), 1, patchHeight, patchWidth,
                **diagnostic_factory),
            "WarpJacobianSigmaMin": torch.ones(
                patch_tokens.size(0), 1, patchHeight, patchWidth,
                **diagnostic_factory),
            "WarpJacobianSigmaMax": torch.ones(
                patch_tokens.size(0), 1, patchHeight, patchWidth,
                **diagnostic_factory),
            "WarpTopologyValid": torch.zeros(
                patch_tokens.size(0), 1, patchHeight, patchWidth,
                **diagnostic_factory),
            "WarpFoldPenalty": torch.zeros((), **diagnostic_factory)}

        camera_motion_from_prev = cameraMotion

        if prevVisualState is not None:
            previous_valid = prevVisualValid.view(-1)
            warp_row_valid = previous_valid
            warp_row_mask = warp_row_valid.view(-1, 1, 1)
            warped_prev_tokens, warp_valid, warp_depth_residual, warp_grid = self.WarpPrevPatchTokens(
                prevVisualState,
                depthState,
                patchHeight,
                patchWidth,
                cameraIntrinsics,
                self.CameraIntrinsicsReferenceSize(),
                camera_motion_from_prev,
                patch_tokens)
            topology_domain = rearrange(
                warp_valid > 0.0,
                "b (h w) c -> b c h w",
                h=patchHeight,
                w=patchWidth)
            topology_domain = (
                topology_domain
                & warp_row_valid.view(-1, 1, 1, 1))
            topology_auxiliary = self.projective_topology(
                warp_grid,
                domainMask=topology_domain)
            topology_valid = rearrange(
                topology_auxiliary["WarpTopologyValid"] > 0.0,
                "b c h w -> b (h w) c")
            warp_valid = torch.where(
                topology_valid,
                warp_valid,
                torch.zeros_like(warp_valid))
            warped_prev_tokens = torch.where(
                warp_row_mask,
                warped_prev_tokens,
                patch_tokens.detach())
            warp_valid = torch.where(
                warp_row_mask, warp_valid, torch.zeros_like(warp_valid))
            warp_depth_residual = torch.where(
                warp_row_mask,
                warp_depth_residual,
                torch.zeros_like(warp_depth_residual))
            topology_auxiliary["WarpTopologyValid"] = torch.where(
                warp_row_valid.view(-1, 1, 1, 1),
                topology_auxiliary["WarpTopologyValid"],
                torch.zeros_like(topology_auxiliary["WarpTopologyValid"]))
            patch_motion = patch_tokens - warped_prev_tokens
        else:
            warped_prev_tokens = torch.zeros_like(patch_tokens)
            warp_valid = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
            warp_depth_residual = patch_tokens.new_zeros(patch_tokens.size(0), patch_tokens.size(1), 1)
            patch_motion = torch.zeros_like(patch_tokens)

        motion_reliability = geometry_reliability * warp_valid
        magno_summary, patch_motion_tokens, motion_weights = self.BuildMotionSummary(
            patch_motion,
            patch_weights,
            motion_reliability)
        object_motion = torch.einsum(
            "bkn,bnd->bkd",
            object_patch_weights,
            patch_motion_tokens * motion_reliability)
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
        position_residual_camera = semantic_nodes.pop(
            "position_residual_camera")
        orientation_camera = semantic_nodes.pop("orientation_camera")
        object_geometry = torch.cat([
            object_geometry[..., :3] + position_residual_camera,
            object_geometry[..., 3:],
        ], dim=-1)
        semantic_nodes["pose_camera"] = torch.cat([
            object_geometry[..., :3],
            orientation_camera], dim=-1)
        topology_auxiliary = {
            name: value if name == "WarpFoldPenalty" else value.detach()
            for name, value in topology_auxiliary.items()}
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
                "PatchMotionReliability": motion_reliability.detach(),
                "PatchMotionWeights": motion_weights.detach(),
                "PatchMotionDepthResidual": warp_depth_residual.detach(),
                "WarpedPrevPatchTokens": warped_prev_tokens.detach(),
                "WarpPrevPatchValid": warp_valid.detach(),
                "CameraMotionFromPrev": camera_motion_from_prev.detach(),
                "PatchGridShape": torch.tensor(
                    [patchHeight, patchWidth],
                    dtype=torch.long),
                "DorsalReliabilityGate": geometry_confidence.detach(),
                **topology_auxiliary,
                **({} if enhancementAuxiliary is None else enhancementAuxiliary)},)

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,) -> VisualState:
        # x: [B, 3, H, W]
        frame = x
        batch_size = int(frame.size(0))
        self.ValidatePreviousVisualMask(frame, prevVisualValid)
        self.EnsureB(batch_size)
        pyramid, enhancement_auxiliary = self.BuildAugmentedPyramid(
            frame,
            prevVisualState,
            prevVisualValid)
        feat, depth_state = self.depth_fusion(
            pyramid["Deep"],
            pyramid["Layer3"],
            pyramid["Layer2"],
            pyramid["Layer1"],
            depth=depth,
            depthValid=depthValid)
        depth_state = self.EnhanceDepthState(frame, depth_state)

        feat = self.cnn_feat_adapter(feat)

        patch_map = self.AddPatchContentProjection(
            feat,
            self.patch_adapter(feat))  # [B, embed_dim, Ph, Pw]
        x, Ph, Pw = self.BuildTransformerInput(pyramid, patch_map)
        camera_intrinsics = self.CameraIntrinsicsBatch(batch_size)
        depth_attention_bias = self.BuildDepthAttentionBias(
            depth_state,
            Ph,
            Pw,
            cameraIntrinsics=camera_intrinsics,
            frameSize=self.CameraIntrinsicsReferenceSize())
        rotary_positions = self.BuildRotaryPositions2D(
            Ph,
            Pw,
            x.device)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(
                x,
                srcMask=depth_attention_bias,
                rotaryPositions2D=rotary_positions)
            x = self.token_adapters[i](x)

        return self.AssembleVisualState(
            frame, x, depth_state, camera_intrinsics, Ph, Pw, topDownContext,
            prevVisualState, cameraMotion, prevVisualValid,
            enhancementAuxiliary=enhancement_auxiliary)

    def ComputePerceptionLoss(
        self,
        visualState: VisualState,
        depthTarget: torch.Tensor,
        depthTargetValid: torch.Tensor,
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        ) -> torch.Tensor:
        loss = visualState.IntegratedFeat.new_zeros(())

        obj = visualState.ObjectTokens
        obj_n = F.normalize(obj, dim=-1, eps=1e-6)
        sim = torch.matmul(obj_n, obj_n.transpose(1, 2))
        eye = torch.eye(obj.size(1), device=obj.device, dtype=torch.bool).unsqueeze(0)
        occupancy = (
            F.softmax(visualState.SemanticNodes["node_logits"], dim=-1)[..., 1]
            * visualState.Auxiliary["ObjectGeometryValid"].squeeze(-1)
        ).detach()
        pair_weight = occupancy.unsqueeze(2) * occupancy.unsqueeze(1)
        pair_weight = pair_weight.masked_fill(eye, 0.0)
        diversity = (
            sim.pow(2) * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1.0)
        loss = loss + 0.05 * diversity

        if prevVisualState is not None:
            motion_target = (visualState.VentralFeat - prevVisualState.VentralFeat.detach()).detach()
            motion_pred = self.motion_decoder(visualState.MotionToken)
            motion_loss = F.smooth_l1_loss(
                motion_pred, motion_target, reduction="none").flatten(1).mean(dim=1)
            motion_valid = prevVisualValid.view(-1)
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
        loss = loss + 0.001 * visualState.Auxiliary.get(
            "WarpFoldPenalty",
            loss.new_zeros(()))

        return loss

    def InitWeights(self):
        for name, m in self.named_modules():
            if isinstance(m, SheafGaugeConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
                nn.init.zeros_(m.bias)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        for module in self.modules():
            if isinstance(module, (HebbianConv2d, HebbianLinear)):
                module.ResetHebbianMemory(doneMask=doneMask)



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
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,) -> VisualState:
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
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,) -> torch.Tensor:
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
        prevVisualValid = kwargs["prevVisualValid"]
        cameraMotion = kwargs["cameraMotion"]
        depth = kwargs["depth"]
        depth_valid = kwargs["depthValid"]
        batch_size = int(frame.size(0))
        self.base.ValidatePreviousVisualMask(frame, prevVisualValid)
        self.base.EnsureB(batch_size)
        camera_intrinsics = self.base.CameraIntrinsicsBatch(batch_size)

        pyramid, enhancement_auxiliary = self.base.BuildAugmentedPyramid(
            frame,
            prevVisualState,
            prevVisualValid)
        feat, depth_state = self.base.depth_fusion(
            pyramid["Deep"],
            pyramid["Layer3"],
            pyramid["Layer2"],
            pyramid["Layer1"],
            depth=depth,
            depthValid=depth_valid)
        depth_state = self.base.EnhanceDepthState(frame, depth_state)
        
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
        patches = self.base.AddPatchContentProjection(feat, patches)

        xTok, Ph, Pw = self.base.BuildTransformerInput(pyramid, patches)
        depth_attention_bias = self.base.BuildDepthAttentionBias(
            depth_state,
            Ph,
            Pw,
            cameraIntrinsics=camera_intrinsics,
            frameSize=self.base.CameraIntrinsicsReferenceSize())
        rotary_positions = self.base.BuildRotaryPositions2D(
            Ph,
            Pw,
            xTok.device)

        for i, layer in enumerate(self.base.transformer_layers):
            xTok = layer(
                xTok,
                srcMask=depth_attention_bias,
                rotaryPositions2D=rotary_positions)
            
            xTok = self.base.token_adapters[i](xTok)
            
            deltaTok2D = deltasPerLayer[i].get("token", None)
            if deltaTok2D is not None:
                xTok = xTok + (xTok @ deltaTok2D.t())

        return self.base.AssembleVisualState(
            frame, xTok, depth_state, camera_intrinsics, Ph, Pw, topDownContext,
            prevVisualState, cameraMotion, prevVisualValid,
            enhancementAuxiliary=enhancement_auxiliary)

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
        self.position_residual_camera_head = nn.Linear(hidden, 3)
        self.orientation_camera_head = nn.Linear(hidden, 4)
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
        nn.init.zeros_(self.position_residual_camera_head.weight)
        nn.init.zeros_(self.position_residual_camera_head.bias)

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
            "position_residual_camera": (
                self.position_residual_camera_head(node_h)),
            "orientation_camera": F.normalize(
                self.orientation_camera_head(node_h).float(),
                dim=-1,
                eps=1e-6).to(node_h.dtype),
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

        patch_grid_shape = visualState.Auxiliary.get("PatchGridShape")
        if patch_grid_shape is None:
            patch_count = int(visualState.PatchTokens.size(1))
            patch_height = int(math.sqrt(patch_count))
            patch_grid_shape = torch.tensor(
                [patch_height, patch_count // max(patch_height, 1)],
                dtype=torch.long)

        return {
            **node_out,
            "patch_grid_shape": patch_grid_shape,
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

    @staticmethod
    def PatchGridShape(
        patchGridShape: Union[torch.Tensor, Tuple[int, int], List[int]],
        numPatches: int,) -> Tuple[int, int]:
        if isinstance(patchGridShape, torch.Tensor):
            values = patchGridShape.detach().cpu().view(-1).tolist()
        else:
            values = list(patchGridShape)
        if len(values) != 2:
            raise ValueError("patch grid shape must contain height and width")
        height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0 or height * width != int(numPatches):
            raise ValueError("patch grid shape does not match patch token count")
        return height, width

    def SemanticTarget(
        self,
        targets: Dict[str, torch.Tensor],
        numPatches: int,
        patchGridShape) -> torch.Tensor:
        tensor = targets["semantic_segmentation"]
        grid = self.PatchGridShape(patchGridShape, numPatches)
        down = F.interpolate(tensor.unsqueeze(1).float(), size=grid, mode="nearest")
        return down[:, 0].reshape(tensor.size(0), numPatches).long()

    def DepthTarget(
        self,
        targets: Dict[str, torch.Tensor],
        numPatches: int,
        patchGridShape) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = targets["depth"]
        valid = targets["depth_valid"].to(tensor.dtype)
        grid = self.PatchGridShape(patchGridShape, numPatches)
        inverse = torch.where(valid > 0.0, tensor.clamp_min(1e-6).reciprocal(), torch.zeros_like(tensor))
        weight = F.adaptive_avg_pool2d(valid, grid)
        pooled_inverse = F.adaptive_avg_pool2d(inverse, grid) / weight.clamp_min(1e-6)
        target = pooled_inverse.clamp_min(1e-6).reciprocal()
        target_valid = weight > 0.0
        target = target[:, 0].reshape(tensor.size(0), numPatches)
        target_valid = target_valid[:, 0].reshape(tensor.size(0), numPatches)
        return target, target_valid

    def NormalTarget(
        self,
        targets: Dict[str, torch.Tensor],
        numPatches: int,
        patchGridShape,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = targets["normal"]
        grid = self.PatchGridShape(patchGridShape, numPatches)
        valid = targets["normal_valid"].to(tensor.dtype)
        weighted = F.adaptive_avg_pool2d(tensor * valid, grid)
        valid_weight = F.adaptive_avg_pool2d(valid, grid)
        normal = weighted / valid_weight.clamp_min(1e-6)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        return (
            rearrange(normal, "b c h w -> b (h w) c"),
            valid_weight[:, 0].reshape(tensor.size(0), numPatches) > 0.0,)

    def NodeMaskTarget(
        self,
        nodeMasks: torch.Tensor,
        gtIndex: torch.Tensor,
        numPatches: int,
        patchGridShape) -> torch.Tensor:
        grid = self.PatchGridShape(patchGridShape, numPatches)
        masks = nodeMasks[gtIndex]
        return F.adaptive_max_pool2d(masks.unsqueeze(1).float(), grid).flatten(1)

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
                recallOut["node_mask_logits"].size(-1),
                recallOut["patch_grid_shape"])
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
        patch_sem = self.SemanticTarget(
            targets,
            recallOut["patch_class_logits"].size(1),
            recallOut["patch_grid_shape"])
        logits = recallOut["patch_class_logits"].reshape(-1, recallOut["patch_class_logits"].size(-1))
        add("patch_semantic", F.cross_entropy(logits, patch_sem.reshape(-1)))
        patch_depth, patch_depth_valid = self.DepthTarget(
            targets,
            recallOut["patch_depth"].size(1),
            recallOut["patch_grid_shape"])
        error = F.smooth_l1_loss(recallOut["patch_depth"], patch_depth, reduction="none")
        valid = patch_depth_valid.to(error.dtype)
        add("patch_depth", (error * valid).sum() / valid.sum().clamp_min(1.0))
        patch_normal, patch_normal_valid = self.NormalTarget(
            targets,
            recallOut["patch_normal"].size(1),
            recallOut["patch_grid_shape"])
        normal_error = 1.0 - (recallOut["patch_normal"] * patch_normal).sum(dim=-1).clamp(-1.0, 1.0)
        normal_valid = patch_normal_valid.to(normal_error.dtype)
        add("patch_normal", (
            normal_error * normal_valid
        ).sum() / normal_valid.sum().clamp_min(1.0))
        losses["loss"] = total
        return losses


class PerceptionTrainer(nn.Module):
    def __init__(
        self,
        cameraIntrinsics: torch.Tensor,
        recallLossKwargs: Optional[Dict[str, Any]] = None,
        **extractorKwargs: Any,):
        super().__init__()
        extractorKwargs = dict(extractorKwargs)
        extractorKwargs["enableRecallAuxiliary"] = True
        self.extractor = PerceiveExtractor(
            cameraIntrinsics=cameraIntrinsics,
            **extractorKwargs)
        recallLossKwargs = {} if recallLossKwargs is None else dict(recallLossKwargs)
        self.recall_loss = PerceptionRecallLoss(**recallLossKwargs)

    @property
    def recall_heads(self) -> PerceptionRecallHeads:
        return self.extractor.recall_heads

    def forward(
        self,
        x: torch.Tensor,
        topDownContext: TopDownContext,
        targets: Dict[str, torch.Tensor],
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,) -> Dict[str, Any]:
        visual_state = self.extractor(
            x,
            topDownContext=topDownContext,
            prevVisualState=prevVisualState,
            depth=depth,
            depthValid=depthValid,
            cameraMotion=cameraMotion,
            prevVisualValid=prevVisualValid)
        recall_out = self.recall_heads(visual_state)
        loss_self = self.extractor.ComputePerceptionLoss(
            visual_state,
            depthTarget=targets["depth"],
            depthTargetValid=targets["depth_valid"],
            cameraMotion=cameraMotion,
            prevVisualValid=prevVisualValid,
            prevVisualState=prevVisualState)
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

    def MakeCameraIntrinsics(self, imageSize: int) -> torch.Tensor:
        focal_length = 0.75 * float(imageSize)
        principal_point = 0.5 * (float(imageSize) - 1.0)
        return torch.tensor([
            [focal_length, 0.0, principal_point],
            [0.0, focal_length, principal_point],
            [0.0, 0.0, 1.0],
        ], device=self.device)

    @staticmethod
    def CameraTemporalInputs(
        reference: torch.Tensor,
        previousVisualState: Optional[VisualState] = None,
        ) -> Dict[str, torch.Tensor]:
        camera_motion = reference.new_zeros(
            reference.size(0), ModuleDim.CameraMotionDim)
        camera_motion[:, 3] = 1.0
        previous_valid = torch.full(
            (reference.size(0),),
            previousVisualState is not None,
            device=reference.device,
            dtype=torch.bool)
        return {
            "cameraMotion": camera_motion,
            "prevVisualValid": previous_valid}

    def MakeRegressionModel(
        self,
        imgSize: int = 32,
        *,
        enableRecallAuxiliary: bool = True,
        ) -> PerceiveExtractor:
        return PerceiveExtractor(
            cameraIntrinsics=self.MakeCameraIntrinsics(imgSize),
            imgSize=imgSize,
            patchSize=1,
            embedDim=32,
            numHeads=4,
            numLayers=1,
            baseChannels=8,
            objectTokenCount=4,
            enableRecallAuxiliary=enableRecallAuxiliary).to(self.device)

    def RunRegressionCheck(self, name: str, check) -> bool:
        try:
            check()
            print(f"{name} passed.")
            return True
        except AssertionError as e:
            print(f"{name} failed: {e}")
            return False
        except Exception as e:
            print(f"{name} error: {e}")
            return False

    def PerceptionForward(self, model, x: torch.Tensor, prevVisualState: Optional[VisualState] = None, predictedVisual: Optional[Dict[str, torch.Tensor]] = None) -> VisualState:
        B, _, H, W = x.shape
        depth = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        return model(
            x,
            topDownContext=self.MakeTopDownContext(model, int(x.size(0)), x.dtype, predictedVisual),
            depth=depth,
            depthValid=torch.ones_like(depth, dtype=torch.bool),
            prevVisualState=prevVisualState,
            **self.CameraTemporalInputs(x, prevVisualState))

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
            "normal_valid": depthValid & (normal.norm(dim=1, keepdim=True) > 0.5),
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
            "contact_point_camera": torch.zeros(B, nodes, 3, device=self.device)}

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
            conv = HebbianConv2d(
                inChannels=3,
                outChannels=16,
                kernelSize=3,
                stride=1,
                padding=1).to(self.device)
            x = torch.randn(4, 3, 32, 32, device=self.device)
            conv.EnsureB(int(x.size(0)))
            expected = conv.conv(x)
            y = conv(x)
            assert y.shape == (4, 16, 32, 32), f"Output shape does not match: {y.shape}"
            assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)
            assert torch.count_nonzero(conv.hebb_memory).item() > 0
            assert "hebb_memory" not in conv.state_dict()
            y_next = conv(x)
            assert not torch.equal(y_next, y)
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
            lin = HebbianLinear(
                inFeatures=32,
                outFeatures=64).to(self.device)
            x = torch.randn(5, 32, device=self.device)
            lin.EnsureB(int(x.size(0)))
            expected = F.linear(x, lin.weight, lin.bias)
            y = lin(x)
            assert y.shape == (5, 64), f"Output shape does not match: {y.shape}"
            assert torch.allclose(y, expected, atol=1e-7, rtol=1e-6)
            assert torch.count_nonzero(lin.hebb_memory).item() > 0
            assert "hebb_memory" not in lin.state_dict()
            y_next = lin(x)
            assert not torch.equal(y_next, y)
            lin.ResetHebbianMemory()
            print("HebbianLinear test passed.")
            return True
        except AssertionError as e:
            print(f"HebbianLinear test failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianLinear test error: {e}")
            return False

    def TestHebbianDecaySignAndCorrelation(self):
        def check():
            conv = HebbianConv2d(2, 1, 1).to(self.device).eval()
            conv.EnsureB(1)
            conv_input = torch.zeros(1, 2, 1, 1, device=self.device)
            conv_input[:, 0] = 1.0
            with torch.no_grad():
                conv.conv.weight.zero_()
                conv.conv.weight[0, 0, 0, 0] = 500.0
                conv.hebb_memory.zero_()
                conv.hebb_memory[0, 0, 1, 0, 0] = 0.5
                _ = conv(conv_input)

            expected_conv_memory = (
                0.5
                * 0.995
                * math.exp(-(5e-6 / 0.995) * 500.0 ** 2))
            conv_memory = float(conv.hebb_memory[0, 0, 1, 0, 0].item())
            assert conv_memory > 0.0
            assert math.isclose(
                conv_memory,
                expected_conv_memory,
                rel_tol=1e-5,
                abs_tol=1e-7)

            linear = HebbianLinear(1, 1).to(self.device).eval()
            linear.EnsureB(1)
            zero_input = torch.zeros(1, 1, device=self.device)
            with torch.no_grad():
                linear.weight.zero_()
                linear.bias.fill_(200.0)
                linear.hebb_memory.fill_(0.5)
                _ = linear(zero_input)

            expected_linear_memory = (
                0.5
                * 0.995
                * math.exp(-(5e-5 / 0.995) * 200.0 ** 2))
            linear_memory = float(linear.hebb_memory.item())
            assert linear_memory > 0.0
            assert math.isclose(
                linear_memory,
                expected_linear_memory,
                rel_tol=1e-5,
                abs_tol=1e-7)

            signed_input = torch.ones(1, 1, device=self.device)
            with torch.no_grad():
                linear.bias.fill_(-1.0)
                linear.hebb_memory.fill_(1e-6)
                signed_output = linear(signed_input)

            assert float(signed_output.item()) < 0.0
            assert float(linear.hebb_memory.item()) < 0.0

        return self.RunRegressionCheck(
            "HebbianDecaySignAndCorrelation",
            check)

    def TestPerceiveExtractor(self):
        try:
            model = PerceiveExtractor(
                cameraIntrinsics=self.MakeCameraIntrinsics(512),
                imgSize=512,
                patchSize=1,
                embedDim=512,
                numHeads=8,
                numLayers=6).to(self.device)
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
                cameraIntrinsics=self.MakeCameraIntrinsics(img_size),
                imgSize=img_size,
                patchSize=1,
                embedDim=embed_dim,
                numHeads=8,
                numLayers=6).to(self.device)
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
                cameraIntrinsics=self.MakeCameraIntrinsics(64),
                imgSize=64,
                patchSize=1,
                embedDim=512,
                numHeads=8,
                numLayers=2,
                baseChannels=16).to(self.device)
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
                state1 = model(
                    x,
                    topDownContext=ctx,
                    depth=depth,
                    depthValid=torch.ones_like(depth, dtype=torch.bool),
                    prevVisualState=state0,
                    **self.CameraTemporalInputs(x, state0))

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
            intrinsics = self.MakeCameraIntrinsics(64)
            model = PerceiveExtractor(
                cameraIntrinsics=intrinsics,
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16).to(self.device)
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
            context = self.MakeTopDownContext(model, B, frames.dtype)

            visual_state = model(
                frames,
                topDownContext=context,
                depth=sensor_depth,
                depthValid=sensor_valid,
                **self.CameraTemporalInputs(frames))
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
                depthTargetValid=target_valid,
                **self.CameraTemporalInputs(frames))
            model.zero_grad(set_to_none=True)
            depth_losses["loss"].backward()
            depth_grad = model.depth_fusion.monocular_head[-1].weight.grad
            assert depth_grad is not None and bool(torch.isfinite(depth_grad).all().item())

            trainer = PerceptionTrainer(
                cameraIntrinsics=intrinsics,
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16).to(self.device)
            assert trainer.recall_heads.enable_auxiliary and hasattr(trainer.recall_heads, "global_trunk")
            targets = self.MakeSyntheticTargets(
                frames, target_depth, target_valid, target_normal, target_semantic)
            assert "camera_pose_world" not in targets and "camera_motion" not in targets
            train_out = trainer(
                frames,
                topDownContext=self.MakeTopDownContext(trainer.extractor, B, frames.dtype),
                depth=sensor_depth,
                depthValid=sensor_valid,
                targets=targets,
                **self.CameraTemporalInputs(frames))
            assert bool((train_out["visual_state"].Auxiliary["ObjectGeometryValid"] > 0).any().item())
            assert tuple(train_out["visual_state"].Auxiliary["ObjectMotion"].shape) == (B, 16, trainer.extractor.embed_dim)
            assert train_out["recall_out"]["node_logits"] is train_out["visual_state"].SemanticNodes["node_logits"]
            assert "loss_node" in train_out and "loss_patch_normal" in train_out
            runtime_model = PerceiveExtractor(
                cameraIntrinsics=intrinsics,
                imgSize=64,
                patchSize=1,
                embedDim=64,
                numHeads=8,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=16).to(self.device)
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
                semantic_nodes = {
                    **heads.ForwardNodes(objects),
                    **heads.ForwardScene(integrated, ventral, dorsal)}
                position_residual_camera = semantic_nodes.pop(
                    "position_residual_camera")
                orientation_camera = semantic_nodes.pop("orientation_camera")
                semantic_nodes["pose_camera"] = torch.cat([
                    objects.new_zeros(B, K, 3)
                    + position_residual_camera,
                    orientation_camera], dim=-1)
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
                    SemanticNodes=semantic_nodes)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
            with torch.no_grad():
                base.perception_enhancement.residual_gain.fill_(0.1)
            base.eval()
            wrapper = PerceptionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.eval()

            x = torch.randn(3, 3, 64, 64, device=self.device)
            with torch.no_grad():
                base.ResetHebbianMemory()
                y_base = self.PerceptionForward(base, x)
                base.ResetHebbianMemory()
                y_wrap = self.PerceptionForward(wrapper, x)

            for name in (
                "IntegratedFeat",
                "PatchTokens",
                "MotionToken",
                "QualityToken",):
                max_abs = (
                    getattr(y_base, name) - getattr(y_wrap, name)
                ).abs().max().item()
                assert max_abs < 1e-6, (
                    f"Wrapper {name} differs when ranks=0: "
                    f"max_abs={max_abs:.3e}")
            for key in ("MetricDepthFullRes", "CorticalEnergy"):
                max_abs = (
                    y_base.Auxiliary[key] - y_wrap.Auxiliary[key]
                ).abs().max().item()
                assert max_abs < 1e-6, (
                    f"Wrapper auxiliary {key} differs when ranks=0: "
                    f"max_abs={max_abs:.3e}")
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(img_size), imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
                base.ResetHebbianMemory()
                y0 = self.PerceptionForward(base, x_chk)
                base.ResetHebbianMemory()
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(img_size), imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(img_size), imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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

            invalid_mask_rejected = False
            try:
                wrapper(
                    x,
                    topDownContext=self.MakeTopDownContext(wrapper, 8),
                    depth=torch.ones(
                        8, 1, img_size, img_size,
                        device=self.device),
                    depthValid=torch.ones(
                        8, 1, img_size, img_size,
                        device=self.device,
                        dtype=torch.bool),
                    cameraMotion=self.CameraTemporalInputs(x)[
                        "cameraMotion"],
                    prevVisualValid=torch.zeros(
                        8,
                        device=self.device))
            except TypeError:
                invalid_mask_rejected = True
            assert invalid_mask_rejected

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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(img_size), imgSize=img_size, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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

    def LossDecreasesWithHebbian(self, steps: int = 80):
        try:
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            assert tail_mean <= 0.5 * start, f"Hebbian insufficient convergence: start={start:.4f}, tail_mean={tail_mean:.4f}"
            print(f"LossDecreasesWithHebbian passed. start={start:.4f} -> end={end:.4f}")
            return True
        except AssertionError as e:
            print(f"LossDecreasesWithHebbian failed: {e}")
            return False
        except Exception as e:
            print(f"LossDecreasesWithHebbian error: {e}")
            return False

    def HebbianMemoryLifecycle(self):
        try:
            conv = HebbianConv2d(
                3, 8, 3, stride=1, padding=1).to(self.device)
            x = torch.randn(4, 3, 32, 32, device=self.device)
            conv.EnsureB(int(x.size(0)))
            n0 = conv.hebb_memory.norm().item()
            for _ in range(3):
                _ = conv(x)
            n1 = conv.hebb_memory.norm().item()
            assert n1 > n0 + 1e-12, f"Conv Hebbian memory no growth: before={n0:.3e}, after={n1:.3e}"
            conv.ResetHebbianMemory()
            n2 = conv.hebb_memory.norm().item()
            assert n2 < 1e-12, f"Conv Hebbian memory unclear zero: now={n2:.3e}"

            lin = HebbianLinear(32, 16).to(self.device)
            z = torch.randn(6, 32, device=self.device)
            lin.EnsureB(int(z.size(0)))
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
            base = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8, numLayers=2, baseChannels=16).to(self.device)
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
            model = PerceiveExtractor(cameraIntrinsics=self.MakeCameraIntrinsics(64), imgSize=64, patchSize=1, embedDim=64, numHeads=8,numLayers=2, baseChannels=16).to(self.device)
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

    def TestEvalDepthLossAuxiliarySchema(self):
        def check():
            model = self.MakeRegressionModel(64).eval()
            frame = torch.rand(1, 3, 64, 64, device=self.device)
            depth = torch.ones(1, 1, 64, 64, device=self.device)
            valid = torch.ones_like(depth, dtype=torch.bool)
            visual = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 1),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            assert "VirtualTarget" in visual.Auxiliary
            assert "EdgeAwareSmoothness" in visual.Auxiliary
            loss = model.ComputePerceptionLoss(
                visual,
                depthTarget=depth,
                depthTargetValid=valid,
                **self.CameraTemporalInputs(frame))
            assert torch.isfinite(loss)
        return self.RunRegressionCheck("EvalDepthLossAuxiliarySchema", check)

    def TestQualityStatisticsDType(self):
        def check():
            model = self.MakeRegressionModel(64).half()
            frame = torch.rand(
                2,
                3,
                64,
                64,
                device=self.device,
                dtype=torch.float16)
            stats = model.QualityStats(frame)
            assert stats.shape == (2, 5)
            assert stats.dtype == torch.float16
            assert model.quality_proj(stats).dtype == torch.float16
        return self.RunRegressionCheck("QualityStatisticsDType", check)

    def TestSpatialFrequencyEntropyAndCorticalSelectivity(self):
        def check():
            size = 64
            coordinate = torch.arange(
                size,
                device=self.device,
                dtype=torch.float32).view(1, 1, 1, size)
            sinusoid = (
                0.5
                + 0.4 * torch.cos(2.0 * math.pi * coordinate / 8.0)
            ).expand(1, 1, size, size)
            constant = torch.full_like(sinusoid, 0.5)
            noise = torch.rand_like(sinusoid)
            frames = torch.cat([
                constant,
                sinusoid,
                noise,
                0.5 * sinusoid,
            ], dim=0).expand(-1, 3, -1, -1)
            entropy = PerceiveExtractor.SpatialFrequencyEntropy(frames)
            assert float(entropy[0]) < 1e-6
            assert float(entropy[1]) < 0.1
            assert float(entropy[2]) > float(entropy[1]) + 0.5
            assert torch.allclose(entropy[1], entropy[3], atol=1e-5)
            singleton_entropy = PerceiveExtractor.SpatialFrequencyEntropy(
                torch.ones(2, 3, 1, 1, device=self.device))
            assert torch.equal(
                singleton_entropy,
                torch.zeros_like(singleton_entropy))
            one_band_entropy = PerceiveExtractor.SpatialFrequencyEntropy(
                frames,
                bandCount=1)
            assert torch.equal(
                one_band_entropy,
                torch.zeros_like(one_band_entropy))

            vision = CorticalEarlyVision(outChannels=8).to(self.device)
            stationary_input = torch.randn(
                1,
                1,
                80,
                80,
                device=self.device)
            stationary_even, stationary_odd = vision.MultiscaleQuadrature(
                stationary_input)
            shifted_even, shifted_odd = vision.MultiscaleQuadrature(
                torch.roll(stationary_input, shifts=1, dims=-1))
            interior = (..., slice(31, -31), slice(31, -31))
            assert torch.equal(
                shifted_even[interior],
                torch.roll(stationary_even, shifts=1, dims=-1)[interior])
            assert torch.equal(
                shifted_odd[interior],
                torch.roll(stationary_odd, shifts=1, dims=-1)[interior])

            calibration_size = 96
            calibration_x = torch.arange(
                calibration_size,
                device=self.device,
                dtype=torch.float32).view(1, 1, 1, calibration_size)
            calibration_input = torch.cat([
                torch.cos(
                    2.0 * math.pi * calibration_x
                    / (4.0 * scale)
                ).expand(1, 1, calibration_size, calibration_size)
                for scale in vision.frequency_scales
            ], dim=0)
            calibration_even, calibration_odd = vision.MultiscaleQuadrature(
                calibration_input)
            calibration_energy = vision.QuadratureAmplitude(
                calibration_even,
                calibration_odd)
            matched_response = torch.stack([
                calibration_energy[
                    index,
                    index,
                    0,
                    31:-31,
                    31:-31].mean()
                for index in range(len(vision.frequency_scales))
            ])
            assert (
                matched_response.max() / matched_response.min()
            ) < 1.001
            _, blank_auxiliary = vision(frames[0:1], None, None)
            assert float(blank_auxiliary["CorticalEnergy"].abs().max()) < 1e-6
            luminance = vision.AntiAliasDownsample(sinusoid)
            even, odd = vision.MultiscaleQuadrature(luminance)
            assert even.shape == (1, 3, vision.orientations, 32, 32)
            energy = vision.QuadratureAmplitude(even, odd)
            orientation_response = energy.mean(dim=(0, 1, 3, 4))
            assert int(orientation_response.argmax().item()) == 0
            assert orientation_response[0] > 4.0 * orientation_response[3]
            phase_congruency, _ = vision.MultiscalePhaseStatistics(
                even,
                odd,
                energy)
            assert float(phase_congruency.max()) <= 1.0 + 1e-4
            orientation_coherence = vision.OrientationCoherence(energy)
            assert float(orientation_coherence.min()) >= 0.0
            assert float(orientation_coherence.max()) <= 1.0 + 1e-4

            structured_luminance = calibration_input[1:2]
            noise_luminance = torch.randn_like(structured_luminance)

            def structure_score(value: torch.Tensor) -> torch.Tensor:
                value_even, value_odd = vision.MultiscaleQuadrature(value)
                value_energy = vision.QuadratureAmplitude(
                    value_even,
                    value_odd)
                value_phase, value_entropy = (
                    vision.MultiscalePhaseStatistics(
                        value_even,
                        value_odd,
                        value_energy))
                value_orientation = vision.OrientationCoherence(
                    value_energy)
                return (
                    value_phase
                    * value_entropy
                    * value_orientation
                )[..., 31:-31, 31:-31].mean()

            assert (
                structure_score(structured_luminance)
                > 2.0 * structure_score(noise_luminance))

            half_vision = CorticalEarlyVision(
                outChannels=8).to(
                    device=self.device,
                    dtype=torch.float16)
            zero_quadrature = torch.zeros(
                1,
                len(half_vision.frequency_scales),
                half_vision.orientations,
                4,
                4,
                device=self.device,
                dtype=torch.float16)
            half_energy = half_vision.QuadratureAmplitude(
                zero_quadrature,
                zero_quadrature)
            half_coherence = half_vision.OrientationCoherence(half_energy)
            assert torch.equal(half_energy, torch.zeros_like(half_energy))
            assert torch.equal(
                half_coherence,
                torch.zeros_like(half_coherence))

            half_generator = torch.Generator(
                device=self.device).manual_seed(137)
            for magnitude in (0.0, 1e-4, 1e-3, 1e-2):
                weak_even = (
                    torch.rand(
                        zero_quadrature.shape,
                        generator=half_generator,
                        device=self.device,
                        dtype=torch.float16)
                    * magnitude).requires_grad_()
                weak_odd = (
                    torch.rand(
                        zero_quadrature.shape,
                        generator=half_generator,
                        device=self.device,
                        dtype=torch.float16)
                    * magnitude).requires_grad_()
                weak_amplitude = half_vision.QuadratureAmplitude(
                    weak_even,
                    weak_odd)
                weak_phase, weak_entropy = (
                    half_vision.MultiscalePhaseStatistics(
                        weak_even,
                        weak_odd,
                        weak_amplitude))
                weak_coherence = half_vision.OrientationCoherence(
                    weak_amplitude)
                (
                    weak_phase.sum()
                    + weak_entropy.sum()
                    + weak_coherence.sum()
                ).backward()
                assert weak_even.grad is not None
                assert weak_odd.grad is not None
                assert torch.isfinite(weak_even.grad).all()
                assert torch.isfinite(weak_odd.grad).all()

            bfloat_vision = CorticalEarlyVision(
                outChannels=8).to(
                    device=self.device,
                    dtype=torch.bfloat16)
            for magnitude in (1e-4, 1.0):
                bfloat_even = (
                    torch.rand(
                        zero_quadrature.shape,
                        generator=half_generator,
                        device=self.device,
                        dtype=torch.bfloat16)
                    * magnitude).requires_grad_()
                bfloat_odd = (
                    torch.rand(
                        zero_quadrature.shape,
                        generator=half_generator,
                        device=self.device,
                        dtype=torch.bfloat16)
                    * magnitude).requires_grad_()
                bfloat_amplitude = bfloat_vision.QuadratureAmplitude(
                    bfloat_even,
                    bfloat_odd)
                bfloat_phase, bfloat_entropy = (
                    bfloat_vision.MultiscalePhaseStatistics(
                        bfloat_even,
                        bfloat_odd,
                        bfloat_amplitude))
                bfloat_coherence = bfloat_vision.OrientationCoherence(
                    bfloat_amplitude)
                for statistic in (
                    bfloat_phase,
                    bfloat_entropy,
                    bfloat_coherence,):
                    assert torch.isfinite(statistic).all()
                    assert float(statistic.min()) >= 0.0
                    assert float(statistic.max()) <= 1.0
                (
                    bfloat_phase.sum()
                    + bfloat_entropy.sum()
                    + bfloat_coherence.sum()
                ).backward()
                assert torch.isfinite(bfloat_even.grad).all()
                assert torch.isfinite(bfloat_odd.grad).all()
            with torch.no_grad():
                vision.spectral_scale_logits.copy_(
                    vision.spectral_scale_logits.new_tensor(
                        [20.0, -20.0, -20.0]))
            collapsed_independent_phase, _ = vision.MultiscalePhaseStatistics(
                even,
                odd,
                energy)
            assert torch.equal(
                collapsed_independent_phase,
                phase_congruency)
            with torch.no_grad():
                vision.spectral_scale_logits.zero_()

            feature, _ = vision(frames[1:2], None, None)
            feature.square().mean().backward()
            for parameter in (
                vision.spectral_scale_logits,
                vision.phase_congruency_gain_raw,
                vision.feature_projection[0].weight,):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert float(parameter.grad.abs().sum()) > 0.0
        return self.RunRegressionCheck(
            "SpatialFrequencyEntropyAndCorticalSelectivity",
            check)

    def TestPatchProjectionContentSensitivity(self):
        def check():
            model = self.MakeRegressionModel(64).eval()
            assert float(model.patch_embed.weight.norm()) > 0.0
            with torch.no_grad():
                black = model.cnn_extractor(
                    torch.zeros(1, 3, 64, 64, device=self.device))["Deep"]
                white = model.cnn_extractor(
                    torch.ones(1, 3, 64, 64, device=self.device))["Deep"]
                black_patch = model.AddPatchContentProjection(
                    black,
                    model.patch_adapter(black))
                white_patch = model.AddPatchContentProjection(
                    white,
                    model.patch_adapter(white))
                assert not torch.equal(black_patch, white_patch)
                model.patch_embed.weight.zero_()
                black_patch = model.AddPatchContentProjection(
                    black,
                    model.patch_adapter(black))
                white_patch = model.AddPatchContentProjection(
                    white,
                    model.patch_adapter(white))
            assert float(model.patch_content_gain) == 0.0
            assert torch.count_nonzero(black_patch) == 0
            assert torch.count_nonzero(white_patch) == 0
        return self.RunRegressionCheck("PatchProjectionContentSensitivity", check)

    def TestGrowableAdapterStateRoundTrip(self):
        def check():
            conv = GrowableLoRAConv2d(
                nn.Conv2d(4, 6, 3, padding=1, bias=False).to(self.device))
            conv.Grow(2)
            restored_conv = GrowableLoRAConv2d(
                nn.Conv2d(4, 6, 3, padding=1, bias=False).to(self.device))
            restored_conv.load_state_dict(conv.state_dict(), strict=True)
            assert len(restored_conv.A_list) == 1
            assert torch.equal(restored_conv.A_list[0], conv.A_list[0])

            feature = GrowableConv1x1Adapter(8).to(self.device)
            feature.Grow(3)
            restored_feature = GrowableConv1x1Adapter(8).to(self.device)
            restored_feature.load_state_dict(feature.state_dict(), strict=True)
            assert len(restored_feature.A_list) == 1
            assert torch.equal(restored_feature.B_list[0], feature.B_list[0])

            token = GrowableTokenAdapter(8).to(self.device)
            token.Grow(2)
            restored_token = GrowableTokenAdapter(8).to(self.device)
            restored_token.load_state_dict(token.state_dict(), strict=True)
            assert len(restored_token.A_list) == 1
            assert torch.equal(restored_token.alpha[0], token.alpha[0])

            model = self.MakeRegressionModel()
            model.cnn_feat_adapter.Grow(2)
            model.patch_adapter.Grow(2)
            model.token_adapters[0].Grow(2)
            restored_model = self.MakeRegressionModel()
            incompatible = restored_model.load_state_dict(
                model.state_dict(),
                strict=True)
            assert incompatible.missing_keys == []
            assert incompatible.unexpected_keys == []
            assert len(restored_model.cnn_feat_adapter.A_list) == 1
            assert len(restored_model.patch_adapter.A_list) == 1
            assert len(restored_model.token_adapters[0].A_list) == 1

            empty_state = self.MakeRegressionModel().state_dict()
            SynchronizeDynamicAdapterTopologiesForFullLoad(
                restored_model,
                empty_state)
            assert len(restored_model.cnn_feat_adapter.A_list) == 0
            assert len(restored_model.patch_adapter.A_list) == 0
            assert len(restored_model.token_adapters[0].A_list) == 0
            incompatible = restored_model.load_state_dict(
                empty_state,
                strict=True)
            assert incompatible.missing_keys == []
            assert incompatible.unexpected_keys == []
            assert len(restored_model.cnn_feat_adapter.A_list) == 0
            assert len(restored_model.patch_adapter.A_list) == 0
            assert len(restored_model.token_adapters[0].A_list) == 0

            markerless_adapter_state = dict(conv.state_dict())
            markerless_adapter_state.pop("topology_count")
            markerless_rejected = False
            try:
                GrowableLoRAConv2d(
                    nn.Conv2d(
                        4, 6, 3, padding=1, bias=False).to(
                            self.device)).load_state_dict(
                                markerless_adapter_state,
                                strict=True)
            except RuntimeError:
                markerless_rejected = True

            incomplete_model_state = dict(
                self.MakeRegressionModel().state_dict())
            incomplete_model_state.pop("patch_content_gain")
            incomplete_model_rejected = False
            try:
                self.MakeRegressionModel().load_state_dict(
                    incomplete_model_state,
                    strict=True)
            except RuntimeError:
                incomplete_model_rejected = True
            assert markerless_rejected
            assert incomplete_model_rejected
        return self.RunRegressionCheck("GrowableAdapterStateRoundTrip", check)

    def TestInternalRegistrationAndTransientState(self):
        def check():
            model = self.MakeRegressionModel()
            model_state_keys = tuple(model.state_dict())
            assert not any(
                name.startswith("patch_adapter.target.")
                for name in model_state_keys)
            assert model.patch_embed is model.patch_adapter.target
            assert model.patch_aggregator[-1].bias is None
            for block in (
                model.cnn_extractor.layer1[0],
                model.cnn_extractor.layer1[1],
                model.cnn_extractor.layer2[1],
                model.cnn_extractor.layer3[1],
                model.cnn_extractor.layer4[1],):
                assert not block.use_downsample
                assert isinstance(block.downsample, nn.Identity)
                assert sum(
                    parameter.numel()
                    for parameter in block.downsample.parameters()) == 0

            conv = HebbianConv2d(
                3,
                4,
                3,
                padding=1).to(self.device)
            conv.EnsureB(2)
            _ = conv(torch.rand(2, 3, 8, 8, device=self.device))
            assert torch.count_nonzero(conv.hebb_memory) > 0
            conv.load_state_dict(conv.state_dict(), strict=True)
            assert torch.count_nonzero(conv.hebb_memory) == 0
            conv.hebb_memory.fill_(1.0)
            conv.ResetHebbianMemory(doneMask=torch.ones(
                2,
                device=self.device,
                dtype=torch.bool))
            assert torch.count_nonzero(conv.hebb_memory) == 0

            linear = HebbianLinear(
                8,
                4).to(self.device)
            assert linear.bias is not None
            linear.EnsureB(2)
            _ = linear(torch.rand(2, 8, device=self.device))
            assert torch.count_nonzero(linear.hebb_memory) > 0
            linear.load_state_dict(linear.state_dict(), strict=True)
            assert torch.count_nonzero(linear.hebb_memory) == 0
            linear.hebb_memory.fill_(1.0)
            linear.ResetHebbianMemory(doneMask=torch.ones(
                2,
                device=self.device,
                dtype=torch.bool))
            assert torch.count_nonzero(linear.hebb_memory) == 0

            trainer = PerceptionTrainer(
                cameraIntrinsics=self.MakeCameraIntrinsics(32),
                imgSize=32,
                patchSize=1,
                embedDim=32,
                numHeads=4,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=4).to(self.device)
            assert trainer.recall_heads is trainer.extractor.recall_heads
            trainer_state_keys = tuple(trainer.state_dict())
            assert not any(
                name.startswith("recall_heads.")
                for name in trainer_state_keys)
            parameter_ids = [
                id(parameter)
                for _, parameter in model.named_parameters(
                    remove_duplicate=False)]
            buffer_ids = [
                id(buffer)
                for _, buffer in model.named_buffers(
                    remove_duplicate=False)]
            assert len(parameter_ids) == len(set(parameter_ids))
            assert len(buffer_ids) == len(set(buffer_ids))
        return self.RunRegressionCheck(
            "InternalRegistrationAndTransientState",
            check)

    def TestPartialAdapterLoadPreservesRank(self):
        def check():
            adapter = GrowableConv1x1Adapter(8).to(self.device)
            adapter.Grow(2)
            adapter.load_state_dict(
                {"anchor_": adapter.anchor_.clone()},
                strict=False)
            assert len(adapter.A_list) == 1
            model = self.MakeRegressionModel()
            model.load_state_dict(
                {"patch_content_gain": torch.ones_like(
                    model.patch_content_gain)},
                strict=False)
            assert torch.equal(
                model.patch_content_gain,
                torch.ones_like(model.patch_content_gain))
        return self.RunRegressionCheck("PartialAdapterLoadPreservesRank", check)

    def TestHebbianPartialRowReset(self):
        def check():
            layer = HebbianConv2d(
                3,
                4,
                3,
                padding=1).to(self.device).train()
            value = torch.rand(3, 3, 8, 8, device=self.device)
            layer.EnsureB(int(value.size(0)))
            _ = layer(value)
            before = layer.hebb_memory.clone()
            layer.ResetHebbianMemory(doneMask=torch.tensor(
                [False, True, False],
                device=self.device))
            assert torch.equal(layer.hebb_memory[0], before[0])
            assert torch.count_nonzero(layer.hebb_memory[1]) == 0
            assert torch.equal(layer.hebb_memory[2], before[2])
        return self.RunRegressionCheck("HebbianPartialRowReset", check)

    def TestHebbianEvalPlasticity(self):
        def check():
            layer = HebbianConv2d(
                3,
                4,
                3,
                padding=1).to(self.device).eval()
            value = torch.rand(2, 3, 8, 8, device=self.device)
            layer.EnsureB(int(value.size(0)))
            expected = layer.conv(value)
            first = layer(value)
            assert torch.allclose(first, expected, atol=1e-6, rtol=1e-5)
            before = layer.hebb_memory.clone()
            _ = layer(value)
            assert torch.count_nonzero(before) > 0
            assert not torch.equal(layer.hebb_memory, before)
        return self.RunRegressionCheck("HebbianEvalPlasticity", check)

    def TestNonSquarePatchSupervision(self):
        def check():
            model = self.MakeRegressionModel(64).train()
            frame = torch.rand(1, 3, 64, 96, device=self.device)
            depth = torch.ones(1, 1, 64, 96, device=self.device)
            valid = torch.ones_like(depth, dtype=torch.bool)
            normal = torch.zeros(1, 3, 64, 96, device=self.device)
            normal[:, 2] = 1.0
            targets = self.MakeSyntheticTargets(
                frame,
                depth,
                valid,
                normal,
                torch.zeros(
                    1,
                    64,
                    96,
                    device=self.device,
                    dtype=torch.long),
                nodes=2)
            visual = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 1),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            recall = model.recall_heads(visual)
            losses = PerceptionRecallLoss(
                identityDim=model.recall_heads.identity_dim).to(self.device)(
                    recall,
                    targets)
            assert torch.isfinite(losses["loss"])
        return self.RunRegressionCheck("NonSquarePatchSupervision", check)

    def TestAnisotropicDiffusionStability(self):
        def check():
            diffusion = StableAnisotropicDiffusion(
                iterations=2).to(self.device)
            value = torch.rand(
                2,
                6,
                17,
                19,
                device=self.device,
                requires_grad=True)
            output = diffusion(value)
            channel_min = value.detach().amin(
                dim=(-2, -1),
                keepdim=True)
            channel_max = value.detach().amax(
                dim=(-2, -1),
                keepdim=True)
            assert torch.all(output.detach() >= channel_min - 1e-6)
            assert torch.all(output.detach() <= channel_max + 1e-6)
            output.mean().backward()
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
            assert 0.0 < float(diffusion.StepSize()) < 0.25
        return self.RunRegressionCheck("AnisotropicDiffusionStability", check)

    def TestCorticalGaborTemporalReset(self):
        def check():
            vision = CorticalEarlyVision(outChannels=8).to(self.device)
            assert vision.frequency_scales == (1.0, 2.0, 4.0)
            assert 0.0 < float(vision.FastDecay()) < float(vision.SlowDecay()) < 1.0
            even = vision.gabor_even.float()
            odd = vision.gabor_odd.float()
            assert torch.allclose(
                even.mean(dim=(-2, -1)),
                torch.zeros_like(even[:, :, 0, 0]),
                atol=1e-5)
            assert torch.allclose(
                odd.mean(dim=(-2, -1)),
                torch.zeros_like(odd[:, :, 0, 0]),
                atol=1e-5)
            assert torch.allclose(
                even.flatten(1).norm(dim=1),
                torch.ones(even.size(0), device=self.device),
                atol=1e-5)
            assert torch.allclose(
                odd.flatten(1).norm(dim=1),
                torch.ones(odd.size(0), device=self.device),
                atol=1e-5)
            original_state = {
                name: value.clone()
                for name, value in vision.state_dict().items()}
            modified_state = {
                name: value.clone()
                for name, value in original_state.items()}
            modified_state["gabor_even"] = (
                modified_state["gabor_even"] + 0.01)
            vision.load_state_dict(modified_state, strict=True)
            assert torch.equal(
                vision.gabor_quadrature,
                torch.cat([vision.gabor_even, vision.gabor_odd], dim=0))
            vision.load_state_dict(original_state, strict=True)
            frame = torch.zeros(2, 3, 64, 64, device=self.device)
            frame[..., 16:48, 20:36] = 1.0
            feature1, state1 = vision(frame, None, None)
            assert feature1.shape == (2, 8, 16, 16)
            assert torch.count_nonzero(
                state1["CorticalTemporalResponse"]) == 0
            moved_frame = torch.roll(frame, shifts=2, dims=-1)
            _, current_state = vision(moved_frame, None, None)
            feature2, state2 = vision(
                moved_frame,
                state1["CorticalFastState"],
                state1["CorticalSlowState"],
                torch.tensor([True, False], device=self.device))
            assert torch.isfinite(feature2).all()
            expected_temporal = (
                vision.SlowDecay() - vision.FastDecay()
            ) * (
                current_state["CorticalFastState"]
                - state1["CorticalFastState"])
            assert torch.allclose(
                state2["CorticalFastState"][0]
                - state2["CorticalSlowState"][0],
                expected_temporal[0],
                atol=1e-6,
                rtol=1e-5)
            assert torch.equal(
                state2["CorticalFastState"][1],
                state2["CorticalSlowState"][1])
            feature2.square().mean().backward()
            for parameter in (
                vision.fast_decay_raw,
                vision.slow_gap_raw,):
                assert parameter.grad is not None
                assert float(parameter.grad.abs()) > 0.0
        return self.RunRegressionCheck("CorticalGaborTemporalReset", check)

    def TestLogDepthResampling(self):
        def check():
            fusion = self.MakeRegressionModel(64).depth_fusion
            depth = torch.tensor(
                [[[[1.0, 4.0]]]],
                device=self.device)
            valid = torch.ones_like(depth, dtype=torch.bool)
            log_depth, weight = fusion.ResampleSensorLogDepth(
                depth,
                valid,
                (1, 1))
            assert torch.allclose(
                log_depth.exp(),
                torch.tensor([[[[2.0]]]], device=self.device),
                atol=1e-6)
            assert torch.equal(weight, torch.ones_like(weight))
        return self.RunRegressionCheck("LogDepthResampling", check)

    def TestSPPAndAxialPositionEncoding(self):
        def check():
            spp = SPPContextAdapter(
                inChannels=32,
                embedDim=24,
                reducedChannels=8).to(self.device)
            context = spp(torch.rand(
                2,
                32,
                5,
                7,
                device=self.device))
            assert context.shape == (2, 24)
            assert torch.isfinite(context).all()

            encoding = AxialPositionEncoding2D(24)(
                3,
                5,
                self.device,
                torch.float32)
            assert encoding.shape == (1, 15, 24)
            assert not torch.equal(encoding[:, 4], encoding[:, 5])

            model = self.MakeRegressionModel()
            positions = model.BuildRotaryPositions2D(
                2,
                3,
                self.device)
            assert positions.shape == (7, 2)
            assert torch.equal(
                positions[0],
                torch.zeros(2, device=self.device))
            assert torch.equal(
                positions[3],
                torch.tensor([1.0, 3.0], device=self.device))
            assert torch.equal(
                positions[4],
                torch.tensor([2.0, 1.0], device=self.device))

            attention = PerceptionRoPEMultiheadAttention(
                embedDim=24,
                numHeads=4).to(self.device).eval()
            tokens = torch.rand(2, 7, 24, device=self.device)
            attended, weights = attention(
                tokens,
                tokens,
                tokens,
                rotaryPositions2D=positions)
            assert attended.shape == tokens.shape
            assert weights.shape == (2, 7, 7)
            assert torch.isfinite(attended).all()
            legacy_attention = RoPEMultiheadAttention(
                embedDim=24,
                numHeads=4).to(self.device)
            legacy_attention.load_state_dict(
                attention.state_dict(),
                strict=True)
            legacy_output, _ = legacy_attention(
                tokens,
                tokens,
                tokens)
            raw_heads = attention.ReshapeHeads(attention.q_proj(tokens))
            rotated_heads = attention.Apply2DRotary(
                raw_heads,
                positions)
            assert torch.allclose(
                rotated_heads.norm(dim=-1),
                raw_heads.norm(dim=-1),
                atol=1e-5,
                rtol=1e-5)
            adjacent_channels = torch.arange(
                6,
                device=self.device,
                dtype=torch.float32)
            assert torch.equal(
                attention.rope.RotateHalf(adjacent_channels),
                torch.tensor(
                    [-1.0, 0.0, -3.0, 2.0, -5.0, 4.0],
                    device=self.device))
            assert not torch.allclose(
                attended,
                legacy_output,
                atol=1e-7,
                rtol=1e-6)

            full_query_mask = torch.zeros(
                7,
                7,
                device=self.device,
                dtype=torch.bool)
            full_query_mask[2] = True
            masked_output, masked_weights = attention(
                tokens,
                tokens,
                tokens,
                rotaryPositions2D=positions,
                attnMask=full_query_mask)
            assert torch.count_nonzero(masked_output[:, 2]).item() == 0
            assert torch.count_nonzero(masked_weights[:, 2]).item() == 0

            full_padding_mask = torch.ones(
                2,
                7,
                device=self.device,
                dtype=torch.bool)
            base_masked_output, base_masked_weights = legacy_attention(
                tokens,
                tokens,
                tokens,
                keyPaddingMask=full_padding_mask)
            assert torch.count_nonzero(base_masked_output).item() == 0
            assert torch.count_nonzero(base_masked_weights).item() == 0

            dropout_attention = PerceptionRoPEMultiheadAttention(
                embedDim=24,
                numHeads=4,
                dropout=0.5).to(self.device).train()
            torch.manual_seed(1)
            _, dropout_weights_1 = dropout_attention(
                tokens,
                tokens,
                tokens,
                rotaryPositions2D=positions)
            torch.manual_seed(2)
            _, dropout_weights_2 = dropout_attention(
                tokens,
                tokens,
                tokens,
                rotaryPositions2D=positions)
            assert torch.equal(dropout_weights_1, dropout_weights_2)
            assert torch.allclose(
                dropout_weights_1.sum(dim=-1),
                torch.ones_like(dropout_weights_1[..., 0]))
        return self.RunRegressionCheck("SPPAndAxialPositionEncoding", check)

    def TestProjectiveTopologyDiagnostics(self):
        def check():
            diagnostics = ProjectiveTopologyDiagnostics().to(self.device)
            height = width = 4
            yy, xx = torch.meshgrid(
                torch.arange(height, device=self.device, dtype=torch.float32),
                torch.arange(width, device=self.device, dtype=torch.float32),
                indexing="ij")
            pixel_x = 50.0 * xx
            pixel_y = 50.0 * yy
            grid = torch.stack([
                2.0 * (pixel_x + 0.5) / width - 1.0,
                2.0 * (pixel_y + 0.5) / height - 1.0,
            ], dim=-1).unsqueeze(0).half()
            output = diagnostics(
                grid,
                domainMask=torch.ones(
                    1,
                    1,
                    height,
                    width,
                    device=self.device))
            assert all(torch.isfinite(value).all() for value in output.values())
            assert torch.allclose(
                output["WarpJacobianDet"],
                torch.full_like(output["WarpJacobianDet"], 2500.0),
                atol=4.0)

            nan_grid = torch.full(
                (1, 2, 2, 2),
                float("nan"),
                device=self.device,
                dtype=torch.float16)
            invalid = diagnostics(
                nan_grid,
                domainMask=torch.ones(
                    1,
                    1,
                    2,
                    2,
                    device=self.device))
            assert float(invalid["WarpFoldPenalty"]) == 0.0
            assert torch.count_nonzero(invalid["WarpTopologyValid"]) == 0
            singleton = diagnostics(torch.zeros(
                1,
                1,
                3,
                2,
                device=self.device))
            assert torch.count_nonzero(singleton["WarpTopologyValid"]) == 0
        return self.RunRegressionCheck("ProjectiveTopologyDiagnostics", check)

    def TestEnhancedPerceptionAuxiliary(self):
        def check():
            model = self.MakeRegressionModel(64)
            with torch.no_grad():
                model.perception_enhancement.residual_gain.fill_(0.1)
            model.eval()
            frame = torch.rand(1, 3, 64, 64, device=self.device)
            depth = torch.ones(1, 1, 64, 64, device=self.device)
            valid = torch.ones_like(depth, dtype=torch.bool)
            context = self.MakeTopDownContext(model, 1)
            previous = model(
                frame,
                topDownContext=context,
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            motion = torch.zeros(
                1, ModuleDim.CameraMotionDim, device=self.device)
            motion[:, 3] = 1.0
            current = model(
                frame,
                topDownContext=context,
                depth=depth,
                depthValid=valid,
                prevVisualState=previous,
                cameraMotion=motion,
                prevVisualValid=torch.ones(
                    1, device=self.device, dtype=torch.bool))
            assert current.Auxiliary["MetricDepthFullRes"].shape[-2:] == (64, 64)
            grid_shape = tuple(
                int(value)
                for value in current.Auxiliary["PatchGridShape"].tolist())
            assert current.PatchTokens.shape == (
                1,
                grid_shape[0] * grid_shape[1],
                model.embed_dim)
            assert "DenseSemanticTensor" not in current.Auxiliary
            for key in (
                "CorticalEnergy",
                "CorticalTemporalResponse",
                "RigidPatchFlow",
                "WarpJacobianDet",
                "WarpJacobianSigmaMin",
                "WarpJacobianSigmaMax",
                "WarpTopologyValid",):
                assert key in current.Auxiliary
                assert torch.isfinite(current.Auxiliary[key]).all()
            assert torch.allclose(
                current.Auxiliary["WarpJacobianDet"],
                torch.ones_like(current.Auxiliary["WarpJacobianDet"]),
                atol=2e-3,
                rtol=2e-3)
        return self.RunRegressionCheck("EnhancedPerceptionAuxiliary", check)

    def TestCameraProjectionGeometry(self):
        def check():
            fusion = self.MakeRegressionModel(3).depth_fusion
            depth = torch.full(
                (1, 1, 3, 3), 2.0, device=self.device)
            intrinsics = torch.tensor([
                [2.0, 0.5, 1.0],
                [0.0, 4.0, 1.0],
                [0.0, 0.0, 1.0],
            ], device=self.device).unsqueeze(0)
            xyz = fusion.BackprojectDepth(depth, intrinsics, (3, 3))
            assert torch.allclose(
                xyz[0, :, 2, 2],
                torch.tensor([0.875, 0.5, 2.0], device=self.device))

            current_depth = torch.ones(1, 1, 4, 4, device=self.device)
            previous_depth = torch.arange(
                4, device=self.device, dtype=current_depth.dtype
            ).view(1, 1, 1, 4).expand_as(current_depth)
            warp_intrinsics = torch.tensor([
                [4.0, 0.0, 1.5],
                [0.0, 4.0, 1.5],
                [0.0, 0.0, 1.0],
            ], device=self.device).unsqueeze(0)
            motion = torch.zeros(
                1, ModuleDim.CameraMotionDim, device=self.device)
            motion[:, 3] = 1.0
            _, sampled_previous, valid = fusion.WarpPrevDepth(
                current_depth,
                previous_depth,
                warp_intrinsics,
                (4, 4),
                motion)
            assert valid[0, 0, 1, 1]
            assert torch.allclose(
                sampled_previous[0, 0, 1, 1],
                torch.tensor(1.0, device=self.device))
        return self.RunRegressionCheck("CameraProjectionGeometry", check)

    def TestMetricDepthAndPoseAuthority(self):
        def check():
            model = self.MakeRegressionModel(64).eval()
            model.depth_fusion.sensor_dropout = 1.0
            assert "camera_intrinsics" not in model.state_dict()
            assert model.recall_heads.orientation_camera_head.out_features == 4
            frame = torch.rand(1, 3, 64, 64, device=self.device)
            depth = torch.full(
                (1, 1, 64, 64), 1.75, device=self.device)
            valid = torch.ones_like(depth, dtype=torch.bool)
            visual = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 1),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            assert torch.allclose(
                visual.Auxiliary["MetricDepth"],
                torch.full_like(visual.Auxiliary["MetricDepth"], 1.75))
            assert torch.allclose(
                visual.SemanticNodes["pose_camera"][..., :3],
                visual.Auxiliary["ObjectGeometry"][..., :3])
            pose_delta = visual.Auxiliary["ObjectGeometry"].new_tensor(
                [0.1, -0.2, 0.3])
            with torch.no_grad():
                model.recall_heads.position_residual_camera_head.bias.copy_(
                    pose_delta)
            refined = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 1),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            assert torch.allclose(
                refined.Auxiliary["ObjectGeometry"][..., :3],
                visual.Auxiliary["ObjectGeometry"][..., :3]
                + pose_delta.view(1, 1, 3),
                atol=1e-6,
                rtol=1e-5)
            assert torch.equal(
                refined.SemanticNodes["pose_camera"][..., :3],
                refined.Auxiliary["ObjectGeometry"][..., :3])

            model.train()
            fallback = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 1),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            assert torch.equal(
                fallback.Auxiliary["SensorDepthValid"],
                torch.ones_like(fallback.Auxiliary["SensorDepthValid"]))
            assert torch.equal(
                fallback.Auxiliary["SensorDepthUsed"],
                torch.zeros_like(fallback.Auxiliary["SensorDepthUsed"]))
            assert torch.allclose(
                fallback.Auxiliary["MetricDepth"],
                fallback.Auxiliary["MonocularDepth"])
        return self.RunRegressionCheck("MetricDepthAndPoseAuthority", check)

    def TestObjectTokensCanEncodeCameraGeometry(self):
        def check():
            model = self.MakeRegressionModel(64).eval()
            patch_tokens = torch.randn(
                1, 9, model.embed_dim, device=self.device)
            geometry_a = torch.randn(1, 9, 6, device=self.device)
            translated_geometry = geometry_a.clone()
            translated_geometry[..., :3] += translated_geometry.new_tensor(
                [3.0, -2.0, 1.0])
            geometry_b = geometry_a.clone()
            geometry_b[:, :3, :3] += geometry_b.new_tensor(
                [0.5, -0.25, 0.75])
            coordinate_valid = torch.ones(1, 9, 1, device=self.device)
            tokens_a, object_geometry_a, _, weights_a = model.BuildObjectTokens(
                patch_tokens,
                geometry_a,
                coordinate_valid)
            _, _, _, translated_weights = model.BuildObjectTokens(
                patch_tokens,
                translated_geometry,
                coordinate_valid)
            tokens_b, object_geometry_b, _, weights_b = model.BuildObjectTokens(
                patch_tokens,
                geometry_b,
                coordinate_valid)
            assert hasattr(model, "object_geometry_key")
            assert hasattr(model, "object_geometry_proj")
            assert torch.allclose(
                weights_a,
                translated_weights,
                atol=1e-6,
                rtol=1e-5)
            assert float((weights_a - weights_b).abs().max()) > 1e-4
            assert float((tokens_a - tokens_b).abs().max()) > 1e-4
            assert float(
                (object_geometry_a - object_geometry_b).abs().max()
            ) > 1e-4

            with torch.no_grad():
                competition_value = model.object_competition_raw.clone()
                model.object_competition_raw.fill_(-20.0)
            _, _, _, independent_weights = model.BuildObjectTokens(
                patch_tokens,
                geometry_a,
                coordinate_valid)
            with torch.no_grad():
                model.object_competition_raw.fill_(20.0)
            _, _, _, competitive_weights = model.BuildObjectTokens(
                patch_tokens,
                geometry_a,
                coordinate_valid)
            with torch.no_grad():
                model.object_competition_raw.copy_(competition_value)
            assert float(
                (independent_weights - competitive_weights).abs().max()
            ) > 1e-4

            zero_valid = torch.zeros_like(coordinate_valid)
            appearance_tokens, _, appearance_valid, appearance_weights = (
                model.BuildObjectTokens(
                    patch_tokens,
                    geometry_a,
                    zero_valid))
            modified_patch_tokens = patch_tokens.clone()
            modified_patch_tokens[:, 0] += 1.0
            modified_appearance, _, _, _ = model.BuildObjectTokens(
                modified_patch_tokens,
                geometry_a,
                zero_valid)
            assert torch.count_nonzero(appearance_valid) == 0
            assert torch.isfinite(appearance_tokens).all()
            assert torch.allclose(
                appearance_weights.sum(dim=-1),
                torch.ones_like(appearance_weights[..., 0]))
            assert float(
                (appearance_tokens - modified_appearance).abs().max()
            ) > 1e-4
            normalized_weights = F.normalize(weights_a, dim=-1, eps=1e-6)
            weight_similarity = torch.matmul(
                normalized_weights,
                normalized_weights.transpose(1, 2))
            off_diagonal = ~torch.eye(
                model.object_token_count,
                device=self.device,
                dtype=torch.bool).unsqueeze(0)
            assert float(weight_similarity[off_diagonal].mean()) < 0.95
            probe = torch.randn_like(tokens_a)
            (tokens_a * probe).sum().backward()
            for parameter in (
                model.object_geometry_key[-1].weight,
                model.object_geometry_proj[-1].weight,
                model.object_relation_attention.in_proj_weight,):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert float(parameter.grad.abs().sum()) > 0.0
        return self.RunRegressionCheck(
            "ObjectTokensCanEncodeCameraGeometry",
            check)

    def TestEvalSkipsZeroGainEnhancements(self):
        def check():
            model = self.MakeRegressionModel(64)
            with torch.no_grad():
                model.perception_enhancement.residual_gain.zero_()
            model.eval()
            calls = {
                "cortical": 0,
                "patch": 0,
                "spp": 0,
                "dense": 0,
                "gauge_gamma": 0,
                "gauge_beta": 0,}

            def count(name):
                def hook(module, inputs, output):
                    del module, inputs, output
                    calls[name] += 1
                return hook

            handles = [
                model.perception_enhancement.early_vision.register_forward_hook(
                    count("cortical")),
                model.patch_content_projection.register_forward_hook(
                    count("patch")),
                model.spp_context.reduce.register_forward_hook(count("spp")),
                model.dense_depth_refiner.trunk.register_forward_hook(
                    count("dense")),
                model.patch_embed.gauge_gamma.register_forward_hook(
                    count("gauge_gamma")),
                model.patch_embed.gauge_beta.register_forward_hook(
                    count("gauge_beta")),]
            frame = torch.rand(1, 3, 64, 64, device=self.device)
            depth = torch.ones(1, 1, 64, 64, device=self.device)
            with torch.no_grad():
                model(
                    frame,
                    topDownContext=self.MakeTopDownContext(model, 1),
                    depth=depth,
                    depthValid=torch.ones_like(depth, dtype=torch.bool),
                    **self.CameraTemporalInputs(frame))
            for handle in handles:
                handle.remove()
            assert calls == {
                "cortical": 0,
                "patch": 0,
                "spp": 0,
                "dense": 0,
                "gauge_gamma": 1,
                "gauge_beta": 1,}
        return self.RunRegressionCheck("EvalSkipsZeroGainEnhancements", check)

    def TestEnhancementGradientCoverage(self):
        def check():
            model = self.MakeRegressionModel(64).train()
            frame = torch.rand(2, 3, 64, 64, device=self.device)
            depth = torch.rand(2, 1, 64, 64, device=self.device) + 0.5
            valid = torch.ones_like(depth, dtype=torch.bool)
            visual = model(
                frame,
                topDownContext=self.MakeTopDownContext(model, 2),
                depth=depth,
                depthValid=valid,
                **self.CameraTemporalInputs(frame))
            loss = model.ComputePerceptionLoss(
                visual,
                depthTarget=depth,
                depthTargetValid=valid,
                **self.CameraTemporalInputs(frame))
            loss.backward()
            for parameter in (
                model.perception_enhancement.residual_gain,
                model.spp_context.residual_gain,
                model.axial_position_gain,
                model.dense_depth_refiner.output.weight,
                model.patch_content_projection.weight,):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            for parameter in (
                model.perception_enhancement.early_vision.spectral_scale_logits,
                model.perception_enhancement.early_vision.phase_congruency_gain_raw,
                model.perception_enhancement.early_vision.feature_projection[0].weight,
                model.object_geometry_key[-1].weight,
                model.object_relation_attention.in_proj_weight,):
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert float(parameter.grad.abs().sum()) > 0.0
            assert model.dense_depth_refiner.output.bias.grad[1].abs() > 0
        return self.RunRegressionCheck("EnhancementGradientCoverage", check)

    def PartialPreviousVisualMask(self):
        try:
            B, H = 2, 32
            model = PerceiveExtractor(
                cameraIntrinsics=self.MakeCameraIntrinsics(H),
                imgSize=H,
                patchSize=1,
                embedDim=32,
                numHeads=4,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=4).to(self.device).eval()
            depth = torch.ones(B, 1, H, H, device=self.device)
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
            top_down = self.MakeTopDownContext(model, B)
            previous = model(
                torch.rand(B, 3, H, H, device=self.device),
                topDownContext=top_down,
                depth=depth,
                depthValid=depth_valid,
                **self.CameraTemporalInputs(depth))
            camera_motion = torch.zeros(
                B, ModuleDim.CameraMotionDim, device=self.device)
            camera_motion[:, 3] = 1.0
            current = model(
                torch.rand(B, 3, H, H, device=self.device),
                topDownContext=top_down,
                depth=depth,
                depthValid=depth_valid,
                prevVisualState=previous,
                cameraMotion=camera_motion,
                prevVisualValid=torch.tensor([False, True], device=self.device))
            invalid_mask_rejected = False
            try:
                model(
                    torch.rand(B, 3, H, H, device=self.device),
                    topDownContext=top_down,
                    depth=depth,
                    depthValid=depth_valid,
                    prevVisualState=previous,
                    cameraMotion=camera_motion,
                    prevVisualValid=torch.tensor(
                        [0.0, 1.0], device=self.device))
            except TypeError:
                invalid_mask_rejected = True
            expected_identity = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                device=self.device)
            ok = (
                invalid_mask_rejected
                and all(bool(torch.isfinite(value).all().item()) for value in (
                    current.IntegratedFeat,
                    current.MotionToken,
                    current.Auxiliary["WarpedPrevPatchTokens"],
                    current.Auxiliary["WarpPrevPatchValid"],
                    current.Auxiliary["PatchMotionDepthResidual"],))
                and torch.equal(
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

    def TestNormalValidityIndependentFromDepth(self):
        try:
            loss_module = PerceptionRecallLoss().to(self.device)
            normal = torch.zeros(1, 3, 4, 4, device=self.device)
            normal[:, 2] = 1.0
            no_normal = {
                "normal": normal,
                "normal_valid": torch.zeros(
                    1, 1, 4, 4, device=self.device, dtype=torch.bool),}
            _, invalid = loss_module.NormalTarget(no_normal, 4, (2, 2))

            valid_normal = dict(no_normal)
            valid_normal["normal_valid"] = torch.ones(
                1, 1, 4, 4, device=self.device, dtype=torch.bool)
            _, valid = loss_module.NormalTarget(valid_normal, 4, (2, 2))
            ok = bool(
                torch.count_nonzero(invalid).item() == 0
                and torch.all(valid).item())
            print(f"NormalValidityIndependentFromDepth {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"NormalValidityIndependentFromDepth error: {e}")
            return False

    def RunAll(self):
        results = {
            "HebbianConv2d": self.TestHebbianConv2d(),
            "HebbianLinear": self.TestHebbianLinear(),
            "HebbianDecaySignAndCorrelation": self.TestHebbianDecaySignAndCorrelation(),
            "PerceiveExtractorForward": self.TestPerceiveExtractor(),
            "PerceiveExtractorIOShapes": self.TestPerceiveExtractorIOShapes(),
            "PerceiveExtractorStructuredState": self.TestPerceiveExtractorStructuredState(),
            "RGBDGeometryAndSupervision": self.TestRGBDGeometryAndSupervision(),
            "EvalDepthLossAuxiliarySchema": self.TestEvalDepthLossAuxiliarySchema(),
            "QualityStatisticsDType": self.TestQualityStatisticsDType(),
            "SpatialFrequencyEntropyAndCorticalSelectivity": self.TestSpatialFrequencyEntropyAndCorticalSelectivity(),
            "PatchProjectionContentSensitivity": self.TestPatchProjectionContentSensitivity(),
            "GrowableAdapterStateRoundTrip": self.TestGrowableAdapterStateRoundTrip(),
            "InternalRegistrationAndTransientState": self.TestInternalRegistrationAndTransientState(),
            "PartialAdapterLoadPreservesRank": self.TestPartialAdapterLoadPreservesRank(),
            "HebbianPartialRowReset": self.TestHebbianPartialRowReset(),
            "HebbianEvalPlasticity": self.TestHebbianEvalPlasticity(),
            "NonSquarePatchSupervision": self.TestNonSquarePatchSupervision(),
            "AnisotropicDiffusionStability": self.TestAnisotropicDiffusionStability(),
            "CorticalGaborTemporalReset": self.TestCorticalGaborTemporalReset(),
            "LogDepthResampling": self.TestLogDepthResampling(),
            "SPPAndAxialPositionEncoding": self.TestSPPAndAxialPositionEncoding(),
            "ProjectiveTopologyDiagnostics": self.TestProjectiveTopologyDiagnostics(),
            "EnhancedPerceptionAuxiliary": self.TestEnhancedPerceptionAuxiliary(),
            "CameraProjectionGeometry": self.TestCameraProjectionGeometry(),
            "MetricDepthAndPoseAuthority": self.TestMetricDepthAndPoseAuthority(),
            "ObjectTokensCanEncodeCameraGeometry": self.TestObjectTokensCanEncodeCameraGeometry(),
            "EvalSkipsZeroGainEnhancements": self.TestEvalSkipsZeroGainEnhancements(),
            "EnhancementGradientCoverage": self.TestEnhancementGradientCoverage(),
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
            "LossDecreasesWithHebbian": self.LossDecreasesWithHebbian(),
            "HebbianMemoryLifecycle": self.HebbianMemoryLifecycle(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "SmallBatchSafety": self.SmallBatchSafety(),
            "NormalValidityIndependentFromDepth": self.TestNormalValidityIndependentFromDepth(),
            "PartialPreviousVisualMask": self.PartialPreviousVisualMask(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nPerception module tests (with wrapper): {passed}/{len(results)} passed.")
        return results
