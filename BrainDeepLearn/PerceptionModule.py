import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Iterable, Tuple, Any
from contextlib import contextmanager



def ProjectFroNorm(tensor: torch.Tensor, maxNorm: Optional[float]):
    if not maxNorm:
        return
    with torch.no_grad():
        n = torch.linalg.vector_norm(tensor, ord=2)
        if torch.isfinite(n) and (n > maxNorm):
            tensor.mul_(float(maxNorm) / (n + 1e-12))


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
        A = init.get("A", torch.randn(addRank, self.cin * ksz) * 1e-4)
        B = init.get("B", torch.zeros(self.cout, addRank))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous())
        B = nn.Parameter(B.contiguous())
        s = nn.Parameter(torch.tensor(float(s)))

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
            delta = delta + s * (B @ A)
        return delta.view(self.cout, self.cin, self.kh, self.kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None:
            w = w + delta
        return F.conv2d(
            x, w, self.target.bias,
            stride=self.target.stride,
            padding=self.target.padding,
            dilation=self.target.dilation,
            groups=self.target.groups)


class GrowableConv1x1Adapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        self.A_list = nn.ParameterList() 
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        A = init.get("A", torch.randn(addRank, self.C, 1, 1) * 1e-4)
        B = init.get("B", torch.zeros(self.C, addRank, 1, 1))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous())
        B = nn.Parameter(B.contiguous())
        s = nn.Parameter(torch.tensor(float(s)))

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
            y = y + s * z
        return y


class GrowableTokenAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.D = dim
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if init is None: init = {}
        A = init.get("A", torch.randn(addRank, self.D) * 1e-4)
        B = init.get("B", torch.zeros(self.D, addRank))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous())
        B = nn.Parameter(B.contiguous())
        s = nn.Parameter(torch.tensor(float(s)))

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
            y = y + s * z
        return y


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
                nn.BatchNorm2d(outChannels))
            
        self.conv1 = HebbianConv2d(inChannels, outChannels, 3, stride=stride, padding=1,bias=False, useHebbian=useHebbian)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = HebbianConv2d(outChannels, outChannels, 3, stride=1, padding=1,bias=False, useHebbian=useHebbian)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=False) 

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
        
        self.bn1 = nn.BatchNorm2d(baseChannels)
        self.relu = nn.ReLU(inplace=False) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(baseChannels, baseChannels, blocks=2, stride=1, useHebbian=useHebbian)
        self.layer2 = self._make_layer(baseChannels, baseChannels*2, blocks=2, stride=2, useHebbian=useHebbian)
        self.layer3 = self._make_layer(baseChannels*2, baseChannels*4, blocks=2, stride=2, useHebbian=useHebbian)
        self.layer4 = self._make_layer(baseChannels*4, baseChannels*8, blocks=2, stride=2, useHebbian=useHebbian)

        self.conv2 = HebbianConv2d(baseChannels*8, baseChannels*16, 3, stride=1, padding=1,
                                   bias=False, useHebbian=useHebbian)
        self.bn2 = nn.BatchNorm2d(baseChannels*16)

    def _make_layer(self, inC, outC, blocks, stride, useHebbian):
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
    def __init__(self,
                 imgSize: int = 224,
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

        self.patch_embed = nn.Conv2d(
            in_channels=cnn_feat_dim,
            out_channels=embedDim,
            kernel_size=patchSize,
            stride=patchSize,
            bias=False)
        
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
            nn.ReLU(),
            nn.Linear(embedDim // 4, 1, bias=True),
            nn.Sigmoid())

        self.output_norm = nn.LayerNorm(embedDim, eps=1e-6)

        self.patch_aggregator = nn.Sequential(
            nn.Linear(embedDim, embedDim // 4, bias= True),
            nn.ReLU(inplace=False),
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
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def ResetHebbianMemory(self):
        for module in self.modules():
            if hasattr(module, 'ResetHebbianMemory'):
                module.ResetHebbianMemory()



class PerceptionOnlineWrapper(nn.Module):
    def __init__(self,
                 base: nn.Module,
                 initFeatRank: int = 8,
                 initPatchRank: int = 8,
                 initTokenRank: int = 8,
                 autoRank: bool = True,
                 evThreshold: float = 0.90,
                 maxRank: int = 64,
                 maxRankFeat: Optional[int] = None,
                 maxRankPatch: Optional[int] = None,
                 maxRankToken: Optional[int] = None,
                 rankBootstrap: int = 1,
                 gradEma: float = 0.9,
                 verifyAtol: float = 1e-6,
                 verifyRtol: float = 1e-4):
        super().__init__()
        self.base = base
        self.device = next(self.base.parameters()).device
        self.dtype  = next(self.base.parameters()).dtype

        self._C = self.base.patch_embed.weight.size(1)
        self._E = self.base.patch_embed.weight.size(0)
        self._kh, self._kw = self.base.patch_embed.weight.shape[2:]
        self._D = self.base.pos_embed.size(-1)
        self._L = len(self.base.transformer_layers)

        self.auto_rank = bool(autoRank)
        self.ev_threshold = float(evThreshold)
        self.max_rank = int(maxRank)
        self.max_rank_feat = int(maxRankFeat) if maxRankFeat is not None else self.max_rank
        self.max_rank_patch = int(maxRankPatch) if maxRankPatch is not None else self.max_rank
        self.max_rank_token = int(maxRankToken) if maxRankToken is not None else self.max_rank
        self.rank_bootstrap = int(rankBootstrap)
        self.grad_ema = float(gradEma)

        self.verify_atol = float(verifyAtol)
        self.verify_rtol = float(verifyRtol)

        for p in self.base.parameters():
            p.requires_grad_(False)

        self._cand = None
        self._grad_ema_buf = None
        self.InitCandidates(initFeatRank, initPatchRank, initTokenRank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ForwardWithCandidates(x, self._cand)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def Update(self, action: str, **kwargs) -> Dict[str, Any]:
        act = str(action).lower()

        if act == "reset":
            self._cand = None
            self._grad_ema_buf = None
            feat = int(kwargs.get("featRank", 0))
            patch = int(kwargs.get("patchRank", 0))
            token = int(kwargs.get("tokenRankEach", 0))
            self.InitCandidates(feat, patch, token)
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "ranks":
            return {"ranks": self.CurrentRanks()}

        elif act == "accumulategrads":
            self.AccumulateGrads()
            return {"ok": True}

        elif act == "grow":
            gf = float(kwargs.get("growFactor", 2.0))
            add_feat = int(kwargs.get("addFeat",  0))
            add_patch = int(kwargs.get("addPatch", 0))
            add_token = int(kwargs.get("addTokenEach", 0))
            self.GrowCandidates(growFactor=gf, addFeat=add_feat, addPatch=add_patch, addTokenEach=add_token)
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "autogrow":
            self.CandidatesAuto()
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "commit":
            stats = self.WritebackToBase(self._cand, sample=kwargs.get("sample", None))
            self._cand = None
            self._grad_ema_buf = None
            self.InitCandidates(0, 0, 0)
            return {"ok": True, "commit_stats": stats, "ranks": self.CurrentRanks()}

        elif act == "rollback":
            self._cand = None
            self._grad_ema_buf = None
            self.InitCandidates(0, 0, 0)
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "set":
            if "evThreshold" in kwargs: self.ev_threshold = float(kwargs["evThreshold"])
            if "maxRank" in kwargs: self.max_rank = int(kwargs["maxRank"])
            if "maxRankFeat" in kwargs: self.max_rank_feat = int(kwargs["maxRankFeat"])
            if "maxRankPatch" in kwargs: self.max_rank_patch = int(kwargs["maxRankPatch"])
            if "maxRankToken" in kwargs: self.max_rank_token = int(kwargs["maxRankToken"])
            if "gradEma" in kwargs: self.grad_ema = float(kwargs["gradEma"])
            return {"ok": True, "settings": {
                "evThreshold": self.ev_threshold,
                "maxRank": self.max_rank,
                "maxRankFeat": self.max_rank_feat,
                "maxRankPatch": self.max_rank_patch,
                "maxRankToken": self.max_rank_token,
                "gradEma": self.grad_ema,}}

        else:
            raise ValueError(f"Unknown action for Update(): {action}")

    def InitCandidates(self, featRank: int, patchRank: int, tokenRankEach: int):
        dev, dt = self.device, self.dtype
        def plist(): return nn.ParameterList()
        def ml(n): return [nn.ParameterList() for _ in range(n)]
        self._cand = {
            "active": True,
            "feat_A": plist(), "feat_B": plist(), "feat_s": plist(),
            "patch_A": plist(), "patch_B": plist(), "patch_s": plist(),
            "token_A": ml(self._L), "token_B": ml(self._L), "token_s": ml(self._L),}
        if featRank > 0:
            A = nn.Parameter(torch.randn(featRank, self._C, 1, 1, device=dev, dtype=dt) * 1e-4)
            B = nn.Parameter(torch.zeros(self._C, featRank, 1, 1, device=dev, dtype=dt))
            s = nn.Parameter(torch.tensor(1e-3, device=dev, dtype=dt))
            self._cand["feat_A"].append(A); self._cand["feat_B"].append(B); self._cand["feat_s"].append(s)
        if patchRank > 0:
            ksz = self._kh * self._kw
            A = nn.Parameter(torch.randn(patchRank, self._C * ksz, device=dev, dtype=dt) * 1e-4)
            B = nn.Parameter(torch.zeros(self._E, patchRank, device=dev, dtype=dt))
            s = nn.Parameter(torch.tensor(1e-3, device=dev, dtype=dt))
            self._cand["patch_A"].append(A); self._cand["patch_B"].append(B); self._cand["patch_s"].append(s)
        if tokenRankEach > 0:
            for i in range(self._L):
                A = nn.Parameter(torch.randn(tokenRankEach, self._D, device=dev, dtype=dt) * 1e-4)
                B = nn.Parameter(torch.zeros(self._D, tokenRankEach, device=dev, dtype=dt))
                s = nn.Parameter(torch.tensor(1e-3, device=dev, dtype=dt))
                self._cand["token_A"][i].append(A); self._cand["token_B"][i].append(B); self._cand["token_s"][i].append(s)

    def CurrentRanks(self) -> Dict[str, Any]:
        eps = 1e-12
        def _eff_rank(plist_A, plist_s):
            r = 0
            for A, s in zip(plist_A, plist_s):
                sval = float(s.detach().item()) if torch.is_tensor(s) else float(s)
                if abs(sval) > eps:
                    r += int(A.size(0))
            return r
        fr = _eff_rank(self._cand["feat_A"],  self._cand["feat_s"])
        pr = _eff_rank(self._cand["patch_A"], self._cand["patch_s"])
        tr = []
        for i in range(self._L):
            tr.append(_eff_rank(self._cand["token_A"][i], self._cand["token_s"][i]))
        return {"feat": fr, "patch": pr, "token": tr}

    def CandParameters(self):
        for k in ["feat_A","feat_B","feat_s","patch_A","patch_B","patch_s"]:
            for p in self._cand[k]:
                if p.requires_grad: yield p
        for i in range(self._L):
            for p in self._cand["token_A"][i]:
                if p.requires_grad: yield p
            for p in self._cand["token_B"][i]:
                if p.requires_grad: yield p
            for p in self._cand["token_s"][i]:
                if p.requires_grad: yield p

    @torch.no_grad()
    def AccumulateGrads(self):
        if self._grad_ema_buf is None:
            self._grad_ema_buf = {"feat":{"A":[], "B":[]}, "patch":{"A":[], "B":[]}, "token":[{"A":[], "B":[]} for _ in range(self._L)]}
        def _ema_store(dst_list, src_tensor_list):
            if len(dst_list) < len(src_tensor_list):
                dst_list.clear()
            out = []
            for i, t in enumerate(src_tensor_list):
                if t is None:
                    out.append(dst_list[i] if i < len(dst_list) else None)
                else:
                    val = t.detach()
                    if i < len(dst_list) and dst_list[i] is not None:
                        val = self.grad_ema * dst_list[i] + (1 - self.grad_ema) * val
                    out.append(val)
            return out
        
        gA = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["feat_A"]]
        gB = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["feat_B"]]
        self._grad_ema_buf["feat"]["A"] = _ema_store(self._grad_ema_buf["feat"]["A"], gA)
        self._grad_ema_buf["feat"]["B"] = _ema_store(self._grad_ema_buf["feat"]["B"], gB)
        gA = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["patch_A"]]
        gB = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["patch_B"]]
        self._grad_ema_buf["patch"]["A"] = _ema_store(self._grad_ema_buf["patch"]["A"], gA)
        self._grad_ema_buf["patch"]["B"] = _ema_store(self._grad_ema_buf["patch"]["B"], gB)
        for i in range(self._L):
            gA = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["token_A"][i]]
            gB = [p.grad.clone() if (p.grad is not None) else None for p in self._cand["token_B"][i]]
            self._grad_ema_buf["token"][i]["A"] = _ema_store(self._grad_ema_buf["token"][i]["A"], gA)
            self._grad_ema_buf["token"][i]["B"] = _ema_store(self._grad_ema_buf["token"][i]["B"], gB)

    @torch.no_grad()
    def GrowCandidates(self, growFactor: float = 2.0, addFeat: int = 0, addPatch: int = 0, addTokenEach: int = 0):
        def _maybe_grow(plist_A, plist_B, plist_s, unit, add_explicit):
            cur = sum(p.size(0) for p in plist_A)
            target = max(cur, cur if growFactor <= 1.0 else int(round(cur * growFactor)))
            target += max(0, add_explicit)
            if cur == 0 and target == 0:
                return
            if unit == "feat":
                target = min(target, self.max_rank_feat)
                add = max(0, target - cur)
                if add <= 0: return
                A = nn.Parameter(torch.randn(add, self._C, 1, 1, device=self.device, dtype=self.dtype) * 1e-4)
                B = nn.Parameter(torch.zeros(self._C, add, 1, 1, device=self.device, dtype=self.dtype))
            elif unit == "patch":
                target = min(target, self.max_rank_patch)
                add = max(0, target - cur)
                if add <= 0: return
                ksz = self._kh * self._kw
                A = nn.Parameter(torch.randn(add, self._C*ksz, device=self.device, dtype=self.dtype) * 1e-4)
                B = nn.Parameter(torch.zeros(self._E, add, device=self.device, dtype=self.dtype))
            else:
                raise ValueError
            s = nn.Parameter(torch.tensor(1e-3, device=self.device, dtype=self.dtype))
            plist_A.append(A); plist_B.append(B); plist_s.append(s)

        _maybe_grow(self._cand["feat_A"], self._cand["feat_B"], self._cand["feat_s"], "feat", addFeat)
        _maybe_grow(self._cand["patch_A"], self._cand["patch_B"], self._cand["patch_s"], "patch", addPatch)

        for i in range(self._L):
            cur = sum(p.size(0) for p in self._cand["token_A"][i])
            target = max(cur, cur if growFactor <= 1.0 else int(round(cur * growFactor)))
            target += max(0, addTokenEach)
            if cur == 0 and target == 0: 
                continue
            target = min(target, self.max_rank_token)
            add = max(0, target - cur)
            if add <= 0: 
                continue
            A = nn.Parameter(torch.randn(add, self._D, device=self.device, dtype=self.dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(self._D, add, device=self.device, dtype=self.dtype))
            s = nn.Parameter(torch.tensor(1e-3, device=self.device, dtype=self.dtype))
            self._cand["token_A"][i].append(A)
            self._cand["token_B"][i].append(B)
            self._cand["token_s"][i].append(s)

    def CandidatesAuto(self):
        dev, dt = self.device, self.dtype

        def _ensure_seed(plist_A, plist_B, plist_s, add, unit, D=None, E=None, C=None, kh=None, kw=None):
            if add <= 0: return
            if unit == "feat":
                A = nn.Parameter(torch.randn(add, C, 1, 1, device=dev, dtype=dt) * 1e-4)
                B = nn.Parameter(torch.zeros(C, add, 1, 1, device=dev, dtype=dt))
            elif unit == "patch":
                ksz = kh * kw
                A = nn.Parameter(torch.randn(add, C * ksz, device=dev, dtype=dt) * 1e-4)
                B = nn.Parameter(torch.zeros(E, add, device=dev, dtype=dt))
            elif unit == "token":
                A = nn.Parameter(torch.randn(add, D, device=dev, dtype=dt) * 1e-4)
                B = nn.Parameter(torch.zeros(D, add, device=dev, dtype=dt))
            else:
                raise ValueError
            s = nn.Parameter(torch.tensor(1e-3, device=dev, dtype=dt))
            plist_A.append(A); plist_B.append(B); plist_s.append(s)

        def _build_delta(plist_A, plist_B, plist_s, out_dim, in_dim):
            if len(plist_A) == 0:
                return torch.zeros(out_dim, in_dim, device=dev, dtype=dt)
            Delta = torch.zeros(out_dim, in_dim, device=dev, dtype=dt)
            for A, B, s in zip(plist_A, plist_B, plist_s):
                A2 = A.reshape(A.shape[0], in_dim)
                B2 = B.reshape(out_dim, A.shape[0])
                Delta = Delta + float(s.detach()) * (B2 @ A2)
            return Delta

        def _svd_truncate(Delta, r_star):
            U, S, Vh = torch.linalg.svd(Delta, full_matrices=False) 
            r = int(min(r_star, S.numel()))
            if r <= 0:
                return None, None, None
            U_r = U[:, :r].contiguous()
            S_r = S[:r].contiguous()
            Vh_r = Vh[:r, :].contiguous() 
            return U_r, S_r, Vh_r

        def _replace_site(plist_A, plist_B, plist_s,
                  U_r, S_r, Vh_r, unit, D=None, E=None, C=None, kh=None, kw=None):
            if U_r is None: 
                for j in range(len(plist_s)):
                    with torch.no_grad():
                        if isinstance(plist_s[j], nn.Parameter):
                            plist_s[j].data.zero_()
                return

            sqrtS = torch.sqrt(S_r).unsqueeze(0)
            B_new_2D = U_r * sqrtS
            A_new_2D = (sqrtS.t() @ Vh_r).contiguous()

            if unit == "feat": 
                A_new = A_new_2D.view(A_new_2D.size(0), C, 1, 1).contiguous()
                B_new = B_new_2D.view(C, B_new_2D.size(1), 1, 1).contiguous()
            elif unit == "patch": 
                ksz = kh * kw
                A_new = A_new_2D.view(A_new_2D.size(0), C * ksz).contiguous()
                B_new = B_new_2D.view(E, B_new_2D.size(1)).contiguous()
            elif unit == "token": 
                A_new = A_new_2D.view(A_new_2D.size(0), D).contiguous()
                B_new = B_new_2D.view(D, B_new_2D.size(1)).contiguous()
            else:
                raise ValueError

            if len(plist_A) == 0:
                plist_A.append(nn.Parameter(A_new))
                plist_B.append(nn.Parameter(B_new))
                plist_s.append(nn.Parameter(torch.tensor(1.0, device=A_new.device, dtype=A_new.dtype)))
            else:
                plist_A[0] = nn.Parameter(A_new)
                plist_B[0] = nn.Parameter(B_new)
                plist_s[0] = nn.Parameter(torch.tensor(1.0, device=A_new.device, dtype=A_new.dtype))
                for j in range(1, len(plist_s)):
                    with torch.no_grad():
                        if isinstance(plist_s[j], nn.Parameter):
                            plist_s[j].data.zero_()

        def _suggest_rank_from_site(gradA_list, gradB_list, A_list, B_list, s_list, out_dim, in_dim, max_rank_site):
            G_acc = None; cnt = 0
            for gA, gB, A, B, s in zip(gradA_list, gradB_list, A_list, B_list, s_list):
                if gA is None and gB is None:
                    continue
                s_val = float(s.detach().item()) if torch.is_tensor(s) else float(s)
                s_val = s_val if abs(s_val) > 1e-12 else 1.0
                A2 = A.reshape(A.shape[0], in_dim)
                B2 = B.reshape(out_dim, A.shape[0])
                Gs = []
                if gA is not None:
                    Bt = B2.t()
                    G_A = (torch.linalg.pinv(Bt) @ gA.reshape_as(A2)) / s_val
                    Gs.append(G_A)
                if gB is not None:
                    At = A2.t()
                    G_B = (gB.reshape(out_dim, A.shape[0]) @ torch.linalg.pinv(At)) / s_val
                    Gs.append(G_B)
                if len(Gs) == 0:
                    continue
                G_est = sum(Gs) / len(Gs) 
                G_acc = G_est if G_acc is None else (G_acc + G_est)
                cnt += 1

            if cnt == 0 or G_acc is None:
                return 0
            G = (G_acc / cnt).detach()
            svals = torch.linalg.svdvals(G)
            if svals.numel() == 0:
                return 0
            ev = (svals**2).cumsum(0) / (svals**2).sum()
            r_need = int((ev < self.ev_threshold).sum().item() + 1)
            r_need = max(0, min(r_need, max_rank_site, min(out_dim, in_dim)))
            return r_need

        ranks = self.CurrentRanks()

        if self._grad_ema_buf is None:
            if ranks["feat"] == 0:
                _ensure_seed(self._cand["feat_A"], self._cand["feat_B"], self._cand["feat_s"],self.rank_bootstrap, unit="feat", C=self._C)
            if ranks["patch"] == 0:
                _ensure_seed(self._cand["patch_A"], self._cand["patch_B"], self._cand["patch_s"], self.rank_bootstrap, unit="patch", C=self._C, E=self._E, kh=self._kh, kw=self._kw)
            for i in range(self._L):
                if ranks["token"][i] == 0:
                    _ensure_seed(self._cand["token_A"][i], self._cand["token_B"][i], self._cand["token_s"][i], self.rank_bootstrap, unit="token", D=self._D)
            return

        GB = self._grad_ema_buf 

        cur_feat = ranks["feat"]
        if cur_feat > 0 or (cur_feat == 0 and len(self._cand["feat_A"]) > 0):
            r_target = _suggest_rank_from_site(
                GB["feat"]["A"], GB["feat"]["B"],
                list(self._cand["feat_A"]), list(self._cand["feat_B"]), list(self._cand["feat_s"]),out_dim=self._C, in_dim=self._C, max_rank_site=self.max_rank_feat)
            if r_target > cur_feat:
                add = r_target - cur_feat
                _ensure_seed(self._cand["feat_A"], self._cand["feat_B"], self._cand["feat_s"],add, unit="feat", C=self._C)
            elif 0 <= r_target < cur_feat:
                Delta = _build_delta(self._cand["feat_A"], self._cand["feat_B"], self._cand["feat_s"],out_dim=self._C, in_dim=self._C)
                U_r, S_r, Vh_r = _svd_truncate(Delta, r_target)
                _replace_site(self._cand["feat_A"], self._cand["feat_B"], self._cand["feat_s"], U_r, S_r, Vh_r, unit="feat", C=self._C)

        cur_patch = ranks["patch"]; ksz = self._kh * self._kw
        if cur_patch > 0 or (cur_patch == 0 and len(self._cand["patch_A"]) > 0):
            r_target = _suggest_rank_from_site(
                GB["patch"]["A"], GB["patch"]["B"],
                list(self._cand["patch_A"]), list(self._cand["patch_B"]), list(self._cand["patch_s"]),out_dim=self._E, in_dim=self._C * ksz, max_rank_site=self.max_rank_patch)
            if r_target > cur_patch:
                add = r_target - cur_patch
                _ensure_seed(self._cand["patch_A"], self._cand["patch_B"], self._cand["patch_s"], add, unit="patch", C=self._C, E=self._E, kh=self._kh, kw=self._kw)
            elif 0 <= r_target < cur_patch:
                Delta = _build_delta(self._cand["patch_A"], self._cand["patch_B"], self._cand["patch_s"], out_dim=self._E, in_dim=self._C * ksz)
                U_r, S_r, Vh_r = _svd_truncate(Delta, r_target)
                _replace_site(self._cand["patch_A"], self._cand["patch_B"], self._cand["patch_s"],U_r, S_r, Vh_r, unit="patch", C=self._C, E=self._E, kh=self._kh, kw=self._kw)

        for i in range(self._L):
            cur_tok = ranks["token"][i]
            if cur_tok > 0 or (cur_tok == 0 and len(self._cand["token_A"][i]) > 0):
                r_target = _suggest_rank_from_site(
                    GB["token"][i]["A"], GB["token"][i]["B"],
                    list(self._cand["token_A"][i]), list(self._cand["token_B"][i]), list(self._cand["token_s"][i]),out_dim=self._D, in_dim=self._D, max_rank_site=self.max_rank_token)
                if r_target > cur_tok:
                    add = r_target - cur_tok
                    _ensure_seed(self._cand["token_A"][i], self._cand["token_B"][i], self._cand["token_s"][i],add, unit="token", D=self._D)
                elif 0 <= r_target < cur_tok:
                    Delta = _build_delta(self._cand["token_A"][i], self._cand["token_B"][i], self._cand["token_s"][i],out_dim=self._D, in_dim=self._D)
                    U_r, S_r, Vh_r = _svd_truncate(Delta, r_target)
                    _replace_site(self._cand["token_A"][i], self._cand["token_B"][i], self._cand["token_s"][i],U_r, S_r, Vh_r, unit="token", D=self._D)

    def ForwardWithCandidates(self, x: torch.Tensor, P: Dict[str, Any]) -> torch.Tensor:
        feat_in = self.base.cnn_extractor(x)
        feat = self.base.cnn_feat_adapter(feat_in)
        if len(P["feat_A"]) > 0:
            y = feat
            for A, Bp, s in zip(P["feat_A"], P["feat_B"], P["feat_s"]):
                z = F.conv2d(feat_in, A, bias=None, stride=1, padding=0)
                z = F.conv2d(z, Bp, bias=None, stride=1, padding=0)
                y = y + s * z
            feat = y

        W = self.base.patch_embed.weight
        W_eff = W
        base_delta = self.base.patch_adapter.DeltaWeight()
        if base_delta is not None:
            W_eff = W_eff + base_delta
        if len(P["patch_A"]) > 0:
            delta = W.new_zeros(self._E, self._C, self._kh, self._kw)
            for A, Bp, s in zip(P["patch_A"], P["patch_B"], P["patch_s"]):
                delta = delta + s * (Bp @ A).view(self._E, self._C, self._kh, self._kw)
            W_eff = W_eff + delta

        patches = F.conv2d(
            feat, W_eff, bias=None,
            stride=self.base.patch_embed.stride,
            padding=self.base.patch_embed.padding,
            dilation=self.base.patch_embed.dilation,
            groups=self.base.patch_embed.groups)

        Bsz, Edim, Ph, Pw = patches.shape
        tokens = rearrange(patches, 'b c h w -> b (h w) c')
        cls_tokens = repeat(self.base.cls_token, '1 1 d -> b 1 d', b=Bsz)
        x_tok = torch.cat([cls_tokens, tokens], dim=1)
        x_tok = x_tok + self.base.pos_embed
        x_tok = self.base.pos_drop(x_tok)

        for i, layer in enumerate(self.base.transformer_layers):
            x_layer = layer(x_tok)
            token_in = x_layer
            base_out = self.base.token_adapters[i](token_in)
            if len(P["token_A"][i]) > 0:
                add = 0.0
                for A, Bp, s in zip(P["token_A"][i], P["token_B"][i], P["token_s"][i]):
                    add = add + s * ((token_in @ A.t()) @ Bp.t())
                x_tok = base_out + add
            else:
                x_tok = base_out

        x_tok = self.base.encoder_norm(x_tok)

        cls_rep = x_tok[:, 0, :]
        mlp_out = self.base.mlp(cls_rep)
        gate = self.base.adaptive_gate(mlp_out)
        out = gate * mlp_out + (1 - gate) * cls_rep
        out = self.base.output_norm(out)

        patch_tokens = x_tok[:, 1:, :]
        patch_scores = self.base.patch_aggregator(patch_tokens).squeeze(-1)
        patch_weights = F.softmax(patch_scores, dim=1)
        global_patch = (patch_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)

        return torch.cat([out, global_patch], dim=1)

    @torch.no_grad()
    def WritebackToBase(self, P: Dict[str, Any], sample: Optional[torch.Tensor]) -> Dict[str, float]:
        if P is None or not P.get("active", False):
            raise RuntimeError("No candidate adapters to commit.")
        if sample is None:
            B = 2; H = getattr(self.base, "img_size", 224)
            sample = torch.randn(B, 3, H, H, device=self.device, dtype=self.dtype)

        with self.HebbDisabled():
            y_before = self.ForwardWithCandidates(sample, P)

            for A, Bp, s in zip(P["feat_A"], P["feat_B"], P["feat_s"]):
                init = {"A": A.detach().clone(), "B": Bp.detach().clone(), "scale": float(s.item())}
                self.base.cnn_feat_adapter.Grow(addRank=A.size(0), init=init, freezeOld=False)
            for A, Bp, s in zip(P["patch_A"], P["patch_B"], P["patch_s"]):
                init = {"A": A.detach().clone(), "B": Bp.detach().clone(), "scale": float(s.item())}
                self.base.patch_adapter.Grow(addRank=A.size(0), init=init, freezeOld=False)
            for i in range(self._L):
                for A, Bp, s in zip(P["token_A"][i], P["token_B"][i], P["token_s"][i]):
                    init = {"A": A.detach().clone(), "B": Bp.detach().clone(), "scale": float(s.item())}
                    self.base.token_adapters[i].Grow(addRank=A.size(0), init=init, freezeOld=False)

            y_after = self.base(sample)

        max_abs = (y_after - y_before).abs().max().item()
        max_rel = ((y_after - y_before).abs() / (y_before.abs() + 1e-9)).max().item()
        ok = (max_abs <= self.verify_atol) or (max_rel <= self.verify_rtol)
        if not ok:
            raise RuntimeError(f"[Commit verify failed] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
        return {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "verified_equal": float(ok)}

    def CollectHebbModules(self):
        mods = []
        for m in self.base.modules():
            if hasattr(m, "enable_hebbian_updates"):
                mods.append(m)
        return mods

    @contextmanager
    def HebbDisabled(self):
        mods = self.CollectHebbModules()
        old = [bool(m.enable_hebbian_updates) for m in mods]
        try:
            for m in mods:
                m.enable_hebbian_updates = False
            yield
        finally:
            for m, v in zip(mods, old):
                m.enable_hebbian_updates = v



class TestPerceptionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)


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
            model = PerceiveExtractor(imgSize=224, patchSize=1, embedDim=512, numHeads=8, numLayers=6, useHebbian=True).to(self.device)
            x = torch.randn(2, 3, 224, 224, device=self.device)
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


    def SumRanksFeat(self, adapter) -> int:
        return sum(int(A.shape[0]) for A in adapter.A_list)

    def SumRanksPatch(self, adapter) -> int:
        return sum(int(A.shape[0]) for A in adapter.A_list)

    def SumRanksToken(self, ta) -> int:
        return sum(int(A.shape[0]) for A in ta.A_list)


    def WrapperForwardEqualWhenNoInitRank(self):
        try:
            base = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8,
                                     numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()
            wrapper = PerceptionOnlineWrapper(base=base, initFeatRank=0, initPatchRank=0, initTokenRank=0).to(self.device)
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
            base = PerceiveExtractor(imgSize=64, patchSize=1, embedDim=64, numHeads=8,
                                     numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()
            wrapper = PerceptionOnlineWrapper(base=base, initFeatRank=0, initPatchRank=0, initTokenRank=0).to(self.device)
            wrapper.train()

            r = wrapper.Update("ranks")["ranks"]
            assert r["feat"] == 0 and r["patch"] == 0 and all(v == 0 for v in r["token"])

            wrapper.Update("grow", growFactor=2.0, addFeat=2, addPatch=2, addTokenEach=1)
            r2 = wrapper.Update("ranks")["ranks"]
            assert r2["feat"] >= 2 and r2["patch"] >= 2 and all(v >= 1 for v in r2["token"])

            wrapper.Update("accumulategrads")

            st = wrapper.Update("set", evThreshold=0.85, maxRank=64, gradEma=0.8)
            assert st["ok"] and abs(st["settings"]["evThreshold"] - 0.85) < 1e-12

            wrapper.Update("rollback")
            r3 = wrapper.Update("ranks")["ranks"]
            assert r3["feat"] == 0 and r3["patch"] == 0 and all(v == 0 for v in r3["token"])

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
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64,numHeads=8, numLayers=2, baseChannels=16,useHebbian=False).to(self.device)
            base.eval()

            wrapper = PerceptionOnlineWrapper(
                base=base, initFeatRank=2, initPatchRank=2, initTokenRank=1,
                verifyAtol=1e-5, verifyRtol=1e-4).to(self.device)
            wrapper.train()

            head = nn.Linear(128, 16).to(self.device).train()

            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=3e-3)

            _ = wrapper.Update("grow", growFactor=2.0)

            for _ in range(10):
                x = torch.randn(8, 3, img_size, img_size, device=self.device)
                y = torch.randn(8, 16, device=self.device)
                pred = head(wrapper(x))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            wrapper.eval()

            sample = torch.randn(4, 3, img_size, img_size, device=self.device)
            res = wrapper.Update("commit", sample=sample)
            assert res["ok"] and res["commit_stats"]["verified_equal"], "Commit verify failed."

            base.eval(); wrapper.eval()
            x = torch.randn(2, 3, img_size, img_size, device=self.device)
            with torch.no_grad():
                y0 = base(x)
                y1 = wrapper(x)
            max_abs = (y0 - y1).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper vs base mismatch after commit: {max_abs:.3e}"

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
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64,numHeads=8, numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            base.eval()

            wrapper = PerceptionOnlineWrapper(
                base=base, initFeatRank=4, initPatchRank=0, initTokenRank=0,
                evThreshold=0.90, maxRankFeat=8, verifyAtol=1e-5, verifyRtol=1e-4).to(self.device)
            wrapper.train()

            ranks_before = wrapper.Update("ranks")["ranks"]["feat"]
            assert ranks_before >= 4, f"expected feat rank >=4, got {ranks_before}"

            L = len(wrapper._cand["feat_A"])
            wrapper._grad_ema_buf = {
                "feat":  {"A": [None]*L, "B": [None]*L},
                "patch": {"A": [], "B": []},
                "token": [{"A": [], "B": []} for _ in range(wrapper._L)],}

            wrapper.Update("autogrow")
            ranks_after = wrapper.Update("ranks")["ranks"]["feat"]
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
            base = PerceiveExtractor(imgSize=img_size, patchSize=1, embedDim=64, numHeads=8,
                                     numLayers=2, baseChannels=16, useHebbian=False).to(self.device)
            wrapper = PerceptionOnlineWrapper(base=base, initFeatRank=0, initPatchRank=0, initTokenRank=0).to(self.device)
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
            "WrapperPipelineCompatible": self.WrapperPipelineCompatible(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\nPerception module tests (with wrapper): {passed}/{len(results)} passed.")
        return results

