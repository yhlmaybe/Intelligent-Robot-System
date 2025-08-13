import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional   


def ProjectFroNorm(tensor: torch.Tensor, maxNorm: Optional[float]):
    if not maxNorm:
        return
    n = torch.linalg.vector_norm(tensor, ord=2)
    
    if torch.isfinite(n) and n > maxNorm:
        tensor.mul_(float(maxNorm) / (n + 1e-12))

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
        useHebbian = False):
        super().__init__()

        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride=stride, padding=padding, bias=bias)

        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap

        self.enable_hebbian_updates = useHebbian  

        self.register_buffer("hebb_memory", torch.zeros_like(self.conv.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_eff = self.conv.weight
        if self.enable_hebbian_updates:
            weight_eff = weight_eff + self.apply_scale * self.hebb_memory  

        out = F.conv2d(
            x, weight_eff, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups)
        
        if self.enable_hebbian_updates:
            with torch.no_grad():
                kH, kW = self.conv.kernel_size
                stride = self.conv.stride
                padding = self.conv.padding

                x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)

                out_unfold = out.view(out.size(0), out.size(1), -1)

                hebb_term = torch.einsum('bik,bjk->ij', out_unfold, x_unfold)

                outC = self.conv.weight.size(0)

                weight_flat = self.conv.weight.view(outC, -1) 

                y2_sum = out_unfold.square().sum(dim=[0, 2])

                decay_term = y2_sum.unsqueeze(1) * weight_flat

                delta_w = self.hebb_rate * (hebb_term - decay_term) # [Cout, Cin*kH*kW]
                delta_w = delta_w.view_as(self.hebb_memory)

                self.hebb_memory.mul_(self.ema_alpha).add_(delta_w, alpha=(1.0 - self.ema_alpha))

                ProjectFroNorm(self.hebb_memory, self.mem_norm_cap) 

        return out

    def ResetHebbianMemory(self):
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
        useHebbian = False):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)
        self.bias = nn.Parameter(torch.zeros(outFeatures)) if bias else None

        self.hebb_rate = float(hebbRate)
        self.ema_alpha = float(emaMomentum)
        self.apply_scale = float(applyScale)
        self.mem_norm_cap = memNormCap

        self.normalize = normalize
        if normalize:
            self.register_buffer("running_mean", torch.zeros(outFeatures))
            self.register_buffer("running_var", torch.ones(outFeatures))
            self.momentum = 0.1

        self.weight_constraint = weightConstraint

        self.enable_hebbian_updates = useHebbian  

        self.register_buffer("hebb_memory", torch.zeros_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight + (self.apply_scale * self.hebb_memory if self.enable_hebbian_updates else 0.0)
        y = F.linear(x, w_eff, self.bias)  # [B, out]

        if self.normalize:
            if self.training:
                with torch.no_grad():
                    mean, var = y.mean(0), y.var(0, unbiased=False)
                    self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
            y_hat = (y - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        else:
            y_hat = y

        if self.enable_hebbian_updates:
            with torch.no_grad():
                hebb_term = torch.einsum('bi,bj->ij', y_hat, x)   
                y_sq_sum = (y_hat ** 2).sum(dim=0)             
                decay_term = torch.einsum('i,ij->ij', y_sq_sum, self.weight)  

                delta_w = self.hebb_rate * (hebb_term - decay_term) 

                self.hebb_memory.mul_(self.ema_alpha).add_(delta_w, alpha=(1.0 - self.ema_alpha))

                if self.weight_constraint == 'clip':
                    self.hebb_memory.copy_(torch.clamp(self.hebb_memory, -1.0, 1.0))
                elif self.weight_constraint == 'norm':
                    self.hebb_memory.copy_(F.normalize(self.hebb_memory, dim=1))

                ProjectFroNorm(self.hebb_memory, self.mem_norm_cap)

        return y_hat

    def ResetHebbianMemory(self) -> None:
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

    def forward(self, src: torch.Tensor,srcMask: Optional[torch.Tensor] = None, srcKeyPaddingMask: Optional[torch.Tensor] = None) -> torch.Tensor:

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
    def __init__(self, inChannels : int, outChannels : int, stride : int = 1, useHebbian : bool = False):
        super().__init__()
        self.downsample = None
        if stride != 1 or inChannels != outChannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels))
        
        self.conv1 = HebbianConv2d(inChannels, outChannels, 3, stride=stride, padding=1, bias=False, useHebbian = useHebbian)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = HebbianConv2d(outChannels, outChannels, 3, stride=1, padding=1, bias=False, useHebbian = useHebbian)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class CNNFeatureExtractor(nn.Module):
    def __init__(self, inChannels : int = 3, baseChannels : int = 64, useHebbian : bool = True):
        super().__init__()

        self.conv1 = HebbianConv2d(inChannels, baseChannels, 7, stride=2, padding=3, bias= False, useHebbian = useHebbian)
        
        self.bn1 = nn.BatchNorm2d(baseChannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.MakeLayer(baseChannels, baseChannels, 2, stride=1, useHebbian=useHebbian)
        self.layer2 = self.MakeLayer(baseChannels, baseChannels*2, 2, stride=2, useHebbian=useHebbian)
        self.layer3 = self.MakeLayer(baseChannels*2, baseChannels*4, 2, stride=2, useHebbian=useHebbian)
        self.layer4 = self.MakeLayer(baseChannels*4, baseChannels*8, 2, stride=2, useHebbian=useHebbian)
        
        self.conv2 = HebbianConv2d(baseChannels*8, baseChannels*16, 3, stride=1, padding=1, bias= False, useHebbian = useHebbian)
        self.bn2 = nn.BatchNorm2d(baseChannels*16)   
    
    def MakeLayer(self, inChannels : int, outChannels : int, blocks : int, stride : int = 1, useHebbian : bool = False) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(inChannels, outChannels, stride, useHebbian))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(outChannels, outChannels, stride=1, useHebbian=useHebbian))
        
        return nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


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

        down_ratio = 32
        fmap_size = imgSize // down_ratio
        assert fmap_size % patchSize == 0, "patch size must divide feature map size"
        num_patches = (fmap_size // patchSize) ** 2
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
            nn.ReLU(inplace=True),
            nn.Linear(embedDim // 4, 1, bias=True))

        self.InitWeights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        feat = self.cnn_extractor(x)  # [B, C, Hf, Wf]
        patches = self.patch_embed(feat)  # [B, embed_dim, Ph, Pw]
        B, C, Ph, Pw = patches.shape
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.transformer_layers:
            x = layer(x)
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

    def ResetFastWeights(self):
        for module in self.modules():
            if hasattr(module, 'ResetHebbianMemory'):
                module.ResetHebbianMemory()



class TestPerceptionMTool:
    def __init__(self):
        pass

    def TestHebbianConv2d(self):
        conv = HebbianConv2d(inChannels=3, outChannels=16, kernelSize=3, stride=1, padding=1, useHebbian=True)
        x = torch.randn(4, 3, 32, 32)
        y = conv(x)
        if y.shape == (4, 16, 32, 32):
            conv.ResetHebbianMemory()
            print("HebbianConv2d test passed.")
            return True
        else:
            print(f"HebbianConv2d output shape mismatch: {y.shape}")
            return False
        

    def TestHebbianLinear(self):
        lin = HebbianLinear(inFeatures=32, outFeatures=64, useHebbian=True)
        x = torch.randn(5, 32)
        y = lin(x)
        if y.shape == (5, 64):
            lin.ResetHebbianMemory()
            print("HebbianLinear test passed.")
            return True
        else:
            print(f"HebbianLinear output shape mismatch: {y.shape}")
            return False


    def TestPerceiveExtractor(self):
        model = PerceiveExtractor(
            imgSize=224,
            patchSize=1,
            embedDim=512,
            numHeads=8,
            numLayers=6,
            useHebbian=True)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        expected_dim = 512 * 2
        if out.shape == (2, expected_dim):
            print("PerceiveExtractor test passed. Output shape:", out.shape)
            return True
        else:
            print(f"PerceiveExtractor output shape mismatch: {out.shape}")
            return False
        




