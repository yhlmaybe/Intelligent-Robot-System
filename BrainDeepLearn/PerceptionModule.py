import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List   
from torch.nn.utils.stateless import functional_call

class HebbianConv2d(nn.Module):
    def __init__(self, inChannels : int, outChannels : int, kernelSize : int, stride : int = 1, padding :int = 0, hebbRate : float = 0.01, bias : bool = False):
        super().__init__()

        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride = stride, padding = padding, bias = bias)
        self.bn = nn.BatchNorm2d(outChannels)
        self.hebb_rate = hebbRate
        self.register_buffer("hebb_memory", torch.zeros_like(self.conv.weight))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = F.relu(self.bn(out))

        with torch.no_grad():
            kH, kW = self.conv.kernel_size
            stride = self.conv.stride
            padding = self.conv.padding
                
            x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
            out_unfold = out.view(out.size(0), out.size(1), -1)
                
            hebb_term = torch.einsum('bik,bjk->ij', out_unfold, x_unfold)

            outC = self.conv.weight.size(0)
            weight_flat = self.conv.weight.view(outC, -1) # [outC, D]
            y2_sum = out_unfold.square().sum(dim=[0,2]) # [outC]
            decay_term = y2_sum.unsqueeze(1) * weight_flat  # [outC, D]

            delta_w = self.hebb_rate * (hebb_term - decay_term)

            delta_w = delta_w.view(self.conv.weight.shape)

            self.hebb_memory.mul_(0.9).add_(delta_w, alpha=0.1)
            self.conv.weight.data.add_(self.hebb_memory) 
        return out
    
    def ResetHebbianMemory(self):
        self.hebb_memory.zero_()

class HebbianLinear(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, hebbRate: float = 0.01, emaMomentum: float = 0.9, normalize: bool = True, weightConstraint: str = 'clip'):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)

        self.bias = nn.Parameter(torch.zeros(outFeatures))

        self.hebb_rate = hebbRate
        self.ema_alpha = emaMomentum
        self.normalize = normalize
        self.weight_constraint = weightConstraint
        
        self.register_buffer("hebb_memory", torch.zeros_like(self.weight))

        if normalize:
            self.register_buffer("running_mean", torch.zeros(outFeatures))
            self.register_buffer("running_var",  torch.ones(outFeatures))
            self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)  # [B, out]

        with torch.no_grad():
            hebb_term = torch.einsum('bi,bj->ij', y, x)      

            y_sq_sum = (y ** 2).sum(dim = 0)
            decay_term = torch.einsum('i,ij->ij', y_sq_sum, self.weight)

            delta_w = self.hebb_rate * (hebb_term - decay_term)

            self.hebb_memory.mul_(self.ema_alpha).add_(delta_w, alpha=(1 - self.ema_alpha))
                
            self.weight.data.add_(self.hebb_memory)
                
            self.ApplyWeightConstraint()
                
            if self.normalize:
                mean, var = y.mean(0), y.var(0)
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)

        if self.normalize:
            y_hat = (y - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        else:
            y_hat = y
            
        return y_hat

    def ApplyWeightConstraint(self):
        if self.weight_constraint == 'clip':
            self.weight.data = torch.clamp(self.weight.data, -1.0, 1.0)
        elif self.weight_constraint == 'norm':
            self.weight.data = F.normalize(self.weight.data, dim=1)
        
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

    def forward(self, src: torch.Tensor,srcMask: torch.Optional[torch.Tensor] = None, srcKeyPaddingMask: torch.Optional[torch.Tensor] = None) -> torch.Tensor:

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
        
        conv_layer = HebbianConv2d if useHebbian else nn.Conv2d
        self.conv1 = conv_layer(inChannels, outChannels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = conv_layer(outChannels, outChannels, 3, stride=1, padding=1, bias=False)
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
        conv_layer = HebbianConv2d if useHebbian else nn.Conv2d
        self.conv1 = conv_layer(inChannels, baseChannels, 7, stride=2, padding=3, bias= False)
        
        self.bn1 = nn.BatchNorm2d(baseChannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.MakeLayer(baseChannels, baseChannels, 2, stride=1, useHebbian=useHebbian)
        self.layer2 = self.MakeLayer(baseChannels, baseChannels*2, 2, stride=2, useHebbian=useHebbian)
        self.layer3 = self.MakeLayer(baseChannels*2, baseChannels*4, 2, stride=2, useHebbian=useHebbian)
        self.layer4 = self.MakeLayer(baseChannels*4, baseChannels*8, 2, stride=2, useHebbian=useHebbian)
        
        self.conv2 = conv_layer(baseChannels*8, baseChannels*16, 3, stride=1, padding=1, bias= False)
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
        if useHebbian:
            layers.append(HebbianLinear(hidden_dim, hidden_dim, hebbRate=hebbRate))
        layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, embedDim, bias=True))
        layers.append(nn.GELU())
        if useHebbian:
            layers.append(HebbianLinear(embedDim, embedDim, hebbRate=hebbRate))
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

    def ResetHebbianMemory(self):
        for module in self.modules():
            if hasattr(module, 'ResetHebbianMemory'):
                module.ResetHebbianMemory()



class TestPerceptionModule:
    def __init__(self):
        pass

    def TestHebbianConv2d(self):
        conv = HebbianConv2d(inChannels=3, outChannels=16, kernelSize=3, stride=1, padding=1)
        x = torch.randn(4, 3, 32, 32)
        y = conv(x)
        if y.shape == (4, 16, 32, 32):
            conv.ResetHebbianMemory()
            print("HebbianConv2d test passed.")
        else:
            print(f"HebbianConv2d output shape mismatch: {y.shape}")
        

    def TestHebbianLinear(self):
        lin = HebbianLinear(inFeatures=32, outFeatures=64)
        x = torch.randn(5, 32)
        y = lin(x)
        if y.shape == (5, 64):
            lin.ResetHebbianMemory()
            print("HebbianLinear test passed.")
        else:
            print(f"HebbianLinear output shape mismatch: {y.shape}")


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
        else:
            print(f"PerceiveExtractor output shape mismatch: {out.shape}")
        




