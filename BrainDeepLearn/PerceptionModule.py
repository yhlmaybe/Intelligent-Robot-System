import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List   

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

        if self.training:
            with torch.no_grad():
                kH, kW = self.conv.kernel_size
                stride = self.conv.stride
                padding = self.conv.padding
                
                x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
                out_unfold = out.view(out.size(0), out.size(1), -1)
                
                hebb_term = torch.einsum('bik,bjk->ij', out_unfold, x_unfold)

                out_sq = out_unfold ** 2

                weight_flat = self.conv.weight.view(self.conv.weight.size(0), -1)
                weight_norm = torch.norm(weight_flat, dim=1) 

                out_sq_sum = out_sq.sum(dim=[0, 2])

                decay_term = torch.einsum('i,i->i', out_sq_sum, weight_norm)  

                decay_term = decay_term.unsqueeze(1)

                delta_w = self.hebb_rate * (hebb_term - decay_term)

                delta_w = delta_w.view(self.conv.weight.shape)

                self.hebb_memory = 0.9 * self.hebb_memory + 0.1 * delta_w
                self.conv.weight.data += self.hebb_memory
        return out
    
    def ResetHebbianMemory(self):
        self.hebb_memory.zero_()

class HebbianLinear(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, hebbRate: float = 0.01, emaMomentum: float = 0.9, normalize: bool = True, weightConstraint: str = 'clip'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outFeatures, inFeatures) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(outFeatures))

        self.hebb_rate   = hebbRate
        self.ema_alpha   = emaMomentum
        self.normalize   = normalize
        self.weight_constraint = weightConstraint
        
        self.register_buffer("hebb_memory", torch.zeros_like(self.weight))

        if normalize:
            self.register_buffer("running_mean", torch.zeros(outFeatures))
            self.register_buffer("running_var",  torch.ones(outFeatures))
            self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)  # [B, out]

        if self.training:
            with torch.no_grad():
                hebb_term  = torch.einsum('bi,bj->ij', x, y)        
                decay_term = torch.einsum('bj,bj,ij->ij', y, y, self.weight)
                delta_w    = self.hebb_rate * (hebb_term - decay_term)

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
    def __init__(self, modelDim : int, headNum : int, dimFeedforward : int = 2048, dropout : float = 0.1):
        super().__init__()

        self.self_atten = nn.MultiheadAttention(modelDim, headNum, dropout = dropout, batch_first = True)

        self.linear1 = nn.Linear(modelDim, dimFeedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dimFeedforward, modelDim)
        
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src : torch.Tensor) -> torch.Tensor:
        src2 = self.self_atten(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    

class ResidualBlock(nn.Module):
    def __init__(self, inChannels : int, outChannels : int, stride : int = 1, useHebbian : bool = False):
        super().__init__()
        self.downsample = None
        if stride != 1 or inChannels != outChannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(outChannels)
            )
        
        conv_layer = HebbianConv2d if useHebbian else nn.Conv2d
        self.conv1 = conv_layer(inChannels, outChannels, kernelSize=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = conv_layer(outChannels, outChannels, kernelSize=3, stride=1, padding=1, bias=False)
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
        self.conv1 = conv_layer(inChannels, baseChannels, kernelSize=7, stride=2, padding=3)
        
        self.bn1 = nn.BatchNorm2d(baseChannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.MakeLayer(baseChannels, baseChannels, 2, stride=1, use_hebbian=useHebbian)
        self.layer2 = self.MakeLayer(baseChannels, baseChannels*2, 2, stride=2, use_hebbian=useHebbian)
        self.layer3 = self.MakeLayer(baseChannels*2, baseChannels*4, 2, stride=2, use_hebbian=useHebbian)
        self.layer4 = self.MakeLayer(baseChannels*4, baseChannels*8, 2, stride=2, use_hebbian=useHebbian)
        
        self.conv2 = conv_layer(baseChannels*8, baseChannels*16, kernelSize=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(baseChannels*16)
    
    def MakeLayer(self, in_channels : int, out_channels : int, blocks : int, stride : int = 1, use_hebbian : bool = False) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_hebbian))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, useHebbian=use_hebbian))
        
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
    def __init__(self, imgSize : int = 224, patchSize : int = 1, embedDim : int = 512, numHeads : int = 8, numLayers : int = 6, hebbRate : float = 0.01, useHebbian : bool = True):
        super().__init__()
        self.img_size = imgSize
        self.patch_size = patchSize
        self.use_hebbian = useHebbian
        
        self.cnn_extractor = CNNFeatureExtractor(inChannels=3, baseChannels=64, useHebbian=useHebbian)
        
        self.down_ratio = 32
        fmap_size = imgSize // self.down_ratio
        assert fmap_size % patchSize == 0, "PerceptionModule PerceptionModule patchSize is not divide fmap_size"

        cnn_feat_dim = 1024  # base_channels * 16
        
        self.patch_embed = nn.Conv2d(cnn_feat_dim, embedDim, kernel_size=patchSize, stride=patchSize)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedDim))
        num_patches = (fmap_size // patchSize) ** 2  
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedDim))
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncode(
                modelDim=embedDim, 
                headNum=numHeads,
                dimFeedforward=embedDim*4
            ) for _ in range(numLayers)
        ])
        
        if useHebbian:
            self.hebb_fc1 = nn.Sequential(
                nn.Linear(embedDim, embedDim * 2),
                nn.GELU(),
                HebbianLinear(embedDim * 2, embedDim * 2, hebbRate=hebbRate)
            )
            self.hebb_fc2 = nn.Sequential(
                nn.Linear(embedDim * 2, embedDim),
                nn.GELU(),
                HebbianLinear(embedDim, embedDim, hebbRate=hebbRate)
            )
        else:
            self.hebb_fc1 = nn.Sequential(
                nn.Linear(embedDim, embedDim * 2),
                nn.GELU()
            )
            self.hebb_fc2 = nn.Sequential(
                nn.Linear(embedDim * 2, embedDim),
                nn.GELU()
            )
        
        self.adaptive_gate = nn.Sequential(
            nn.Linear(embedDim, embedDim // 4),
            nn.ReLU(),
            nn.Linear(embedDim // 4, 1),
            nn.Sigmoid()
        )
        
        self.output_norm = nn.LayerNorm(embedDim)
        
        self.InitWeights()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # [B, 3, H, W] -> [B, 512]
        cnn_features = self.cnn_extractor(x)
        
        patches = self.patch_embed(cnn_features)  # [B, embed_dim, num_patches_h, num_patches_w]
        patches = rearrange(patches, 'b c h w -> b (h w) c')
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=patches.shape[0])
        x = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        cls_rep = x[:, 0, :]
        
        features = self.hebb_fc1(cls_rep)
        features = self.hebb_fc2(features)
        
        gate = self.adaptive_gate(features)
        features = gate * features + (1 - gate) * cls_rep
        
        return self.output_norm(features)
    

    def InitWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        for m in self.adaptive_gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def ResetHebbianMemory(self):
        for module in self.modules():
            if isinstance(module, HebbianConv2d) or isinstance(module, HebbianLinear):
                module.ResetHebbianMemory()



class PerceiveExtractorMetaWrapper(nn.Module):
    def __init__(self, baseExtractor : PerceiveExtractor, innerLr : float = 0.01, numInnerSteps : int = 3):
        super().__init__()
        self.base_extractor = baseExtractor
        self.inner_lr = innerLr
        self.num_inner_steps = numInnerSteps
        self.meta_optimizer = torch.optim.Adam(self.base_extractor.parameters(), lr=1e-3)
        
    def Adapt(self, support_set : Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        fast_weights = list(self.base_extractor.parameters())
        
        for _ in range(self.num_inner_steps):
            features = self.base_extractor(support_set['images'])
            loss = self.ContrastiveLoss(features, support_set['labels'])
            
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        return fast_weights
    
    def MetaUpdate(self, tasks : List[Dict[str, Dict[str, torch.Tensor]]]) -> None:
        meta_grads = []
        
        for task in tasks:
            fast_weights = self.Adapt(task['support'])
            
            with torch.set_grad_enabled(True):
                original_params = list(self.base_extractor.parameters())
                for param, fast_param in zip(self.base_extractor.parameters(), fast_weights):
                    param.data = fast_param.data
                
                query_features = self.base_extractor(task['query']['images'])
                meta_loss = self.ContrastiveLoss(query_features, task['query']['labels'])
                
                meta_grad = torch.autograd.grad(meta_loss, original_params)
                meta_grads.append(meta_grad)
                
                for param, orig_param in zip(self.base_extractor.parameters(), original_params):
                    param.data = orig_param.data
        
        self.meta_optimizer.zero_grad()
        for param, grads in zip(self.base_extractor.parameters(), zip(*meta_grads)):
            grad = torch.stack([g for g in grads]).mean(dim=0)
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            param.grad += grad
        
        self.meta_optimizer.step()
    
    def ContrastiveLoss(self, features : torch.Tensor, labels : torch.Tensor, temperature : float = 0.1):
        features = F.normalize(features, dim=1)
        
        sim_matrix = torch.mm(features, features.t()) / temperature
        
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = label_matrix.fill_diagonal_(False) 
        
        exp_sim = torch.exp(sim_matrix)
        
        pos_term = -torch.log(exp_sim[positive_mask].sum(dim=1) / positive_mask.sum(dim=1))
        
        neg_term = torch.log(exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix)))
        
        loss = (pos_term + neg_term).mean()
        return loss
