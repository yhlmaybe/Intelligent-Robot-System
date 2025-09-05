import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional   


def ProjectFroNorm(tensor: torch.Tensor, maxNorm: Optional[float]):
    if not maxNorm:
        return
    with torch.no_grad():
        n = torch.linalg.vector_norm(tensor, ord=2)
        if torch.isfinite(n) and (n > maxNorm):
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
            nn.ReLU(inplace=False),
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
            print("PerceiveExtractor test passed. Output shape:", out.shape)
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
            assert grads_ok, "There are parameters whose gradient is None or a non-finite number of parameters"

            opt.step()
            print("Perception TrainStepSmoke passed.")
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
                        assert torch.isfinite(p.grad).all(), f"step {t} Gradient is not finite: {n}"
                opt.step()
            print("Perception NoNanAfterManySteps passed.")
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
            assert changed, "The norm of key parameters has hardly changed, and it is suspected that they have not been updated."
            print("Perception ParamsActuallyChange passed.")
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
            assert end <= 0.8 * start, "Training did not show sufficient convergence (decline < 20%)"
            print("Perception TestNormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print(f"TestNormalTrainingConvergence failed: {e}")
            return False
        except Exception as e:
            print(f"TestNormalTrainingConvergence error: {e}")
            return False

    def RunAll(self):
        results = {
            "HebbianConv2d": self.TestHebbianConv2d(),
            "HebbianLinear": self.TestHebbianLinear(),
            "PerceiveExtractorForward": self.TestPerceiveExtractor(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "ParamsActuallyChange": self.ParamsActuallyChange(),
            "NormalTrainingConvergence": self.TestNormalTrainingConvergence(),}
        
        passed = sum(1 for v in results.values() if v)
        print(f"\nPerception module tests: {passed}/{len(results)} passed.")
        return results

        




