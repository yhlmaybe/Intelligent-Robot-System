from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MetaPlasticityController(nn.Module):
    #Generates five neuromodulatory scalars (a, b, bias, fusion, importance)
    def __init__(self, metaInDim: int = 9, hiddenDim: int = 96):
        """
        Args:
            metaInDim: Dimension of input meta-features
            hiddenDim: Hidden state dimension of the GRU controller
        """
        super().__init__()
        self.rnn = nn.GRUCell(metaInDim, hiddenDim)
        self.fc_out = nn.Linear(hiddenDim, 5)
        self.h_state: Optional[torch.Tensor] = None
        self.memory_utilization = 0.0   # Current memory utilization rate

    def UpdateMemoryUtilization(self, util: float):
        self.memory_utilization = util

    def forward(self, metaFeat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            meta_feat: Input meta-features [B, metaInDim]
        Returns:
            a: Learning rate scaling factor for Hebbian updates (range: 0.5-1.0)
            b: Decay rate scaling factor for fast weights (range: 0.5-1.0)
            bias: Gate bias for output modulation (range: -1 to 1)
            fusion_gate: Weighting factor for memory fusion (range: 0-1)
            importance: Significance score for current state (range: 0-1)
        """
        B = metaFeat.size(0)
        if self.h_state is None or self.h_state.size(0) != B:
            self.h_state = torch.zeros(B, self.rnn.hidden_size, device=metaFeat.device)

        self.h_state = self.rnn(metaFeat, self.h_state)
        a_hat, b_hat, bias_hat, fus_hat, imp_hat = self.fc_out(self.h_state).chunk(5, dim=1)
        a = 0.5 + 0.5 * torch.tanh(a_hat)
        b = 0.5 + 0.5 * torch.sigmoid(b_hat)
        bias = torch.tanh(bias_hat)
        fusion_gate = torch.sigmoid(fus_hat)
        importance = torch.sigmoid(imp_hat)
        return a.squeeze(-1), b.squeeze(-1), bias.squeeze(-1), fusion_gate.squeeze(-1), importance.squeeze(-1)

    def Reset(self):
        self.h_state = None
        self.memory_utilization = 0.0



class MemoryExtractor(nn.Module):
    """State-Space backbone + Hebbian fast weights + KV memory + optional meta-controller."""
    def __init__(
        self,
        inputDim: int = 512,              # Dimension of input features
        ssmStateDim: int = 512,           # Dimension of SSM hidden state
        memoryDim: int = 768,             # Dimension of memory keys/values
        memorySize: int = 200,            # Capacity of the key-value memory
        outputDim: int = 768,             # Dimension of output features
        hebbAlpha: float = 0.15,          # Base learning rate for Hebbian updates
        decayFactor: float = 0.95,        # Base decay rate for fast weights
        topk: int = 8,                    # Number of top memories to retrieve
        tdScale: float = 5.0,             # Scaling factor for TD-error neuromodulation
        softBeta: float = 0.2,            # Decay factor for soft reset
        useMeta: bool = True,             # Whether to use meta-plasticity controller
        compressThreshold: float = 0.75,  # Memory fill ratio threshold for compression
        useAmp: bool = True,              # Whether to use automatic mixed precision
        svdInterval: int = 10,            # Steps between SVD normalizations
        svdMin: float = 0.1,              # Minimum value for singular value clipping
        svdMax: float = 1.5) -> None:     # Maximum value for singular value clipping

        super().__init__()
        self.ssm_state_dim = ssmStateDim
        self.memory_dim = memoryDim
        self.output_dim = outputDim
        self.memory_size = memorySize
        self.topk = min(topk, memorySize)
        self.hebb_alpha = hebbAlpha
        self.decay = decayFactor
        self.td_scale = tdScale
        self.soft_beta = softBeta
        self.use_meta = useMeta
        self.compress_threshold = compressThreshold
        self.use_amp = useAmp
        self.svd_interval = max(1, svdInterval)
        self.svd_min = svdMin
        self.svd_max = svdMax
        
        # State transition matrix (orthogonal initialization)
        A_init = torch.empty(ssmStateDim, ssmStateDim)
        nn.init.orthogonal_(A_init, gain=0.8)
        self.A_full = nn.Parameter(A_init * 0.05)

        # Input projection matrix
        self.B_mat = nn.Linear(inputDim, ssmStateDim, bias=False)

        # State output projection
        self.C_mat = nn.Linear(ssmStateDim, outputDim, bias=False)

        # Skip connection (input to output)
        self.D_mat = nn.Linear(inputDim, outputDim, bias=False)
        for p in (self.B_mat, self.C_mat, self.D_mat):
            nn.init.xavier_uniform_(p.weight)

        self.h_state = torch.zeros(1, ssmStateDim)

        # Hebbian associative matrix (fast weights)
        self.register_buffer("fast_weights", torch.zeros(memoryDim, memoryDim))

        # Key-value memory storage (using half precision to save memory)
        self.register_buffer("memory_keys", torch.zeros(memorySize, memoryDim, dtype=torch.float16))
        self.register_buffer("memory_values", torch.zeros(memorySize, memoryDim, dtype=torch.float16))
        self.register_buffer("memory_importance", torch.zeros(memorySize))

        # Timesteps of storage
        self.register_buffer("memory_steps", torch.zeros(memorySize, dtype=torch.long))

        # Correlation scores
        self.register_buffer("memory_corr", torch.zeros(memorySize))

        self.mem_ptr = 0                  # Current write pointer
        self.time_step = 0                # Global timestep counter
        self.memory_filled = 0            # Number of filled memory slots
        self.memory_usage = 0.0           # Current memory utilization rate
        self.last_compress_step = 0       # Last timestep when compression occurred
        self._steps_since_svd = 0         # Counter for SVD interval

        self.state2mem = nn.Linear(ssmStateDim, memoryDim)
        self.state2val = nn.Linear(ssmStateDim, memoryDim)
        self.importance_net = nn.Sequential(
            nn.Linear(ssmStateDim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.local_gate = nn.Sequential(
            nn.Linear(ssmStateDim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.fusion_gate_net = nn.Sequential(
            nn.Linear(memoryDim * 3, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

        self.meta_ctrl = MetaPlasticityController(hidden_dim=96) if useMeta else None

        self.fusion = nn.Sequential(
            nn.Linear(outputDim + memoryDim, 1024), nn.GELU(),
            nn.Linear(1024, outputDim)
        )
        self.norm = nn.LayerNorm(outputDim)
        self.grad_bridge = nn.Parameter(torch.tensor(0.3))

        with torch.no_grad():
            self.grad_bridge.clamp_(0.1, 0.9)


    def forward(
        self,
        x: torch.Tensor,                            # Input features [B, inputDim]
        *,
        tdError: Optional[torch.Tensor] = None,     # Temporal difference error [B]
        entropy:   Optional[torch.Tensor] = None,   # Policy entropy [B]
        reward:    Optional[torch.Tensor] = None,   # Immediate reward [B]
        uncertainty: Optional[torch.Tensor] = None, # Agent uncertainty [B]
        reset: bool = False,                        # All reset 
        softReset: bool = False,                    # Soft(Part) reset 
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        amp_enable = self.use_amp and x.is_cuda
        with torch.autocast(device_type=x.device.type,
            dtype=torch.float16 if x.is_cuda else torch.bfloat16, 
            enabled=amp_enable):

            B, device = x.size(0), x.device
            
            if self.h_state.device != device:
                self.h_state = self.h_state.to(device)
            
            if reset:
                self.ResetAll()
            elif softReset:
                self.SoftReset()
                
            self.time_step += 1

            if self.h_state.size(0) != B:
                self.h_state = torch.zeros(B, self.ssm_state_dim, device=device)

            # SSM state update
            h_new : torch.Tensor = self.h_state @ self.A_full.t() + self.B_mat(x)
            y_ssm : torch.Tensor = self.C_mat(h_new) + self.D_mat(x)
            
            self.h_state = h_new * self.grad_bridge + h_new.detach() * (1 - self.grad_bridge)

            # Project to memory space
            key = F.normalize(self.state2mem(h_new), dim=-1)
            val = F.normalize(self.state2val(h_new), dim=-1)
            importance = self.importance_net(h_new)
            gate_local = self.local_gate(h_new)

            # Neuromodulation and meta-signals
            neuromod = self.GetNeuromod(tdError)
            self.UpdateMemoryUtilization()
            self.AutoCompress()
            
            # Get meta-control signals
            a, b, gate_bias, fusion_gate, meta_imp = self.GetMetaSignals(tdError, entropy, reward, uncertainty, B, device)
            
            # Combine importance scores
            importance = 0.7 * importance + 0.3 * meta_imp.view(-1, 1)

            # Hebbian fast weights update
            self.HebbianUpdate(key, gate_local, neuromod, a, b)
            
            # Write to key-value memory
            self.KvWrite(key, val, importance)
            
            mem_recall = self.Retrieve(key, fusion_gate)

            # Output gating based on TD-error
            if tdError is not None:
                mem_recall = self.ApplyOutputGate(mem_recall, tdError, gate_bias)

            fused = self.fusion(torch.cat([y_ssm, mem_recall], dim=-1))
            output = self.norm(fused)
            
        return output.float(), mem_recall.float()
 

    def GetNeuromod(self, tdError: Optional[torch.Tensor]) -> torch.Tensor:
        """Generates neuromodulation signal from TD-error"""
        if tdError is None:
            return torch.ones(1, 1, 1, device=self.h_state.device)
        return torch.tanh(tdError / self.td_scale).view(-1, 1, 1)

    def GetMetaSignals(
            self, 
            tdError: Optional[torch.Tensor], 
            entropy: Optional[torch.Tensor], 
            reward: Optional[torch.Tensor], 
            uncertainty: Optional[torch.Tensor], 
            B: int, 
            device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Gets meta-control signals from controller or defaults"""
        if self.meta_ctrl is None or tdError is None:
            return (
                torch.ones(B, device=device),
                torch.ones(B, device=device),
                torch.zeros(B, device=device),
                torch.full((B,), 0.5, device=device),
                torch.full((B,), 0.5, device=device))
        
        # Compute memory statistics
        mem_fill_ratio = self.memory_filled / self.memory_size if self.memory_size > 0 else 0.0
        imp_mean = self.memory_importance[:self.memory_filled].mean().item() if self.memory_filled > 0 else 0.0
        corr_mean = self.memory_corr[:self.memory_filled].mean().item() if self.memory_filled > 0 else 0.0
        
        # Prepare meta-features: [δ, |δ|, entropy, reward, mem_util, uncertainty, fill_ratio, imp_mean, corr_mean]
        meta_feat = torch.stack([
            tdError,
            tdError.abs(),
            entropy if entropy is not None else torch.full_like(tdError, -1.0),  
            reward if reward is not None else torch.full_like(tdError, -1.0),   
            torch.full_like(tdError, self.memory_usage),
            uncertainty if uncertainty is not None else torch.full_like(tdError, -1.0),  
            torch.full_like(tdError, mem_fill_ratio),
            torch.full_like(tdError, imp_mean),
            torch.full_like(tdError, corr_mean),], dim=-1)
        
        return self.meta_ctrl(meta_feat)

    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        """Applies output gating based on TD-error and bias"""
        gate_out = (1.0 + torch.tanh(tdError / self.td_scale + gateBias)).view(-1, 1) / 2.0
        return gate_out * memRecall

    def SoftReset(self):
        self.h_state = self.h_state * self.soft_beta
        self.fast_weights.mul_(self.soft_beta)

        if self.memory_importance is not None and self.memory_filled > 0:
            self.memory_importance[:self.memory_filled] *= self.soft_beta

    @torch.no_grad()
    def HebbianUpdate(self, key: torch.Tensor, gateLocal: torch.Tensor, neuromod: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
        """
        Hebbian update with conditional SVD normalization
        Args:
            key: Memory key vectors [B, memory_dim]
            gateLocal: Local gating signal [B, 1]
            neuromod: Neuromodulation factor [B, 1, 1]
            a: Learning rate scaling factor [B]
            b: Decay rate scaling factor [B]
        """
        a = a.view(-1, 1, 1)
        b = b.view(-1, 1, 1)
        
        outer = torch.einsum('bi,bj->bij', key, key)
        update = (neuromod * self.hebb_alpha * a * gateLocal.view(-1, 1, 1) * outer).sum(0)
        
        self.fast_weights.mul_(self.decay * b.mean()).add_(update)
        
        self._steps_since_svd += 1
        
        if self._steps_since_svd >= self.svd_interval:
            self._steps_since_svd = 0  
            
            with torch.autocast(device_type=self.fast_weights.device.type, enabled=False):
                fw_fp32 = self.fast_weights.float()

                fw_fp32 = fw_fp32 + 1e-4 * torch.eye(fw_fp32.size(0), device=fw_fp32.device,dtype=fw_fp32.dtype)

                U, S, Vh = torch.linalg.svd(fw_fp32, full_matrices=False)
                
                S_clamped = torch.clamp(S, self.svd_min, self.svd_max)
                diag_S = torch.diag(S_clamped)
                
                fw_proj = U @ diag_S @ Vh
                self.fast_weights.copy_(fw_proj.to(self.fast_weights.dtype))

    @torch.no_grad()
    def KvWrite(self, key: torch.Tensor, val: torch.Tensor, importance: torch.Tensor) -> None:
        """
        Writes to key-value memory (using half precision storage)
        Args:
            key: Key vectors [B, memory_dim]
            val: Value vectors [B, memory_dim]
            importance: Importance scores [B, 1]
        """
        n = key.size(0)
        device = key.device
    
        if self.memory_filled < self.memory_size:
            n = min(n, self.memory_size - self.memory_filled) 
            start = self.memory_filled
            idx = torch.arange(start, start + n, device=device)
            self.memory_filled += n
            self.mem_ptr = (start + n) % self.memory_size  
        else:
            imp_slice = self.memory_importance[:self.memory_filled]
            _, replace_idx = torch.topk(-imp_slice, min(n, self.memory_filled), largest=True)
            idx = replace_idx
    
        if self.memory_filled > 0:
            mask = torch.ones(self.memory_filled, dtype=torch.bool, device=device)
            if self.memory_filled < self.memory_size: 
                mask[idx] = False
            else: 
                mask[replace_idx] = False
        
            valid_keys = self.memory_keys[mask].float()
        
            if valid_keys.numel() > 0:
                corr = torch.mm(key[:n], valid_keys.t()).mean(dim=1)
                self.memory_corr[idx] = corr.detach()
            else:
                self.memory_corr[idx].fill_(1.0)
        else:
            self.memory_corr[idx].fill_(1.0)
    
        self.memory_keys[idx] = key[:n].detach().half()
        self.memory_values[idx] = val[:n].detach().half()
        self.memory_importance[idx] = importance[:n].squeeze().detach()
        self.memory_steps[idx] = self.time_step
    


    def Retrieve(self, query: torch.Tensor, fusionGate: torch.Tensor) -> torch.Tensor:
        """
        Retrieves memory using content-based addressing
        Args:
            query: Query vector [B, memoryDim]
            fusionGate: Fusion weight from meta-controller [B]
            
        Returns:
            Combined memory vector [B, memoryDim]
        """
        fast_part = query @ self.fast_weights
        
        if self.memory_filled == 0:
            kv_part = torch.zeros_like(fast_part)
        else:
            keys = self.memory_keys[:self.memory_filled].float()
            values = self.memory_values[:self.memory_filled].float()
            importance = self.memory_importance[:self.memory_filled]
            corr = self.memory_corr[:self.memory_filled]
            steps = self.memory_steps[:self.memory_filled]
            
            sim = query @ keys.t()
            sim = sim * importance.unsqueeze(0) * corr.unsqueeze(0)
            
            age = (self.time_step - steps).clamp(min=0).float()
            sim = sim * torch.exp(-0.05 * age).unsqueeze(0)
            
            k = max(1, min(self.topk, self.memory_filled))
            top_sim, top_idx = torch.topk(sim, k, dim=-1)
            
            th = top_sim.mean(dim=-1, keepdim=True) - 0.5 * top_sim.std(dim=-1, keepdim=True)
            th = torch.where(torch.isfinite(th), th, torch.zeros_like(th))  
            top_sim = top_sim * (top_sim > th)
            
            attn_weights = F.softmax(top_sim, dim=-1)
            vals = values[top_idx]
            kv_part = torch.einsum('bk,bkd->bd', attn_weights, vals)
        
        fusion_input = torch.cat([query, fast_part, kv_part], dim=-1)
        gate = self.fusion_gate_net(fusion_input)
        gate = 0.5 * gate + 0.5 * fusionGate.view(-1, 1)
        return gate * fast_part + (1 - gate) * kv_part

    @torch.no_grad()
    def UpdateMemoryUtilization(self):
        window_size = max(50, min(200, self.memory_size // 5))
        min_step = max(1, self.time_step - window_size)
        
        accessed = (
            (self.memory_steps >= min_step) & 
            (self.memory_steps > 0) &
            (torch.arange(self.memory_size, device=self.memory_steps.device) < self.memory_filled)
            )
        
        accessed_count = accessed.sum().item()
        
        self.memory_usage = (
            min(1.0, accessed_count / self.memory_filled) 
            if self.memory_filled > 0 else 0.0
        )
        
        if self.meta_ctrl:
            self.meta_ctrl.UpdateMemoryUtilization(self.memory_usage)

    @torch.no_grad()
    def AutoCompress(self):
        """Compresses memory by keeping only most important entries"""
        current_thresh = max(0.6, min(0.9, 0.7 + self.memory_usage * 0.2))
        if self.memory_filled < self.memory_size * current_thresh:
            return
        if self.time_step - self.last_compress_step < 100:  
            return
        
        if self.memory_filled > 0:
            time_diff = self.time_step - self.memory_steps[:self.memory_filled]
            decay_factor = torch.exp(-0.01 * time_diff.float())
            self.memory_importance[:self.memory_filled] *= decay_factor
            
        importances = self.memory_importance[:self.memory_filled]
        _, sorted_idx = torch.sort(importances, descending=True)
        
        keep_num = min(int(self.memory_size * 0.7), self.memory_filled)
        sorted_idx = sorted_idx[:keep_num]
        
        new_keys = self.memory_keys[sorted_idx]
        new_values = self.memory_values[sorted_idx]
        new_imp = importances[sorted_idx]
        new_steps = self.memory_steps[sorted_idx]
        new_corr = self.memory_corr[sorted_idx]
        
        self.memory_keys[:keep_num] = new_keys
        self.memory_values[:keep_num] = new_values
        self.memory_importance[:keep_num] = new_imp
        self.memory_steps[:keep_num] = new_steps
        self.memory_corr[:keep_num] = new_corr
        
        self.memory_filled = keep_num
        self.mem_ptr = keep_num % self.memory_size
        self.last_compress_step = self.time_step

    def ResetAll(self):
        self.fast_weights.zero_()
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_importance.zero_()
        self.memory_steps.zero_()
        self.memory_corr.zero_()
        
        self.mem_ptr = 0
        self.time_step = 0
        self.memory_filled = 0
        self.memory_usage = 0.0
        self.last_compress_step = 0
        self._steps_since_svd = 0
        
        self.h_state.zero_()
        
        if self.meta_ctrl:
            self.meta_ctrl.Reset()