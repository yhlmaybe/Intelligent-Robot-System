import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import PerceptionModule
import AttentionModule
import MemoryModule
import DecisionModule
import ValueEstimationModule


class BrainDeepLearnModule(nn.Module):
    def __init__(self, imgSize : int=224,useHebbian : bool =True, useMetalearning : bool =False,numActions : int =10): 
        super().__init__()

        self.perception = PerceptionModule.PerceiveExtractor(imgSize=imgSize,useHebbian=useHebbian)

        self.attention = AttentionModule.AttentionExtractor()

        self.memory = MemoryModule.MemoryExtractor()

        self.decision = DecisionModule.DecisionExtractor()

        self.value = ValueEstimationModule.ValueEstimationExtractor()

        #another module

        self.use_metalearning = useMetalearning
        if useMetalearning:
            self.metalearn_wrapper = PerceptionModule.PerceiveExtractorMetaWrapper(self.perception)

        
    def forward(self, frame_sequence : torch.Tensor):
        #frame_sequence: [B, T, C, H, W]

        B, T = frame_sequence.shape[:2]
        
        frame_features = []
        for b in range(B):
            features = self.perception(frame_sequence[:, b])
            frame_features.append(features)
        frame_features = torch.stack(frame_features, dim=0)  # [B, T, D]
        
        attn_output = self.attention(frame_features)

        #another module
        
        return 
    
    def ResetHebbianMemory(self):
        self.perception.ResetHebbianMemory()
        self.attention.ResetHebbianMemory()
    
    def ResetMemoryState(self):
        pass
    

def Train(model : BrainDeepLearnModule, trainLoader, valLoader, epochs=10, lr=1e-3, useMetalearning=False,device='cuda'):
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in trainLoader:
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            actions = batch['action'].to(device)  # [B]
            
            if useMetalearning:
                task = {
                    'support': {
                        'images': frames[:len(frames)//2],
                        'labels': actions[:len(actions)//2]
                    },
                    'query': {
                        'images': frames[len(frames)//2:],
                        'labels': actions[len(actions)//2:]
                    }
                }
                
                model.metalearn_wrapper.MetaUpdate([task])
                
                with torch.no_grad():
                    logits, _ = model(task['query']['images'])
                    loss = criterion(logits, task['query']['labels'])
                    _, predicted = logits.max(1)
                    
                train_loss += loss.item()
                correct += predicted.eq(task['query']['labels']).sum().item()
                total += len(task['query']['labels'])
            
            else:
                optimizer.zero_grad()
                
                memory_state = model.ResetMemoryState(frames.size(0), device)
                
                logits, _ = model(frames, memory_state)
                loss = criterion(logits, actions)
                
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(actions).sum().item()
                total += actions.size(0)
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(trainLoader)
        
        val_loss, val_acc = evaluate(model, valLoader, device)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
    return model

def evaluate(model : BrainDeepLearnModule, dataLoader, device='cuda'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataLoader:
            frames = batch['frames'].to(device)
            actions = batch['action'].to(device)
            
            memory_state = model.ResetMemoryState(frames.size(0), device)
            
            logits, _ = model(frames, memory_state)
            loss = criterion(logits, actions)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(actions).sum().item()
            total += actions.size(0)
    
    avg_loss = total_loss / len(dataLoader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def deploy(model, frameSequence, memoryState=None, device='cuda'):
    model.eval()
    
    # [T, C, H, W] -> [1, T, C, H, W]
    if frameSequence.dim() == 4:
        frameSequence = frameSequence.unsqueeze(0)
    
    with torch.no_grad():
        frameSequence = frameSequence.to(device)
        
        action_logits, new_memory_state = model(frameSequence, memoryState)
        
        action_probs = F.softmax(action_logits, dim=-1)
        predicted_action = action_probs.argmax(dim=-1).item()
        
        return predicted_action, new_memory_state

