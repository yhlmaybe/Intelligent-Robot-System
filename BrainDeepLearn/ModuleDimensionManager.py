

class ModuleDim:
    PerceptionEmbed: int = 512
    PerceptionFeat: int = 2*PerceptionEmbed

    AttentionFeat: int = PerceptionFeat

    WorldFeat: int = 512
    WorldOutHState: int = 512
    WorldOutZState: int = 64

    MemoryFeat=AttentionFeat

    MemoryItem: int = MemoryFeat  
    WorldMemoryItem: int = WorldFeat  

    ConsciousnessState: int = MemoryFeat

    IntentionFeat:int = 512

    ValueEstimationOutEmotion=64