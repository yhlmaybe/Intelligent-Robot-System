from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import threading
import queue
import random
import json

import numpy as np
import torch
import torch.nn as nn
import traceback
import os
import math
import copy
import inspect
import tempfile
from types import SimpleNamespace

#import debugpy

from dataclasses import dataclass, field, replace
from collections import deque

from Config import BasicParameters
from CoreTypes import (
    AgentActInput,
    AgentActOutput,
    BrainStepInput,
    BrainStepOutput,
    CameraCalibration,
    CognitionGoalStage,
    DECISION_REQUEST_PROVENANCE_FIELDS,
    DECISION_WIRE_SCHEMA_VERSION,
    ExecutionStage,
    PerceptionPhysicalStage,
    RobotState,
    TEXT_TRUST_OPERATOR_COMMAND,
    TEXT_TRUST_UNSAFE_EXTERNAL,
    ValueMemoryWorldStage)
from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper, PerceptionRecallLoss, TopDownContext, VisualState
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper
from MemoryModule import MemoryExtractor, MemoryType
from DecisionModule import DecisionExtractor, DecisionPlannerExtractor
from DecisionDecoupler import ApplyPoseDelta, CanonicalizeQuaternion, DecoupledDecision, DecisionActionMask, DecisionDecouplerV2, EndpointControlEncoder, EndpointPoseEncoder, FlattenActiveDecisionTensor, MotionCommand, NormalizePose, QuatConjugate, QuatMultiply, QuatRotate, RelativePose, RelativePoseError, SAFETY_MARGIN_NAMES
from WorldModule import (
    KLDiagNormal,
    RSSMWorldModel,
    WORLD_MEMORY_TENSOR_FIELDS,
    WorldOnlineWrapper)
from ValueEstimationModule import ValueEstimationExtractor,ValueEstimationOnlineWrapper
from ConsciousnessModule import ConsciousnessExtractor, ConsciousnessOutput
from IntentionModule import IntentionExtractor, IntentionOnlineWrapper
from OCRModule import OCREngineExtractor
from PhysicalStateModule import PhysicalStateExtractor, PhysicalStateLoss
from GoalModule import GoalGrounding, FourLevelGoalManager
from NeuroSymbolicModule import FAILURE_CAUSES, OPERATORS, PREDICATES, NeuroSymbolicExtractor, NeuroSymbolicRobotStateEncoder
from TemporalExecutionModule import (
    CANCEL,
    CONTINUE,
    DISPATCH,
    REDISPATCH,
    TEMPORAL_REASON_NAMES,
    TemporalDecisionEnvelope,
    TemporalExecutionGateExtractor)
from ModuleMessagerManager import ModuleDim, ModuleMessagerManager
from FunctionTools import SynchronizeDynamicAdapterTopologiesForFullLoad
 

BRAIN_RUNTIME_SCHEMA_VERSION = 15

BRAIN_RUNTIME_BUFFER_FIELDS = frozenset({
    "schema_version",
    "prev_mem",
    "prev_attn",
    "prev_option_logit",
    "prev_entropy",
    "prev_decision_state",
    "prev_latent_control",
    "prev_target_endpoint_pose",
    "prev_target_endpoint_valid",
    "prev_measured_endpoint_pose",
    "prev_measured_endpoint_valid",
    "active_option_policy_input",
    "active_option_prior_logit",
    "active_option_goal_mid",
    "active_option_index",
    "active_option_valid",
    "prev_belief_prediction_state",
    "prev_belief_prediction_valid",
    "temporal_active_mask",
    "temporal_action_age_steps",
    "temporal_action_epoch",
    "temporal_invoke_drift",
    "temporal_active_kind",
    "active_motion_command",
    "prev_mapper_hidden",
    "prev_td_error",
    "world_state",
    "world_robot_physical_encoding",
    "mem_state",
    "mem_pending",
    "attn_state",
    "critic_state",
    "perc_buffer",
    "prev_visual_state",
    "prev_visual_valid",
    "prospective_visual_prediction",
    "prev_precision",
    "prev_goal_bias",
    "prev_self_sem",
    "prev_intent_sem",
    "visual_state_buffer",
    "visual_state_valid_buffer",
    "ocr_state",
    "history",
    "extra_mem",
    "neuro_symbolic_plan",
    "adaptive_runtime_buffers",
    "prev_failure_count",
    "slow_step_count",
    "slow_cache",
    "thread_end",
})

AGENT_RUNTIME_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "calibration_id",
    "world_frame_id",
    "batch_size",
    "online_learning",
    "brain",
    "online_candidates",
    "buffers",
    "world_memory",
    "memory_durable",
    "rng_py",
    "rng_np",
    "rng_torch",
    "rng_cuda_all",
})

AGENT_TRAINING_CHECKPOINT_FIELDS = frozenset({
    "opt_actor",
    "opt_critic",
    "opt_world",
})

WORLD_RUNTIME_STATE_NAMES = frozenset(
    {f"_{name}" for name in WORLD_MEMORY_TENSOR_FIELDS}
    | {"_robot_physical_state", "s4.x"})


def IsWorldRuntimeStateKey(name: str) -> bool:
    return any(
        name == runtime_name or name.endswith(f".{runtime_name}")
        for runtime_name in WORLD_RUNTIME_STATE_NAMES)


def BrainNonModelStateKeys(brain: nn.Module) -> frozenset[str]:
    parameter_names = frozenset(name for name, _ in brain.named_parameters())
    excluded = {
        name
        for name in brain.state_dict()
        if (
            IsWorldRuntimeStateKey(name)
            or name.startswith(("mem_copy.", "attn_copy.", "critic_copy."))
            or (
                name.startswith(("mem.", "critic."))
                and name not in parameter_names))}
    adaptive_modules_fn = getattr(brain, "AdaptiveRuntimeModules", None)
    if callable(adaptive_modules_fn):
        adaptive_module_ids = {
            id(module)
            for module in adaptive_modules_fn().values()}
        for module_path, module in brain.named_modules():
            if id(module) not in adaptive_module_ids:
                continue
            prefix = f"{module_path}." if module_path else ""
            excluded.update(
                f"{prefix}{buffer_name}"
                for buffer_name, _ in module.named_buffers())
    return frozenset(excluded)


def ExportBrainModelState(brain: nn.Module) -> Dict[str, torch.Tensor]:
    """Return model state only; runtime and durable memories use separate artifacts."""
    excluded = BrainNonModelStateKeys(brain)
    return {
        name: value
        for name, value in brain.state_dict().items()
        if name not in excluded}


ONLINE_WRAPPER_ROOTS = ("perc", "attn", "world", "critic", "intention")


def ExportDeploymentModelState(brain: nn.Module) -> Dict[str, torch.Tensor]:
    """Export the canonical non-wrapper parameter namespace used by deployment."""
    state = ExportBrainModelState(brain)
    if not bool(getattr(brain, "is_online_learning", False)):
        return state
    for root in ONLINE_WRAPPER_ROOTS:
        wrapper = getattr(brain, root)
        if hasattr(wrapper, "CandParameters") and any(
            True for _ in wrapper.CandParameters()
        ):
            raise RuntimeError(
                "online candidates must be materialized before deployment export")
    canonical: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        mapped = name
        for root in ONLINE_WRAPPER_ROOTS:
            prefix = f"{root}.base."
            if name.startswith(prefix):
                mapped = f"{root}.{name[len(prefix):]}"
                break
        if mapped in canonical:
            raise RuntimeError(f"duplicate canonical deployment parameter: {mapped}")
        canonical[mapped] = value
    return canonical


def LoadDeploymentModelState(
    brain: nn.Module,
    state: Dict[str, torch.Tensor],) -> None:
    """Load canonical deployment parameters into either wrapped or base modules."""
    if not bool(getattr(brain, "is_online_learning", False)):
        LoadBrainModelState(brain, state)
        return
    wrapped_roots = tuple(
        root
        for root in ONLINE_WRAPPER_ROOTS
        if hasattr(getattr(brain, root), "base"))
    wrapped: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        mapped = name
        for root in wrapped_roots:
            prefix = f"{root}."
            if name.startswith(prefix):
                mapped = f"{root}.base.{name[len(prefix):]}"
                break
        if mapped in wrapped:
            raise RuntimeError(f"duplicate wrapped training parameter: {mapped}")
        wrapped[mapped] = value
    LoadBrainModelState(brain, wrapped)


def LoadBrainModelState(
    brain: nn.Module,
    state: Dict[str, torch.Tensor],) -> None:
    if type(state) is not dict:
        raise TypeError("brain model state must be a dictionary")
    non_model_keys = BrainNonModelStateKeys(brain)
    runtime_keys = sorted(name for name in state if name in non_model_keys)
    if runtime_keys:
        raise ValueError(
            f"brain parameter artifact contains non-model state: {runtime_keys}")
    SynchronizeDynamicAdapterTopologiesForFullLoad(brain, state)
    expected = ExportBrainModelState(brain)
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"brain model state fields do not match: missing={missing}, "
            f"unexpected={unexpected}")
    result = brain.load_state_dict(state, strict=False)
    expected_runtime = sorted(non_model_keys)
    if sorted(result.missing_keys) != expected_runtime or result.unexpected_keys:
        raise RuntimeError(
            "brain model load crossed the runtime-state boundary")


@dataclass
class BrainStepTrace:
    ObsImg: Optional[torch.Tensor] = None
    ActionEmbed: Optional[torch.Tensor] = None

    PercBuffer: Optional[list[torch.Tensor]] = None
    VisualBuffer: Optional[List[VisualState]] = None
    VisualStateNow: Optional[VisualState] = None
    OcrSemantic: Optional[torch.Tensor] = None
    IntentHint: Optional[torch.Tensor] = None
    PercFeat: Optional[torch.Tensor] = None
    AttnFeat: Optional[torch.Tensor] = None
    MemFeat: Optional[torch.Tensor] = None
    WorldState: Optional[torch.Tensor] = None
    WorldDeltaTransport: Optional[torch.Tensor] = None
    WorldDeltaPhysics: Optional[torch.Tensor] = None
    ConsciousnessState: Optional[torch.Tensor] = None
    IntentionState: Optional[torch.Tensor] = None
    Reward: Optional[torch.Tensor] = None
    Done: Optional[torch.Tensor] = None
    ValueFeat: Optional[torch.Tensor] = None
    ActionEntropy: Optional[torch.Tensor] = None

    EntropyPrev: Optional[torch.Tensor] = None
    UncertaintyPrev: Optional[torch.Tensor] = None
    TdErrorPrev: Optional[torch.Tensor] = None

    extras: Dict[str, Any] = field(default_factory=dict)


class BrainCore(nn.Module):
    def __init__(
        self,
        calibration: CameraCalibration,
        device: Optional[torch.device] = None,
        *,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        plasticHebbian: bool = True,
        prioritizeExtStr: bool = True,
        plasticOnlineLearning: bool = False,
        usePlanner: bool = True,
        plannerTeacherMode: bool = True,
        enablePerceptionSupervision: bool = False,
        saveModuleMessagerOutput: bool = True,
        needTrace: bool = True,
        slowPeriod: int = 4,):
        super().__init__()
        self.SEQ_LEN = seqLen
        self.slow_period = int(slowPeriod)
        self.is_online_learning = plasticOnlineLearning
        self.prioritize_ext_str = prioritizeExtStr
        self.need_trace = bool(needTrace)
        self.use_planner = bool(usePlanner)
        self.planner_teacher_mode = bool(plannerTeacherMode)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calibration_id = calibration.calibration_id

        self.perc = PerceiveExtractor(
            cameraIntrinsics=calibration.intrinsics,
            imgSize=BasicParameters.IMAGE_SIZE,
            embedDim=ModuleDim.PerceptionEmbed,
            objectTokenCount=ModuleDim.PstObservedSlots,
            useHebbian=plasticHebbian,
            enableRecallAuxiliary=enablePerceptionSupervision)
        self.perception_recall_loss = PerceptionRecallLoss() if enablePerceptionSupervision else None
        
        self.attn = AttentionExtractor(
            embedDim=ModuleDim.AttentionFeat,
            sequenceLength=seqLen, 
            hebbianRate=(0.01 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian,
            structuredDim=ModuleDim.PerceptionEmbed,
            goalDim=ModuleDim.IntentionFeat,
            objectTokenCount=self.perc.object_token_count)
        
        self.mem = MemoryExtractor(
            inputDim=ModuleDim.AttentionFeat,
            ssmStateDim=ModuleDim.MemoryItem,
            memoryDim=ModuleDim.MemoryItem,
            outputDim=ModuleDim.MemoryFeat,
            hebbAlpha=(0.15 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian,
            emotionDim=ModuleDim.ValueEstimationOutEmotion)
        
        self.value_tensor_dim = 512
        self.actor = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            includeNoSkill=True,
            useHebb=plasticHebbian,
            valueTensorDim=self.value_tensor_dim,
            vNextTensorDim=self.value_tensor_dim,
            beliefDim=ModuleDim.DecisionBeliefDim,
            decisionDynDim=ModuleDim.DecisionDynDim,
            latentControlDim=ModuleDim.LatentControlDim,
            mapperEmbedDim=ModuleDim.MapperHiddenDim,)
        
        self.world = RSSMWorldModel(
            visionDim=ModuleDim.AttentionFeat, 
            deterDim=ModuleDim.WorldOutHState,
            stochDim=ModuleDim.WorldOutZState,
            stateDim=ModuleDim.WorldFeat,
            ssmDim=ModuleDim.WorldOutXState,
            useMemory=True,
            globalFeatDim=ModuleDim.PerceptionFeat,
            objectTokenDim=ModuleDim.PerceptionEmbed,
            numObjectTokens=self.perc.object_token_count,
            motionPredDim=ModuleDim.PerceptionEmbed,
            integratedFeatDim=ModuleDim.PerceptionFeat,
            # K_PST is decoupled from K_observed: PST is a persistent world memory
            # that must accumulate identities across frames, so it gets its own slot count.
            physicalSlots=ModuleDim.PstSlots,
            physicalSlotDim=ModuleDim.PstSlotDim,
            physicalPoseDim=ModuleDim.PstPoseDim,
            physicalAttrDim=ModuleDim.PstAttrDim,
            physicalIdDim=ModuleDim.PstIdDim,
            physicalRelDim=ModuleDim.PstRelDim,
            physicalRelationClasses=ModuleDim.PstRelationClasses,
            physicalSemanticDim=ModuleDim.PstSemanticDim,
            physicalStateDim=ModuleDim.PstStateDim,
            physicalAffordanceDim=ModuleDim.PstAffordanceDim,
            physicalTextDim=ModuleDim.PstTextDim,
            physicalSymbolDim=ModuleDim.PstSymbolClasses)

        self.critic = ValueEstimationExtractor(
            memoryDim=ModuleDim.MemoryFeat,
            attnDim=ModuleDim.AttentionFeat,
            stateDim=ModuleDim.WorldFeat,
            emotionDim=ModuleDim.ValueEstimationOutEmotion,
            useHebb=plasticHebbian,
            valueTensorDim=self.value_tensor_dim,)
        
        self.conscious = ConsciousnessExtractor(
            memItemDim=ModuleDim.MemoryItem,
            worldItemDim=ModuleDim.WorldMemoryItem,
            intentDim=ModuleDim.ConsciousnessState,
            useHebb=plasticHebbian)

        self.intention = IntentionExtractor(
            dimSem=ModuleDim.IntentionFeat,
            consSelfDim=int(self.conscious.self_dim),
            consIntentDim=int(self.conscious.intent_dim),
            ocrDictPath=BasicParameters.OCR_DICT_PATH)

        # Embodied-AGI v2: tensorized physical state, hierarchical goals, coarse-to-fine.
        self.pst_builder = PhysicalStateExtractor(
            inObjectDim=ModuleDim.PerceptionEmbed)
        self.pst_loss = PhysicalStateLoss()
        world_latent_dim = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState
        self.goal_manager = FourLevelGoalManager(
            worldLatentDim=world_latent_dim,
            pstSummaryDim=ModuleDim.PstSlotDim,
            intentDim=ModuleDim.IntentionFeat)
        self.goal_grounding = GoalGrounding()
        self.decision_decoupler = DecisionDecouplerV2(decisionDim=ModuleDim.DecisionBeliefDim)
        self.neuro_symbolic = NeuroSymbolicExtractor()
        self.temporal_gate = TemporalExecutionGateExtractor()
        self.execution_satisfaction_threshold = 0.5

        self.OCR = OCREngineExtractor()

        self.history_len = int(BasicParameters.MEMORY_CALLBACK_LEN)

        if plasticOnlineLearning:
            self.perc = PerceptionOnlineWrapper(self.perc)
            self.attn = AttentionOnlineWrapper(self.attn)
            self.world = WorldOnlineWrapper(self.world)
            self.critic =ValueEstimationOnlineWrapper(self.critic)
            self.intention = IntentionOnlineWrapper(self.intention)

        self.planner = None
        if self.use_planner or self.planner_teacher_mode:
            self.planner = DecisionPlannerExtractor().BuildPlanner(
                worldModel=self.world,
                wmIsOnlineWrapper=plasticOnlineLearning,
                decisionDecoupler=self.decision_decoupler,
                N=64, elite=8, iters=3,
                temperature=1.0, momentum=0.15,
                minVar=1e-4)

        self.buf_B = 0

        self.extra_mem = None
        self.thread_end = True
        self.smooth_generation = 0
        self.smooth_queue: queue.Queue = queue.Queue()
        self.smooth_worker = threading.Thread(target=self.SmoothWorkerLoop, daemon=True)
        self.smooth_worker.start()

        self.mem_copy = copy.deepcopy(self.mem)
        self.attn_copy = copy.deepcopy(self.attn)
        self.critic_copy = copy.deepcopy(self.critic)
        self.moduleMessager = ModuleMessagerManager(maxSteps=256)
        self.save_module_messager_output = bool(saveModuleMessagerOutput)

        self.ResetBuffers(B=1, isOnlineLearning=self.is_online_learning,device=self.device)

    def SetModuleMessagerEnabled(self, enabled: bool):
        self.save_module_messager_output = bool(enabled)

    def SetTraceEnabled(self, enabled: bool):
        self.need_trace = bool(enabled)

    def RuntimeModule(self, mod: nn.Module) -> nn.Module:
        return mod.base if hasattr(mod, "base") else mod

    def SuspendTransientTrainingGraph(self) -> Dict[str, Any]:
        return self.RuntimeModule(self.critic).SuspendTransientTrainingGraph()

    def RestoreTransientTrainingGraph(self, state: Dict[str, Any]) -> None:
        self.RuntimeModule(self.critic).RestoreTransientTrainingGraph(state)

    def PreviewWorldPrior(
        self,
        *,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotPhysicalState: torch.Tensor,
        cameraMotion: torch.Tensor,) -> Dict[str, torch.Tensor]:
        # Prospective inference must retain online candidate deltas when a wrapper is active.
        return self.world.StepPriorOnly(
            hPrev=hPrev,
            zPrev=zPrev,
            s4xPrev=s4xPrev,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotPhysicalState=robotPhysicalState,
            cameraMotion=cameraMotion,
            sample=False)

    def ExportRuntimeWorldMemoryBank(self, topk: int) -> Optional[Dict[str, torch.Tensor]]:
        # Durable memory belongs to the base world model, not its transient online wrapper.
        return self.RuntimeModule(self.world).ExportWorldMemoryBank(topk=topk)

    def RunWorldTrainingStep(
        self,
        *,
        visionIn: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        transitionPhysicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotPhysicalState: torch.Tensor,
        transitionRobotPhysicalState: torch.Tensor,
        cameraMotion: torch.Tensor,
        reward: Optional[torch.Tensor],
        done: Optional[torch.Tensor],) -> Dict[str, torch.Tensor]:
        update_runtime_memory = bool(self.training)
        kwargs = {
            "actionEnc": actionEnc,
            "reward": reward,
            "done": done,
            "physicalState": physicalState,
            "transitionPhysicalState": transitionPhysicalState,
            "robotPhysicalState": robotPhysicalState,
            "transitionRobotPhysicalState": transitionRobotPhysicalState,
            "cameraMotion": cameraMotion,
            "sample": update_runtime_memory,
            "updateMemory": update_runtime_memory,}
        if self.is_online_learning:
            return self.world(visionIn, **kwargs)
        return self.world.ForwardTrain(visionIn=visionIn, **kwargs)

    def PlannerInitialState(
        self,
        worldOutput: Dict[str, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            worldOutput["h_next"].detach(),
            worldOutput["z_next"].detach(),
            worldOutput["x_next"].detach(),)

    def ComputeAliveWorldPredictionLoss(
        self,
        *,
        prevWorldH: torch.Tensor,
        prevWorldZ: torch.Tensor,
        prevWorldX: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotPhysicalState: torch.Tensor,
        cameraMotion: torch.Tensor,
        targetVisualState: VisualState,
        precision: torch.Tensor,
        aliveMask: torch.Tensor,) -> Dict[str, torch.Tensor]:
        alive_mask = aliveMask.view(-1)
        if not bool(alive_mask.any().item()):
            return {}
        prediction = self.world.PredictNextVisualFromPosterior(
            prevWorldH,
            prevWorldZ,
            prevWorldX,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotPhysicalState=robotPhysicalState,
            cameraMotion=cameraMotion,
            sample=False,)
        return self.world.ComputePredictionLoss(
            predictedVisual=prediction["predicted_visual"],
            reconstructedVisualState=prediction["reconstructed_visual_state"],
            targetVisualState=targetVisualState,
            precision=precision,
            sampleMask=alive_mask,)

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        self.RuntimeModule(self.perc).ResetHebbianMemory(doneMask=doneMask)
        self.RuntimeModule(self.attn).ResetHebbianMemory(doneMask=doneMask)
        self.RuntimeModule(self.actor).ResetHebbianMemory(doneMask=doneMask)
        self.RuntimeModule(self.critic).ResetHebbianMemory(doneMask=doneMask)
        self.RuntimeModule(self.mem).ResetHebbianMemory(doneMask=doneMask)
        self.RuntimeModule(self.conscious).ResetHebbianMemory(doneMask=doneMask)

    @torch.no_grad()
    def ResizeStateBuffersForLoad(self, stateDict: Dict[str, Any]) -> None:
        modules = dict(self.named_modules())

        for key, value in stateDict.items():
            if not isinstance(value, torch.Tensor):
                continue

            parts = str(key).split(".")
            if len(parts) < 2:
                continue

            module_name = ".".join(parts[:-1])
            buffer_name = parts[-1]
            module = modules.get(module_name, None)
            if module is None or buffer_name not in module._buffers:
                continue

            current = module._buffers.get(buffer_name)
            if not isinstance(current, torch.Tensor):
                continue
            if tuple(current.shape) == tuple(value.shape):
                continue

            module._buffers[buffer_name] = torch.zeros(
                tuple(value.shape),
                device=current.device,
                dtype=current.dtype)

    def DetachVisualState(self, state: Optional[VisualState], *, clone: bool = False) -> Optional[VisualState]:
        if state is None:
            return None

        def d(t: torch.Tensor) -> torch.Tensor:
            out = t.detach()
            return out.clone() if clone else out

        return VisualState(
            IntegratedFeat=d(state.IntegratedFeat),
            GlobalFeat=d(state.GlobalFeat),
            VentralFeat=d(state.VentralFeat),
            DorsalFeat=d(state.DorsalFeat),
            MotionToken=d(state.MotionToken),
            QualityToken=d(state.QualityToken),
            PredErrorToken=d(state.PredErrorToken),
            ObjectTokens=d(state.ObjectTokens),
            PatchTokens=d(state.PatchTokens),
            SemanticNodes={k: d(v) for k, v in state.SemanticNodes.items()},
            Auxiliary={k: d(v) for k, v in state.Auxiliary.items() if isinstance(v, torch.Tensor)},)

    def DetachRuntimeObject(self, obj: Any, *, clone: bool = False) -> Any:
        if isinstance(obj, torch.Tensor):
            out = obj.detach()
            return out.clone() if clone else out
        if isinstance(obj, dict):
            return {k: self.DetachRuntimeObject(v, clone=clone) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.DetachRuntimeObject(v, clone=clone) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.DetachRuntimeObject(v, clone=clone) for v in obj)
        if isinstance(obj, deque):
            return deque((self.DetachRuntimeObject(v, clone=clone) for v in obj), maxlen=obj.maxlen)
        if hasattr(obj, "__dataclass_fields__"):
            vals = {
                name: self.DetachRuntimeObject(getattr(obj, name), clone=clone)
                for name in obj.__dataclass_fields__.keys()}
            return type(obj)(**vals)
        return obj

    def MoveRuntimeStateToModel(self, obj: Any) -> Any:
        """Move restored runtime tensors to this model's execution placement."""
        reference = next(self.parameters(), None)
        if reference is None:
            reference = next(self.buffers(), None)
        if reference is None:
            raise RuntimeError("cannot restore runtime state for a module without tensors")
        device = reference.device
        floating_dtype = (
            reference.dtype
            if reference.dtype.is_floating_point
            else torch.float32)

        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    return value.to(device=device, dtype=floating_dtype)
                return value.to(device=device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, deque):
                return deque((move(item) for item in value), maxlen=value.maxlen)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                fields = {
                    name: move(getattr(value, name))
                    for name in value.__dataclass_fields__.keys()}
                return type(value)(**fields)
            return value

        return move(obj)

    def ClearRuntimeRows(self, obj: Any, resetMask: torch.Tensor) -> Any:
        """Clone a nested runtime object and zero tensor rows selected by ``resetMask``."""
        if isinstance(obj, torch.Tensor):
            out = obj.detach().clone()
            if out.ndim > 0 and out.size(0) == resetMask.numel():
                out[resetMask] = 0
            return out
        if isinstance(obj, dict):
            return {key: self.ClearRuntimeRows(value, resetMask) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.ClearRuntimeRows(value, resetMask) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self.ClearRuntimeRows(value, resetMask) for value in obj)
        if isinstance(obj, deque):
            return deque(
                (self.ClearRuntimeRows(value, resetMask) for value in obj),
                maxlen=obj.maxlen)
        if hasattr(obj, "__dataclass_fields__"):
            values = {
                name: self.ClearRuntimeRows(getattr(obj, name), resetMask)
                for name in obj.__dataclass_fields__.keys()}
            return type(obj)(**values)
        return obj

    def BuildTopDownContext(
        self,
        realizedVisualPrior: Optional[Dict[str, torch.Tensor]],) -> TopDownContext:
        return TopDownContext(
            PredictedVisual=realizedVisualPrior,
            Precision=self.prev_precision,
            MemoryCue=self.prev_mem,)

    @torch.no_grad()
    def BuildRealizedVisualPrior(
        self,
        *,
        prevWorldH: torch.Tensor,
        prevWorldZ: torch.Tensor,
        prevWorldX: torch.Tensor,
        transitionPhysicalState: Dict[str, torch.Tensor],
        measuredActionEnc: torch.Tensor,
        transitionRobotPhysicalState: torch.Tensor,
        cameraMotion: torch.Tensor,
        transitionValid: torch.Tensor,) -> Optional[Dict[str, torch.Tensor]]:
        valid = transitionValid.view(-1).bool()
        if not bool(valid.any().item()):
            return None
        prediction = self.world.PredictNextVisualFromPosterior(
            prevWorldH,
            prevWorldZ,
            prevWorldX,
            physicalState=transitionPhysicalState,
            actionEnc=measuredActionEnc,
            robotPhysicalState=transitionRobotPhysicalState,
            cameraMotion=cameraMotion,
            sample=False,)["reconstructed_visual_state"]
        prior = self.DetachRuntimeObject(prediction)
        confidence = prior["PriorConfidence"]
        prior["PriorConfidence"] = confidence.masked_fill(
            ~valid.view(-1, 1), 0.0)
        return prior

    def BuildVisualSequenceTensors(
        self,
        visualStates: List[VisualState],
        *,
        validMasks: Optional[List[torch.Tensor]] = None,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recent = visualStates[-self.SEQ_LEN:]
        recent_valid = (
            validMasks[-self.SEQ_LEN:]
            if validMasks is not None
            else [torch.ones(batchSize, device=device, dtype=torch.bool) for _ in recent])
        if len(recent_valid) != len(recent):
            raise ValueError("validMasks must have one batch mask per visual state")
        T = len(recent)
        start = self.SEQ_LEN - T

        integrated = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionFeat, device=device, dtype=dtype)
        perc_mod = self.RuntimeModule(self.perc)
        object_token_count = int(getattr(perc_mod, "object_token_count", 16))
        object_seq = torch.zeros(batchSize, self.SEQ_LEN, object_token_count, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        motion_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        quality_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        pred_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        key_padding_mask = torch.ones(batchSize, self.SEQ_LEN, device=device, dtype=torch.bool)

        for idx, (vs, valid) in enumerate(zip(recent, recent_valid), start=start):
            integrated[:, idx] = vs.IntegratedFeat
            object_seq[:, idx] = vs.ObjectTokens
            motion_seq[:, idx] = vs.MotionToken
            quality_seq[:, idx] = vs.QualityToken
            pred_seq[:, idx] = vs.PredErrorToken
            key_padding_mask[:, idx] = ~valid

        return integrated.contiguous(), object_seq.contiguous(), motion_seq.contiguous(), quality_seq.contiguous(), pred_seq.contiguous(), key_padding_mask

    def EncodeOcrSemantic(self, ocrTexts: Optional[List[List[str]]], *, batchSize: int, device: torch.device) -> torch.Tensor:
        intention_mod = self.RuntimeModule(self.intention)
        if ocrTexts is None:
            merged = [""] * batchSize
        else:
            merged = intention_mod.MergeOcrTexts(ocrTexts)
        if len(merged) != batchSize:
            merged = (merged + [""] * batchSize)[:batchSize]
        with torch.no_grad():
            sem, _, _ = intention_mod.EncodeStringsWithSlots(merged, device=device)
        return sem.detach()

    @torch.no_grad()
    def ResetBuffers(self, B: int = 1, isOnlineLearning: bool = False, device: Optional[torch.device] = None):
        device = device or self.device

        def z(*s, dtype=torch.float32):
            return torch.zeros(B, *s, device=device, dtype=dtype)

        self.prev_mem = z(ModuleDim.MemoryFeat)
        self.prev_attn = z(ModuleDim.AttentionFeat)

        self.prev_world_h = z(ModuleDim.WorldOutHState)
        self.prev_world_z = z(ModuleDim.WorldOutZState)
        self.prev_world_x = z(ModuleDim.WorldOutXState)
        self.prev_world_s = z(ModuleDim.WorldFeat)
        self.prev_done_flag = torch.ones(B, device=device, dtype=torch.bool)

        if isOnlineLearning and hasattr(self.actor, "base"):
            self.prev_option_logit = z(self.actor.base.num_options)
            actor_runtime = self.actor.base
        else:
            self.prev_option_logit = z(self.actor.num_options)
            actor_runtime = self.actor

        self.prev_decision_state = z(int(actor_runtime.dyn_dim))
        self.prev_latent_control = z(int(actor_runtime.u_dim))
        self.prev_mapper_hidden = z(int(actor_runtime.mapper_hidden_dim))
        self.prev_td_error = torch.zeros(B, device=device, dtype=torch.float32)
        option_policy_dim = int(actor_runtime.dyn_dim + actor_runtime.u_dim + actor_runtime.mapper_hidden_dim)
        self.active_option_policy_input = z(option_policy_dim)
        self.active_option_prior_logit = z(int(actor_runtime.num_options))
        self.active_option_goal_mid = z(ModuleDim.GoalMidDim)
        self.active_option_index = torch.zeros(B, device=device, dtype=torch.long)
        self.active_option_valid = torch.zeros(B, device=device, dtype=torch.bool)
        self.prev_belief_prediction_state = z(int(actor_runtime.dyn_dim))
        self.prev_belief_prediction_valid = torch.zeros(B, device=device, dtype=torch.bool)
        self.temporal_active_mask = z()
        self.temporal_action_age_steps = torch.zeros(
            B, device=device, dtype=torch.long)
        self.temporal_action_epoch = torch.zeros(B, device=device, dtype=torch.long)
        self.temporal_invoke_drift = z()
        self.temporal_active_kind = torch.zeros(B, device=device, dtype=torch.long)
        self.active_motion_command = None
        # Last frame's committed target pose. Next frame the measured endpoint pose is compared
        # against it to expose command-vs-achieved tracking error.
        self.prev_target_endpoint_pose = z(ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim)
        self.prev_target_endpoint_pose[..., 6] = 1.0
        self.prev_target_endpoint_valid = torch.zeros(B, device=device, dtype=torch.bool)
        self.prev_measured_endpoint_pose = z(
            ModuleDim.RobotStateEndpointCount,
            ModuleDim.DecisionEndpointPoseDim)
        self.prev_measured_endpoint_pose[..., 6] = 1.0
        self.prev_measured_endpoint_valid = torch.zeros(B, device=device, dtype=torch.bool)

        self.prev_entropy = z()

        self.prev_visual_state = None
        self.prev_visual_valid = torch.zeros(
            B, device=device, dtype=torch.bool)
        self.prospective_visual_prediction = None
        self.prev_precision = torch.ones(B, device=device, dtype=torch.float32)
        self.prev_goal_bias = z(ModuleDim.IntentionFeat)
        self.prev_self_sem = None
        self.prev_intent_sem = z(ModuleDim.IntentionFeat)

        self.prev_failure_count = z()

        self.perc_buffer = []
        self.visual_state_buffer = []
        self.visual_state_valid_buffer = []

        self.history = deque(maxlen=self.history_len)

        self.slow_step_count = 0
        self.slow_cache = None

        self.buf_B = B

    def RelativeCameraMotion(
        self,
        prevPose: torch.Tensor,
        curPose: torch.Tensor,
        prevValid: torch.Tensor) -> torch.Tensor:
        """Derive rotation-only inter-frame camera motion for the fixed robot."""
        motion = CanonicalizeQuaternion(
            QuatMultiply(
                QuatConjugate(prevPose[:, 3:7]),
                curPose[:, 3:7]))
        identity = motion.new_zeros(motion.shape)
        identity[:, 3] = 1.0
        return torch.where(prevValid.view(-1, 1), motion, identity)

    def HasTrustedExternalText(
        self,
        textExt: Optional[List[Optional[str]]],
        textTrust: Optional[List[str]]) -> bool:
        if textExt is None:
            return False
        trust = (
            [TEXT_TRUST_UNSAFE_EXTERNAL for _ in range(len(textExt))]
            if textTrust is None
            else [str(item) for item in textTrust])
        return any(
            trust_item == TEXT_TRUST_OPERATOR_COMMAND
            and text_item is not None
            and str(text_item).strip() != ""
            for text_item, trust_item in zip(textExt, trust))

    def RunPerceptionPhysicalStage(self, **kwargs) -> PerceptionPhysicalStage:
        return PerceptionPhysicalStage(**kwargs)

    def RunValueMemoryWorldStage(self, **kwargs) -> ValueMemoryWorldStage:
        return ValueMemoryWorldStage(**kwargs)

    def RunCognitionGoalStage(self, **kwargs) -> CognitionGoalStage:
        return CognitionGoalStage(**kwargs)

    def RunDecisionExecutionStage(self, **kwargs) -> ExecutionStage:
        return ExecutionStage(**kwargs)

    def BuildTrainingLosses(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return losses

    @staticmethod
    def ComputeTemporalKindSupervisionLoss(
        executionKindScores: torch.Tensor,
        targetKind: torch.Tensor,
        targetValid: torch.Tensor,
        activeMask: torch.Tensor,
        overrideApplied: torch.Tensor,) -> torch.Tensor:
        inactive = activeMask <= 0.5
        illegal_while_inactive = (
            targetKind.eq(CONTINUE)
            | targetKind.eq(CANCEL)
            | targetKind.eq(REDISPATCH))
        # A hard safety override is a non-learned software boundary.  It must not
        # turn the pre-override policy proposal into a supervised FAILSAFE label.
        learnable = (
            targetValid
            & ~(inactive & illegal_while_inactive)
            & ~overrideApplied)
        if not bool(learnable.any().item()):
            # DISPATCH is always a legal finite execution coordinate; retaining
            # this zero graph keeps the empty-label batch differentiable without
            # summing the -inf entries used for illegal actions.
            return executionKindScores[:, DISPATCH].sum() * 0.0
        # Index before cross entropy because illegal execution coordinates are
        # represented by -inf; multiplying an invalid per-row loss by zero would
        # still leave an inf * 0 NaN.
        return nn.functional.cross_entropy(
            executionKindScores[learnable],
            targetKind[learnable])

    def BuildExecutionSatisfaction(
        self,
        robotState: Dict[str, torch.Tensor],
        grounding: Dict[str, torch.Tensor],) -> torch.Tensor:
        planner_success = robotState["planner_reached"].view(-1) * torch.exp(-robotState["planner_tracking_error"].view(-1))
        return planner_success * grounding["reference_confidence"].detach()

    def BuildExecutedActionFeedback(
        self,
        endpointPose: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(endpointPose.size(0))
        measured_valid = self.prev_measured_endpoint_valid.view(B, 1, 1)
        decision_endpoint_pose = endpointPose[
            :, ModuleDim.RobotStateControlledEndpointSlice]
        previous_decision_endpoint_pose = self.prev_measured_endpoint_pose[
            :, ModuleDim.RobotStateControlledEndpointSlice]
        executed_decision_tensor = self.decision_decoupler.MaskDecisionTensor(
            RelativePoseError(
                previous_decision_endpoint_pose,
                decision_endpoint_pose)) * measured_valid
        feedback = self.decision_decoupler.EncodeEndpointAction(
            executed_decision_tensor)
        feedback = feedback * self.prev_measured_endpoint_valid.view(B, 1)
        return executed_decision_tensor, feedback

    def MaterializeMotionCommand(
        self,
        decision: DecoupledDecision,
        currentEndpointPoseWorld: torch.Tensor,) -> MotionCommand:
        """Cross the execution seam from local learned action to a world target."""
        decision_tensor = self.decision_decoupler.MaskDecisionTensor(
            decision.decision_tensor)
        return MotionCommand(
            decision_tensor=decision_tensor,
            target_endpoint_pose=ApplyPoseDelta(
                currentEndpointPoseWorld,
                decision_tensor),
            endpoint_names=tuple(ModuleDim.DecisionEndpointNames),
            decision_dof_mask=(
                self.decision_decoupler.action_projector.action_mask
                .expand(decision_tensor.size(0), -1, -1).bool()),
            gripper_cmd=decision.gripper_cmd,
            gripper_valid=decision.gripper_valid,
            mode_logits=decision.mode_logits,
            mode_valid=decision.mode_valid,
            safety_scores=decision.safety_scores,
            safety_names=SAFETY_MARGIN_NAMES,)

    def RebaseWorldMotionCommand(
        self,
        command: MotionCommand,
        currentEndpointPoseWorld: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> MotionCommand:
        """Express an already materialized world target from the current endpoint pose."""
        remaining_decision = self.decision_decoupler.MaskDecisionTensor(
            RelativePoseError(
                currentEndpointPoseWorld,
                command.target_endpoint_pose))
        reachable_target = ApplyPoseDelta(
            currentEndpointPoseWorld,
            remaining_decision)
        return replace(
            command,
            decision_tensor=remaining_decision,
            target_endpoint_pose=reachable_target,
            safety_scores=self.decision_decoupler.SafetyScores(
                remaining_decision,
                risk,
                confidence,
                precision))

    @staticmethod
    def BodyEndpointPoseRelative(endpointPose: torch.Tensor) -> torch.Tensor:
        body_pose = endpointPose[:, ModuleDim.RobotStateBodyEndpointSlice]
        reference_pose = endpointPose[
            :,
            ModuleDim.DecisionEndpointReferenceRobotStateIndices]
        return RelativePose(reference_pose, body_pose)

    @staticmethod
    def CameraPhysicalReference(
        cameraPoseWorld: torch.Tensor,
        baseOrientationWorld: torch.Tensor,
        gravityDirectionWorld: torch.Tensor,) -> torch.Tensor:
        """Build gauge-free camera rotation and gravity without fixed translation."""
        camera_orientation_world = cameraPoseWorld[..., 3:7]
        world_orientation_camera = QuatConjugate(camera_orientation_world)
        base_orientation_camera = CanonicalizeQuaternion(
            QuatMultiply(
                world_orientation_camera,
                baseOrientationWorld))
        gravity_camera = QuatRotate(
            world_orientation_camera,
            gravityDirectionWorld)
        return torch.cat([base_orientation_camera, gravity_camera], dim=-1)

    @torch.no_grad()
    def ResetDecisionRuntimeRows(
        self,
        doneMask: torch.Tensor,
        allDone: Optional[bool] = None,) -> None:
        keep = (~doneMask).float()
        keep_feature = keep.unsqueeze(-1)
        keep_pose = keep.view(-1, 1, 1)

        self.prev_mem.mul_(keep_feature)
        self.prev_attn.mul_(keep_feature)
        self.prev_option_logit.mul_(keep_feature)
        self.prev_entropy.mul_(keep)
        self.prev_decision_state.mul_(keep_feature)
        self.prev_latent_control.mul_(keep_feature)
        self.prev_mapper_hidden.mul_(keep_feature)
        self.prev_td_error.mul_(keep)
        self.prev_failure_count.mul_(keep)
        self.prev_precision = torch.where(
            doneMask,
            torch.ones_like(self.prev_precision),
            self.prev_precision)
        self.prev_goal_bias.mul_(keep_feature)
        self.prev_self_sem = self.ClearRuntimeRows(
            self.prev_self_sem, doneMask)
        self.prev_intent_sem.mul_(keep_feature)
        self.active_option_policy_input.mul_(keep_feature)
        self.active_option_prior_logit.mul_(keep_feature)
        self.active_option_goal_mid.mul_(keep_feature)
        self.active_option_index.masked_fill_(doneMask, 0)
        self.active_option_valid.logical_and_(~doneMask)
        self.prev_belief_prediction_state.mul_(keep_feature)
        self.prev_belief_prediction_valid.logical_and_(~doneMask)

        self.temporal_active_mask.mul_(keep)
        self.temporal_action_age_steps.masked_fill_(doneMask, 0)
        self.temporal_action_epoch.masked_fill_(doneMask, 0)
        self.temporal_active_kind.masked_fill_(doneMask, 0)
        self.temporal_invoke_drift.mul_(keep)

        self.prev_target_endpoint_pose.mul_(keep_pose)
        self.prev_target_endpoint_pose[..., 6].add_(1.0 - keep.view(-1, 1))
        self.prev_target_endpoint_valid.logical_and_(~doneMask)
        self.prev_measured_endpoint_pose.mul_(keep_pose)
        self.prev_measured_endpoint_pose[..., 6].add_(1.0 - keep.view(-1, 1))
        self.prev_measured_endpoint_valid.logical_and_(~doneMask)

        if allDone is None:
            allDone = bool(doneMask.all().item())
        if allDone:
            self.active_motion_command = None
        else:
            self.active_motion_command = self.ClearRuntimeRows(
                self.active_motion_command, doneMask)
            if self.active_motion_command is not None:
                self.active_motion_command.target_endpoint_pose[doneMask, :, 6] = 1.0

    def Step(
        self,
        step: BrainStepInput,) -> BrainStepOutput:
        frame = step.frame
        textExt = step.text_ext
        textTrust = step.text_trust
        rewardExt = step.reward_ext
        doneFlag = step.done_flag
        isTrain = step.is_train
        sampleActions = step.sample_actions
        deterministicActor = step.deterministic_actor
        depth = step.depth
        depthValid = step.depth_valid
        perceptionTargets = step.perception_targets
        robotState = step.robot_state
        compute_critic_loss = bool(step.compute_critic_loss)

        if self.is_online_learning and not isTrain:
            raise RuntimeError(f"Wrappers can only be used during training, but isTrain is {isTrain}, isUseWrappers is {self.is_online_learning}")

        isBeginStep = True

        def saveModuleOutput(moduleName: str, output: Any):
            nonlocal isBeginStep
            if not self.save_module_messager_output:
                return
            self.moduleMessager.SaveModuleOutput(moduleName, output, isBeginStep=isBeginStep)
            isBeginStep = False

        B, dev = frame.size(0), frame.device
        if self.buf_B != B:
            self.ResetBuffers(B=B, isOnlineLearning=self.is_online_learning, device=dev)
            self.RuntimeModule(self.world).ResetState(batchSize=B)
        # Current proprioception closes the previous transition. During offline training the
        # recorded endpoint poses are executed behavior, never the model's counterfactual command.
        endpoint_pose = robotState["endpoint_pose"]
        camera_pose_world = endpoint_pose[
            :, ModuleDim.RobotStateCameraEndpointIndex]
        robot_physical_reference = self.CameraPhysicalReference(
            camera_pose_world,
            robotState["base_orientation_world"],
            robotState["gravity_direction_world"])
        prev_camera_pose_world = self.prev_measured_endpoint_pose[
            :, ModuleDim.RobotStateCameraEndpointIndex]
        camera_motion_from_prev = self.RelativeCameraMotion(
            prev_camera_pose_world,
            camera_pose_world,
            self.prev_measured_endpoint_valid)
        decision_endpoint_pose = endpoint_pose[
            :, ModuleDim.RobotStateControlledEndpointSlice]
        body_endpoint_pose_relative = self.BodyEndpointPoseRelative(endpoint_pose)
        planner_expected_endpoint_pose = robotState["planner_expected_endpoint_pose"]
        executed_decision_tensor, world_action_feedback = self.BuildExecutedActionFeedback(
            endpoint_pose)
        reported_model_command_executed = (
            robotState["model_command_executed"].view(B).eq(1.0))
        executed_action_id = robotState["executed_action_id"].view(B)
        expected_action_id = self.temporal_action_epoch
        executed_action_matches = (
            (expected_action_id > 0)
            & (executed_action_id == expected_action_id))
        if bool((reported_model_command_executed & ~executed_action_matches).any().item()):
            raise ValueError(
                "RobotState executed_action_id does not identify the preceding "
                "model command")
        model_command_executed = (
            reported_model_command_executed & executed_action_matches)
        endpoint_tracking_error = (
            self.decision_decoupler.MaskDecisionTensor(
                RelativePoseError(
                    self.prev_target_endpoint_pose,
                    decision_endpoint_pose))
            * (self.prev_target_endpoint_valid & model_command_executed).view(B, 1, 1))
        planner_endpoint_tracking_error = (
            self.decision_decoupler.MaskDecisionTensor(
                RelativePoseError(
                    planner_expected_endpoint_pose,
                    decision_endpoint_pose))
            * robotState["planner_executing"].view(B, 1, 1))
        decision_robot_state_encoding = self.decision_decoupler.EncodeRobotState(
            body_endpoint_pose_relative,
            endpoint_tracking_error,
            planner_endpoint_tracking_error)
        decision_action_feedback = self.decision_decoupler.EncodeDecisionFeedback(
            executed_decision_tensor,
            decision_robot_state_encoding,
            robot_physical_reference)
        decision_action_feedback = decision_action_feedback * (
            self.prev_measured_endpoint_valid.view(B, 1))
        # Candidate commands remain prospective; only measured behavior explains this frame.
        world_runtime = self.RuntimeModule(self.world)
        world_robot_physical_encoding = world_runtime.EncodeRobotPhysicalState(
            body_endpoint_pose_relative,
            robot_physical_reference)

        # External feedback was strictly validated at the preprocessing seam.
        reward_ext = None if rewardExt is None else rewardExt.detach().view(B)
        done_ext = None if doneFlag is None else doneFlag.detach().view(B)
        
        if self.extra_mem and self.thread_end:
            self.mem.MergeMemoryState(self.extra_mem)
            self.extra_mem =None


        def init_shadow_module_parms():
            mem_state = self.mem.ExportState()
            self.mem_copy.EnsureB(B)
            self.mem_copy.ImportState(mem_state)
            self.mem_copy.load_state_dict(self.mem.state_dict(), strict=True)
            self.mem_copy.pending = self.DetachRuntimeObject(self.mem.pending, clone=True)
            attn_state = self.attn.ExportState()
            self.attn_copy.ImportState(attn_state)
            self.attn_copy.load_state_dict(self.attn.state_dict(), strict=True)
            critic_state = self.critic.ExportState()
            self.critic_copy.ImportState(critic_state)
            self.critic_copy.load_state_dict(self.critic.state_dict(), strict=True)

        if self.need_trace and not isTrain and reward_ext is not None and self.history and self.thread_end:
            self.thread_end = False
            init_shadow_module_parms()
            # Traces are detached/cloned at creation, so a shallow snapshot is enough.
            self.smooth_queue.put((
                list(self.history), reward_ext, "Reward",
                self.attn_copy, self.mem_copy, self.critic_copy,
                self.smooth_generation))

        if self.need_trace and not isTrain and done_ext is not None and self.history and self.thread_end:
            self.thread_end = False
            init_shadow_module_parms()
            self.smooth_queue.put((
                list(self.history), done_ext, "Done",
                self.attn_copy, self.mem_copy, self.critic_copy,
                self.smooth_generation))

        credit_option_policy_input = self.active_option_policy_input.detach()
        credit_option_prior_logit = self.active_option_prior_logit.detach()
        credit_option_goal_mid = self.active_option_goal_mid.detach()
        credit_option_index = self.active_option_index.detach()
        credit_option_valid = (
            self.active_option_valid & model_command_executed).detach()
        credit_option_weight = credit_option_valid.float()
        belief_prediction_state_prev = self.prev_belief_prediction_state.detach()
        belief_prediction_valid_prev = self.prev_belief_prediction_valid.detach()

        prev_world_h_for_prediction = self.prev_world_h.detach()
        prev_world_z_for_prediction = self.prev_world_z.detach()
        prev_world_x_for_prediction = self.prev_world_x.detach()
        prev_done_for_prediction = self.prev_done_flag.detach().clone()
        prev_physical_state_for_prediction = world_runtime.BuildModelPhysicalState(
            world_runtime.ExportPhysicalState(),
            prev_camera_pose_world)
        prev_world_robot_physical_encoding_for_prediction = (
            world_runtime.ExportRobotPhysicalState()["RobotPhysicalState"])

        realized_visual_prior = self.BuildRealizedVisualPrior(
            prevWorldH=prev_world_h_for_prediction,
            prevWorldZ=prev_world_z_for_prediction,
            prevWorldX=prev_world_x_for_prediction,
            transitionPhysicalState=prev_physical_state_for_prediction,
            measuredActionEnc=world_action_feedback,
            transitionRobotPhysicalState=(
                prev_world_robot_physical_encoding_for_prediction),
            cameraMotion=camera_motion_from_prev,
            transitionValid=(
                ~prev_done_for_prediction
                & self.prev_measured_endpoint_valid),)
        top_down = self.BuildTopDownContext(realized_visual_prior)
        previous_visual_state = self.prev_visual_state
        previous_visual_valid = self.prev_visual_valid
        visual_state = self.perc(
            frame,
            prevVisualState=previous_visual_state,
            prevVisualValid=previous_visual_valid,
            topDownContext=top_down,
            depth=depth,
            depthValid=depthValid,
            cameraMotion=camera_motion_from_prev)
        perc_feats = visual_state.IntegratedFeat # [B, D_perc]
        # Slow/fast split: OCR, consciousness, intention and the long/mid goal stack run
        # every slow_period steps; an external text command forces an immediate refresh.
        text_control_refresh = self.HasTrustedExternalText(textExt, textTrust)
        slow_refresh = (
            self.slow_cache is None
            or (self.slow_step_count % self.slow_period == 0)
            or text_control_refresh)
        if not slow_refresh:
            refresh_event = robotState["planner_failed"].eq(1.0)
            if done_ext is not None:
                refresh_event = refresh_event | done_ext.eq(1.0)
            slow_refresh = bool(refresh_event.any().item())
        self.slow_step_count += 1
        if slow_refresh:
            ocr_items = self.OCR(frame)
            fuse_ocr = self.OCR.ExportFusedTexts() # List[List[str]]
            ocr_semantic = self.EncodeOcrSemantic(fuse_ocr, batchSize=B, device=dev)
        else:
            ocr_items = self.slow_cache["ocr_items"]
            fuse_ocr = self.slow_cache["fuse_ocr"]
            ocr_semantic = self.slow_cache["ocr_semantic"]

        visual_seq_src = self.visual_state_buffer + [visual_state]
        visual_valid_src = self.visual_state_valid_buffer + [
            torch.ones(B, device=dev, dtype=torch.bool)]
        if len(visual_seq_src) > self.SEQ_LEN:
            visual_seq_src = visual_seq_src[-self.SEQ_LEN:]
            visual_valid_src = visual_valid_src[-self.SEQ_LEN:]

        percs_seq, object_seq, motion_seq, quality_seq, pred_error_seq, key_padding_mask = self.BuildVisualSequenceTensors(
            visual_seq_src,
            validMasks=visual_valid_src,
            batchSize=B,
            device=dev,
            dtype=frame.dtype)

        self.visual_state_buffer = [
            self.DetachVisualState(v)
            for v in visual_seq_src]
        self.visual_state_valid_buffer = [valid.detach().clone() for valid in visual_valid_src]
        self.perc_buffer = [v.IntegratedFeat for v in self.visual_state_buffer if v is not None]
        self.prev_visual_state = self.DetachVisualState(visual_state)
        self.prev_visual_valid.fill_(True)

        saveModuleOutput("Perception", {
            "feat": perc_feats,
            "visual_state": {
                "global": visual_state.GlobalFeat,
                "ventral": visual_state.VentralFeat,
                "dorsal": visual_state.DorsalFeat,
                "motion": visual_state.MotionToken,
                "quality": visual_state.QualityToken,
                "pred_error": visual_state.PredErrorToken,
                "objects_mean": visual_state.ObjectTokens.mean(dim=1),
                "metric_depth": visual_state.Auxiliary["MetricDepth"],
                "monocular_depth": visual_state.Auxiliary["MonocularDepth"],
                "sensor_depth_reliability": visual_state.Auxiliary["SensorDepthReliability"],
                "object_geometry": visual_state.Auxiliary["ObjectGeometry"],},
            "key_padding_mask": key_padding_mask,})

        # --- Embodied-AGI v2: build and merge the Physical State Tensor before world inference ---
        semantic_view = self.pst_builder.SemanticWorldView(visual_state.SemanticNodes)
        node_mask = semantic_view["NodePresence"]
        geometry_valid = visual_state.Auxiliary["ObjectGeometryValid"]
        physical_out = self.pst_builder(
            visual_state.ObjectTokens,
            visual_state.Auxiliary["ObjectMotion"],
            visual_state.Auxiliary["ObjectGeometry"],
            node_mask,
            geometry_valid)
        observed_pst = {
            **physical_out,
            **semantic_view,
            "ObservedSlotMask": physical_out["ObservationMask"]}
        pst = world_runtime.UpdatePhysicalState(
            observed_pst,
            cameraPoseWorld=camera_pose_world,
            robotPhysicalState=world_robot_physical_encoding)
        pst["U"] = self.mem.usage_bank.SlotReadout(
            pst["IdentityKey"],
            pst["ARaw"],
            pst["SlotPresence"] * pst["MphysRaw"]) * pst["SlotPresence"].unsqueeze(-1)
        active_physical_mask = pst["SlotPresence"] * pst["MphysRaw"]
        pst_summary = self.pst_builder.SlotSummary(pst["SlotState"], active_physical_mask)
        perception_physical_stage = self.RunPerceptionPhysicalStage(
            visual_state=visual_state,
            perc_feats=perc_feats,
            percs_seq=percs_seq,
            object_seq=object_seq,
            motion_seq=motion_seq,
            quality_seq=quality_seq,
            pred_error_seq=pred_error_seq,
            key_padding_mask=key_padding_mask,
            prev_visual_for_loss=previous_visual_state,
            ocr_items=ocr_items,
            fuse_ocr=fuse_ocr,
            ocr_semantic=ocr_semantic,
            slow_refresh=slow_refresh,
            text_control_refresh=text_control_refresh,
            pst=pst,
            observed_pst=observed_pst,
            pst_summary=pst_summary,
            world_action_feedback=world_action_feedback)
        
        w_preview = self.PreviewWorldPrior(
            hPrev=self.prev_world_h,
            zPrev=self.prev_world_z,
            s4xPrev=self.prev_world_x,
            physicalState=prev_physical_state_for_prediction,
            actionEnc=world_action_feedback,
            robotPhysicalState=prev_world_robot_physical_encoding_for_prediction,
            cameraMotion=camera_motion_from_prev)
        saveModuleOutput("WorldPreview", w_preview)

        s_t = w_preview["s_next"] # [B, D_world]
        r_t = w_preview["r_pred"].detach() # [B]
        d_t = w_preview["d_prob"].detach() # [B]
        d_tr = w_preview["d_tr"] # Optional[[B, D_world]]
        d_ph = w_preview["d_ph"] # Optional[[B, D_world]]

        value_reward = reward_ext if (isTrain and reward_ext is not None) else r_t
        value_done = done_ext if (isTrain and done_ext is not None) else d_t

        if self.is_online_learning:
            value_kwargs = {
                "rewardModel": value_reward,
                "computeLoss": isTrain and compute_critic_loss,
                "policyEntropyPrev": self.prev_entropy,
                "doneModel": value_done,
                "worldDeltaTransport": d_tr,
                "worldDeltaPhysics": d_ph}
            value_x = {"memoryPrev": self.prev_mem,"attnPrev": self.prev_attn, "state": s_t} # memoryPrev:[B, D_mem], attnPrev:[B, D_attn], state:[B, D_world]
            critic_out = self.critic(x=value_x, **value_kwargs)
        else:
            critic_out = self.critic(memoryPrev=self.prev_mem,attnPrev=self.prev_attn,state=s_t,
                                     rewardModel=value_reward,doneModel=value_done,
                                     computeLoss=isTrain and compute_critic_loss,
                                     policyEntropyPrev=self.prev_entropy,
                                     worldDeltaTransport=d_tr,worldDeltaPhysics=d_ph,)
        saveModuleOutput("ValueEstimation", critic_out)

        value_current = critic_out.value
        value_next_current = critic_out.valueNext

        # ``tdError`` is geometric transition surprise.  Reward credit is the
        # one-step Bellman residual carried by the dedicated return branch.
        td_sig = critic_out.tdError.detach() # [B]
        return_advantage_sig = critic_out.returnAdvantage.detach() # [B]
        unc_sig = critic_out.uncertainty.detach() # [B]
        precision_sig = critic_out.precision.detach() # [B]
        emotion_sig = critic_out.emotion.detach() # [B, D_emotion]
        value_comps = critic_out.rComps
        risk_sig = value_comps["risk"].detach()
        confidence_sig = value_comps["confidence"].detach()
        if not slow_refresh and bool((risk_sig > 0.85).any().item()):
            slow_refresh = True
            ocr_items = self.OCR(frame)
            fuse_ocr = self.OCR.ExportFusedTexts()
            ocr_semantic = self.EncodeOcrSemantic(fuse_ocr, batchSize=B, device=dev)
            perception_physical_stage.ocr_items = ocr_items
            perception_physical_stage.fuse_ocr = fuse_ocr
            perception_physical_stage.ocr_semantic = ocr_semantic
            perception_physical_stage.slow_refresh = True

        self.prev_precision = precision_sig.detach()

        atten_out = self.attn(
            percs_seq,
            keyPaddingMask=key_padding_mask,
            tdError=td_sig,
            uncertainty=unc_sig,
            objectSeq=object_seq,
            motionSeq=motion_seq,
            qualitySeq=quality_seq,
            predErrorSeq=pred_error_seq,
            goalBias=self.prev_goal_bias,
            precision=precision_sig,
            applyPlasticity=True) # [B, D_attn]
        saveModuleOutput("Attention", atten_out)

        intent_hint_for_memory = self.prev_intent_sem
        memory_reward = value_reward.detach()
        mem_feat = self.mem(
            atten_out,
            tdError=td_sig,
            emotion=emotion_sig,
            reward=memory_reward,
            visualState=visual_state,
            ocrSemantic=ocr_semantic,
            intentHint=intent_hint_for_memory,
            uncertainty=unc_sig,
            risk=risk_sig,
            confidence=confidence_sig) # [B, D_mem], [B, D_mem]
        saveModuleOutput("Memory", mem_feat)

        if isTrain:
            w_out = self.RunWorldTrainingStep(
                visionIn=atten_out,
                physicalState=pst,
                transitionPhysicalState=prev_physical_state_for_prediction,
                actionEnc=world_action_feedback,
                robotPhysicalState=world_robot_physical_encoding,
                transitionRobotPhysicalState=prev_world_robot_physical_encoding_for_prediction,
                cameraMotion=camera_motion_from_prev,
                reward=reward_ext,
                done=done_ext)
        else:
            w_out = self.world.StepPosterior(visionIn=atten_out,
                                             actionEnc=world_action_feedback,
                                             physicalState=pst,
                                             transitionPhysicalState=prev_physical_state_for_prediction,
                                             robotPhysicalState=world_robot_physical_encoding,
                                             transitionRobotPhysicalState=prev_world_robot_physical_encoding_for_prediction,
                                             cameraMotion=camera_motion_from_prev,
                                             sample=False)
        saveModuleOutput("World", w_out)

        s_t = w_out["s_next"] # [B, D_world]
        r_t = w_out["r_pred"].detach() # [B]
        d_t = w_out["d_prob"].detach() # [B]
        d_tr = w_out["d_tr"] # Optional[[B, D_world]]
        d_ph = w_out["d_ph"] # Optional[[B, D_world]]

        done_now = (
            done_ext.eq(1.0)
            if done_ext is not None
            else torch.zeros(B, device=dev, dtype=torch.bool))
        terminal_stop = (
            done_now
            if done_ext is not None
            else d_t > 0.5)
        hard_stop = torch.maximum(
            (risk_sig > 0.98).float(),
            terminal_stop.float())

        self.prev_world_s = s_t.detach()
        self.prev_done_flag = done_now.detach()

        if slow_refresh:
            memory_bank = self.mem.ExportMemoryBank(
                topk=BasicParameters.CONSCIOUSNESSTEM,
                includeMeta=False) # Optional[Dict[str, Tensor]]
            world_bank = self.ExportRuntimeWorldMemoryBank(topk=BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]

            conscious_out = self.conscious(memoryBank=memory_bank, worldBank=world_bank) # self_sem/intention_sem: [B, D_cons]

            intent_sem, sym_probs, intention_extras = self.intention(
                conscious_out.self_sem,
                conscious_out.intent_sem,
                ocrTexts=fuse_ocr,
                extTexts=textExt,
                prioritizeExt=self.prioritize_ext_str,
                textTrust=textTrust,) # [B, D_intent], [B, K_sym], Dict[str, Tensor]
        else:
            conscious_out = ConsciousnessOutput(
                self_sem=self.slow_cache["self_sem"],
                intent_sem=self.slow_cache["cons_intent_sem"],
                extras=self.slow_cache["cons_extras"],)
            intent_sem = self.slow_cache["intent_sem"]
            sym_probs = self.slow_cache["sym_probs"]
            intention_extras = self.slow_cache["intention_extras"]

        saveModuleOutput("Consciousness", {
            "self_sem": conscious_out.self_sem,
            "intent_sem": conscious_out.intent_sem,
            "extras": conscious_out.extras,})

        saveModuleOutput("OCR", {
            "items": ocr_items,
            "texts": fuse_ocr,})

        intention_texts = [] if intention_extras is None else intention_extras.get("recall_texts", [])
        saveModuleOutput("Intention", {
            "intent_sem": intent_sem,
            "sym_probs": sym_probs,
            "extras": intention_extras,
            "ocr_texts": fuse_ocr,
            "ext_texts": textExt,})

        self.prev_self_sem = conscious_out.self_sem.detach()
        self.prev_intent_sem = intent_sem.detach()
        self.prev_goal_bias = intent_sem.detach()

        world_hzx_now = torch.cat([
            w_out["h_next"].detach(),
            w_out["z_next"],
            w_out["x_next"].detach(),], dim=-1)

        if slow_refresh:
            goals = self.goal_manager(
                worldLatent=world_hzx_now,
                pstSummary=pst_summary,
                intentEmbed=intent_sem,)
            self.slow_cache = {
                "ocr_items": ocr_items,
                "fuse_ocr": fuse_ocr,
                "ocr_semantic": ocr_semantic.detach(),
                "self_sem": conscious_out.self_sem.detach(),
                "cons_intent_sem": conscious_out.intent_sem.detach(),
                "cons_extras": self.DetachRuntimeObject(conscious_out.extras),
                "intent_sem": intent_sem.detach(),
                "sym_probs": sym_probs.detach(),
                "intention_extras": self.DetachRuntimeObject(intention_extras),
                "g_ultimate": goals["g_ultimate"].detach(),
                "g_long": goals["g_long"].detach(),
                "g_mid": goals["g_mid"].detach(),}
        else:
            g_short = self.goal_manager.ShortGoal(
                self.slow_cache["g_ultimate"],
                self.slow_cache["g_long"],
                self.slow_cache["g_mid"],
                pst_summary)
            fused_goals = self.goal_manager.FuseGoals(
                self.slow_cache["g_ultimate"],
                self.slow_cache["g_long"],
                self.slow_cache["g_mid"],
                g_short)
            goals = {
                "g_ultimate": self.slow_cache["g_ultimate"],
                "g_long": self.slow_cache["g_long"],
                "g_mid": self.slow_cache["g_mid"],
                "g_short": g_short,}
            goals.update(fused_goals)
        grounding = self.goal_grounding(goals["goal_symbolic"], intent_sem, pst, observed_pst)
        cognition_goal_stage = self.RunCognitionGoalStage(
            conscious_out=conscious_out,
            intent_sem=intent_sem,
            sym_probs=sym_probs,
            intention_extras=intention_extras,
            intention_texts=intention_texts,
            world_hzx_now=world_hzx_now,
            goals=goals,
            grounding=grounding)

        saveModuleOutput("PhysicalState", {
            "PoseCamera": pst["PoseCamera"], "ARaw": pst["ARaw"], "SlotPresence": pst["SlotPresence"], "MphysRaw": pst["MphysRaw"], "PairwiseRelationCamera": pst["PairwiseRelationCamera"], "U": pst["U"],
            "LevelProb": pst["LevelProb"],
            "ObjectClassProb": pst["ObjectClassProb"],
            "PartClassProb": pst["PartClassProb"],
            "ParentProb": pst["ParentProb"],
            "Size": pst["Size"], "StateRaw": pst["StateRaw"], "AffordanceRaw": pst["AffordanceRaw"],
            "ExternalRelationProbRaw": pst["ExternalRelationProbRaw"],
            "MotionCameraRaw": pst["MotionCameraRaw"], "MovingProbRaw": pst["MovingProbRaw"],
            "ContactProbRaw": pst["ContactProbRaw"], "ContactForceRaw": pst["ContactForceRaw"],
            "Visibility": pst["Visibility"], "Occlusion": pst["Occlusion"],
            "HasTextProb": pst["HasTextProb"], "TextEmbed": pst["TextEmbed"],
            "SymbolProb": pst["SymbolProb"],
            "Observed": pst["Observed"], "LastSeen": pst["LastSeen"],
            "WorldRobotPhysicalEncoding": world_robot_physical_encoding,
            "pst_binding": w_out["pst_binding"], "summary": pst_summary,
            "slot_presence_mask": pst["SlotPresence"],
            "physical_entity_mask": active_physical_mask})
        saveModuleOutput("Goals", {
            "g_ultimate": goals["g_ultimate"],
            "g_long": goals["g_long"],
            "g_mid": goals["g_mid"],
            "g_short": goals["g_short"]})
        saveModuleOutput("GoalGrounding", {
            "referenced_object_probs": grounding["referenced_object_probs"],
            "referenced_slot_summary": grounding["referenced_slot_summary"],
            "reference_confidence": grounding["reference_confidence"],
            "no_slot_prob": grounding["no_slot_prob"],
            "reference_distribution": grounding["reference_distribution"]})

        self.RuntimeModule(self.actor).ClearInvalidEligibility(~credit_option_valid)
        aug_actor_kwargs = {
            "uncertainty": unc_sig,
            "confidence": confidence_sig,
            "precision": precision_sig,
            "risk": risk_sig,
            "worldHzx": world_hzx_now,
            "prevDecisionState": self.prev_decision_state,
            "prevLatentControl": self.prev_latent_control,
            "prevActionEmbed": decision_action_feedback,
            "prevMapperHidden": self.prev_mapper_hidden,
            # Fast plasticity may only reinforce an eligibility trace whose command
            # actually generated the transition that closes at this observation.
            "feedbackTdError": return_advantage_sig * credit_option_weight,}
        base_act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                  deterministic=deterministicActor,prevOptionLogit=self.prev_option_logit,
                                  valueTensor=value_current, vNextTensor=value_next_current,
                                  **aug_actor_kwargs)
        belief_prediction_delayed = self.RuntimeModule(self.actor).PredictBelief(
            belief_prediction_state_prev)
        base_act_out["prediction_error"] = (
            base_act_out["belief"] - belief_prediction_delayed)
        decision_uncertainty = base_act_out["decision_uncertainty"].detach()
        world_abstract = self.RuntimeModule(self.world).BuildWorldAbstract(
            w_out,
            pst,
            pst_summary,
            unc_sig,
            confidence_sig,)

        # --- Embodied-AGI v2: endpoint pose feature -> neuro-symbolic plan -> decoupled endpoint command ---
        satisfaction_prob = self.BuildExecutionSatisfaction(robotState, grounding)

        revalidated_active_command = None
        active_safety_risk = risk_sig
        if self.active_motion_command is not None:
            revalidated_active_command = self.RebaseWorldMotionCommand(
                self.active_motion_command,
                decision_endpoint_pose,
                risk_sig,
                confidence_sig,
                precision_sig)
            active_safety_risk = (
                1.0 - revalidated_active_command.safety_scores.amin(dim=-1))
        execution_safety_risk = torch.where(
            self.temporal_active_mask > 0.5,
            active_safety_risk,
            risk_sig)

        reference_distribution = grounding["reference_distribution"]
        reference_uncertainty = -(
            reference_distribution
            * reference_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(reference_distribution.size(-1))
        recent_failure = (self.prev_failure_count / 5.0).clamp(0.0, 1.0)
        temporal_context = self.temporal_gate.BuildContext(
            activeMask=self.temporal_active_mask,
            actionAgeSteps=self.temporal_action_age_steps,
            noSlotProb=grounding["no_slot_prob"].detach(),
            referenceConfidence=grounding["reference_confidence"].detach(),
            satisfactionProb=satisfaction_prob.detach(),
            safetyRisk=execution_safety_risk,
            candidateSafetyRisk=risk_sig,
            interruptRisk=torch.maximum(risk_sig, decision_uncertainty),
            observationFreshness=1.0 - grounding["no_slot_prob"].detach(),
            canInterrupt=torch.ones_like(risk_sig),
            hardStop=hard_stop,
            plannerProgress=robotState["planner_progress"],
            plannerTrackingError=robotState["planner_tracking_error"],
            plannerExecuting=robotState["planner_executing"],
            plannerReached=robotState["planner_reached"],
            plannerFailed=robotState["planner_failed"],)
        temporal_goal = self.goal_manager.TemporalGoal(goals["goal_temporal"], temporal_context.feat)
        goals["temporal_goal"] = temporal_goal
        belief_feat = base_act_out["belief"]
        neuro_symbolic_out = self.neuro_symbolic(
            pst=pst,
            observedPst=observed_pst,
            goalEmbed=goals["goal_symbolic"],
            worldBelief=world_hzx_now,
            decisionBelief=belief_feat,
            bodyProprioception=body_endpoint_pose_relative,
            robotPhysicalReference=robot_physical_reference,
            targetTrackingError=endpoint_tracking_error,
            plannerTrackingError=planner_endpoint_tracking_error,
            uncertainty=torch.maximum(unc_sig, decision_uncertainty),
            novelty=risk_sig,
            recentFailure=recent_failure,
            referenceUncertainty=reference_uncertainty,
            satisfactionProb=satisfaction_prob.detach(),
            referenced=grounding["referenced_object_probs"],
            referenceConfidence=grounding["reference_confidence"],
            noSlotProb=grounding["no_slot_prob"],
            temporalContextFeat=temporal_context.feat,
            returnExplain=self.save_module_messager_output,)
        act_out = self.actor.RefineWithNeuroSymbolic(
            base_act_out,
            neuro_symbolic_out,
            world_abstract["world_hzx"],
            world_abstract["pst_summary"],
            goals["goal_decision"],
            temporal_goal,
            decision_robot_state_encoding.body_pose.body_pose_feat,
            decision_robot_state_encoding.endpoint_control.control_feedback_feat,
            robot_physical_reference,)
        saveModuleOutput("Decision", act_out)

        decoupled_decision = self.decision_decoupler(
            decisionBackbone=act_out["decision_feature"],
            planLatent=act_out["decoder_plan_latent"],
            subgoalFeature=act_out["decoder_subgoal_feature"],
            constraintTokens=act_out["decoder_constraint_tokens"],
            robotStateEncoding=decision_robot_state_encoding,
            robotPhysicalReference=robot_physical_reference,
            risk=risk_sig,
            confidence=confidence_sig,
            precision=precision_sig,)

        # Kept before the planner override so the CEM elites can supervise the network heads.
        network_decision_tensor = decoupled_decision.decision_tensor
        planner_prior = None
        if self.planner is not None:
            with torch.no_grad():
                planner_h0, planner_z0, planner_x0 = self.PlannerInitialState(w_out)
                planner_prior = self.planner.Plan(
                    decisionLatent=decoupled_decision.decision_latent.detach(),
                    h0=planner_h0,
                    z0=planner_z0,
                    x0=planner_x0,
                    physicalState=pst,
                    worldRobotPhysicalEncoding=world_robot_physical_encoding,
                    returnDiagnostics=self.planner_teacher_mode,)
            if self.use_planner:
                planned_decision_latent = planner_prior["decision_latent"].detach()
                planned_decision_tensor = planner_prior["decision_tensor"].detach()
                planner_prior["decision_tensor"] = planned_decision_tensor
                decoupled_decision = self.decision_decoupler.ReplaceAction(
                    decoupled_decision,
                    planned_decision_latent,
                    planned_decision_tensor,
                    risk_sig,
                    confidence_sig,
                    precision_sig,)
        candidate_motion_command = self.MaterializeMotionCommand(
            decoupled_decision,
            decision_endpoint_pose)
        candidate_safety_risk = 1.0 - decoupled_decision.safety_scores.amin(dim=-1)
        execution_safety_risk = torch.where(
            self.temporal_active_mask > 0.5,
            active_safety_risk,
            candidate_safety_risk)
        gate_context = self.temporal_gate.BuildContext(
            activeMask=self.temporal_active_mask,
            actionAgeSteps=self.temporal_action_age_steps,
            noSlotProb=grounding["no_slot_prob"].detach(),
            referenceConfidence=grounding["reference_confidence"].detach(),
            satisfactionProb=satisfaction_prob.detach(),
            safetyRisk=execution_safety_risk,
            candidateSafetyRisk=candidate_safety_risk,
            interruptRisk=torch.maximum(act_out["temporal_decision"]["p_interrupt"], risk_sig),
            observationFreshness=1.0 - grounding["no_slot_prob"].detach(),
            canInterrupt=torch.ones_like(risk_sig),
            hardStop=hard_stop,
            plannerProgress=robotState["planner_progress"],
            plannerTrackingError=robotState["planner_tracking_error"],
            plannerExecuting=robotState["planner_executing"],
            plannerReached=robotState["planner_reached"],
            plannerFailed=robotState["planner_failed"],)
        active_motion_command = (
            revalidated_active_command
            if revalidated_active_command is not None
            else candidate_motion_command)
        # STAC-style drift integral over the active option's lifetime:
        # per-step invoke drift plus same-operator target-binding drift.
        invoke_drift = (
            self.temporal_invoke_drift
            + act_out["temporal_decision"]["invoke_delta"]
            + act_out["temporal_decision"]["reference_drift"])
        temporal_envelope = self.temporal_gate(
            gate_context,
            act_out["temporal_decision"],
            candidate_motion_command,
            active_motion_command,
            decision_endpoint_pose,
            self.temporal_action_epoch,
            invoke_drift,)
        motion_command = temporal_envelope.motion_command
        prospective_action_embed = self.decision_decoupler.EncodeEndpointAction(
            motion_command.decision_tensor)
        prospective_camera_motion = (
            self.decision_decoupler.CameraMotionFromDecisionTensor(
                motion_command.decision_tensor))
        prospective_visual_prediction = self.world.PredictNextVisualFromPosterior(
            w_out["h_next"],
            w_out["z_next"],
            w_out["x_next"],
            physicalState=pst,
            actionEnc=prospective_action_embed,
            robotPhysicalState=world_robot_physical_encoding,
            cameraMotion=prospective_camera_motion,
            sample=False,)["reconstructed_visual_state"]
        done_single = B == 1 and bool(done_now.item())
        if done_single:
            self.prospective_visual_prediction = None
        else:
            self.prospective_visual_prediction = self.DetachRuntimeObject(
                prospective_visual_prediction)

        value_memory_world_stage = self.RunValueMemoryWorldStage(
            w_preview=w_preview,
            w_out=w_out,
            s_t=s_t,
            r_t=r_t,
            d_t=d_t,
            d_tr=d_tr,
            d_ph=d_ph,
            done_now=done_now,
            critic_out=critic_out,
            value_current=value_current,
            value_next_current=value_next_current,
            td_sig=td_sig,
            unc_sig=unc_sig,
            precision_sig=precision_sig,
            emotion_sig=emotion_sig,
            risk_sig=risk_sig,
            confidence_sig=confidence_sig,
            atten_out=atten_out,
            mem_feat=mem_feat,
            memory_reward=memory_reward,
            intent_hint_for_memory=intent_hint_for_memory,
            prospective_visual_prediction=prospective_visual_prediction)
        kind_id = temporal_envelope.kind_id
        start_bool = (kind_id == DISPATCH) | (kind_id == REDISPATCH)
        continue_bool = kind_id == CONTINUE
        inactive_bool = ~(start_bool | continue_bool)
        start_mask = start_bool.float()
        continue_mask = continue_bool.float()
        self.temporal_action_epoch = temporal_envelope.action_epoch.detach()
        temporal_active_next = torch.maximum(start_mask, continue_mask * self.temporal_active_mask)
        self.temporal_action_age_steps = torch.where(
            continue_bool,
            self.temporal_action_age_steps + 1,
            torch.zeros_like(self.temporal_action_age_steps))
        self.temporal_invoke_drift = (invoke_drift * continue_mask).detach()
        self.temporal_active_mask = temporal_active_next
        self.temporal_active_kind = kind_id.detach()
        self.active_motion_command = self.DetachRuntimeObject(motion_command, clone=True)
        self.prev_target_endpoint_pose = motion_command.target_endpoint_pose.detach().clone()
        self.prev_target_endpoint_valid = ~done_now

        option_policy_input = act_out["option"]["policy_input"].detach()
        option_prior_logit = act_out["option"]["prior_logits"].detach()
        option_index = act_out["option"]["opt_idx"].detach()
        self.active_option_policy_input = (
            start_mask.unsqueeze(-1) * option_policy_input
            + continue_mask.unsqueeze(-1) * self.active_option_policy_input)
        self.active_option_prior_logit = (
            start_mask.unsqueeze(-1) * option_prior_logit
            + continue_mask.unsqueeze(-1) * self.active_option_prior_logit)
        self.active_option_goal_mid = (
            start_mask.unsqueeze(-1) * goals["g_mid"].detach()
            + continue_mask.unsqueeze(-1) * self.active_option_goal_mid)
        self.active_option_index = torch.where(
            start_bool,
            option_index,
            self.active_option_index)
        self.active_option_index = torch.where(
            inactive_bool,
            torch.zeros_like(self.active_option_index),
            self.active_option_index)
        self.active_option_valid = (
            start_bool
            | (continue_bool & self.active_option_valid))

        self.prev_belief_prediction_state = act_out[
            "decision_state_next"].detach()
        self.prev_belief_prediction_valid = ~done_now
        self.RuntimeModule(self.actor).CommitEligibility(
            act_out["eligibility"]["pre"],
            act_out["eligibility"]["post"],
            start_mask,)

        act_out["satisfaction_prob"] = satisfaction_prob
        act_out["neuro_symbolic"] = neuro_symbolic_out
        act_out["decoupled_decision"] = decoupled_decision
        act_out["candidate_motion_command"] = candidate_motion_command
        act_out["motion_command"] = motion_command
        act_out["candidate_option_index"] = option_index
        act_out["scheduled_option_index"] = self.active_option_index
        act_out["scheduled_option_valid"] = self.active_option_valid
        act_out["credited_option_index"] = credit_option_index
        act_out["credited_option_valid"] = credit_option_valid
        act_out["previous_model_command_executed"] = model_command_executed
        act_out["previous_executed_action_id"] = executed_action_id
        act_out["expected_previous_action_id"] = expected_action_id
        act_out["temporal_context"] = gate_context
        act_out["temporal_envelope"] = temporal_envelope
        act_out["goals"] = goals
        act_out["physical_state"] = pst
        act_out["world_abstract"] = world_abstract
        act_out["observed_physical_state"] = observed_pst
        act_out["goal_grounding"] = grounding
        act_out["planner_prior"] = planner_prior
        execution_stage = self.RunDecisionExecutionStage(
            act_out=act_out,
            motion_command=motion_command,
            candidate_motion_command=candidate_motion_command,
            temporal_envelope=temporal_envelope,
            temporal_context=gate_context,
            satisfaction_prob=satisfaction_prob,
            neuro_symbolic_out=neuro_symbolic_out,
            decoupled_decision=decoupled_decision,
            planner_prior=planner_prior,
            network_decision_tensor=network_decision_tensor,
            prospective_action_embed=prospective_action_embed,
            temporal_goal=temporal_goal,
            world_abstract=world_abstract,
            reference_uncertainty=reference_uncertainty)

        self.prev_failure_count = (
            self.prev_failure_count + robotState["planner_failed"].view(-1))
        self.prev_failure_count = (
            self.prev_failure_count * (1.0 - robotState["planner_reached"].view(-1)))

        saveModuleOutput("DecisionDecoupler", decoupled_decision)
        saveModuleOutput("TemporalExecution", temporal_envelope)
        saveModuleOutput("MotionCommand", motion_command)
        saveModuleOutput("NeuroSymbolic", neuro_symbolic_out)

        entropy_actor = act_out["entropy"] # [B]
        next_option_logit = act_out["prevOptionLogit_next"].detach() # [B, K_option]

        self.prev_option_logit = (
            start_mask.unsqueeze(-1) * next_option_logit
            + continue_mask.unsqueeze(-1) * self.prev_option_logit) # [B, K_option]

        self.prev_mem = mem_feat.detach().clone() # [B, D_mem]
        self.prev_attn = atten_out.detach().clone() # [B, D_attn]
        self.prev_world_h = w_out["h_next"].detach() # [B, D_world_h]
        self.prev_world_z = w_out["z_next"].detach() # [B, D_world_z] (PST-conditioned stochastic latent)
        self.prev_world_x = w_out["x_next"].detach() # [B, D_world_x]
        self.prev_entropy = entropy_actor.detach().clone() # [B]
        self.prev_decision_state = act_out["decision_state_next"].clone()
        self.prev_latent_control = act_out["latent_control_next"].clone()
        self.prev_mapper_hidden = act_out["mapper"]["hidden_next"].clone()
        self.prev_td_error = td_sig.detach().clone()
        self.prev_measured_endpoint_pose = endpoint_pose.detach().clone()
        self.prev_measured_endpoint_valid = ~done_now
        self.prev_visual_valid.logical_and_(~done_now)
        done_count = int(done_single) if B == 1 else int(done_now.sum().item())
        if done_count > 0:
            self.mem.ResetEpisodeState(done_now)
            self.RuntimeModule(self.critic).ResetState(doneMask=done_now)
            self.conscious.ResetState(doneMask=done_now)
            self.OCR.ResetTemporal(doneMask=done_now)
            self.RuntimeModule(self.world).ResetEpisodeState(done_now)
            world_keep = (~done_now).unsqueeze(-1)
            self.prev_world_h = self.prev_world_h * world_keep
            self.prev_world_z = self.prev_world_z * world_keep
            self.prev_world_x = self.prev_world_x * world_keep
            self.prev_world_s = self.prev_world_s * world_keep
            self.ResetDecisionRuntimeRows(done_now, allDone=(done_count == B))
            self.ResetHebbianMemory(doneMask=done_now)
            self.neuro_symbolic.ResetPlan(doneMask=done_now)
            self.slow_step_count = 0
            # These stores have no row/episode ownership metadata. Any episode boundary is a
            # batch-wide invalidation barrier; retaining them would mix a restarted row with
            # history or identity entries from its preceding episode.
            self.slow_cache = None
            self.history.clear()
            self.extra_mem = None
            self.smooth_generation += 1
            if self.perception_recall_loss is not None:
                self.perception_recall_loss.ResetIdentityBank()
            # Row-addressable visual state remains valid for unfinished batch rows.
            if done_count == B:
                self.prev_visual_state = None
                self.prospective_visual_prediction = None
                self.visual_state_buffer = []
                self.visual_state_valid_buffer = []
                self.perc_buffer = []
            else:
                self.prev_visual_state = self.ClearRuntimeRows(
                    self.prev_visual_state, done_now)
                self.prospective_visual_prediction = self.ClearRuntimeRows(
                    self.prospective_visual_prediction, done_now)
                self.visual_state_buffer = [
                    self.ClearRuntimeRows(state, done_now)
                    for state in self.visual_state_buffer]
                self.visual_state_valid_buffer = [
                    valid & ~done_now
                    for valid in self.visual_state_valid_buffer]
                self.perc_buffer = [
                    self.ClearRuntimeRows(value, done_now)
                    for value in self.perc_buffer]

        if self.need_trace and not isTrain:
            def trace_tensor(t: torch.Tensor) -> torch.Tensor:
                return t.detach() if isinstance(t, torch.Tensor) else t

            trace = BrainStepTrace(
                PercBuffer=copy.deepcopy(self.perc_buffer), # List[[B, D_perc]]
                VisualBuffer=[
                    self.DetachVisualState(v, clone=True)
                    for v in self.visual_state_buffer],
                VisualStateNow=self.DetachVisualState(visual_state, clone=True),
                OcrSemantic=trace_tensor(ocr_semantic),
                IntentHint=trace_tensor(intent_hint_for_memory),
                ObsImg=fuse_ocr, # List[List[str]]
                ActionEmbed=trace_tensor(decision_action_feedback),

                PercFeat=trace_tensor(perc_feats), # [B, D_perc]
                AttnFeat=trace_tensor(atten_out), # [B, D_attn]
                MemFeat=trace_tensor(mem_feat), # [B, D_mem]
                WorldState=trace_tensor(s_t), # [B, D_world]
                WorldDeltaTransport=trace_tensor(d_tr), # [B, D_world]
                WorldDeltaPhysics=trace_tensor(d_ph), # [B, D_world]
                ConsciousnessState=trace_tensor(conscious_out.intent_sem), # [B, D_cons]
                IntentionState=trace_tensor(intent_sem), # [B, D_intent]
                Reward=trace_tensor(memory_reward), # [B]
                Done=trace_tensor(d_t), # [B]
                ActionEntropy=trace_tensor(entropy_actor), # [B]

                extras= {},)
        
            self.history.append(trace)

        losses = {}

        if isTrain:
            world_loss = w_out["loss"]

            mem_loss = self.mem.GetInternalLoss()

            critic_loss = critic_out.loss if (critic_out.loss is not None) else world_loss.new_zeros(())
            critic_current_loss = critic_loss
            critic_transport_delayed_loss = world_loss.new_zeros(())
            if critic_out.extras is not None:
                critic_current_loss = critic_out.extras.get("loss_current_graph", critic_current_loss)
                critic_transport_delayed_loss = critic_out.extras.get(
                    "loss_transport_delayed_graph",
                    critic_transport_delayed_loss)

            conscious_loss = world_loss.new_zeros(())
            intention_loss = world_loss.new_zeros(())
            if slow_refresh:
                if conscious_out.extras is not None:
                    candidate_conscious_loss = conscious_out.extras.get("loss")
                    if candidate_conscious_loss is not None:
                        conscious_loss = candidate_conscious_loss
                intention_loss, _ = self.intention.GetInternalLoss(sym_probs)

            perception_loss = world_loss.new_zeros(())
            perception_recall_losses: Dict[str, torch.Tensor] = {}
            if perceptionTargets is not None:
                perception_loss = self.perc.ComputePerceptionLoss(
                    visual_state,
                    depthTarget=perceptionTargets["depth"],
                    depthTargetValid=perceptionTargets["depth_valid"],
                    prevVisualState=previous_visual_state,
                    prevVisualValid=previous_visual_valid,
                    cameraMotion=camera_motion_from_prev)
                if self.perception_recall_loss is not None and "node_valid" in perceptionTargets:
                    recall_out = self.RuntimeModule(self.perc).recall_heads(visual_state)
                    perception_recall_losses = self.perception_recall_loss(recall_out, perceptionTargets)
                    perception_loss = perception_loss + perception_recall_losses["loss"]

            world_prediction_loss = world_loss.new_zeros(())
            world_prediction_losses: Dict[str, torch.Tensor] = {}
            alive_prediction_mask = (
                ~prev_done_for_prediction
                & previous_visual_valid)
            world_prediction_losses = self.ComputeAliveWorldPredictionLoss(
                prevWorldH=prev_world_h_for_prediction,
                prevWorldZ=prev_world_z_for_prediction,
                prevWorldX=prev_world_x_for_prediction,
                physicalState=prev_physical_state_for_prediction,
                actionEnc=world_action_feedback,
                robotPhysicalState=prev_world_robot_physical_encoding_for_prediction,
                cameraMotion=camera_motion_from_prev,
                targetVisualState=visual_state,
                precision=precision_sig,
                aliveMask=alive_prediction_mask,)
            world_prediction_loss = world_prediction_losses.get("loss_pred_total", world_prediction_loss)

            actor_loss = world_loss.new_zeros(())
            world_hzx_prev = torch.cat([
                prev_world_h_for_prediction,
                prev_world_z_for_prediction,
                prev_world_x_for_prediction], dim=-1)
            alive_prev = 1.0 - prev_done_for_prediction.float()
            world_delta = (world_hzx_now.detach() - world_hzx_prev) * alive_prev.unsqueeze(-1)
            credited_goal_progress = self.goal_manager.ProjectedProgress(
                world_delta,
                credit_option_goal_mid)
            realized_information_gain = KLDiagNormal(
                w_out["mu_q"].detach(),
                w_out["logstd_q"].detach(),
                w_out["mu_p"].detach(),
                w_out["logstd_p"].detach()) / float(w_out["mu_q"].size(-1))
            advantage = (
                return_advantage_sig
                + 0.1 * credited_goal_progress
                + 0.05 * realized_information_gain).detach()
            if bool(credit_option_valid.any().item()):
                credited_logp = self.RuntimeModule(self.actor).OptionLogProb(
                    credit_option_policy_input,
                    credit_option_prior_logit,
                    credit_option_index)
                actor_loss = -(
                    advantage * credited_logp * credit_option_weight
                ).sum() / credit_option_weight.sum()
            goal_progress_weight = credit_option_weight * alive_prev
            if bool(goal_progress_weight.any().item()):
                goal_progress_loss = -(
                    credited_goal_progress * goal_progress_weight
                ).sum() / goal_progress_weight.sum()
            else:
                goal_progress_loss = world_loss.new_zeros(())

            decision_prediction_loss = world_loss.new_zeros(())
            if bool(belief_prediction_valid_prev.any().item()):
                prediction_weight = belief_prediction_valid_prev.float() * precision_sig.detach()
                prediction_per_row = nn.functional.smooth_l1_loss(
                    belief_prediction_delayed,
                    base_act_out["belief"].detach(),
                    reduction="none").mean(dim=-1)
                decision_prediction_loss = (
                    prediction_per_row * prediction_weight
                ).sum() / prediction_weight.sum()

            planner_distill_loss = world_loss.new_zeros(())
            decision_energy_loss = world_loss.new_zeros(())
            if planner_prior is not None and self.planner_teacher_mode:
                planner_distill_per_row = nn.functional.smooth_l1_loss(
                    FlattenActiveDecisionTensor(network_decision_tensor),
                    FlattenActiveDecisionTensor(
                        planner_prior["decision_tensor"].detach()),
                    reduction="none").mean(dim=-1)
                planner_std = planner_prior["diagnostics"]["std"]
                planner_convergence = torch.exp(
                    -planner_std.detach().mean(dim=-1))
                teacher_weight = (
                    (~done_now).float()
                    * precision_sig.detach()
                    * (1.0 - risk_sig.detach())
                    * planner_convergence)
                if bool((teacher_weight > 0.0).any().item()):
                    planner_distill_loss = (
                        planner_distill_per_row * teacher_weight
                    ).sum() / teacher_weight.sum()
                    energy_per_row = nn.functional.smooth_l1_loss(
                        act_out["decision_energy"],
                        -planner_prior["expected_return"].detach(),
                        reduction="none")
                    decision_energy_loss = (
                        energy_per_row * teacher_weight
                    ).sum() / teacher_weight.sum()

            # Embodied-AGI v2 auxiliary objectives. Goal/codebook terms only exist on
            # slow-refresh steps where the long/mid heads actually ran.
            codebook_util_loss = world_loss.new_zeros(())
            if slow_refresh:
                codebook_util_loss = (
                    self.goal_manager.ultimate_head.UtilizationLoss(goals["ultimate_logits"])
                    + self.goal_manager.long_head.UtilizationLoss(goals["long_logits"])
                    + self.goal_manager.mid_head.UtilizationLoss(goals["mid_logits"]))
            symbolic_invocation_target = self.RuntimeModule(
                self.neuro_symbolic
            ).InvocationNeedTarget(
                torch.maximum(unc_sig, decision_uncertainty),
                risk_sig,
                recent_failure,
                reference_uncertainty,
                satisfaction_prob.detach(),
                grounding["no_slot_prob"].detach(),)
            symbolic_invocation_loss = nn.functional.binary_cross_entropy(
                neuro_symbolic_out.invoke_mask,
                symbolic_invocation_target.detach())
            no_invocation_need = 1.0 - symbolic_invocation_target.detach()
            symbolic_sparsity_loss = (
                neuro_symbolic_out.invoke_mask * no_invocation_need
            ).sum() / no_invocation_need.sum().clamp_min(1.0)
            grounding_loss = grounding["grounding_consistency_loss"]
            temporal_kind_loss = world_loss.new_zeros(())
            if perceptionTargets is not None and "temporal_kind" in perceptionTargets:
                temporal_kind_loss = self.ComputeTemporalKindSupervisionLoss(
                    temporal_envelope.execution_kind_scores,
                    perceptionTargets["temporal_kind"],
                    perceptionTargets["temporal_kind_valid"],
                    gate_context.active_mask,
                    temporal_envelope.override_applied,)
            temporal_duration_loss = world_loss.new_zeros(())
            if perceptionTargets is not None and "temporal_duration_ms" in perceptionTargets:
                temporal_duration_valid = perceptionTargets[
                    "temporal_duration_valid"]
                if bool(temporal_duration_valid.any().item()):
                    temporal_duration_per_row = nn.functional.smooth_l1_loss(
                        temporal_envelope.duration_ms / 1000.0,
                        perceptionTargets["temporal_duration_ms"] / 1000.0,
                        reduction="none")
                    temporal_duration_weight = temporal_duration_valid.to(
                        temporal_duration_per_row.dtype)
                    temporal_duration_loss = (
                        temporal_duration_per_row * temporal_duration_weight
                    ).sum() / temporal_duration_weight.sum()
            v2_aux_loss = (
                0.05 * symbolic_invocation_loss
                + 0.01 * symbolic_sparsity_loss
                + 0.05 * grounding_loss
                + 0.05 * goal_progress_loss
                + 0.01 * codebook_util_loss
                + 0.5 * temporal_kind_loss
                + 0.05 * temporal_duration_loss)

            physical_loss = world_loss.new_zeros(())
            physical_losses: Dict[str, torch.Tensor] = {}
            if perceptionTargets is not None and "node_valid" in perceptionTargets:
                physical_losses = self.pst_loss(observed_pst, perceptionTargets)
                physical_loss = physical_losses["loss"]

            world_optimization_loss = (
                world_loss
                + 0.05 * world_prediction_loss)
            critic_optimization_loss = critic_current_loss
            policy_optimization_loss = (
                mem_loss
                + conscious_loss
                + intention_loss
                + 0.05 * perception_loss
                + 0.1 * actor_loss
                + planner_distill_loss
                + 0.05 * decision_prediction_loss
                + 0.05 * decision_energy_loss
                + 0.05 * v2_aux_loss
                + physical_loss)
            total_current_loss = (
                world_optimization_loss
                + critic_optimization_loss
                + policy_optimization_loss)
            total_loss = total_current_loss

            losses["symbolic_sparsity_loss"] = symbolic_sparsity_loss
            losses["symbolic_invocation_loss"] = symbolic_invocation_loss
            losses["grounding_loss"] = grounding_loss
            losses["goal_progress_loss"] = goal_progress_loss
            losses["codebook_util_loss"] = codebook_util_loss
            losses["planner_distill_loss"] = planner_distill_loss
            losses["decision_prediction_loss"] = decision_prediction_loss
            losses["decision_energy_loss"] = decision_energy_loss
            losses["temporal_kind_loss"] = temporal_kind_loss
            losses["temporal_duration_loss"] = temporal_duration_loss
            losses["v2_aux_loss"] = v2_aux_loss
            losses["physical_loss"] = physical_loss
            for name, value in perception_recall_losses.items():
                losses[f"perception_recall_{name}"] = value
            for name, value in physical_losses.items():
                if name != "loss":
                    losses[f"physical_{name}"] = value

            losses["world_loss"] = world_loss
            losses["memory_loss"] = mem_loss
            losses["critic_loss"] = critic_loss
            losses["critic_current_loss"] = critic_current_loss
            losses["critic_transport_delayed_loss"] = critic_transport_delayed_loss
            losses["conscious_loss"] = conscious_loss
            losses["intention_loss"] = intention_loss
            losses["perception_loss"] = perception_loss
            losses["world_prediction_loss"] = world_prediction_loss
            losses["actor_loss"] = actor_loss
            losses["world_optimization_loss"] = world_optimization_loss
            losses["critic_optimization_loss"] = critic_optimization_loss
            losses["policy_optimization_loss"] = policy_optimization_loss
            for name, value in world_prediction_losses.items():
                losses[f"world_{name}"] = value
            losses["total_current_loss"] = total_current_loss
            losses["total_loss"] = total_loss
            saveModuleOutput("Losses", losses)

        losses = self.BuildTrainingLosses(losses)
        return BrainStepOutput(
            decision=act_out,
            world={"state": s_t, "reward": r_t, "done": d_t}, # state:[B, D_world], reward/done:[B]
            critic=critic_out,
            features={
                "perc": percs_seq,
                "attn": atten_out,
                "mem": mem_feat,
                "visualState": visual_state,
                "precision": precision_sig,
                "topDown": top_down,
                "keyPaddingMask": key_padding_mask}, # perc:[B, T, D_perc], attn:[B, D_attn], mem:[B, D_mem]
            ocr=ocr_items,
            intention_texts=intention_texts,
            losses=losses,
            stages={
                "perception_physical": perception_physical_stage,
                "value_memory_world": value_memory_world_stage,
                "cognition_goal": cognition_goal_stage,
                "execution": execution_stage})
        

    def AdaptiveRuntimeModules(self) -> Dict[str, nn.Module]:
        modules = {
            "perception": self.RuntimeModule(self.perc),
            "actor": self.RuntimeModule(self.actor),
            "consciousness": self.RuntimeModule(self.conscious),
            "goal_manager": self.goal_manager,}
        if self.perception_recall_loss is not None:
            modules["perception_recall_loss"] = self.perception_recall_loss
        return modules

    @torch.no_grad()
    def ExportAdaptiveRuntimeBuffers(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            module_name: {
                buffer_name: value.detach().clone()
                for buffer_name, value in module.named_buffers()
                if buffer_name.rsplit(".", 1)[-1] != "camera_intrinsics"}
            for module_name, module in self.AdaptiveRuntimeModules().items()}

    @torch.no_grad()
    def ImportAdaptiveRuntimeBuffers(
        self,
        state: Dict[str, Dict[str, torch.Tensor]],) -> None:
        modules = self.AdaptiveRuntimeModules()
        if type(state) is not dict or set(state) != set(modules):
            raise ValueError(
                "adaptive runtime module fields do not match the current schema")
        for module_name, root in modules.items():
            saved_buffers = state[module_name]
            expected_buffers = {
                name
                for name, _ in root.named_buffers()
                if name.rsplit(".", 1)[-1] != "camera_intrinsics"}
            if type(saved_buffers) is not dict or set(saved_buffers) != expected_buffers:
                raise ValueError(
                    f"adaptive runtime buffers for {module_name!r} do not match "
                    "the current schema")
            named_modules = dict(root.named_modules())
            for buffer_name, saved in saved_buffers.items():
                parts = buffer_name.split(".")
                owner_name = ".".join(parts[:-1])
                local_name = parts[-1]
                owner = named_modules[owner_name]
                current = owner._buffers[local_name]
                if not isinstance(current, torch.Tensor) or not torch.is_tensor(saved):
                    raise TypeError(
                        f"adaptive runtime buffer {module_name}.{buffer_name} "
                        "must be a tensor")
                if saved.dtype != current.dtype:
                    raise ValueError(
                        f"adaptive runtime buffer {module_name}.{buffer_name} "
                        "dtype does not match")
                owner._buffers[local_name] = saved.detach().clone()

    @torch.no_grad()
    def ExportSlowRuntimeState(self) -> Dict[str, Any]:
        return {
            "prev_failure_count": self.prev_failure_count.detach().clone(),
            "slow_step_count": int(self.slow_step_count),
            "slow_cache": self.DetachRuntimeObject(self.slow_cache, clone=True),
            "thread_end": bool(self.thread_end),}

    @torch.no_grad()
    def ImportSlowRuntimeState(self, state: Dict[str, Any]) -> None:
        self.prev_failure_count = state["prev_failure_count"].detach().clone()
        self.slow_step_count = int(state["slow_step_count"])
        self.slow_cache = self.DetachRuntimeObject(
            state["slow_cache"], clone=True)
        self.thread_end = bool(state["thread_end"])

    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        def runtime_module(mod: nn.Module) -> nn.Module:
            return mod.base if hasattr(mod, "base") else mod

        world_mod = runtime_module(self.world)
        attn_mod = runtime_module(self.attn)
        critic_mod = runtime_module(self.critic)

        h, z, x = world_mod.ExportState()
        A_prev = getattr(world_mod, "_A_prev", None)

        return {
            "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,
            "prev_mem": self.prev_mem.detach().clone(),
            "prev_attn": self.prev_attn.detach().clone(),
            "prev_option_logit": self.prev_option_logit.detach().clone(),
            "prev_entropy": self.prev_entropy.detach().clone(),
            "prev_decision_state": self.prev_decision_state.detach().clone(),
            "prev_latent_control": self.prev_latent_control.detach().clone(),
            "prev_target_endpoint_pose": self.prev_target_endpoint_pose.detach().clone(),
            "prev_target_endpoint_valid": self.prev_target_endpoint_valid.detach().clone(),
            "prev_measured_endpoint_pose": self.prev_measured_endpoint_pose.detach().clone(),
            "prev_measured_endpoint_valid": self.prev_measured_endpoint_valid.detach().clone(),
            "active_option_policy_input": self.active_option_policy_input.detach().clone(),
            "active_option_prior_logit": self.active_option_prior_logit.detach().clone(),
            "active_option_goal_mid": self.active_option_goal_mid.detach().clone(),
            "active_option_index": self.active_option_index.detach().clone(),
            "active_option_valid": self.active_option_valid.detach().clone(),
            "prev_belief_prediction_state": self.prev_belief_prediction_state.detach().clone(),
            "prev_belief_prediction_valid": self.prev_belief_prediction_valid.detach().clone(),
            "temporal_active_mask": self.temporal_active_mask.detach().clone(),
            "temporal_action_age_steps": self.temporal_action_age_steps.detach().clone(),
            "temporal_action_epoch": self.temporal_action_epoch.detach().clone(),
            "temporal_invoke_drift": self.temporal_invoke_drift.detach().clone(),
            "temporal_active_kind": self.temporal_active_kind.detach().clone(),
            "active_motion_command": self.DetachRuntimeObject(self.active_motion_command, clone=True),
            "prev_mapper_hidden": self.prev_mapper_hidden.detach().clone(),
            "prev_td_error": self.prev_td_error.detach().clone(),
            "world_state": {
                "h": h.detach().clone(),
                "z": z.detach().clone(),
                "x": x.detach().clone(),
                "s": self.prev_world_s.detach().clone(),
                "done": self.prev_done_flag.detach().clone(),
                "A_prev": None if A_prev is None else A_prev.detach().clone(),},
            "world_robot_physical_encoding": world_mod.ExportRobotPhysicalState(),
            "mem_state": self.mem.ExportTransientState(),
            "mem_pending": copy.deepcopy(self.mem.pending),
            "attn_state": attn_mod.ExportState(),
            "critic_state": critic_mod.ExportState(),
            "perc_buffer": [t.detach().clone() for t in self.perc_buffer],
            "prev_visual_state": self.DetachVisualState(self.prev_visual_state, clone=True),
            "prev_visual_valid": self.prev_visual_valid.detach().clone(),
            "prospective_visual_prediction": self.DetachRuntimeObject(
                self.prospective_visual_prediction, clone=True),
            "prev_precision": self.prev_precision.detach().clone(),
            "prev_goal_bias": self.prev_goal_bias.detach().clone(),
            "prev_self_sem": None if self.prev_self_sem is None else self.prev_self_sem.detach().clone(),
            "prev_intent_sem": self.prev_intent_sem.detach().clone(),
            "visual_state_buffer": [
                self.DetachVisualState(v, clone=True)
                for v in self.visual_state_buffer
                if v is not None],
            "visual_state_valid_buffer": [
                valid.detach().clone()
                for valid in self.visual_state_valid_buffer],
            "ocr_state": {
                "temporal_step": int(self.OCR._temporal_step),
                "last_batch_size": int(self.OCR._last_batch_size),
                "last_ocr_texts_batch": copy.deepcopy(self.OCR._last_ocr_texts_batch),
                "tracks_by_bi": copy.deepcopy(self.OCR._tracks_by_bi),},
            "history": copy.deepcopy(list(self.history)),
            "extra_mem": None if self.extra_mem is None else copy.deepcopy(self.extra_mem),
            "neuro_symbolic_plan": self.neuro_symbolic.ExportPlanState(),
            "adaptive_runtime_buffers": self.ExportAdaptiveRuntimeBuffers(),
            **self.ExportSlowRuntimeState(),}

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        def runtime_module(mod: nn.Module) -> nn.Module:
            return mod.base if hasattr(mod, "base") else mod

        if type(state) is not dict or set(state) != BRAIN_RUNTIME_BUFFER_FIELDS:
            raise ValueError(
                "brain runtime buffer fields do not match the current schema")
        if (
            type(state["schema_version"]) is not int
            or state["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported brain runtime schema {state['schema_version']!r}; "
                f"expected {BRAIN_RUNTIME_SCHEMA_VERSION}")
        if (
            type(state["world_state"]) is not dict
            or set(state["world_state"]) != {"h", "z", "x", "s", "done", "A_prev"}
        ):
            raise ValueError("world runtime state fields do not match the current schema")
        if (
            type(state["ocr_state"]) is not dict
            or set(state["ocr_state"]) != {
                "temporal_step",
                "last_batch_size",
                "last_ocr_texts_batch",
                "tracks_by_bi",
            }
        ):
            raise ValueError("OCR runtime state fields do not match the current schema")

        device = next(self.parameters()).device
        state = copy.deepcopy(state)
        state = self.MoveRuntimeStateToModel(state)

        world_mod = runtime_module(self.world)
        attn_mod = runtime_module(self.attn)
        critic_mod = runtime_module(self.critic)

        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_option_logit = state["prev_option_logit"]
        if tuple(state["prev_entropy"].shape) != (int(state["prev_mem"].size(0)),):
            raise ValueError("prev_entropy must have shape [B]")
        self.prev_entropy = state["prev_entropy"]

        self.prev_decision_state = state["prev_decision_state"]
        self.prev_latent_control = state["prev_latent_control"]
        self.prev_target_endpoint_pose = state["prev_target_endpoint_pose"]
        self.prev_target_endpoint_valid = state["prev_target_endpoint_valid"]
        self.prev_measured_endpoint_pose = state["prev_measured_endpoint_pose"]
        self.prev_measured_endpoint_valid = state["prev_measured_endpoint_valid"]
        self.active_option_policy_input = state["active_option_policy_input"]
        self.active_option_prior_logit = state["active_option_prior_logit"]
        self.active_option_goal_mid = state["active_option_goal_mid"]
        self.active_option_index = state["active_option_index"]
        self.active_option_valid = state["active_option_valid"]
        self.prev_belief_prediction_state = state["prev_belief_prediction_state"]
        self.prev_belief_prediction_valid = state["prev_belief_prediction_valid"]
        self.temporal_active_mask = state["temporal_active_mask"]
        self.temporal_action_age_steps = state["temporal_action_age_steps"]
        self.temporal_action_epoch = state["temporal_action_epoch"]
        self.temporal_invoke_drift = state["temporal_invoke_drift"]
        self.temporal_active_kind = state["temporal_active_kind"]
        self.active_motion_command = state["active_motion_command"]
        self.prev_mapper_hidden = state["prev_mapper_hidden"]
        self.prev_td_error = state["prev_td_error"]

        world_state = state["world_state"]
        world_mod.ImportState(world_state["h"], world_state["z"], world_state["x"])
        world_mod._A_prev = None if world_state["A_prev"] is None else world_state["A_prev"].detach().clone()
        world_mod.ImportRobotPhysicalState(state["world_robot_physical_encoding"])
        self.prev_world_h = world_state["h"].detach().clone()
        self.prev_world_z = world_state["z"].detach().clone()
        self.prev_world_x = world_state["x"].detach().clone()
        self.prev_world_s = world_state["s"]
        self.prev_done_flag = world_state["done"]

        self.mem.EnsureB(int(self.prev_mem.size(0)))
        self.mem.ImportTransientState(state["mem_state"])
        self.mem.pending = state["mem_pending"]
        attn_mod.ImportState(state["attn_state"])
        critic_mod.ImportState(state["critic_state"])

        self.perc_buffer = state["perc_buffer"]
        self.prev_visual_state = state["prev_visual_state"]
        self.prev_visual_valid = state["prev_visual_valid"]
        self.prospective_visual_prediction = state[
            "prospective_visual_prediction"]
        self.prev_precision = state["prev_precision"]
        self.prev_goal_bias = state["prev_goal_bias"]
        self.prev_self_sem = state["prev_self_sem"]
        self.prev_intent_sem = state["prev_intent_sem"]
        self.visual_state_buffer = state["visual_state_buffer"]
        self.visual_state_valid_buffer = state["visual_state_valid_buffer"]

        ocr_state = state["ocr_state"]
        self.OCR._temporal_step = int(ocr_state["temporal_step"])
        self.OCR._last_batch_size = int(ocr_state["last_batch_size"])
        self.OCR._last_ocr_texts_batch = ocr_state["last_ocr_texts_batch"]
        self.OCR._tracks_by_bi = ocr_state["tracks_by_bi"]

        self.history = deque(state["history"], maxlen=self.history_len)
        self.extra_mem = state["extra_mem"]
        self.neuro_symbolic.ImportPlanState(state["neuro_symbolic_plan"])
        self.ImportAdaptiveRuntimeBuffers(state["adaptive_runtime_buffers"])
        self.ImportSlowRuntimeState(state)
        self.buf_B = int(self.prev_mem.size(0))


    def SmoothCorrection(
        self,
        wmSeq: torch.Tensor,
        extLast: torch.Tensor,
        *,
        q: float = 0.05,
        rWm: float = 0.5,
        rExt: float = 0.05,
        initVar: float = 1.0,) -> torch.Tensor:
        B, T = wmSeq.shape

        device = wmSeq.device
        dtype = wmSeq.dtype

        q_t = torch.as_tensor(float(q), device=device, dtype=dtype)
        r_wm_t = torch.as_tensor(float(rWm), device=device, dtype=dtype)
        r_ext_t = torch.as_tensor(float(rExt), device=device, dtype=dtype)

        x_filt = wmSeq.new_zeros(B, T)
        P_filt = wmSeq.new_zeros(B, T)

        x_filt[:, 0] = wmSeq[:, 0]
        P_filt[:, 0] = wmSeq.new_full((B,), float(initVar))

        for t in range(1, T):
            x_prior = x_filt[:, t - 1]
            P_prior = P_filt[:, t - 1] + q_t

            z_t = wmSeq[:, t]

            K_t = P_prior / (P_prior + r_wm_t + 1e-8)

            x_post = x_prior + K_t * (z_t - x_prior)
            P_post = (1.0 - K_t) * P_prior

            x_filt[:, t] = x_post
            P_filt[:, t] = P_post

        x_T = x_filt[:, -1]
        P_T = P_filt[:, -1]
        ext_vec = extLast

        K_ext = P_T / (P_T + r_ext_t + 1e-8)  
        x_T_corr = x_T + K_ext * (ext_vec - x_T) 
        P_T_corr = (1.0 - K_ext) * P_T 

        x_filt[:, -1] = x_T_corr
        P_filt[:, -1] = P_T_corr

        x_smooth = x_filt.clone()
        P_smooth = P_filt.clone()

        for t in range(T - 2, -1, -1):
            P_t = P_filt[:, t]
            P_tp = P_t + q_t
            C_t = P_t / (P_tp + 1e-8)

            x_smooth[:, t] = x_filt[:, t] + C_t * (x_smooth[:, t + 1] - x_filt[:, t])
            P_smooth[:, t] = P_t + C_t * C_t * (P_smooth[:, t + 1] - P_tp)

        return x_smooth  # [B,T]

    def SmoothWorkerLoop(self):
        while True:
            task = self.smooth_queue.get()
            self.SmoothWork(*task)

    def SmoothWork(
        self,
        historyRef,
        lastRef,
        signal: str,
        attenModule: nn.Module,
        memModule: nn.Module,
        criticModule: nn.Module,
        generation: int,):
        try:
            visual_buffer_list = []
            visual_state_list = []
            ocr_semantic_list = []
            intent_hint_list = []
            atten_list = []
            mem_list = []
            world_state_list = []
            world_dtr_list = []
            world_dph_list = []
            reward_list = []
            done_list = []
            entropy_list = []

            for tr in historyRef:
                visual_buffer_list.append(tr.VisualBuffer)
                visual_state_list.append(tr.VisualStateNow)
                ocr_semantic_list.append(tr.OcrSemantic)
                intent_hint_list.append(tr.IntentHint)
                atten_list.append(tr.AttnFeat)
                mem_list.append(tr.MemFeat)
                world_state_list.append(tr.WorldState)
                world_dtr_list.append(tr.WorldDeltaTransport)
                world_dph_list.append(tr.WorldDeltaPhysics)
                reward_list.append(tr.Reward) 
                done_list.append(tr.Done)  
                entropy_list.append(tr.ActionEntropy)

            ref_seq = next((t for t in reward_list + done_list if isinstance(t, torch.Tensor)), None)
            if ref_seq is None:
                return

            B = int(ref_seq.size(0))
            reward_list = [x.detach().view(B) for x in reward_list]
            done_list = [x.detach().view(B) for x in done_list]
            last_ref = lastRef.detach().view(B)

            if signal == "Reward":
                seq_list = reward_list
            elif signal == "Done":
                seq_list = done_list
            else:
                return

            if len(seq_list) <= 1:
                return

            wm_seq = torch.stack(seq_list, dim=1).contiguous()  # [B, T]

            smoothed = self.SmoothCorrection(wmSeq=wm_seq, extLast=last_ref)

            smoothed_list = list(smoothed.unbind(dim=1))

            start = int(memModule.time_step.min().item())

            with torch.no_grad():
                for i in range(1, len(smoothed_list)):
                    if signal == "Reward":
                        reward_in = smoothed_list[i]
                    else: # Done
                        reward_in = reward_list[i]

                    value = criticModule(
                        memoryPrev=mem_list[i-1],
                        attnPrev=atten_list[i-1],
                        state=world_state_list[i],
                        rewardModel=reward_list[i],
                        policyEntropyPrev=entropy_list[i-1],
                        doneModel=done_list[i],
                        worldDeltaTransport=world_dtr_list[i],
                        worldDeltaPhysics=world_dph_list[i],)

                    td_sig = value.tdError.detach()
                    unc_sig = value.uncertainty.detach()
                    precision_sig = value.precision.detach()
                    emotion_sig = value.emotion.detach()
                    value_comps = value.rComps
                    risk_sig = value_comps["risk"].detach()
                    confidence_sig = value_comps["confidence"].detach()

                    percs_seq, object_seq, motion_seq, quality_seq, pred_error_seq, key_padding_mask = self.BuildVisualSequenceTensors(
                        visual_buffer_list[i],
                        batchSize=reward_in.size(0),
                        device=reward_in.device,
                        dtype=reward_in.dtype)
                    atten_out = attenModule(
                        percs_seq,
                        keyPaddingMask=key_padding_mask,
                        tdError=td_sig,
                        uncertainty=unc_sig,
                        objectSeq=object_seq,
                        motionSeq=motion_seq,
                        qualitySeq=quality_seq,
                        predErrorSeq=pred_error_seq,
                        goalBias=intent_hint_list[i],
                        precision=precision_sig,
                        applyPlasticity=True)
                    memModule(
                        atten_out,
                        tdError=td_sig,
                        emotion=emotion_sig,
                        reward=reward_in,
                        visualState=visual_state_list[i],
                        ocrSemantic=ocr_semantic_list[i],
                        intentHint=intent_hint_list[i],
                        uncertainty=unc_sig,
                        risk=risk_sig,
                        confidence=confidence_sig,
                        sourceLabel=MemoryType.SRC_IMAGINE)

                memModule.FlushPendingWrites()

            extra_state = memModule.ExportState(step=start)
            extra_state["memory_delta_base_step"] = torch.tensor(start, device=last_ref.device, dtype=torch.long)
            extra_state["memory_delta_new_step"] = memModule.time_step.detach().max()
            extra_state["memory_delta_kind"] = torch.tensor(1 if signal == "Reward" else 2, device=last_ref.device, dtype=torch.long)
            if generation == self.smooth_generation:
                self.extra_mem = extra_state

        except Exception as e:
            print("[SmoothWork] error:", repr(e))
        finally:
            self.thread_end = True
            


class Agent:
    def __init__(self,
        brain: BrainCore,
        isTrain: bool,
        device: Union[str, torch.device] = "cpu",
        *,
        worldMemoryPath: str = None,
        memMemoryPath: str = None):

        self.device = torch.device(device)

        self.brain = brain
        self.is_train = isTrain

        self.wm_mem_path = worldMemoryPath
        self.mem_mem_path = memMemoryPath
        self.world_frame_id: Optional[str] = None
        self.world_memory_batch_size: Optional[int] = None

        self.brain.to(self.device)

        self.ResetHebbianMemory()

        if isTrain:
            actor_params = self.CollectTrainableParams(*self.ActorOptimizerModules())
        
            self.opt_actor = torch.optim.Adam(actor_params, lr=3e-4)

            critic_params = self.CollectTrainableParams(self.brain.critic)
            if not critic_params:
                # A rank-zero online critic has no trainable candidates yet. Adam requires a
                # non-empty initial list; the placeholder is frozen and removed on first sync.
                critic_params = [next(self.brain.critic.parameters())]
            self.opt_critic = torch.optim.Adam(critic_params, lr=2e-4)
            self.transport_manual_lr = 2e-4
            self.transport_manual_max_norm = 1.0
            self.transport_manual_weight_decay = 0.0

            world_params = self.CollectTrainableParams(*self.WorldOptimizerModules())
            self.opt_world = torch.optim.Adam(world_params, lr=2e-4)

    def ActorOptimizerModules(self) -> Tuple[nn.Module, ...]:
        return (
            self.brain.perc,
            self.brain.attn,
            self.brain.mem,
            self.brain.actor,
            self.brain.conscious,
            self.brain.intention,
            self.brain.pst_builder,
            self.brain.goal_manager,
            self.brain.goal_grounding,
            self.brain.decision_decoupler,
            self.brain.neuro_symbolic,
            self.brain.temporal_gate)

    def WorldOptimizerModules(self) -> Tuple[nn.Module, ...]:
        return (self.brain.world,)

    def CollectTrainableParams(self, *modules: nn.Module) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for mod in modules:
            for p in mod.parameters():
                if (not p.requires_grad) or (id(p) in seen):
                    continue
                seen.add(id(p))
                params.append(p)
            # BaseOnlineWrapper candidates intentionally live outside the registered module
            # tree so uncommitted adaptations are not serialized as durable model weights.
            # They still must be owned by the corresponding optimizer.
            if hasattr(mod, "CandParameters"):
                for p in mod.CandParameters():
                    if (not p.requires_grad) or (id(p) in seen):
                        continue
                    seen.add(id(p))
                    params.append(p)
        return params

    def SyncOptimizerParameters(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: List[nn.Parameter],) -> None:
        if len(optimizer.param_groups) != 1:
            raise RuntimeError("dynamic wrapper synchronization expects one optimizer parameter group")
        desired: List[nn.Parameter] = []
        desired_ids: set[int] = set()
        for parameter in parameters:
            if id(parameter) in desired_ids:
                continue
            desired_ids.add(id(parameter))
            desired.append(parameter)
        optimizer.param_groups[0]["params"] = desired
        for parameter in list(optimizer.state.keys()):
            if id(parameter) not in desired_ids:
                del optimizer.state[parameter]

    def SyncTrainableOptimizers(self) -> None:
        if not self.is_train:
            return
        self.SyncOptimizerParameters(
            self.opt_actor,
            self.CollectTrainableParams(*self.ActorOptimizerModules()))
        self.SyncOptimizerParameters(
            self.opt_critic,
            self.CollectTrainableParams(self.brain.critic))
        self.SyncOptimizerParameters(
            self.opt_world,
            self.CollectTrainableParams(*self.WorldOptimizerModules()))

    def ClearTrainableOptimizerState(self) -> int:
        """Discard moments that no longer correspond to newly loaded parameters."""
        if not self.is_train:
            return 0
        cleared = 0
        for optimizer in (self.opt_actor, self.opt_critic, self.opt_world):
            cleared += len(optimizer.state)
            optimizer.state.clear()
        return cleared

    def OptimizerParameters(self, optimizers=None) -> List[nn.Parameter]:
        if not self.is_train:
            return []
        if optimizers is None:
            optimizers = (self.opt_actor, self.opt_critic, self.opt_world)
        parameters: List[nn.Parameter] = []
        seen: set[int] = set()
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if (not parameter.requires_grad) or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters
    
    def EnsureFile(self, path: str) -> bool:
        dir_ = os.path.dirname(path)
        if dir_ and (not os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
            print(f"[EnsureFile] created directory: {os.path.abspath(dir_)}")

        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[EnsureFile] file already exists: {os.path.abspath(path)} (size={size} bytes)")
            return False

        return True

    def GetRuntimeWorld(self):
        return self.brain.world.base if self.brain.is_online_learning else self.brain.world

    def NamedOnlineWrappers(self) -> Tuple[Tuple[str, nn.Module], ...]:
        if not self.brain.is_online_learning:
            return ()
        return tuple(
            (name, getattr(self.brain, name))
            for name in ONLINE_WRAPPER_ROOTS
            if hasattr(getattr(self.brain, name), "ExportCandidateState"))

    @torch.no_grad()
    def ExportOnlineCandidateState(self) -> Dict[str, Any]:
        return {
            name: wrapper.ExportCandidateState()
            for name, wrapper in self.NamedOnlineWrappers()}

    @torch.no_grad()
    def ImportOnlineCandidateState(self, state: Dict[str, Any]) -> None:
        if type(state) is not dict:
            raise TypeError("online candidate checkpoint state must be a dictionary")
        wrappers = dict(self.NamedOnlineWrappers())
        if set(state) != set(wrappers):
            raise ValueError("online candidate checkpoint wrappers do not match the brain")
        for name, wrapper in wrappers.items():
            wrapper.ImportCandidateState(state[name])

    @torch.no_grad()
    def ResetOnlineCandidateState(self) -> None:
        for _, wrapper in self.NamedOnlineWrappers():
            wrapper.Update("rollback")
        self.SyncTrainableOptimizers()

    def BindWorldMemoryContext(
        self,
        worldFrameId: str,
        *,
        batchSize: int,
        loadPersistent: bool = True,) -> None:
        if type(worldFrameId) is not str or not worldFrameId:
            raise ValueError("worldFrameId must be a non-empty string")
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be a positive integer")
        if self.world_frame_id is not None:
            if (
                self.world_frame_id != worldFrameId
                or self.world_memory_batch_size != batchSize
            ):
                raise RuntimeError(
                    "Agent world memory context cannot change after binding")
            return

        world = self.GetRuntimeWorld()
        world.BindMemoryContext(self.brain.calibration_id, worldFrameId)
        self.world_frame_id = worldFrameId
        self.world_memory_batch_size = batchSize
        if self.wm_mem_path is not None:
            world._use_memory = True
            world._mem_path = self.wm_mem_path
            if loadPersistent:
                self.LoadWorldMemory(self.wm_mem_path)
            else:
                world.EnsureB(batchSize)
        if self.mem_mem_path is not None:
            if loadPersistent:
                self.LoadAgentMemory(
                    self.mem_mem_path,
                    batchSize=batchSize)
            else:
                self.brain.mem.EnsureB(batchSize)

    def LoadWorldMemory(self, path: str):
        if self.world_frame_id is None:
            raise RuntimeError(
                "bind the Agent world memory context before loading WorldMemory")
        world = self.GetRuntimeWorld()
        world._use_memory = True
        world._mem_path = path
        if os.path.exists(path):
            world.LoadMemory(
                path,
                batchSize=self.world_memory_batch_size,
                mapLocation=None)
        else:
            world.EnsureB(self.world_memory_batch_size)
            world.SaveMemory(path)

    def LoadAgentMemory(self, path: str, *, batchSize: int):
        mem = self.brain.mem
        if os.path.exists(path) and os.path.getsize(path) > 0:
            mem.LoadState(path, expectedBatch=batchSize)
            return

        self.EnsureFile(path)
        mem.EnsureB(batchSize)
        mem.SaveState(path)

    def SaveRuntimeMemories(self):
        if self.wm_mem_path is not None:
            if self.world_frame_id is None:
                raise RuntimeError(
                    "bind the Agent world memory context before saving WorldMemory")
            world = self.GetRuntimeWorld()
            world._use_memory = True
            world._mem_path = self.wm_mem_path
            world.SaveMemory(self.wm_mem_path)

        if self.mem_mem_path is not None:
            self.brain.mem.SaveState(self.mem_mem_path)

    def LoadTorchPayload(self, path: str):
        return torch.load(path, map_location=self.device, weights_only=True)

    def LoadBrainWeights(self, path: str):
        payload = self.LoadTorchPayload(path)
        expected_fields = {"schema_version", "calibration_id", "brain"}
        if type(payload) is not dict or set(payload) != expected_fields:
            raise TypeError(
                f"checkpoint {path} brain-weight fields do not match the current schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported brain parameter schema {payload['schema_version']!r}")
        if payload["calibration_id"] != self.brain.calibration_id:
            raise ValueError(
                "brain parameter calibration_id does not match configured K")
        brain_state = payload["brain"]
        if type(brain_state) is not dict:
            raise TypeError("brain model state must be a dictionary")

        if not self.brain.perc.recall_heads.enable_auxiliary:
            auxiliary_prefixes = (
                "perc.recall_heads.reconstruction_head.",
                "perc.recall_heads.patch_trunk.",
                "perc.recall_heads.patch_class_logits.",
                "perc.recall_heads.patch_depth.",
                "perc.recall_heads.patch_normal.")
            brain_state = {
                name: value
                for name, value in brain_state.items()
                if not name.startswith(auxiliary_prefixes)}
        LoadDeploymentModelState(self.brain, brain_state)
        self.ResetOnlineCandidateState()
        self.ClearTrainableOptimizerState()

    def ExportModuleMessagerData(self, nSteps: int = 0):
        return self.brain.moduleMessager.ExportDict(nSteps=nSteps)

    def Act(
        self,
        request: AgentActInput,) -> AgentActOutput:

        brain_step = BrainStepInput(
            frame=request.frame,
            text_ext=request.text_ext,
            reward_ext=request.reward,
            done_flag=request.done,
            is_train=self.is_train,
            sample_actions=request.sample_actions,
            deterministic_actor=request.deterministic_actor,
            depth=request.depth,
            depth_valid=request.depth_valid,
            perception_targets=request.perception_targets,
            robot_state=request.robot_state,
            text_trust=request.text_trust,
            compute_critic_loss=request.compute_critic_loss)
        if self.is_train:
            step_out = self.brain.Step(brain_step)
            motion_command = step_out.decision["motion_command"]
            return AgentActOutput(
                motion_command=motion_command,
                temporal_envelope=step_out.decision["temporal_envelope"],
                decision=step_out.decision,
                loss=step_out.losses["total_current_loss"],
                optimization_losses={
                    "world": step_out.losses["world_optimization_loss"],
                    "critic": step_out.losses["critic_optimization_loss"],
                    "policy": step_out.losses["policy_optimization_loss"],},
                transport_delayed_loss=step_out.losses["critic_transport_delayed_loss"],
                total_loss=step_out.losses["total_loss"],
                physical_loss=step_out.losses["physical_loss"],
                ocr=step_out.ocr,
                intention_texts=step_out.intention_texts)
        else:
            with torch.no_grad():
                step_out = self.brain.Step(brain_step)
                motion_command = step_out.decision["motion_command"]
                return AgentActOutput(
                    motion_command=motion_command,
                    temporal_envelope=step_out.decision["temporal_envelope"],
                    decision=step_out.decision,
                    loss=None,
                    ocr=step_out.ocr,
                    intention_texts=step_out.intention_texts)


    def UnpackActPacked(
        self,
        actOut: AgentActOutput,
        *,
        requestProvenance: Dict[str, Any],) -> str:
        if (
            type(requestProvenance) is not dict
            or set(requestProvenance) != set(DECISION_REQUEST_PROVENANCE_FIELDS)
        ):
            raise ValueError(
                "decision request provenance fields do not match the current schema")
        for field_name in (
            "stream_id",
            "frame_id",
            "calibration_id",
            "world_frame_id",
        ):
            if (
                type(requestProvenance[field_name]) is not str
                or not requestProvenance[field_name]
            ):
                raise ValueError(
                    f"decision request {field_name} must be a non-empty string")
        if (
            type(requestProvenance["sequence_index"]) is not int
            or requestProvenance["sequence_index"] < 0
        ):
            raise ValueError(
                "decision request sequence_index must be a non-negative integer")
        intention_texts: List[str] = []
        motion_command_value = actOut.motion_command
        intention_texts_value = actOut.intention_texts
        if isinstance(intention_texts_value, list):
            intention_texts = [str(item) for item in intention_texts_value]
        else:
            intention_texts = [str(intention_texts_value)]

        def to_json_scalar_or_list(value: Any) -> Any:
            if torch.is_tensor(value):
                tensor = value.detach().cpu()
                if tensor.dim() > 0:
                    tensor = tensor[0]
                if tensor.numel() == 1:
                    if tensor.dtype == torch.bool:
                        return bool(tensor.reshape(-1)[0].item())
                    if not tensor.dtype.is_floating_point:
                        return int(tensor.reshape(-1)[0].item())
                    return float(tensor.reshape(-1)[0].item())
                return tensor.tolist()
            if isinstance(value, dict):
                return {str(k): to_json_scalar_or_list(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_json_scalar_or_list(v) for v in value]
            return value

        motion_command = {
            "decision_tensor": to_json_scalar_or_list(motion_command_value.decision_tensor),
            "target_endpoint_pose": to_json_scalar_or_list(motion_command_value.target_endpoint_pose),
            "endpoint_names": list(motion_command_value.endpoint_names),
            "decision_dof_mask": to_json_scalar_or_list(
                motion_command_value.decision_dof_mask),
            "gripper_cmd": to_json_scalar_or_list(motion_command_value.gripper_cmd),
            "gripper_valid": to_json_scalar_or_list(motion_command_value.gripper_valid),
            "mode_logits": to_json_scalar_or_list(motion_command_value.mode_logits),
            "mode_valid": to_json_scalar_or_list(motion_command_value.mode_valid),
            "safety_scores": to_json_scalar_or_list(motion_command_value.safety_scores),
            "safety_names": list(motion_command_value.safety_names),}
        temporal_value = actOut.temporal_envelope
        kind_id_tensor = temporal_value.kind_id.detach().cpu()
        kind_id0 = int(kind_id_tensor.reshape(-1)[0].item())
        execution_scores = temporal_value.execution_kind_scores.detach().cpu()
        if execution_scores.dim() > 1:
            execution_scores = execution_scores[0]
        execution_scores_json = [
            float(score.item()) if bool(torch.isfinite(score).item()) else None
            for score in execution_scores]
        execution_legal_json = [
            bool(value.item())
            for value in torch.isfinite(execution_scores)]
        temporal_envelope = {
            "kind": temporal_value.kind_names[kind_id0],
            "kind_id": to_json_scalar_or_list(temporal_value.kind_id),
            "kind_logits": to_json_scalar_or_list(temporal_value.kind_logits),
            "execution_kind_scores": execution_scores_json,
            "execution_kind_legal": execution_legal_json,
            "primitive_names": list(temporal_value.kind_names),
            "override_applied": to_json_scalar_or_list(temporal_value.override_applied),
            "action_id": to_json_scalar_or_list(temporal_value.action_id),
            "action_epoch": to_json_scalar_or_list(temporal_value.action_epoch),
            "reason_scores": to_json_scalar_or_list(temporal_value.reason_scores),
            "reason_names": list(temporal_value.reason_names),
            "duration_ms": to_json_scalar_or_list(temporal_value.duration_ms),
            "soft_timeout_ms": to_json_scalar_or_list(temporal_value.soft_timeout_ms),
            "hard_timeout_ms": to_json_scalar_or_list(temporal_value.hard_timeout_ms),
            "action_age_steps": to_json_scalar_or_list(
                actOut.decision["temporal_context"].action_age_steps),
            "timeouts_apply_to_action_id": to_json_scalar_or_list(
                temporal_value.action_id),
            "latch_timeout_budget": to_json_scalar_or_list(
                temporal_value.publish_motion_command > 0.5),
            "publish_motion_command": to_json_scalar_or_list(
                temporal_value.publish_motion_command > 0.5),
            "reuse_active_motion_command": to_json_scalar_or_list(
                temporal_value.reuse_active_motion_command > 0.5),
            "publish_stop_command": to_json_scalar_or_list(
                temporal_value.publish_stop_command > 0.5),
            "publish_hold_command": to_json_scalar_or_list(
                temporal_value.publish_hold_command > 0.5),
            "same_operator": to_json_scalar_or_list(temporal_value.same_operator),
            "operator_changed": to_json_scalar_or_list(temporal_value.operator_changed),
            "invoke_delta": to_json_scalar_or_list(temporal_value.invoke_delta),
            "reference_drift": to_json_scalar_or_list(temporal_value.reference_drift),
            "invoke_drift": to_json_scalar_or_list(temporal_value.invoke_drift),}

        decision_value = actOut.decision
        option_assignment = {
            "candidate_option_index": to_json_scalar_or_list(
                decision_value["candidate_option_index"]),
            "scheduled_option_index": to_json_scalar_or_list(
                decision_value["scheduled_option_index"]),
            "scheduled_option_valid": to_json_scalar_or_list(
                decision_value["scheduled_option_valid"]),
            "credited_option_index": to_json_scalar_or_list(
                decision_value["credited_option_index"]),
            "credited_option_valid": to_json_scalar_or_list(
                decision_value["credited_option_valid"]),
            "previous_model_command_executed": to_json_scalar_or_list(
                decision_value["previous_model_command_executed"]),
            "previous_executed_action_id": to_json_scalar_or_list(
                decision_value["previous_executed_action_id"]),
            "expected_previous_action_id": to_json_scalar_or_list(
                decision_value["expected_previous_action_id"]),}

        command_contract = {
            "authority": "proposal",
            "physical_execution_ready": False,
            "repository_motion_bridge": "not_connected",
            "hardware_validation": "not_performed",
            "external_validation_required": True,
            "model_margin_semantics": "advisory",
            "model_margin_scope": (
                "endpoint_step_bounds_and_learned_current_state_risk"),
            "required_executor_checks": [
                "supported_endpoint_and_dof",
                "frame_transform",
                "inverse_kinematics",
                "joint_and_actuator_limits",
                "self_collision",
                "environment_collision",
                "atomic_command_preflight",
            ],
            "decision_tensor_frame": "endpoint_local_body",
            "decision_translation_unit": "meter",
            "decision_rotation": "axis_angle_radian",
            "target_pose_frame": "world",
            "target_pose_unit": "meter",
            "target_quaternion_order": "xyzw",
            "timeout_unit": "ms",
            "timeout_anchor": "dispatch",
            "timeout_clock": "executor_monotonic",
            "continue_renews_timeout": False,
            "soft_timeout_effect": "request_redispatch",
            "hard_timeout_effect": "fail_closed_software_stop",
            "timeout_enforcement": "external_executor_required",
            "failsafe_stop_semantics": "software_stop_request_not_certified_estop",
        }

        return json.dumps({
            "schema_version": DECISION_WIRE_SCHEMA_VERSION,
            **requestProvenance,
            "command_contract": command_contract,
            "motion_command": motion_command,
            "temporal_envelope": temporal_envelope,
            "option_assignment": option_assignment,
            "intention_texts": intention_texts,}, ensure_ascii=False, allow_nan=False)


    def Save(self, path: str):
        if self.world_frame_id is None:
            raise RuntimeError(
                "bind the Agent world memory context before saving a runtime checkpoint")
        if self.world_memory_batch_size is None:
            raise RuntimeError("Agent runtime batch size is not bound")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,
            "calibration_id": self.brain.calibration_id,
            "world_frame_id": self.world_frame_id,
            "batch_size": self.world_memory_batch_size,
            "online_learning": bool(self.brain.is_online_learning),
            "brain": ExportBrainModelState(self.brain),
            "online_candidates": self.ExportOnlineCandidateState(),
            "buffers": self.brain.ExportBuffers(),
            "world_memory": self.GetRuntimeWorld().ExportMemoryPayload(),
            "memory_durable": self.brain.mem.ExportDurableState(),
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None),}
        
        if self.is_train:
            payload["opt_actor"] = self.opt_actor.state_dict()
            payload["opt_critic"] = self.opt_critic.state_dict()
            payload["opt_world"] = self.opt_world.state_dict()

        directory = os.path.dirname(path) or "."
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory)
        os.close(fd)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        self.SaveRuntimeMemories()

    def Load(
        self,
        path: str,
        mapLocation: Optional[Union[str, torch.device]] = None,) -> None:
        payload = torch.load(path, map_location=mapLocation or self.device, weights_only=False)

        expected_fields = set(AGENT_RUNTIME_CHECKPOINT_FIELDS)
        if self.is_train:
            expected_fields.update(AGENT_TRAINING_CHECKPOINT_FIELDS)
        if type(payload) is not dict or set(payload) != expected_fields:
            raise TypeError("agent checkpoint fields do not match the current schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported agent checkpoint schema {payload['schema_version']!r}")
        if payload["calibration_id"] != self.brain.calibration_id:
            raise ValueError(
                "agent checkpoint calibration_id does not match configured K")
        if self.world_frame_id is None:
            raise RuntimeError(
                "bind the Agent world memory context before loading a runtime checkpoint")
        if payload["world_frame_id"] != self.world_frame_id:
            raise ValueError(
                "agent checkpoint world_frame_id does not match the active world")
        if (
            type(payload["batch_size"]) is not int
            or payload["batch_size"] != self.world_memory_batch_size
        ):
            raise ValueError("agent checkpoint batch_size does not match the runtime")
        if (
            type(payload["online_learning"]) is not bool
            or payload["online_learning"] != bool(self.brain.is_online_learning)
        ):
            raise ValueError("agent checkpoint online_learning mode does not match the brain")
        brain_state = payload["brain"]
        if type(brain_state) is not dict:
            raise TypeError("brain model state must be a dictionary")
        world = self.GetRuntimeWorld()
        world_batch_size, _ = world._ValidateMemoryPayload(payload["world_memory"])
        if world_batch_size != self.world_memory_batch_size:
            raise ValueError("agent checkpoint World memory batch size is invalid")
        self.brain.mem.ValidateDurableState(
            payload["memory_durable"],
            expectedBatch=self.world_memory_batch_size)
        LoadBrainModelState(self.brain, brain_state)
        self.ImportOnlineCandidateState(payload["online_candidates"])
        self.SyncTrainableOptimizers()
        self.ClearTrainableOptimizerState()

        if self.is_train:
            self.opt_actor.load_state_dict(payload["opt_actor"])
            self.opt_critic.load_state_dict(payload["opt_critic"])
            self.opt_world.load_state_dict(payload["opt_world"])

        world.ImportMemoryPayload(
            payload["world_memory"],
            batchSize=self.world_memory_batch_size)
        self.brain.mem.ImportDurableState(payload["memory_durable"])
        self.brain.ImportBuffers(payload["buffers"])

        random.setstate(payload["rng_py"])
        np.random.set_state(payload["rng_np"])
        torch.set_rng_state(payload["rng_torch"].cpu())
        if torch.cuda.is_available() and payload["rng_cuda_all"] is not None:
            torch.cuda.set_rng_state_all(payload["rng_cuda_all"])

        self.SaveRuntimeMemories()


    def ResetBrainState(self, B: int = 1, isOnlineLearning: Optional[bool] = None):
        if isOnlineLearning is None:
            isOnlineLearning = self.brain.is_online_learning

        if isOnlineLearning:
            self.brain.world.base.ResetState(batchSize=B)
            self.brain.world.base.ResetPhysicalState()
            self.brain.critic.base.ResetState()
        else:
            self.brain.world.ResetState(batchSize=B)
            self.brain.world.ResetPhysicalState()
            self.brain.critic.ResetState()

        self.brain.mem.ResetEpisodeState()
        self.brain.conscious.ResetState()
        self.brain.RuntimeModule(
            self.brain.intention
        ).ResetTransientLossCache()
        self.brain.OCR.ResetTemporal()
        self.brain.ResetHebbianMemory()

        self.brain.extra_mem = None
        self.brain.thread_end = True
        self.brain.smooth_generation += 1

        self.brain.ResetBuffers(B=B, isOnlineLearning=isOnlineLearning, device=self.device)

    def ResetHebbianMemory(self):
        self.brain.ResetHebbianMemory()

    def CaptureCriticTransportGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "CaptureTransportGrad"):
            return critic.CaptureTransportGrad(clearParamGrad=True)
        return {"captured": 0.0, "grad_norm": 0.0, "accum_steps": 0.0}

    def ApplyCriticTransportManualGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "ApplyTransportManualGrad"):
            return critic.ApplyTransportManualGrad(
                lr=self.transport_manual_lr,
                maxNorm=self.transport_manual_max_norm,
                weightDecay=self.transport_manual_weight_decay,
                clear=True)
        return {"updated": 0.0, "grad_norm": 0.0, "scale": 1.0}

    def ClearCriticTransportGradAccumulator(self) -> None:
        critic = self.brain.critic
        if hasattr(critic, "ClearTransportGradAccumulator"):
            critic.ClearTransportGradAccumulator()

    def AfterOptimizerStep(self):
        if hasattr(self.brain.critic, "AfterOptimizerStep"):
            self.brain.critic.AfterOptimizerStep()





    def UpdateWrappers(self, wrappers, action: str, **kwargs):
        results = []
        for w in wrappers:
            if not hasattr(w, "Update"):
                continue
            out = w.Update(action, **kwargs)
            results.append(out)
        self.SyncTrainableOptimizers()
        return results
    
    def UpdateAllWrappers(self, action: str, **kwargs):
        wrappers = [self.brain.perc, self.brain.attn, self.brain.actor, self.brain.world, self.brain.critic, self.brain.intention]
        results = []
        for w in wrappers:
            if not hasattr(w, "Update"):
                continue
            out = w.Update(action, **kwargs)
            results.append(out)
        self.SyncTrainableOptimizers()
        return results

    def GetModuleParamsCount(self, onlyTrainable: bool = True):
        def count_module_params(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0

            total = 0
            for p in module.parameters():
                if onlyTrainable and not p.requires_grad:
                    continue
                total += p.numel()
            return total

        core_module_names = (
            "perc",
            "attn",
            "mem",
            "actor",
            "world",
            "critic",
            "conscious",
            "intention",
            "OCR",)

        per_module_counts: Dict[str, int] = {}
        for name in core_module_names:
            module = getattr(self.brain, name, None)
            if isinstance(module, nn.Module):
                per_module_counts[name] = count_module_params(module)

        aux_module_counts: Dict[str, int] = {}
        for child_name, child in self.brain.named_children():
            if child_name in per_module_counts:
                continue
            aux_module_counts[child_name] = count_module_params(child)

        core_total = sum(per_module_counts.values())
        aux_total = sum(aux_module_counts.values())
        full_total = count_module_params(self.brain)

        kind = "trainable" if onlyTrainable else "all"
        print(f"===== Core module parameter counts ({kind}) =====")
        for name, n in per_module_counts.items():
            print(f"{name:15s}: {n:,}")
        print("----------------------------------")
        print(f"TOTAL core {kind} params: {core_total:,}")

        if aux_module_counts:
            print("===== Auxiliary module parameter counts =====")
            for name, n in aux_module_counts.items():
                print(f"{name:15s}: {n:,}")
            print("----------------------------------")
            print(f"TOTAL auxiliary {kind} params: {aux_total:,}")

        if full_total != (core_total + aux_total):
            print(f"TOTAL full {kind} params: {full_total:,}")

        return core_total


class TestAGICoreMTool:
    @staticmethod
    def MakeRobotPhysicalEncodingWorld(
        bodyPoseDim: int = (
            ModuleDim.DecisionBodyEndpointCount
            * ModuleDim.DecisionEndpointPoseDim),
        outDim: int = ModuleDim.PstSlotDim,) -> RSSMWorldModel:
        from WorldModule import WorldRobotPhysicalEncoder

        world = object.__new__(RSSMWorldModel)
        nn.Module.__init__(world)
        world.robot_physical_encoder = WorldRobotPhysicalEncoder(
            bodyPoseDim=bodyPoseDim,
            outDim=outDim)
        return world

    def TestDataclassContracts(self) -> bool:
        try:
            step_sig = inspect.signature(BrainCore.Step)
            act_sig = inspect.signature(Agent.Act)
            assert list(step_sig.parameters.keys()) == ["self", "step"]
            assert list(act_sig.parameters.keys()) == ["self", "request"]
            brain_fields = set(BrainStepInput.__dataclass_fields__)
            agent_fields = set(AgentActInput.__dataclass_fields__)
            assert not any(name.startswith("robot_") and name.endswith("_context") for name in brain_fields)
            assert ("interaction" + "_context") not in brain_fields
            assert not any(name.startswith("robot_") and name.endswith("_context") for name in agent_fields)
            assert ("interaction" + "_context") not in agent_fields

            sample = torch.zeros(1, 1)
            out = BrainStepOutput(
                decision={},
                world={"state": sample, "reward": sample.view(1), "done": sample.view(1)},
                critic=None,
                features={},
                ocr=[],
                intention_texts=[],
                losses={},
                stages={})
            assert isinstance(out.stages, dict)
            print("AGICore dataclass contracts passed.")
            return True
        except AssertionError as e:
            print(f"AGICore dataclass contracts failed: {e}")
            return False
        except Exception as e:
            print(f"AGICore dataclass contracts error: {e}")
            return False

    def TestWorldRobotPhysicalEncodingExcludesControlFeedback(self) -> bool:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            control_encoder = EndpointControlEncoder().to(device).eval()
            world = self.MakeRobotPhysicalEncodingWorld().to(device).eval()
            endpoint_pose = torch.zeros(
                2,
                ModuleDim.DecisionBodyEndpointCount,
                ModuleDim.DecisionEndpointPoseDim,
                device=device)
            endpoint_pose[..., 6] = 1.0
            physical_reference = torch.zeros(
                2, ModuleDim.RobotPhysicalReferenceDim, device=device)
            physical_reference[:, 3] = 1.0
            physical_reference[:, 6] = -1.0
            state0 = world.EncodeRobotPhysicalState(
                endpoint_pose,
                physical_reference)
            zero_error = torch.zeros(
                2,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionActionDim,
                device=device)
            control0 = control_encoder(zero_error, zero_error)
            changed_error = zero_error.clone()
            changed_error[:, 0, 0] = 0.25
            control1 = control_encoder(changed_error, zero_error)
            state_from_changed_control = world.EncodeRobotPhysicalState(
                endpoint_pose,
                physical_reference)
            changed_pose = endpoint_pose.clone()
            changed_pose[:, 0, 0] = 0.25
            state1 = world.EncodeRobotPhysicalState(
                changed_pose,
                physical_reference)
            changed_reference = physical_reference.clone()
            changed_reference[:, 0] = 0.5
            state2 = world.EncodeRobotPhysicalState(
                endpoint_pose,
                changed_reference)
            rotated_reference = physical_reference.clone()
            sqrt_half = math.sqrt(0.5)
            rotated_reference[:, 0:4] = rotated_reference.new_tensor([
                0.0, 0.0, sqrt_half, sqrt_half])
            state_rotation = world.EncodeRobotPhysicalState(
                endpoint_pose,
                rotated_reference)
            changed_gravity = physical_reference.clone()
            changed_gravity[:, 4:7] = changed_gravity.new_tensor([
                0.0, -1.0, 0.0])
            state_gravity = world.EncodeRobotPhysicalState(
                endpoint_pose,
                changed_gravity)
            ok = (
                tuple(state0.shape) == (2, ModuleDim.PstSlotDim)
                and torch.equal(state0, state_from_changed_control)
                and not torch.allclose(
                    control0.control_feedback_feat,
                    control1.control_feedback_feat,
                    atol=1e-6)
                and tuple(state1.shape) == (2, ModuleDim.PstSlotDim)
                and not torch.allclose(state0, state1, atol=1e-6)
                and not torch.allclose(state0, state2, atol=1e-6)
                and not torch.allclose(state0, state_rotation, atol=1e-6)
                and not torch.allclose(state0, state_gravity, atol=1e-6))
            print(f"AGICore world robot physical encoding {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore world robot physical encoding error: {e}")
            return False

    def TestCameraPhysicalReferenceGaugeInvariance(self) -> bool:
        try:
            sqrt_half = math.sqrt(0.5)
            camera_world = torch.tensor([[
                0.3, -0.2, 1.1,
                0.0, 0.0, sqrt_half, sqrt_half]])
            base_world = torch.tensor([[0.1, 0.4, 0.2, 0.0, sqrt_half, 0.0, sqrt_half]])
            gravity_world = torch.tensor([[0.0, 0.0, -1.0]])
            world_gauge = torch.tensor([[1.2, -0.7, 0.5, sqrt_half, 0.0, 0.0, sqrt_half]])

            def compose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
                return NormalizePose(torch.cat([
                    parent[..., :3] + QuatRotate(
                        parent[..., 3:7], child[..., :3]),
                    QuatMultiply(
                        parent[..., 3:7], child[..., 3:7])], dim=-1))

            reference = BrainCore.CameraPhysicalReference(
                camera_world,
                base_world[..., 3:7],
                gravity_world)
            transformed_reference = BrainCore.CameraPhysicalReference(
                compose(world_gauge, camera_world),
                compose(world_gauge, base_world)[..., 3:7],
                QuatRotate(world_gauge[..., 3:7], gravity_world))

            camera_sign_flipped = camera_world.clone()
            camera_sign_flipped[..., 3:7] *= -1.0
            camera_sign_reference = BrainCore.CameraPhysicalReference(
                camera_sign_flipped,
                base_world[..., 3:7],
                gravity_world)
            base_sign_reference = BrainCore.CameraPhysicalReference(
                camera_world,
                -base_world[..., 3:7],
                gravity_world)
            translated_camera = camera_world.clone()
            translated_camera[..., :3] += torch.tensor([3.0, -2.0, 1.0])
            translated_reference = BrainCore.CameraPhysicalReference(
                translated_camera,
                base_world[..., 3:7],
                gravity_world)
            reconstructed_base_orientation = QuatMultiply(
                camera_world[..., 3:7],
                reference[..., :4])
            reconstructed_gravity_world = QuatRotate(
                camera_world[..., 3:7],
                reference[..., 4:7])
            reconstructed_base_quaternion_match = (
                reconstructed_base_orientation
                * base_world[..., 3:7]
            ).sum(dim=-1).abs()

            endpoint_world = torch.tensor([[0.6, 0.1, 0.8, 0.0, 0.0, 0.0, 1.0]])
            local_delta = torch.tensor([[0.02, -0.01, 0.03, 0.05, -0.02, 0.01]])
            target_world = ApplyPoseDelta(endpoint_world, local_delta)
            transformed_target = ApplyPoseDelta(
                compose(world_gauge, endpoint_world),
                local_delta)
            expected_transformed_target = compose(world_gauge, target_world)

            endpoint_system = torch.zeros(
                1,
                ModuleDim.RobotStateEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_system[..., 6] = 1.0
            endpoint_system[..., 0] = torch.linspace(
                -0.3, 0.3, ModuleDim.RobotStateEndpointCount)
            endpoint_system[:, ModuleDim.RobotStateCameraEndpointIndex] = camera_world
            transformed_endpoint_system = compose(
                world_gauge.unsqueeze(1),
                endpoint_system)
            body_relative = BrainCore.BodyEndpointPoseRelative(endpoint_system)
            transformed_body_relative = BrainCore.BodyEndpointPoseRelative(
                transformed_endpoint_system)
            world = self.MakeRobotPhysicalEncodingWorld().eval()
            physical_state = world.EncodeRobotPhysicalState(
                body_relative,
                reference)
            transformed_physical_state = world.EncodeRobotPhysicalState(
                transformed_body_relative,
                transformed_reference)

            ok = (
                torch.allclose(
                    reference,
                    transformed_reference,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    reference,
                    camera_sign_reference,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    reference,
                    base_sign_reference,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.equal(reference, translated_reference)
                and tuple(reference.shape) == (
                    1, ModuleDim.RobotPhysicalReferenceDim)
                and torch.allclose(
                    reconstructed_base_quaternion_match,
                    torch.ones_like(reconstructed_base_quaternion_match),
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    reconstructed_gravity_world,
                    gravity_world,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    transformed_target,
                    expected_transformed_target,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    body_relative,
                    transformed_body_relative,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    physical_state,
                    transformed_physical_state,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    reference[..., 4:7].norm(dim=-1),
                    torch.ones(1),
                    atol=1e-5,
                    rtol=1e-5))
            print(
                f"AGICore camera physical reference gauge invariance "
                f"{'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(
                f"AGICore camera physical reference gauge invariance error: {e}")
            return False

    def TestBodyEndpointReferenceMapping(self) -> bool:
        try:
            sqrt_half = math.sqrt(0.5)

            def compose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
                return NormalizePose(torch.cat([
                    QuatRotate(parent[..., 3:7], child[..., :3]) + parent[..., :3],
                    QuatMultiply(parent[..., 3:7], child[..., 3:7]),
                ], dim=-1))

            endpoint_pose = torch.zeros(
                1,
                ModuleDim.RobotStateEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            camera = torch.tensor([[
                0.7, -0.4, 1.2,
                0.0, sqrt_half, 0.0, sqrt_half]])
            left_wrist_camera = torch.tensor([[
                0.25, -0.10, 0.35,
                0.0, 0.0, sqrt_half, sqrt_half]])
            right_wrist_camera = torch.tensor([[
                -0.20, 0.15, 0.40,
                sqrt_half, 0.0, 0.0, sqrt_half]])
            left_wrist = compose(camera, left_wrist_camera)
            right_wrist = compose(camera, right_wrist_camera)
            endpoint_pose[:, ModuleDim.DecisionLeftWristEndpointIndex] = left_wrist
            endpoint_pose[:, ModuleDim.DecisionRightWristEndpointIndex] = right_wrist
            endpoint_pose[:, ModuleDim.RobotStateCameraEndpointIndex] = camera

            left_local = torch.zeros(1, 5, 7)
            left_local[..., 6] = 1.0
            left_local[0, :, :3] = torch.tensor([
                [0.10, 0.01, 0.02],
                [0.20, 0.02, 0.01],
                [0.30, 0.03, 0.00],
                [0.40, 0.04, -0.01],
                [0.50, 0.05, -0.02]])
            left_local[..., 3:7] = torch.tensor([
                0.0, sqrt_half, 0.0, sqrt_half])
            right_local = torch.zeros(1, 5, 7)
            right_local[..., 6] = 1.0
            right_local[0, :, :3] = torch.tensor([
                [0.01, 0.10, -0.02],
                [0.02, 0.20, -0.01],
                [0.03, 0.30, 0.00],
                [0.04, 0.40, 0.01],
                [0.05, 0.50, 0.02]])
            right_local[..., 3:7] = torch.tensor([
                0.0, 0.0, sqrt_half, sqrt_half])
            endpoint_pose[:, :5] = compose(
                left_wrist.unsqueeze(1), left_local)
            endpoint_pose[:, 5:10] = compose(
                right_wrist.unsqueeze(1), right_local)

            body_relative = BrainCore.BodyEndpointPoseRelative(endpoint_pose)
            expected = torch.cat([
                left_local,
                right_local,
                left_wrist_camera.unsqueeze(1),
                right_wrist_camera.unsqueeze(1),
            ], dim=1)
            quaternion_match = (
                body_relative[..., 3:7]
                * expected[..., 3:7]
            ).sum(dim=-1).abs()
            ok = (
                tuple(body_relative.shape) == (1, 12, 7)
                and torch.allclose(
                    body_relative[..., :3],
                    expected[..., :3],
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    quaternion_match,
                    torch.ones_like(quaternion_match),
                    atol=1e-5,
                    rtol=1e-5))
            print(
                f"AGICore body endpoint reference mapping "
                f"{'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore body endpoint reference mapping error: {e}")
            return False

    def TestCameraMotionConvention(self) -> bool:
        try:
            sqrt_half = math.sqrt(0.5)

            def compose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
                return NormalizePose(torch.cat([
                    QuatRotate(parent[..., 3:7], child[..., :3]) + parent[..., :3],
                    QuatMultiply(parent[..., 3:7], child[..., 3:7]),
                ], dim=-1))

            previous = torch.tensor([[
                0.4, -0.3, 0.8,
                sqrt_half, 0.0, 0.0, sqrt_half]])
            current = torch.tensor([[
                0.4, -0.3, 0.8,
                0.0, sqrt_half, 0.0, sqrt_half]])
            motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                previous,
                current,
                torch.tensor([True]))
            first_frame_motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                previous,
                current,
                torch.tensor([False]))
            expected_motion = CanonicalizeQuaternion(
                QuatMultiply(
                    QuatConjugate(previous[:, 3:7]),
                    current[:, 3:7]))
            world_gauge = torch.tensor([[
                -0.6, 0.9, 0.2,
                0.0, 0.0, sqrt_half, sqrt_half]])
            gauged_motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                compose(world_gauge, previous),
                compose(world_gauge, current),
                torch.tensor([True]))
            previous_sign_flipped = previous.clone()
            previous_sign_flipped[:, 3:7] *= -1.0
            current_sign_flipped = current.clone()
            current_sign_flipped[:, 3:7] *= -1.0
            previous_sign_motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                previous_sign_flipped,
                current,
                torch.tensor([True]))
            current_sign_motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                previous,
                current_sign_flipped,
                torch.tensor([True]))
            translated_current = current.clone()
            translated_current[:, :3] += torch.tensor([2.0, -1.0, 4.0])
            translated_motion = BrainCore.RelativeCameraMotion(
                object.__new__(BrainCore),
                previous,
                translated_current,
                torch.tensor([True]))
            point_current = torch.tensor([[0.2, -0.4, 0.7]])
            point_previous = QuatRotate(motion, point_current)
            point_world = (
                QuatRotate(current[:, 3:7], point_current)
                + current[:, :3])
            expected_point_previous = QuatRotate(
                QuatConjugate(previous[:, 3:7]),
                point_world - previous[:, :3])
            expected_identity = torch.tensor([[
                0.0, 0.0, 0.0, 1.0]])
            ok = (
                torch.allclose(
                    motion,
                    expected_motion,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    motion,
                    gauged_motion,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    motion, previous_sign_motion, atol=1e-6, rtol=1e-6)
                and torch.allclose(
                    motion, current_sign_motion, atol=1e-6, rtol=1e-6)
                and torch.equal(motion, translated_motion)
                and torch.allclose(
                    point_previous,
                    expected_point_previous,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.equal(first_frame_motion, expected_identity))
            print(
                f"AGICore camera motion convention "
                f"{'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore camera motion convention error: {e}")
            return False

    def TestMotionCommandWorldBoundary(self) -> bool:
        try:
            sqrt_half = math.sqrt(0.5)

            def compose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
                return NormalizePose(torch.cat([
                    QuatRotate(parent[..., 3:7], child[..., :3]) + parent[..., :3],
                    QuatMultiply(parent[..., 3:7], child[..., 3:7]),
                ], dim=-1))

            class FakeDecoupler:
                def __init__(self):
                    self.last_remaining = None
                    self.action_projector = SimpleNamespace(
                        action_mask=DecisionActionMask().view(
                            1,
                            ModuleDim.DecisionEndpointCount,
                            ModuleDim.DecisionActionDim))

                def MaskDecisionTensor(self, decisionTensor):
                    return decisionTensor * self.action_projector.action_mask

                def SafetyScores(self, remaining, risk, confidence, precision):
                    self.last_remaining = remaining
                    return remaining.square().sum(dim=-1)

            B = 1
            current = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            current[..., 6] = 1.0
            current[..., :3] = torch.linspace(
                -0.25,
                0.30,
                ModuleDim.DecisionEndpointCount).view(1, -1, 1) * torch.tensor([
                    1.0, -0.5, 0.25])
            current[..., 3:7] = torch.tensor([
                0.0, 0.0, sqrt_half, sqrt_half])
            local_action = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionActionDim)
            local_action[..., 0] = 0.03
            local_action[..., 1] = torch.linspace(
                -0.02, 0.02, ModuleDim.DecisionEndpointCount)
            local_action[..., 3] = 0.05
            local_action[..., 4] = -0.03
            local_action[..., 5] = 0.04
            decision = SimpleNamespace(
                decision_tensor=local_action,
                gripper_cmd=torch.zeros(B, ModuleDim.ArmCount, 1),
                gripper_valid=torch.ones(B, dtype=torch.bool),
                mode_logits=torch.zeros(B, ModuleDim.ActTypeDim),
                mode_valid=torch.ones(B, dtype=torch.bool),
                safety_scores=torch.ones(B, len(SAFETY_MARGIN_NAMES)))
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.decision_decoupler = FakeDecoupler()

            command = BrainCore.MaterializeMotionCommand(
                brain,
                decision,
                current)
            expected_mask = DecisionActionMask().view(
                1,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionActionDim)
            expected_action = local_action * expected_mask
            expected_target = ApplyPoseDelta(current, expected_action)
            moved_current = ApplyPoseDelta(current, -0.25 * local_action)
            rebased = BrainCore.RebaseWorldMotionCommand(
                brain,
                command,
                moved_current,
                risk=torch.zeros(B),
                confidence=torch.ones(B),
                precision=torch.ones(B))
            expected_remaining = (
                RelativePoseError(moved_current, expected_target)
                * expected_mask)
            expected_reachable_target = ApplyPoseDelta(
                moved_current,
                expected_remaining)
            safety_remaining = brain.decision_decoupler.last_remaining.clone()

            world_gauge = torch.tensor([[
                0.7, -0.4, 0.2,
                sqrt_half, 0.0, 0.0, sqrt_half]])
            gauged_command = replace(
                command,
                target_endpoint_pose=compose(
                    world_gauge.unsqueeze(1),
                    command.target_endpoint_pose))
            gauged_rebased = BrainCore.RebaseWorldMotionCommand(
                brain,
                gauged_command,
                compose(world_gauge.unsqueeze(1), moved_current),
                risk=torch.zeros(B),
                confidence=torch.ones(B),
                precision=torch.ones(B))
            camera_index = ModuleDim.DecisionCameraEndpointIndex
            camera_delta = RelativePoseError(
                current[:, camera_index],
                command.target_endpoint_pose[:, camera_index])
            ok = (
                tuple(command.decision_tensor.shape) == (
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim)
                and torch.allclose(
                    command.decision_tensor,
                    expected_action,
                    atol=1e-7,
                    rtol=1e-7)
                and torch.allclose(
                    command.target_endpoint_pose,
                    expected_target,
                    atol=1e-6,
                    rtol=1e-6)
                and command.endpoint_names == tuple(ModuleDim.DecisionEndpointNames)
                and torch.equal(
                    command.decision_dof_mask,
                    expected_mask.expand(B, -1, -1).bool())
                and torch.allclose(
                    command.target_endpoint_pose[:, camera_index, :3],
                    current[:, camera_index, :3],
                    atol=1e-7,
                    rtol=1e-7)
                and torch.allclose(
                    camera_delta,
                    expected_action[:, camera_index],
                    atol=1e-6,
                    rtol=1e-6)
                and torch.count_nonzero(
                    camera_delta[:, [3, 4, 5]]).item() == 3
                and torch.count_nonzero(
                    camera_delta[:, [0, 1, 2]]).item() == 0
                and torch.allclose(
                    rebased.target_endpoint_pose,
                    expected_reachable_target,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    rebased.decision_tensor,
                    expected_remaining,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    safety_remaining,
                    rebased.decision_tensor,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    ApplyPoseDelta(
                        moved_current,
                        rebased.decision_tensor),
                    rebased.target_endpoint_pose,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.count_nonzero(
                    rebased.decision_tensor[
                        :, camera_index, [0, 1, 2]]).item() == 0
                and not torch.allclose(
                    rebased.decision_tensor,
                    command.decision_tensor,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.equal(rebased.gripper_cmd, command.gripper_cmd)
                and torch.equal(rebased.mode_logits, command.mode_logits)
                and torch.allclose(
                    gauged_rebased.decision_tensor,
                    expected_remaining,
                    atol=1e-5,
                    rtol=1e-5)
                and torch.allclose(
                    gauged_rebased.target_endpoint_pose,
                    compose(
                        world_gauge.unsqueeze(1),
                        expected_reachable_target),
                    atol=1e-5,
                    rtol=1e-5))
            print(
                f"AGICore motion-command world boundary "
                f"{'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore motion-command world boundary error: {e}")
            return False

    def TestEndpointIdentitySurvivesPooling(self) -> bool:
        try:
            torch.manual_seed(7)
            encoder = EndpointPoseEncoder(
                endpointCount=3,
                poseDim=7,
                embedDim=16,
                hidden=32).eval()
            endpoint_pose = torch.zeros(2, 3, 7)
            endpoint_pose[..., 6] = 1.0
            endpoint_pose[:, 0, 0] = 0.25
            endpoint_pose[:, 1, 1] = -0.5
            original = encoder(endpoint_pose)
            permutation = torch.tensor([1, 0, 2])
            permuted = encoder(endpoint_pose[:, permutation])
            ok = (
                tuple(original.endpoint_pose_tokens.shape) == (2, 3, 16)
                and tuple(original.body_pose_feat.shape) == (2, 16)
                and not torch.allclose(
                    original.body_pose_feat,
                    permuted.body_pose_feat,
                    atol=1e-6,
                    rtol=1e-5))
            print(f"AGICore endpoint identity pooling {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore endpoint identity pooling error: {e}")
            return False

    def TestEquivalentQuaternionEncoding(self) -> bool:
        try:
            torch.manual_seed(11)
            encoder = EndpointPoseEncoder(
                endpointCount=3,
                poseDim=7,
                embedDim=16,
                hidden=32).eval()
            endpoint_pose = torch.randn(2, 3, 7)
            endpoint_pose[..., 3:7] = torch.nn.functional.normalize(
                endpoint_pose[..., 3:7],
                dim=-1)
            equivalent_pose = endpoint_pose.clone()
            equivalent_pose[:, 1, 3:7] = -equivalent_pose[:, 1, 3:7]
            normalized = NormalizePose(endpoint_pose)
            equivalent_normalized = NormalizePose(equivalent_pose)
            original = encoder(endpoint_pose)
            equivalent = encoder(equivalent_pose)
            canonical_index = normalized[..., 3:7].abs().argmax(dim=-1, keepdim=True)
            canonical_component = torch.gather(
                normalized[..., 3:7],
                dim=-1,
                index=canonical_index)
            ok = (
                torch.all(canonical_component >= 0.0)
                and torch.allclose(normalized, equivalent_normalized, atol=1e-6, rtol=1e-5)
                and torch.allclose(
                    original.endpoint_pose_tokens,
                    equivalent.endpoint_pose_tokens,
                    atol=1e-6,
                    rtol=1e-5)
                and torch.allclose(
                    original.body_pose_feat,
                    equivalent.body_pose_feat,
                    atol=1e-6,
                    rtol=1e-5))
            print(f"AGICore equivalent quaternion encoding {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore equivalent quaternion encoding error: {e}")
            return False

    def TestWorldRobotPhysicalEncoderOptimizerOwnership(self) -> bool:
        try:
            class MinimalBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.mem = nn.Identity()
                    self.actor = nn.Linear(1, 1)
                    self.conscious = nn.Identity()
                    self.intention = nn.Identity()
                    self.pst_builder = nn.Identity()
                    self.goal_manager = nn.Identity()
                    self.goal_grounding = nn.Identity()
                    self.decision_decoupler = nn.Identity()
                    self.neuro_symbolic = nn.Identity()
                    self.temporal_gate = nn.Identity()
                    self.critic = nn.Linear(1, 1)
                    self.world = TestAGICoreMTool.MakeRobotPhysicalEncodingWorld(
                        bodyPoseDim=8,
                        outDim=4)

                def ResetHebbianMemory(self):
                    return None

            brain = MinimalBrain()
            agent = Agent(brain, isTrain=True, device="cpu")
            robot_params = list(brain.world.robot_physical_encoder.parameters())
            robot_ids = {id(param) for param in robot_params}
            world_ids = {
                id(param)
                for group in agent.opt_world.param_groups
                for param in group["params"]}
            actor_ids = {
                id(param)
                for group in agent.opt_actor.param_groups
                for param in group["params"]}

            before = [param.detach().clone() for param in robot_params]
            body_pose = torch.randn(3, 8)
            physical_reference = torch.randn(
                3, ModuleDim.RobotPhysicalReferenceDim)
            target = torch.randn(3, 4)
            agent.opt_world.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                brain.world.EncodeRobotPhysicalState(
                    body_pose,
                    physical_reference),
                target)
            loss.backward()
            had_grad = any(param.grad is not None for param in robot_params)
            agent.opt_world.step()
            changed = any(
                not torch.equal(old, param.detach())
                for old, param in zip(before, robot_params))
            agent.opt_world.zero_grad(set_to_none=True)
            cleared = all(param.grad is None for param in robot_params)

            ok = (
                not hasattr(brain, "world_robot_physical_encoder")
                and agent.WorldOptimizerModules() == (brain.world,)
                and robot_ids.issubset(world_ids)
                and robot_ids.isdisjoint(actor_ids)
                and had_grad
                and changed
                and cleared)
            print(f"AGICore world robot encoder optimizer ownership {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore world robot encoder optimizer ownership error: {e}")
            return False

    def TestDecisionStructureOptimizerOwnership(self) -> bool:
        try:
            class MinimalBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.mem = nn.Identity()
                    self.actor = DecisionExtractor(
                        stateDim=16,
                        intentDim=16,
                        valueTensorDim=8,
                        vNextTensorDim=8,
                        worldHDim=4,
                        worldZDim=2,
                        worldXDim=2,
                        beliefDim=32,
                        decisionDynDim=8,
                        latentControlDim=4,
                        mapperEmbedDim=8,
                        actionEmbedDim=6,
                        hiddenDim=32,
                        psiDim=8,
                        optionNum=3,
                        useHebb=False)
                    self.conscious = nn.Identity()
                    self.intention = nn.Identity()
                    self.pst_builder = nn.Identity()
                    self.goal_manager = nn.Identity()
                    self.goal_grounding = nn.Identity()
                    self.decision_decoupler = DecisionDecouplerV2(
                        decisionDim=32)
                    self.neuro_symbolic = NeuroSymbolicRobotStateEncoder()
                    self.temporal_gate = nn.Identity()
                    self.critic = nn.Linear(1, 1)
                    self.world = nn.Linear(1, 1)

                def ResetHebbianMemory(self):
                    return None

            brain = MinimalBrain()
            agent = Agent(brain, isTrain=True, device="cpu")
            actor_ids = {
                id(parameter)
                for group in agent.opt_actor.param_groups
                for parameter in group["params"]}
            critic_ids = {
                id(parameter)
                for group in agent.opt_critic.param_groups
                for parameter in group["params"]}
            world_ids = {
                id(parameter)
                for group in agent.opt_world.param_groups
                for parameter in group["params"]}
            structure_parameters = tuple(
                brain.actor.belief_assembler.source_gate.parameters()
            ) + tuple(
                brain.actor.option_transition_prior.parameters()
            ) + tuple(
                brain.decision_decoupler.endpoint_action_refiner.parameters()
            ) + tuple(
                brain.neuro_symbolic.parameters())
            ok = (
                all(id(parameter) in actor_ids for parameter in structure_parameters)
                and all(id(parameter) not in critic_ids for parameter in structure_parameters)
                and all(id(parameter) not in world_ids for parameter in structure_parameters))
            print(f"AGICore decision structure optimizer ownership {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore decision structure optimizer ownership error: {e}")
            return False

    def TestOnlineCandidateOptimizerOwnership(self) -> bool:
        try:
            class CandidateWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.base_weight = nn.Parameter(torch.ones(1))
                    self.candidates: List[nn.Parameter] = [nn.Parameter(torch.ones(1))]

                def CandParameters(self):
                    yield from self.candidates

                def Update(self, action: str, **kwargs):
                    if str(action).lower() == "grow":
                        self.candidates.append(nn.Parameter(torch.ones(1)))
                    return {"ok": True}

            class MinimalBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.mem = nn.Identity()
                    self.actor = nn.Linear(1, 1)
                    self.conscious = nn.Identity()
                    self.intention = nn.Identity()
                    self.pst_builder = nn.Identity()
                    self.goal_manager = nn.Identity()
                    self.goal_grounding = nn.Identity()
                    self.decision_decoupler = nn.Identity()
                    self.neuro_symbolic = nn.Identity()
                    self.temporal_gate = nn.Identity()
                    self.critic = nn.Linear(1, 1)
                    self.world = CandidateWorld()

                def ResetHebbianMemory(self):
                    return None

            brain = MinimalBrain()
            agent = Agent(brain, isTrain=True, device="cpu")

            def optimizer_ids() -> set[int]:
                return {
                    id(parameter)
                    for group in agent.opt_world.param_groups
                    for parameter in group["params"]}

            first = brain.world.candidates[0]
            initial_owned = id(first) in optimizer_ids()
            agent.UpdateWrappers([brain.world], "grow")
            second = brain.world.candidates[1]
            grown_owned = id(second) in optimizer_ids()

            before = [parameter.detach().clone() for parameter in brain.world.candidates]
            agent.opt_world.zero_grad(set_to_none=True)
            sum(parameter.square().sum() for parameter in brain.world.candidates).backward()
            agent.opt_world.step()
            changed = all(
                not torch.equal(old, parameter.detach())
                for old, parameter in zip(before, brain.world.candidates))
            ok = initial_owned and grown_owned and changed
            print(f"AGICore online candidate optimizer ownership {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore online candidate optimizer ownership error: {e}")
            return False

    def TestAgentWeightLoadsResyncOnlineOptimizer(self) -> bool:
        try:
            import io
            from FunctionTools import _TestOnlineBase, _TestOnlineWrapper

            class DirectHeadBase(_TestOnlineBase):
                def __init__(self):
                    super().__init__()
                    self.direct_head = nn.Linear(3, 2)

                def _ValidateMemoryPayload(self, state):
                    return int(state["batch_size"]), 1

                def ImportMemoryPayload(self, state, *, batchSize):
                    if state["batch_size"] != batchSize:
                        raise ValueError("invalid test World memory")

            class DirectHeadWrapper(_TestOnlineWrapper):
                def __init__(self, base, initRankEach):
                    super().__init__(base, initRankEach=initRankEach)
                    self.RestoreBaseTrainabilityAfterCommit()

                def RestoreBaseTrainabilityAfterCommit(self):
                    for parameter in self.base.parameters():
                        parameter.requires_grad_(False)
                    for parameter in self.base.direct_head.parameters():
                        parameter.requires_grad_(True)

            class RecallHeads(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enable_auxiliary = True

            class MinimalPerception(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.recall_heads = RecallHeads()

            class MinimalMemory(nn.Module):
                def ValidateDurableState(self, state, *, expectedBatch):
                    if state != {"batch_size": expectedBatch}:
                        raise ValueError("invalid test durable memory")

                def ImportDurableState(self, state):
                    return None

            class MinimalBrain(nn.Module):
                def __init__(self, candidateRank: int):
                    super().__init__()
                    self.is_online_learning = True
                    self.calibration_id = "test-calibration"
                    self.perc = MinimalPerception()
                    self.attn = nn.Identity()
                    self.mem = MinimalMemory()
                    self.actor = nn.Linear(1, 1)
                    self.conscious = nn.Identity()
                    self.intention = nn.Identity()
                    self.pst_builder = nn.Identity()
                    self.goal_manager = nn.Identity()
                    self.goal_grounding = nn.Identity()
                    self.decision_decoupler = nn.Identity()
                    self.neuro_symbolic = nn.Identity()
                    self.temporal_gate = nn.Identity()
                    self.critic = nn.Linear(1, 1)
                    self.world = DirectHeadWrapper(
                        DirectHeadBase(), initRankEach=candidateRank)

                def ResetHebbianMemory(self):
                    return None

                def ImportBuffers(self, state):
                    return None

            source = MinimalBrain(candidateRank=1)
            committed_result = source.world.Update("commit")
            assert committed_result["commit_stats"]["committed_triples"] == 1.0
            deployment_state = ExportDeploymentModelState(source)
            checkpoint_state = ExportBrainModelState(source)

            brain = MinimalBrain(candidateRank=1)
            agent = Agent(brain, isTrain=True, device="cpu")
            agent.ResetOnlineCandidateState()
            empty_optimizer_states = {
                "opt_actor": copy.deepcopy(agent.opt_actor.state_dict()),
                "opt_critic": copy.deepcopy(agent.opt_critic.state_dict()),
                "opt_world": copy.deepcopy(agent.opt_world.state_dict()),
            }
            original_sync = agent.SyncTrainableOptimizers
            sync_calls = 0

            def counted_sync():
                nonlocal sync_calls
                sync_calls += 1
                original_sync()

            agent.SyncTrainableOptimizers = counted_sync

            def seed_stale_optimizer_state() -> None:
                agent.opt_actor.state[brain.actor.weight]["stale_moment"] = torch.ones(())

            def optimizer_state_is_clear() -> bool:
                return not any(
                    optimizer.state
                    for optimizer in (agent.opt_actor, agent.opt_critic, agent.opt_world))

            def contract_holds() -> bool:
                adapter = brain.world.base.adapter
                committed = (
                    list(adapter.A_list)
                    + list(adapter.B_list)
                    + list(adapter.alpha))
                candidates = list(brain.world.CandParameters())
                optimizer_ids = {
                    id(parameter)
                    for group in agent.opt_world.param_groups
                    for parameter in group["params"]}
                return (
                    bool(committed)
                    and not any(parameter.requires_grad for parameter in committed)
                    and all(
                        parameter.requires_grad
                        for parameter in brain.world.base.direct_head.parameters())
                    and {id(parameter) for parameter in candidates}.issubset(optimizer_ids)
                    and {id(parameter) for parameter in committed}.isdisjoint(optimizer_ids))

            seed_stale_optimizer_state()
            agent.LoadTorchPayload = lambda path: {
                "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,
                "calibration_id": brain.calibration_id,
                "brain": deployment_state}
            agent.LoadBrainWeights("unused.pth")
            weights_path_ok = (
                sync_calls == 1
                and contract_holds()
                and optimizer_state_is_clear())

            sync_calls = 0
            seed_stale_optimizer_state()
            payload = io.BytesIO()
            torch.save({
                "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,
                "calibration_id": brain.calibration_id,
                "world_frame_id": "test-world",
                "batch_size": 1,
                "online_learning": True,
                "brain": checkpoint_state,
                "online_candidates": {
                    "world": source.world.ExportCandidateState()},
                "buffers": {},
                "world_memory": {"batch_size": 1},
                "memory_durable": {"batch_size": 1},
                "rng_py": random.getstate(),
                "rng_np": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda_all": None,
                **empty_optimizer_states}, payload)
            payload.seek(0)
            agent.world_frame_id = "test-world"
            agent.world_memory_batch_size = 1
            agent.Load(payload)
            checkpoint_path_ok = (
                sync_calls == 1
                and contract_holds()
                and optimizer_state_is_clear())

            ok = weights_path_ok and checkpoint_path_ok
            print(f"AGICore online load optimizer sync {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore online load optimizer sync error: {e}")
            return False

    def TestResizeStateBuffersPreservesRuntimeDtype(self) -> bool:
        try:
            class BufferedModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(1, dtype=torch.float16))
                    self.register_buffer(
                        "runtime_state",
                        torch.zeros(1, dtype=torch.float16))

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.buffered = BufferedModule()
            BrainCore.ResizeStateBuffersForLoad(
                brain,
                {"buffered.runtime_state": torch.zeros(3, dtype=torch.float32)})
            ok = (
                tuple(brain.buffered.runtime_state.shape) == (3,)
                and brain.buffered.runtime_state.dtype == torch.float16)
            print(f"AGICore buffer resize dtype {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore buffer resize dtype error: {e}")
            return False

    def TestRuntimeRestoreUsesModelDtype(self) -> bool:
        try:
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.anchor = nn.Parameter(torch.ones(1, dtype=torch.float32))
            restored = BrainCore.MoveRuntimeStateToModel(
                brain,
                {
                    "float": torch.ones(2, dtype=torch.float16),
                    "nested": [torch.ones(1, dtype=torch.float64)],
                    "step": torch.ones(2, dtype=torch.long),
                    "valid": torch.ones(2, dtype=torch.bool)})
            ok = (
                restored["float"].dtype == torch.float32
                and restored["nested"][0].dtype == torch.float32
                and restored["step"].dtype == torch.long
                and restored["valid"].dtype == torch.bool)
            print(f"AGICore runtime restore dtype {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore runtime restore dtype error: {e}")
            return False

    def TestSlowRuntimeSnapshotRoundTrip(self) -> bool:
        try:
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.prev_failure_count = torch.tensor([2.0, 5.0])
            brain.slow_step_count = 7
            brain.slow_cache = {"intent": torch.tensor([[3.0], [4.0]])}
            brain.thread_end = False

            state = BrainCore.ExportSlowRuntimeState(brain)
            brain.prev_failure_count.zero_()
            brain.slow_step_count = 0
            brain.slow_cache = None
            brain.thread_end = True
            BrainCore.ImportSlowRuntimeState(brain, state)

            ok = (
                torch.equal(brain.prev_failure_count, torch.tensor([2.0, 5.0]))
                and brain.slow_step_count == 7
                and torch.equal(
                    brain.slow_cache["intent"],
                    torch.tensor([[3.0], [4.0]]))
                and brain.thread_end is False)
            print(f"AGICore slow runtime snapshot {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore slow runtime snapshot error: {e}")
            return False

    def TestAdaptiveRuntimeBufferSnapshotRoundTrip(self) -> bool:
        try:
            class MutableModule(nn.Module):
                def __init__(self, value: float):
                    super().__init__()
                    self.register_buffer("runtime", torch.tensor([value]))

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.perc = MutableModule(1.0)
            brain.actor = MutableModule(2.0)
            brain.conscious = MutableModule(3.0)
            brain.goal_manager = MutableModule(4.0)
            brain.perception_recall_loss = MutableModule(5.0)
            brain.perc.register_buffer(
                "camera_intrinsics",
                torch.eye(3),
                persistent=False)

            class WorldSnapshot(nn.Module):
                def ExportState(self):
                    state = torch.zeros(1, 1)
                    return state, state, state

                def ExportPhysicalState(self):
                    return {}

                def ExportRobotPhysicalState(self):
                    return {}

            class StateSnapshot(nn.Module):
                def ExportState(self):
                    return {}

            class MemorySnapshot(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pending = []

                def ExportTransientState(self):
                    return {}

            brain.world = WorldSnapshot()
            brain.attn = StateSnapshot()
            brain.critic = StateSnapshot()
            brain.mem = MemorySnapshot()
            brain.OCR = SimpleNamespace(
                _temporal_step=0,
                _last_batch_size=0,
                _last_ocr_texts_batch=[],
                _tracks_by_bi=[])
            brain.neuro_symbolic = SimpleNamespace(
                ExportPlanState=lambda: {})

            tensor_state_names = (
                "prev_mem", "prev_attn", "prev_option_logit", "prev_entropy",
                "prev_decision_state", "prev_latent_control",
                "prev_target_endpoint_pose", "prev_target_endpoint_valid",
                "prev_measured_endpoint_pose", "prev_measured_endpoint_valid",
                "prev_visual_valid",
                "active_option_policy_input", "active_option_prior_logit",
                "active_option_goal_mid", "active_option_index",
                "active_option_valid", "prev_belief_prediction_state",
                "prev_belief_prediction_valid", "temporal_active_mask",
                "temporal_action_age_steps", "temporal_action_epoch",
                "temporal_invoke_drift", "temporal_active_kind",
                "prev_mapper_hidden", "prev_td_error", "prev_world_s",
                "prev_done_flag", "prev_precision", "prev_goal_bias",
                "prev_intent_sem", "prev_failure_count")
            for name in tensor_state_names:
                setattr(brain, name, torch.zeros(1))
            brain.prev_measured_endpoint_valid = torch.tensor([False, True])
            brain.prev_visual_valid = torch.tensor([True, False])
            brain.active_motion_command = None
            brain.prev_visual_state = None
            brain.prospective_visual_prediction = None
            brain.prev_self_sem = None
            brain.perc_buffer = []
            brain.visual_state_buffer = []
            brain.visual_state_valid_buffer = []
            brain.history = deque()
            brain.extra_mem = None
            brain.slow_step_count = 0
            brain.slow_cache = None
            brain.thread_end = True

            exported_buffers = BrainCore.ExportBuffers(brain)
            state = exported_buffers["adaptive_runtime_buffers"]
            for module in (
                brain.perc,
                brain.actor,
                brain.conscious,
                brain.goal_manager,
                brain.perception_recall_loss,):
                module._buffers["runtime"] = torch.tensor([10.0, 11.0])
            BrainCore.ImportAdaptiveRuntimeBuffers(brain, state)

            observed = [
                float(module.runtime.item())
                for module in (
                    brain.perc,
                    brain.actor,
                    brain.conscious,
                    brain.goal_manager,
                    brain.perception_recall_loss,)]
            state_dict_has_intrinsics = any(
                name.rsplit(".", 1)[-1] == "camera_intrinsics"
                for name in brain.state_dict())

            def has_intrinsics_key(value: Any) -> bool:
                if isinstance(value, dict):
                    return any(
                        str(name).rsplit(".", 1)[-1] == "camera_intrinsics"
                        or has_intrinsics_key(child)
                        for name, child in value.items())
                if isinstance(value, (list, tuple)):
                    return any(has_intrinsics_key(child) for child in value)
                return False

            runtime_has_intrinsics = has_intrinsics_key(exported_buffers)
            ok = (
                observed == [1.0, 2.0, 3.0, 4.0, 5.0]
                and exported_buffers["prev_measured_endpoint_valid"].tolist()
                == [False, True]
                and exported_buffers["prev_visual_valid"].tolist()
                == [True, False]
                and not state_dict_has_intrinsics
                and not runtime_has_intrinsics)
            print(f"AGICore adaptive runtime buffers {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore adaptive runtime buffers error: {e}")
            return False

    def TestPartialBatchRuntimeMasking(self) -> bool:
        try:
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.SEQ_LEN = 2

            class FakePerception:
                object_token_count = 2

            brain.perc = FakePerception()
            done = torch.tensor([True, False])
            cleared = BrainCore.ClearRuntimeRows(
                brain,
                {"value": torch.tensor([[1.0], [2.0]])},
                done)

            previous_camera = torch.zeros(2, 7)
            previous_camera[:, 6] = 1.0
            current_camera = previous_camera.clone()
            sqrt_half = math.sqrt(0.5)
            current_camera[1, 5] = sqrt_half
            current_camera[1, 6] = sqrt_half
            camera_motion = BrainCore.RelativeCameraMotion(
                brain,
                previous_camera,
                current_camera,
                torch.tensor([False, True]))

            def visual(value: float) -> VisualState:
                return VisualState(
                    IntegratedFeat=torch.full((2, ModuleDim.PerceptionFeat), value),
                    GlobalFeat=torch.zeros(2, 1),
                    VentralFeat=torch.zeros(2, 1),
                    DorsalFeat=torch.zeros(2, 1),
                    MotionToken=torch.full((2, ModuleDim.PerceptionEmbed), value),
                    QualityToken=torch.full((2, ModuleDim.PerceptionEmbed), value),
                    PredErrorToken=torch.full((2, ModuleDim.PerceptionEmbed), value),
                    ObjectTokens=torch.full((2, 2, ModuleDim.PerceptionEmbed), value),
                    PatchTokens=torch.zeros(2, 1, 1),
                    SemanticNodes={},
                    Auxiliary={})

            sequence = BrainCore.BuildVisualSequenceTensors(
                brain,
                [visual(1.0), visual(2.0)],
                validMasks=[torch.tensor([True, False]), torch.tensor([False, True])],
                batchSize=2,
                device=torch.device("cpu"),
                dtype=torch.float32)
            padding_mask = sequence[-1]
            expected_padding = torch.tensor([[False, True], [True, False]])
            ok = (
                torch.equal(cleared["value"], torch.tensor([[0.0], [2.0]]))
                and torch.equal(
                    camera_motion[0],
                    torch.tensor([0.0, 0.0, 0.0, 1.0]))
                and torch.allclose(
                    camera_motion[1],
                    torch.tensor([0.0, 0.0, sqrt_half, sqrt_half]))
                and torch.equal(padding_mask, expected_padding))
            print(f"AGICore partial-batch runtime masking {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore partial-batch runtime masking error: {e}")
            return False

    def TestTrainBatchResizeResetsWorldBeforePhysicalSnapshot(self) -> bool:
        try:
            class SnapshotReached(Exception):
                pass

            class BatchAwareWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.batch_size = 2
                    self.reset_calls = []

                def ResetState(self, batchSize: int = 1):
                    self.batch_size = int(batchSize)
                    self.reset_calls.append(self.batch_size)

                def EncodeRobotPhysicalState(
                    self,
                    bodyEndpointPoseRelative: torch.Tensor,
                    robotPhysicalReference: torch.Tensor,) -> torch.Tensor:
                    return torch.zeros(
                        bodyEndpointPoseRelative.size(0),
                        ModuleDim.PstSlotDim)

                def ExportPhysicalState(self) -> Dict[str, torch.Tensor]:
                    return {"SlotState": torch.zeros(self.batch_size, 1, 1)}

                def BuildModelPhysicalState(
                    self,
                    physicalState: Dict[str, torch.Tensor],
                    cameraPoseWorld: torch.Tensor,) -> Dict[str, torch.Tensor]:
                    if physicalState["SlotState"].size(0) != cameraPoseWorld.size(0):
                        raise RuntimeError("World physical state retained the previous batch")
                    raise SnapshotReached

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.device = torch.device("cpu")
            brain.is_online_learning = False
            brain.actor = SimpleNamespace(
                num_options=2,
                dyn_dim=3,
                u_dim=4,
                mapper_hidden_dim=5)
            brain.world = BatchAwareWorld()
            brain.history_len = 2
            brain.need_trace = False
            brain.extra_mem = None
            brain.thread_end = True
            brain.save_module_messager_output = False
            BrainCore.ResetBuffers(
                brain,
                B=2,
                isOnlineLearning=False,
                device=brain.device)
            brain.BuildExecutedActionFeedback = lambda endpointPose: (
                torch.zeros(3, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
                torch.zeros(3, ModuleDim.EndpointActionEmbedDim))

            endpoint_pose = torch.zeros(
                3,
                ModuleDim.RobotStateEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            planner_expected = endpoint_pose[
                :, ModuleDim.RobotStateControlledEndpointSlice].clone()
            robot_state = {
                "endpoint_pose": endpoint_pose,
                "base_orientation_world": endpoint_pose[:, 0, 3:7].clone(),
                "gravity_direction_world": torch.tensor(
                    [[0.0, 0.0, -1.0]]).expand(3, -1).clone(),
                "planner_expected_endpoint_pose": planner_expected,
                "model_command_executed": torch.zeros(3),
                "executed_action_id": torch.zeros(3, dtype=torch.long),
                "planner_executing": torch.zeros(3),}
            request = BrainStepInput(
                frame=torch.zeros(3, 3, 1, 1),
                text_ext=None,
                reward_ext=None,
                done_flag=None,
                is_train=True,
                sample_actions=False,
                deterministic_actor=True,
                depth=torch.ones(3, 1, 1, 1),
                depth_valid=torch.ones(3, 1, 1, 1, dtype=torch.bool),
                perception_targets=None,
                robot_state=robot_state)

            reached_snapshot = False
            try:
                BrainCore.Step(brain, request)
            except SnapshotReached:
                reached_snapshot = True

            ok = reached_snapshot and brain.world.reset_calls == [3]
            print(f"AGICore train batch resize {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore train batch resize error: {e}")
            return False

    def TestWorldRuntimeRoutingAndEvalIsolation(self) -> bool:
        try:
            class FakeBaseWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.memory_exports = 0
                    self.encoder_calls = 0
                    self.encoder_inputs = None

                def EncodeRobotPhysicalState(
                    self,
                    bodyProprioception,
                    robotPhysicalReference):
                    self.encoder_calls += 1
                    self.encoder_inputs = (
                        bodyProprioception,
                        robotPhysicalReference)
                    return torch.cat([
                        bodyProprioception.flatten(1),
                        robotPhysicalReference], dim=-1)

                def StepPriorOnly(self, *args, **kwargs):
                    raise AssertionError("preview bypassed the online wrapper")

                def ExportWorldMemoryBank(self, topk):
                    self.memory_exports += 1
                    return {"topk": torch.tensor(topk)}

            class FakeOnlineWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.base = FakeBaseWorld()
                    self.preview_calls = 0
                    self.preview_kwargs: Dict[str, Any] = {}
                    self.forward_kwargs: Dict[str, Any] = {}

                def StepPriorOnly(self, *args, **kwargs):
                    self.preview_calls += 1
                    self.preview_kwargs = dict(kwargs)
                    return {"source": "wrapper"}

                def forward(self, vision, **kwargs):
                    self.forward_kwargs = dict(kwargs)
                    return {"source": "wrapper_train"}

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.world = FakeOnlineWorld()
            brain.is_online_learning = True
            brain.eval()

            body_proprioception = torch.tensor([
                [[11.0]],
                [[21.0]]])
            robot_physical_reference = torch.tensor([
                [12.0],
                [22.0]])
            world_robot_physical_encoding = brain.RuntimeModule(
                brain.world).EncodeRobotPhysicalState(
                    body_proprioception,
                    robot_physical_reference)
            endpoint_action = torch.tensor([
                [31.0, 32.0],
                [41.0, 42.0]])

            preview = BrainCore.PreviewWorldPrior(
                brain,
                hPrev=torch.zeros(2, 1),
                zPrev=torch.zeros(2, 1),
                s4xPrev=torch.zeros(2, 1),
                physicalState={"SlotPresence": torch.ones(2, 1)},
                actionEnc=endpoint_action,
                robotPhysicalState=world_robot_physical_encoding,
                cameraMotion=torch.tensor([
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]]))
            bank = BrainCore.ExportRuntimeWorldMemoryBank(brain, topk=5)
            posterior_physical_state = {
                "SlotPresence": torch.ones(2, 1)}
            transition_physical_state = {
                "SlotPresence": torch.zeros(2, 1)}
            transition_robot_physical_encoding = (
                world_robot_physical_encoding + 1.0)
            trained = BrainCore.RunWorldTrainingStep(
                brain,
                visionIn=torch.zeros(2, 1),
                physicalState=posterior_physical_state,
                transitionPhysicalState=transition_physical_state,
                actionEnc=endpoint_action,
                robotPhysicalState=world_robot_physical_encoding,
                transitionRobotPhysicalState=(
                    transition_robot_physical_encoding),
                cameraMotion=torch.tensor([
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]]),
                reward=torch.zeros(2),
                done=torch.zeros(2))
            kwargs = brain.world.forward_kwargs
            preview_kwargs = brain.world.preview_kwargs
            ok = (
                preview["source"] == "wrapper"
                and brain.world.preview_calls == 1
                and brain.world.base.encoder_calls == 1
                and brain.world.base.encoder_inputs[0] is body_proprioception
                and brain.world.base.encoder_inputs[1] is robot_physical_reference
                and not hasattr(brain.world, "EncodeRobotPhysicalState")
                and torch.equal(
                    preview_kwargs["robotPhysicalState"],
                    world_robot_physical_encoding)
                and torch.equal(
                    preview_kwargs["actionEnc"],
                    endpoint_action)
                and int(bank["topk"].item()) == 5
                and brain.world.base.memory_exports == 1
                and trained["source"] == "wrapper_train"
                and kwargs["physicalState"] is posterior_physical_state
                and kwargs["transitionPhysicalState"] is transition_physical_state
                and torch.equal(
                    kwargs["robotPhysicalState"],
                    world_robot_physical_encoding)
                and torch.equal(
                    kwargs["transitionRobotPhysicalState"],
                    transition_robot_physical_encoding)
                and torch.equal(kwargs["actionEnc"], endpoint_action)
                and kwargs.get("sample") is False
                and kwargs.get("updateMemory") is False)
            print(f"AGICore world runtime routing {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore world runtime routing error: {e}")
            return False

    def TestPlannerStartsFromCurrentPosterior(self) -> bool:
        try:
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            current = {
                "h_next": torch.full((2, 3), 2.0, requires_grad=True),
                "z_next": torch.full((2, 4), 3.0, requires_grad=True),
                "x_next": torch.full((2, 5), 4.0, requires_grad=True),}
            h0, z0, x0 = BrainCore.PlannerInitialState(brain, current)
            ok = (
                torch.equal(h0, current["h_next"])
                and torch.equal(z0, current["z_next"])
                and torch.equal(x0, current["x_next"])
                and not h0.requires_grad
                and not z0.requires_grad
                and not x0.requires_grad)
            print(f"AGICore planner posterior start {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore planner posterior start error: {e}")
            return False

    def TestBatchPredictionAliveMask(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sample_mask = None

                def PredictNextVisualFromPosterior(self, *args, **kwargs):
                    B = int(args[0].size(0))
                    return {
                        "predicted_visual": torch.zeros(B, 1),
                        "reconstructed_visual_state": {"value": torch.zeros(B, 1)},}

                def ComputePredictionLoss(self, *, sampleMask, **kwargs):
                    self.sample_mask = sampleMask.detach().clone()
                    return {"loss_pred_total": sampleMask.float().sum()}

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.world = FakeWorld()
            losses = BrainCore.ComputeAliveWorldPredictionLoss(
                brain,
                prevWorldH=torch.zeros(2, 1),
                prevWorldZ=torch.zeros(2, 1),
                prevWorldX=torch.zeros(2, 1),
                physicalState={"SlotPresence": torch.ones(2, 1)},
                actionEnc=torch.zeros(2, 1),
                robotPhysicalState=torch.zeros(2, 1),
                cameraMotion=torch.tensor([
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]]),
                targetVisualState=object(),
                precision=torch.ones(2),
                aliveMask=torch.tensor([False, True]))
            ok = (
                torch.equal(brain.world.sample_mask, torch.tensor([False, True]))
                and float(losses["loss_pred_total"].item()) == 1.0)
            print(f"AGICore batch prediction alive mask {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore batch prediction alive mask error: {e}")
            return False

    def TestPerceptionPriorUsesRealizedTransition(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.calls = []

                def PredictNextVisualFromPosterior(self, *args, **kwargs):
                    self.calls.append((args, kwargs))
                    B = int(args[0].size(0))
                    return {
                        "reconstructed_visual_state": {
                            "PriorConfidence": torch.ones(B, 1),
                            "Marker": kwargs["actionEnc"].clone(),}}

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.world = FakeWorld()
            brain.prev_precision = torch.tensor([0.4, 0.8])
            brain.prev_mem = torch.zeros(2, 3)
            brain.prospective_visual_prediction = {
                "PriorConfidence": torch.ones(2, 1),
                "Marker": torch.full((2, 2), 99.0)}

            h = torch.full((2, 3), 1.0)
            z = torch.full((2, 4), 2.0)
            x = torch.full((2, 5), 3.0)
            physical = {"SlotPresence": torch.tensor([[1.0], [0.0]])}
            measured_action = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
            robot = torch.full((2, 6), 8.0)
            camera_motion = torch.tensor([
                [0.2, 0.0, 0.0, 0.9797959],
                [0.0, 0.3, 0.0, 0.9539392]])
            realized = BrainCore.BuildRealizedVisualPrior(
                brain,
                prevWorldH=h,
                prevWorldZ=z,
                prevWorldX=x,
                transitionPhysicalState=physical,
                measuredActionEnc=measured_action,
                transitionRobotPhysicalState=robot,
                cameraMotion=camera_motion,
                transitionValid=torch.tensor([True, False]))
            top_down = BrainCore.BuildTopDownContext(brain, realized)
            args, kwargs = brain.world.calls[0]
            invalid = BrainCore.BuildRealizedVisualPrior(
                brain,
                prevWorldH=h,
                prevWorldZ=z,
                prevWorldX=x,
                transitionPhysicalState=physical,
                measuredActionEnc=measured_action,
                transitionRobotPhysicalState=robot,
                cameraMotion=camera_motion,
                transitionValid=torch.tensor([False, False]))
            ok = (
                len(brain.world.calls) == 1
                and args[0] is h
                and args[1] is z
                and args[2] is x
                and kwargs["physicalState"] is physical
                and kwargs["actionEnc"] is measured_action
                and kwargs["robotPhysicalState"] is robot
                and kwargs["cameraMotion"] is camera_motion
                and top_down.PredictedVisual is realized
                and top_down.PredictedVisual is not brain.prospective_visual_prediction
                and torch.equal(realized["Marker"], measured_action)
                and realized["PriorConfidence"].view(-1).tolist() == [1.0, 0.0]
                and invalid is None)
            print(
                f"AGICore realized perception prior "
                f"{'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore realized perception prior error: {e}")
            return False

    def TestExecutedFeedbackUsesMeasuredTransition(self) -> bool:
        try:
            class FakeDecoupler:
                def __init__(self):
                    self.action_mask = DecisionActionMask().view(
                        1,
                        ModuleDim.DecisionEndpointCount,
                        ModuleDim.DecisionActionDim)

                def MaskDecisionTensor(self, decisionTensor):
                    return decisionTensor * self.action_mask

                def EncodeEndpointAction(self, decisionTensor):
                    return FlattenActiveDecisionTensor(decisionTensor)

            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            brain.decision_decoupler = FakeDecoupler()
            B = 2
            previous = torch.zeros(B, ModuleDim.RobotStateEndpointCount, ModuleDim.DecisionEndpointPoseDim)
            previous[..., 6] = 1.0
            measured_delta = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionActionDim)
            measured_delta[1, :, 0] = torch.linspace(
                0.01,
                0.13,
                ModuleDim.DecisionEndpointCount)
            measured_delta[1, :, 1] = -0.02
            measured_delta[1, :, 2] = 0.03
            measured_delta[1, :, 3] = 0.04
            measured_delta[1, :, 4] = -0.03
            measured_delta[1, :, 5] = 0.06
            current = ApplyPoseDelta(previous, measured_delta)
            brain.prev_measured_endpoint_pose = previous
            brain.prev_measured_endpoint_valid = torch.tensor([False, True])
            brain.prev_target_endpoint_pose = previous[
                :, ModuleDim.RobotStateControlledEndpointSlice].clone()
            brain.prev_target_endpoint_pose[:, :, 1] = 100.0

            delta0, feedback0 = BrainCore.BuildExecutedActionFeedback(brain, current)
            brain.prev_target_endpoint_pose[:, :, 2] = -100.0
            delta1, feedback1 = BrainCore.BuildExecutedActionFeedback(brain, current)
            raw_measured_delta = RelativePoseError(previous, current)
            expected_delta = (
                raw_measured_delta
                * DecisionActionMask().view(
                    1,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim))
            expected_delta[0] = 0.0
            camera_index = ModuleDim.DecisionCameraEndpointIndex
            ok = (
                tuple(delta0.shape) == (
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim)
                and tuple(feedback0.shape) == (
                    B,
                    ModuleDim.DecisionActiveDofCount)
                and torch.count_nonzero(delta0[0]).item() == 0
                and torch.count_nonzero(feedback0[0]).item() == 0
                and torch.allclose(
                    delta0,
                    expected_delta,
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    feedback0,
                    FlattenActiveDecisionTensor(expected_delta),
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    delta0[
                        1,
                        :ModuleDim.DecisionBodyEndpointCount],
                    raw_measured_delta[
                        1,
                        :ModuleDim.DecisionBodyEndpointCount],
                    atol=1e-6,
                    rtol=1e-6)
                and torch.count_nonzero(
                    delta0[1, camera_index, [0, 1, 2]]).item() == 0
                and torch.count_nonzero(
                    delta0[1, camera_index, [3, 4, 5]]).item() == 3
                and torch.equal(delta0, delta1)
                and torch.equal(feedback0, feedback1))
            print(f"AGICore measured executed feedback {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore measured executed feedback error: {e}")
            return False

    def TestDecisionRuntimePartialDoneReset(self) -> bool:
        try:
            brain = object.__new__(BrainCore)
            nn.Module.__init__(brain)
            B = 2

            def feature(width: int):
                return torch.stack([torch.ones(width), torch.full((width,), 2.0)])

            brain.prev_mem = feature(4)
            brain.prev_attn = feature(4)
            brain.prev_option_logit = feature(3)
            brain.prev_entropy = torch.tensor([1.0, 2.0])
            brain.prev_decision_state = feature(4)
            brain.prev_latent_control = feature(3)
            brain.prev_mapper_hidden = feature(4)
            brain.prev_td_error = torch.tensor([1.0, 2.0])
            brain.prev_failure_count = torch.tensor([1.0, 2.0])
            brain.prev_precision = torch.tensor([0.25, 0.75])
            brain.prev_goal_bias = feature(4)
            brain.prev_self_sem = feature(4)
            brain.prev_intent_sem = feature(4)
            brain.active_option_policy_input = feature(6)
            brain.active_option_prior_logit = feature(3)
            brain.active_option_goal_mid = feature(ModuleDim.GoalMidDim)
            brain.active_option_index = torch.tensor([1, 2])
            brain.active_option_valid = torch.tensor([True, True])
            brain.prev_belief_prediction_state = feature(4)
            brain.prev_belief_prediction_valid = torch.tensor([True, True])
            brain.temporal_active_mask = torch.tensor([1.0, 2.0])
            brain.temporal_action_age_steps = torch.tensor([1, 2])
            brain.temporal_action_epoch = torch.tensor([1, 2])
            brain.temporal_active_kind = torch.tensor([1, 2])
            brain.temporal_invoke_drift = torch.tensor([1.0, 2.0])
            pose = torch.zeros(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim)
            pose[..., 0] = torch.tensor([1.0, 2.0]).view(B, 1)
            pose[..., 6] = 1.0
            brain.prev_target_endpoint_pose = pose.clone()
            brain.prev_target_endpoint_valid = torch.tensor([True, True])
            measured_pose = torch.zeros(
                B,
                ModuleDim.RobotStateEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            measured_pose[..., 0] = torch.tensor([1.0, 2.0]).view(B, 1)
            measured_pose[..., 6] = 1.0
            brain.prev_measured_endpoint_pose = measured_pose
            brain.prev_measured_endpoint_valid = torch.tensor([True, True])
            action = torch.stack([
                torch.ones(ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
                torch.full((ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim), 2.0)])
            brain.active_motion_command = MotionCommand(
                decision_tensor=action,
                target_endpoint_pose=pose.clone(),
                endpoint_names=ModuleDim.DecisionEndpointNames,
                decision_dof_mask=DecisionActionMask().view(
                    1,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim).expand(B, -1, -1).bool(),
                gripper_cmd=feature(ModuleDim.ArmCount).unsqueeze(-1),
                gripper_valid=torch.tensor([True, True]),
                mode_logits=feature(ModuleDim.ActTypeDim),
                mode_valid=torch.tensor([True, True]),
                safety_scores=feature(5),
                safety_names=SAFETY_MARGIN_NAMES)

            BrainCore.ResetDecisionRuntimeRows(brain, torch.tensor([True, False]))
            row_fields = [
                brain.prev_mem,
                brain.prev_attn,
                brain.prev_option_logit,
                brain.prev_decision_state,
                brain.prev_latent_control,
                brain.prev_mapper_hidden,
                brain.prev_goal_bias,
                brain.prev_self_sem,
                brain.prev_intent_sem,
                brain.active_option_policy_input,
                brain.active_option_prior_logit,
                brain.active_option_goal_mid,
                brain.prev_belief_prediction_state,]
            ok = all(
                torch.count_nonzero(value[0]).item() == 0
                and torch.count_nonzero(value[1]).item() > 0
                for value in row_fields)
            ok &= brain.prev_entropy.tolist() == [0.0, 2.0]
            ok &= brain.prev_td_error.tolist() == [0.0, 2.0]
            ok &= brain.prev_failure_count.tolist() == [0.0, 2.0]
            ok &= brain.prev_precision.tolist() == [1.0, 0.75]
            ok &= brain.temporal_action_age_steps.tolist() == [0, 2]
            ok &= brain.active_option_index.tolist() == [0, 2]
            ok &= brain.active_option_valid.tolist() == [False, True]
            ok &= brain.prev_target_endpoint_valid.tolist() == [False, True]
            ok &= brain.prev_measured_endpoint_valid.tolist() == [False, True]
            ok &= torch.count_nonzero(brain.prev_target_endpoint_pose[0, :, :6]).item() == 0
            ok &= torch.all(brain.prev_target_endpoint_pose[0, :, 6] == 1.0).item()
            ok &= torch.count_nonzero(brain.active_motion_command.decision_tensor[0]).item() == 0
            ok &= torch.count_nonzero(brain.active_motion_command.decision_tensor[1]).item() > 0
            print(f"AGICore decision partial-done reset {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore decision partial-done reset error: {e}")
            return False

    def TestDecisionJsonWireContract(self) -> bool:
        try:
            B = 1
            pose = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            pose[..., 6] = 1.0
            command = MotionCommand(
                decision_tensor=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim),
                target_endpoint_pose=pose,
                endpoint_names=ModuleDim.DecisionEndpointNames,
                decision_dof_mask=DecisionActionMask().view(
                    1,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim).expand(B, -1, -1).bool(),
                gripper_cmd=torch.full((B, ModuleDim.ArmCount, 1), 0.5),
                gripper_valid=torch.zeros(B, dtype=torch.bool),
                mode_logits=torch.zeros(B, ModuleDim.ActTypeDim),
                mode_valid=torch.zeros(B, dtype=torch.bool),
                safety_scores=torch.ones(B, len(SAFETY_MARGIN_NAMES)),
                safety_names=SAFETY_MARGIN_NAMES)
            execution_scores = torch.zeros(B, ModuleDim.TemporalPrimitiveCount)
            execution_scores[:, CONTINUE] = -torch.inf
            envelope = TemporalDecisionEnvelope(
                kind_logits=torch.zeros(B, ModuleDim.TemporalPrimitiveCount),
                execution_kind_scores=execution_scores,
                kind_id=torch.tensor([DISPATCH]),
                kind_names=ModuleDim.TemporalPrimitiveNames,
                override_applied=torch.tensor([False]),
                action_id=torch.tensor([3]),
                action_epoch=torch.tensor([3]),
                reason_scores=torch.zeros(B, ModuleDim.TemporalReasonDim),
                reason_names=TEMPORAL_REASON_NAMES,
                duration_ms=torch.tensor([1000.0]),
                soft_timeout_ms=torch.tensor([1000.0]),
                hard_timeout_ms=torch.tensor([5000.0]),
                publish_motion_command=torch.tensor([1.0]),
                reuse_active_motion_command=torch.tensor([0.0]),
                publish_stop_command=torch.tensor([0.0]),
                publish_hold_command=torch.tensor([0.0]),
                same_operator=torch.tensor([1.0]),
                operator_changed=torch.tensor([0.0]),
                invoke_delta=torch.tensor([0.0]),
                reference_drift=torch.tensor([0.0]),
                invoke_drift=torch.tensor([0.0]),
                motion_command=command)
            decision = {
                "candidate_option_index": torch.tensor([4]),
                "scheduled_option_index": torch.tensor([4]),
                "scheduled_option_valid": torch.tensor([True]),
                "credited_option_index": torch.tensor([2]),
                "credited_option_valid": torch.tensor([False]),
                "previous_model_command_executed": torch.tensor([False]),
                "previous_executed_action_id": torch.tensor([0]),
                "expected_previous_action_id": torch.tensor([2]),
                "temporal_context": SimpleNamespace(
                    action_age_steps=torch.tensor([2])),}
            packed = Agent.UnpackActPacked(
                object.__new__(Agent),
                AgentActOutput(
                    motion_command=command,
                    temporal_envelope=envelope,
                    decision=decision,
                    loss=None,
                    ocr=[],
                    intention_texts=[]),
                requestProvenance={
                    "stream_id": "test-stream",
                    "sequence_index": 7,
                    "frame_id": "frame-0007",
                    "calibration_id": "test-calibration",
                    "world_frame_id": "test-world",
                })
            parsed = json.loads(packed)
            temporal = parsed["temporal_envelope"]
            option = parsed["option_assignment"]
            contract = parsed["command_contract"]
            decision_dof_mask = parsed["motion_command"][
                "decision_dof_mask"]
            ok = (
                parsed["schema_version"] == DECISION_WIRE_SCHEMA_VERSION
                and parsed["stream_id"] == "test-stream"
                and parsed["sequence_index"] == 7
                and parsed["frame_id"] == "frame-0007"
                and parsed["calibration_id"] == "test-calibration"
                and parsed["world_frame_id"] == "test-world"
                and contract["authority"] == "proposal"
                and contract["physical_execution_ready"] is False
                and contract["hardware_validation"] == "not_performed"
                and contract["external_validation_required"] is True
                and contract["continue_renews_timeout"] is False
                and all(
                    all(endpoint_mask)
                    for endpoint_mask in decision_dof_mask[
                        :ModuleDim.DecisionBodyEndpointCount])
                and decision_dof_mask[
                    ModuleDim.DecisionCameraEndpointIndex] == [
                        False, False, False, True, True, True]
                and temporal["execution_kind_scores"][CONTINUE] is None
                and temporal["execution_kind_legal"][CONTINUE] is False
                and temporal["action_age_steps"] == 2
                and temporal["timeouts_apply_to_action_id"] == 3
                and temporal["latch_timeout_budget"] is True
                and isinstance(temporal["kind_id"], int)
                and isinstance(temporal["publish_motion_command"], bool)
                and isinstance(option["scheduled_option_index"], int)
                and isinstance(option["scheduled_option_valid"], bool))
            print(f"AGICore decision JSON wire {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore decision JSON wire error: {e}")
            return False

    def TestModelStateExcludesWorldRuntime(self) -> bool:
        try:
            class RuntimeWorld(nn.Module):
                def __init__(self, weight: float, runtime: float):
                    super().__init__()
                    self.weight = nn.Parameter(torch.tensor([weight]))
                    self.register_buffer(
                        "_pst_pose_world",
                        torch.tensor([runtime]))
                    self.register_buffer(
                        "_mem_keys",
                        torch.tensor([runtime + 1.0]))
                    self.s4 = nn.Module()
                    self.s4.register_buffer(
                        "x",
                        torch.tensor([runtime + 2.0]))

            class BoundaryBrain(nn.Module):
                def __init__(self, weight: float, runtime: float):
                    super().__init__()
                    self.world = RuntimeWorld(weight, runtime)
                    self.planner = nn.Module()
                    self.planner.wm = self.world

            source = BoundaryBrain(3.0, 7.0)
            state = ExportBrainModelState(source)
            target = BoundaryBrain(0.0, 19.0)
            LoadBrainModelState(target, state)

            runtime_rejected = False
            polluted = dict(state)
            polluted["planner.wm._pst_pose_world"] = torch.tensor([7.0])
            try:
                LoadBrainModelState(target, polluted)
            except ValueError:
                runtime_rejected = True

            missing_rejected = False
            try:
                LoadBrainModelState(target, {})
            except ValueError:
                missing_rejected = True

            ok = (
                set(state) == {"world.weight", "planner.wm.weight"}
                and not any(
                    IsWorldRuntimeStateKey(name)
                    for name in state)
                and torch.equal(target.world.weight, torch.tensor([3.0]))
                and torch.equal(
                    target.world._pst_pose_world,
                    torch.tensor([19.0]))
                and runtime_rejected
                and missing_rejected)
            print(f"AGICore model/runtime state boundary {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore model/runtime state boundary error: {e}")
            return False

    def TestDeploymentStateCanonicalizesOnlineWrapper(self) -> bool:
        try:
            from FunctionTools import _TestOnlineBase, _TestOnlineWrapper

            class OnlineBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.is_online_learning = True
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.world = _TestOnlineWrapper(
                        _TestOnlineBase(),
                        initRankEach=1)
                    self.critic = nn.Identity()
                    self.intention = nn.Identity()

            class DeploymentBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.is_online_learning = False
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.world = _TestOnlineBase()
                    self.critic = nn.Identity()
                    self.intention = nn.Identity()

            torch.manual_seed(31)
            source = OnlineBrain()
            sample = torch.randn(4, 3)
            expected = source.world(sample).detach()
            source.world.Update("commit")
            canonical = ExportDeploymentModelState(source)
            target = DeploymentBrain()
            LoadDeploymentModelState(target, canonical)
            actual = target.world.adapter(sample).detach()
            ok = (
                all(".base." not in name for name in canonical)
                and any(name.startswith("world.adapter.A_list.") for name in canonical)
                and torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
            print(f"AGICore canonical deployment state {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore canonical deployment state error: {e}")
            return False

    def TestWorldMemoryBindingIsDeferredUntilWorldFrame(self) -> bool:
        try:
            import tempfile

            events = []

            class MemoryWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self._use_memory = False
                    self._mem_path = None
                    self.device = torch.device("cpu")
                    self.dtype = torch.float32

                def BindMemoryContext(self, calibrationId, worldFrameId):
                    events.append(("bind", calibrationId, worldFrameId))

                def SaveMemory(self, path):
                    events.append(("save", path))

                def LoadMemory(self, path, *, batchSize, mapLocation=None):
                    events.append(("load", path, batchSize, mapLocation))

                def EnsureB(self, batchSize):
                    events.append(("ensure", batchSize))

            class MinimalBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.calibration_id = "test-calibration"
                    self.is_online_learning = False
                    self.world = MemoryWorld()

                def ResetHebbianMemory(self):
                    return None

            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "new-world-memory.pt")
                agent = Agent(
                    MinimalBrain(),
                    isTrain=False,
                    device="cpu",
                    worldMemoryPath=path)
                deferred = events == []
                agent.BindWorldMemoryContext("test-world", batchSize=1)
                agent.BindWorldMemoryContext("test-world", batchSize=1)
                different_batch_rejected = False
                try:
                    agent.BindWorldMemoryContext("test-world", batchSize=2)
                except RuntimeError:
                    different_batch_rejected = True
                different_world_rejected = False
                try:
                    agent.BindWorldMemoryContext("other-world", batchSize=1)
                except RuntimeError:
                    different_world_rejected = True

            ok = (
                deferred
                and events == [
                    ("bind", "test-calibration", "test-world"),
                    ("ensure", 1),
                    ("save", path),
                ]
                and different_batch_rejected
                and different_world_rejected)
            print(f"AGICore deferred world-memory binding {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"AGICore deferred world-memory binding error: {e}")
            return False

    def TestTemporalKindSupervisionMatchesExecutionSemantics(self) -> bool:
        try:
            execution_scores = torch.tensor([
                [0.0, 3.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, float("-inf"), float("-inf"), 0.0, float("-inf")],
                [0.0, 0.0, 1.0, 3.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, -1.0, 0.0],
            ], requires_grad=True)
            target = torch.tensor([DISPATCH, CONTINUE, CANCEL, CANCEL])
            valid = torch.ones(4, dtype=torch.bool)
            active = torch.tensor([1.0, 0.0, 1.0, 1.0])
            override = torch.tensor([False, False, True, False])

            loss = BrainCore.ComputeTemporalKindSupervisionLoss(
                execution_scores,
                target,
                valid,
                active,
                override,)
            expected = nn.functional.cross_entropy(
                execution_scores[[0, 3]],
                target[[0, 3]])
            gradient = torch.autograd.grad(
                loss,
                execution_scores,
                retain_graph=True)[0]

            no_label_loss = BrainCore.ComputeTemporalKindSupervisionLoss(
                execution_scores,
                target,
                valid,
                active,
                torch.ones_like(override),)
            no_label_gradient = torch.autograd.grad(
                no_label_loss,
                execution_scores)[0]
            ok = bool(
                torch.isfinite(loss).item()
                and torch.allclose(loss, expected, atol=1e-7, rtol=1e-6)
                and float(gradient[[0, 3]].abs().sum().item()) > 0.0
                and float(gradient[[1, 2]].abs().sum().item()) == 0.0
                and float(no_label_loss.detach().item()) == 0.0
                and float(no_label_gradient.abs().sum().item()) == 0.0)
            print(
                "AGICore temporal-kind execution supervision "
                f"{'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"AGICore temporal-kind execution supervision error: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "DataclassContracts": self.TestDataclassContracts(),
            "WorldRobotPhysicalEncodingExcludesControlFeedback": self.TestWorldRobotPhysicalEncodingExcludesControlFeedback(),
            "CameraPhysicalReferenceGaugeInvariance": self.TestCameraPhysicalReferenceGaugeInvariance(),
            "BodyEndpointReferenceMapping": self.TestBodyEndpointReferenceMapping(),
            "CameraMotionConvention": self.TestCameraMotionConvention(),
            "MotionCommandWorldBoundary": self.TestMotionCommandWorldBoundary(),
            "EndpointIdentitySurvivesPooling": self.TestEndpointIdentitySurvivesPooling(),
            "EquivalentQuaternionEncoding": self.TestEquivalentQuaternionEncoding(),
            "WorldRobotPhysicalEncoderOptimizerOwnership": self.TestWorldRobotPhysicalEncoderOptimizerOwnership(),
            "DecisionStructureOptimizerOwnership": self.TestDecisionStructureOptimizerOwnership(),
            "OnlineCandidateOptimizerOwnership": self.TestOnlineCandidateOptimizerOwnership(),
            "AgentWeightLoadsResyncOnlineOptimizer": self.TestAgentWeightLoadsResyncOnlineOptimizer(),
            "ResizeStateBuffersPreservesRuntimeDtype": self.TestResizeStateBuffersPreservesRuntimeDtype(),
            "RuntimeRestoreUsesModelDtype": self.TestRuntimeRestoreUsesModelDtype(),
            "SlowRuntimeSnapshotRoundTrip": self.TestSlowRuntimeSnapshotRoundTrip(),
            "AdaptiveRuntimeBufferSnapshotRoundTrip": self.TestAdaptiveRuntimeBufferSnapshotRoundTrip(),
            "PartialBatchRuntimeMasking": self.TestPartialBatchRuntimeMasking(),
            "TrainBatchResizeResetsWorldBeforePhysicalSnapshot": self.TestTrainBatchResizeResetsWorldBeforePhysicalSnapshot(),
            "WorldRuntimeRoutingAndEvalIsolation": self.TestWorldRuntimeRoutingAndEvalIsolation(),
            "PlannerStartsFromCurrentPosterior": self.TestPlannerStartsFromCurrentPosterior(),
            "BatchPredictionAliveMask": self.TestBatchPredictionAliveMask(),
            "PerceptionPriorUsesRealizedTransition": self.TestPerceptionPriorUsesRealizedTransition(),
            "ExecutedFeedbackUsesMeasuredTransition": self.TestExecutedFeedbackUsesMeasuredTransition(),
            "DecisionRuntimePartialDoneReset": self.TestDecisionRuntimePartialDoneReset(),
            "DecisionJsonWireContract": self.TestDecisionJsonWireContract(),
            "ModelStateExcludesWorldRuntime": self.TestModelStateExcludesWorldRuntime(),
            "DeploymentStateCanonicalizesOnlineWrapper": self.TestDeploymentStateCanonicalizesOnlineWrapper(),
            "TemporalKindSupervisionMatchesExecutionSemantics": self.TestTemporalKindSupervisionMatchesExecutionSemantics(),
            "WorldMemoryBindingIsDeferredUntilWorldFrame": self.TestWorldMemoryBindingIsDeferredUntilWorldFrame(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nAGICore tests: {passed}/{len(results)} passed.")
        return results
