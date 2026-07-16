from __future__ import annotations

from typing import Any, Dict
import ast
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BasicParameters:
    IMAGE_SIZE = 512

    IMAGE_SEQ_LEN = 16

    IMAGE_RM_LEN = math.ceil(IMAGE_SEQ_LEN / 10)

    MEMORY_CALLBACK_LEN = 16

    REWARD_MIN = -10.0

    REWARD_MAX = 10.0

    CONSCIOUSNESSTEM = 1024

    SAVE_EVERY_SAMPLE_COUNT = 500

    DATA_ROOT_PATH = "BrainDeepLearn/Data"
    CAMERA_CALIBRATION_PATH = str(PROJECT_ROOT / "Configure/Camera_Calibration.json")
    DATA_SENSOR_MANIFEST_PATH = "BrainDeepLearn/Data/sensor_manifest.json"
    OCR_DATA_ROOT_PATH = "BrainDeepLearn/Data/OCR"

    DATA_FRAMES_PATH = "BrainDeepLearn/Data/frames"
    DATA_REWARD_PATH = "BrainDeepLearn/Data/reward"
    DATA_DONE_PATH = "BrainDeepLearn/Data/done"
    DATA_DEPTH_PATH = "BrainDeepLearn/Data/depth"
    DATA_DEPTH_VALID_PATH = "BrainDeepLearn/Data/depth_valid"
    DATA_ROBOT_STATE_PATH = "BrainDeepLearn/Data/robot_state"
    DATA_NORMAL_PATH = "BrainDeepLearn/Data/normal"
    DATA_SEMANTIC_SEGMENTATION_PATH = "BrainDeepLearn/Data/semantic_segmentation"
    DATA_INSTANCE_SEGMENTATION_PATH = "BrainDeepLearn/Data/instance_segmentation"
    DATA_SYNTHETIC_SUPERVISION_PATH = "BrainDeepLearn/Data/synthetic_supervision"
    DATA_TEXTS_PATH = "BrainDeepLearn/Data/texts"
    OCR_FRAMES_PATH = "BrainDeepLearn/Data/OCR/frames"
    OCR_TEXTS_PATH = "BrainDeepLearn/Data/OCR/OCRTexts"
    OCR_DICT_PATH = "BrainDeepLearn/ModuleSetting/OCRKeys.txt"
    OCR_RECOGNIZER_FRAMES_PATH = "BrainDeepLearn/Data/OCRRecognition/frames"
    OCR_RECOGNIZER_TEXTS_PATH = "BrainDeepLearn/Data/OCRRecognition/OCRTexts"

    MEMORY_MEMORY_PATH = "BrainDeepLearn/Data/MemoryMemory.pt"
    MEMORY_MEMORY_PATH_TRAIN = "BrainDeepLearn/Data/MemoryMemory_train.pt"
    WORLD_MEMORY_PATH = "BrainDeepLearn/Data/WorldMemory.pt"
    WORLD_MEMORY_PATH_TRAIN = "BrainDeepLearn/Data/WorldMemory_train.pt"
    MODULEPARAMETER_PATH = "BrainDeepLearn/Data/module_parameter.pth"
    OCR_MODULEPARAMETER_PATH = "BrainDeepLearn/Data/ocr_module_parameter.pth"
    OCR_RECOGNIZER_MODULEPARAMETER_PATH = "BrainDeepLearn/Data/ocr_recognizer_parameter.pth"

    OCR_RECOGNIZER_DATA_ROOT_PATH = "BrainDeepLearn/Data/OCRRecognition"
    CKPT_PATH_TRAIN = "BrainDeepLearn/Data/training_checkpoint.pth"
    OCR_CKPT_PATH_TRAIN = "BrainDeepLearn/Data/ocr_training_checkpoint.pth"
    OCR_RECOGNIZER_CKPT_PATH_TRAIN = "BrainDeepLearn/Data/ocr_recognizer_training_checkpoint.pth"

    MEMORY_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/MemoryMemory.pt"
    MEMORY_MEMORY_PATH_TEST_TRAIN = "BrainDeepLearn/TestData/MemoryMemory_train.pt"
    WORLD_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/WorldMemory.pt"
    WORLD_MEMORY_PATH_TEST_TRAIN = "BrainDeepLearn/TestData/WorldMemory_train.pt"
    MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/module_parameter.pth"
    OCR_MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/ocr_module_parameter.pth"
    OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/ocr_recognizer_parameter.pth"
    DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData"
    DATA_SENSOR_MANIFEST_PATH_TEST = "BrainDeepLearn/TestData/sensor_manifest.json"
    DATA_DEPTH_PATH_TEST = "BrainDeepLearn/TestData/depth"
    DATA_DEPTH_VALID_PATH_TEST = "BrainDeepLearn/TestData/depth_valid"
    DATA_ROBOT_STATE_PATH_TEST = "BrainDeepLearn/TestData/robot_state"
    DATA_NORMAL_PATH_TEST = "BrainDeepLearn/TestData/normal"
    DATA_SEMANTIC_SEGMENTATION_PATH_TEST = "BrainDeepLearn/TestData/semantic_segmentation"
    DATA_INSTANCE_SEGMENTATION_PATH_TEST = "BrainDeepLearn/TestData/instance_segmentation"
    DATA_SYNTHETIC_SUPERVISION_PATH_TEST = "BrainDeepLearn/TestData/synthetic_supervision"
    OCR_DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData/OCR"
    OCR_RECOGNIZER_DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData/OCRRecognition"
    CKPT_PATH_TEST = "BrainDeepLearn/TestData/training_test_checkpoint.pth"
    OCR_CKPT_PATH_TEST = "BrainDeepLearn/TestData/ocr_training_checkpoint.pth"
    OCR_RECOGNIZER_CKPT_PATH_TEST = "BrainDeepLearn/TestData/ocr_recognizer_training_checkpoint.pth"

    @classmethod
    def Get(cls, name: str):
        attr_name = str(name).strip()
        if not cls.IsConfigAttribute(attr_name):
            raise AttributeError(f"BasicParameters has no attribute: {attr_name}")
        return getattr(cls, attr_name)

    @classmethod
    def Set(cls, name: str, value: str) -> bool:
        attr_name = str(name).strip()
        if not cls.IsConfigAttribute(attr_name):
            raise AttributeError(f"BasicParameters has no attribute: {attr_name}")
        if not isinstance(value, str):
            raise TypeError(f"BasicParameters.Set value must be str, got {type(value).__name__}")

        current_value = getattr(cls, attr_name)
        parsed_value = cls.ParseValueFromString(value, current_value)
        setattr(cls, attr_name, parsed_value)
        cls.RefreshDerivedParameters(attr_name)
        return True

    @classmethod
    def GetStringDict(cls) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for name in cls.__dict__.keys():
            if cls.IsConfigAttribute(name):
                result[str(name)] = str(getattr(cls, name))
        return result

    @classmethod
    def ParseValueFromString(cls, value: str, currentValue: Any):
        text = value.strip()

        if isinstance(currentValue, bool):
            text_lower = text.lower()
            if text_lower in ("true", "1", "yes", "on"):
                return True
            if text_lower in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot parse bool from: {value}")

        if isinstance(currentValue, int) and not isinstance(currentValue, bool):
            return int(text)

        if isinstance(currentValue, float):
            return float(text)

        if isinstance(currentValue, str):
            return value

        parsed = ast.literal_eval(text)

        if currentValue is None:
            return parsed

        current_type = type(currentValue)
        if isinstance(parsed, current_type):
            return parsed

        return current_type(parsed)

    @classmethod
    def RefreshDerivedParameters(cls, changedName: str = ""):
        if changedName == "IMAGE_SEQ_LEN":
            cls.IMAGE_RM_LEN = math.ceil(cls.IMAGE_SEQ_LEN / 10)

    @classmethod
    def IsConfigAttribute(cls, name: str) -> bool:
        if str(name).strip() == "":
            return False
        if name not in cls.__dict__:
            return False
        value = cls.__dict__[name]
        if isinstance(value, (classmethod, staticmethod)):
            return False
        return not callable(getattr(cls, name))
