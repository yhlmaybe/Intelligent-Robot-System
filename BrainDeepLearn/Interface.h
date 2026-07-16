#ifndef BRAINDEEPLEARNINTERFACE_H
#define BRAINDEEPLEARNINTERFACE_H

#include <python3.8/Python.h>
#include <string>
#include <thread>
#include <map>
#include <vector>
#include <boost/variant.hpp>
#include <functional>
#include <stdexcept>
#include <memory>
#include <optional>
#include <cstdio>
#include <mutex>
#include <atomic>

#include "../include/PythonInteraction.h"
#include "../include/IRSFunction.h"

#define CALL_METHOD_RET_BOOL(name, fmt, ...) \
    ([&]{ \
        if (!pManagerObj) return false; \
        PyGILState_STATE g = PyGILState_Ensure(); \
        PyObject* r = PyObject_CallMethod(pManagerObj, name, fmt, __VA_ARGS__); \
        bool ok = r && PyObject_IsTrue(r); \
        Py_XDECREF(r); \
        PyGILState_Release(g); \
        return ok; \
    }())

#define CALL_METHOD_RET_STATUSVALUE(outValue, name, fmt, ...) \
    ([&]{ \
        if (!pManagerObj) return false; \
        PyGILState_STATE g = PyGILState_Ensure(); \
        PyObject* r = PyObject_CallMethod(pManagerObj, name, fmt, __VA_ARGS__); \
        bool ok = false; \
        if (r) \
        { \
            if (PyLong_Check(r)) \
            { \
                outValue = (int)PyLong_AsLong(r); \
                ok = (PyErr_Occurred() == nullptr); \
            } \
            else if (PyFloat_Check(r)) \
            { \
                outValue = PyFloat_AsDouble(r); \
                ok = (PyErr_Occurred() == nullptr); \
            } \
            else if (PyUnicode_Check(r)) \
            { \
                const char* text = PyUnicode_AsUTF8(r); \
                if (text) \
                { \
                    outValue = std::string(text); \
                    ok = true; \
                } \
            } \
        } \
        else \
        { \
            PyErr_Print(); \
        } \
        Py_XDECREF(r); \
        PyGILState_Release(g); \
        return ok; \
    }())

#define CALL_METHOD_RET_STATUSMAP(outMap, name) \
    ([&]{ \
        if (!pManagerObj) return false; \
        PyGILState_STATE g = PyGILState_Ensure(); \
        PyObject* r = PyObject_CallMethod(pManagerObj, name, nullptr); \
        bool ok = false; \
        if (r && PyDict_Check(r)) \
        { \
            outMap.clear(); \
            PyObject* key = nullptr; \
            PyObject* val = nullptr; \
            Py_ssize_t pos = 0; \
            while (PyDict_Next(r, &pos, &key, &val)) \
            { \
                if (!PyUnicode_Check(key)) continue; \
                std::string k = PyUnicode_AsUTF8(key); \
                if (PyLong_Check(val)) outMap[k] = (int)PyLong_AsLong(val); \
                else if (PyFloat_Check(val)) outMap[k] = PyFloat_AsDouble(val); \
                else if (PyUnicode_Check(val)) outMap[k] = std::string(PyUnicode_AsUTF8(val)); \
            } \
            ok = true; \
        } \
        else if (!r) \
        { \
            PyErr_Print(); \
        } \
        Py_XDECREF(r); \
        PyGILState_Release(g); \
        return ok; \
    }())

#define CALL_METHOD_NOARG(name) \
    ([&]{ \
        if (!pManagerObj) return false;  \
        PyGILState_STATE g = PyGILState_Ensure(); \
        PyObject* r = PyObject_CallMethod(pManagerObj, name, nullptr); \
        bool ok = r && PyObject_IsTrue(r); \
        Py_XDECREF(r); \
        PyGILState_Release(g); \
        return ok; \
    }())

class BrainDeepLearnInterface 
{
public:
    using StatusValue = boost::variant<int, double, std::string>;
    using StatusMap = std::map<std::string, StatusValue>;
    using PyTask = std::function<void()>;

    struct VisualStatus
    {
        int width = 0;
        int height = 0;
        std::vector<unsigned char> bitmapRgb;
        std::string text;
        double updatedAt = 0.0;
    };

    BrainDeepLearnInterface(std::shared_ptr<PythonInteraction::Manager> mag, std::function<void(std::string)> printCallBack = nullptr);
    ~BrainDeepLearnInterface();

    static IRSThreadTools::ThreadSafeQueue<std::string> &GetJsonQueue();

    bool TrainModule(bool isOnlineLearning = false, int epochs = 5, int batchSize = 1, double valSplit = 0.1, bool resume = true);

    bool TrainOCRModule(int epochs = 400, int batchSize = 1, double valSplit = 0.1, bool resume = true);

    bool DeployModule(bool useHebbian = true, bool usePlanner = true);

    bool ExportParmFromCheckpoint(bool isOverride);

    bool Stop();

    bool Pause();

    bool Resume();

    bool ResetHebbianMemory();

    bool SetJsonQueue();

    bool SetParameterReceiver(std::optional<double> reward = std::nullopt, std::optional<double> done = std::nullopt, std::optional<std::string> textExt = std::nullopt);

    bool InitAgentHandle(bool useHebbian = true, bool usePlanner = true);

    // sensorPacketJson carries one synchronized RGB-D frame, stream_id, monotonically
    // increasing sequence_index, frame_id, calibration_id,
    // rgb_encoding="rgb8", depth_unit="meter", and an explicit depth-valid mask;
    // robotStateJson carries the matching stream/sequence/frame/calibration identifiers,
    // a stable world_frame_id, and 13 pose-carrier rows: 10 fingertips, 2 wrists,
    // and the camera's fixed optical-center translation plus its 3-DOF orientation;
    // plus q_world_base (orientation only, no base/camera translation input) and the unit world-frame direction of
    // gravitational acceleration (down). It must also carry all 13 planner target poses,
    // planner progress/tracking/terminal state, and the model_command_executed plus
    // executed_action_id feedback pair identifying whether the preceding proposal actually ran.
    // All measurements refer to the sensor-frame exposure;
    // gravity is dimensionless and does not contain the 9.81 m/s^2 magnitude. The returned motion
    // proposal contains 12 full-SE(3) hand/wrist endpoints plus all 3 camera
    // rotation DOFs and no camera translation: exactly 75 active DOFs. It echoes the request's
    // stream/sequence/frame/calibration/world-frame identity. This transport layer does not
    // perform IK, collision/actuator validation, timeout enforcement, or a certified
    // emergency stop; a downstream executor must validate before actuation.
    bool AgentHandleForward(
        StatusValue& result,
        const std::string& sensorPacketJson,
        const std::string& robotStateJson,
        std::optional<double> reward = std::nullopt,
        std::optional<double> done = std::nullopt);

    bool ResetAgentHandleHebbian();

    bool RunPythonAsync(PyTask task);

    bool GetCurrentStatus(StatusMap& status);

    bool GetCurrentVisualStatus(VisualStatus& status);

    bool SetVisualStateEnabled(bool enabled);

    bool SetOverrideCheckpointWithModuleParams(bool enabled);

    bool SetBasicParameters(std::string name, std::string value);

    bool GetBasicParameters(std::string name, StatusValue& value);

    bool GetBasicParametersDict(StatusMap& parameters);

    bool TestPerceptionModule();

    bool TestAttentionModule();

    bool TestMemoryModule();

    bool TestDecisionModule();

    bool TestWorldModule();

    bool TestValueEstimationModule();

    bool TestConsciousnessModule();

    bool TestIntentionModule();

    bool TestOCRModule();

    bool TestModuleTrain(bool onlineLearning);

    bool TestOCRModuleTrain();

    bool TestOCRRecognitionTrain();

private:

    std::shared_ptr<PythonInteraction::Manager> pyManager = nullptr;

    std::function<void(std::string)> printMessageCB = nullptr;

    PyObject* pModule = nullptr;
    PyObject* pManagerObj = nullptr;

    std::thread brThread;
    std::atomic<bool> brThreadRunning{false};

    static bool ExtractUtf8String(PyObject* value, std::string& out);
    static double ExtractFloatValue(PyObject* value, double defaultValue = 0.0);
    static bool ParseBitmapList(PyObject* bitmapObj, VisualStatus& status);

    void Init();

    void PrintMessage(std::string str);
};


#endif  
