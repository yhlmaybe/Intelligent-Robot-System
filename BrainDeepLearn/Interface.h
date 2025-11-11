#ifndef BRAINDEEPLEARNINTERFACE_H
#define BRAINDEEPLEARNINTERFACE_H

#include <python3.8/Python.h>
#include <string>
#include <thread>
#include <map>
#include <boost/variant.hpp>
#include <functional>
#include <stdexcept>
#include <memory>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <thread>

#include "../include/PythonInteraction.h"

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

    BrainDeepLearnInterface(std::shared_ptr<PythonInteraction::Manager> mag, std::function<void(std::string)> printCallBack = nullptr);
    ~BrainDeepLearnInterface();

    bool StartTraining(std::string& root, int epochs = 5, int batchSize = 32, double valSplit = 0.1, int imagineHorizon = 5, bool resume = true);
    bool Stop();
    bool Pause();
    bool Resume();

    bool RunPythonAsync(PyTask task);

    bool GetTrainingStatus(StatusMap& status);

    bool TestPerceptionModule();

    bool TestAttentionModule();

    bool TestMemoryModule();

    bool TestDecisionModule();

    bool TestWorldModule();

    bool TestValueEstimationModule();

    bool TestModuleTrain(bool onlineLearning);

private:

    std::shared_ptr<PythonInteraction::Manager> pyManager = nullptr;

    std::function<void(std::string)> printMessageCB = nullptr;

    PyObject* pModule = nullptr;
    PyObject* pManagerObj = nullptr;

    std::thread brThread;
    std::atomic<bool> brThreadRunning{false};

    void Init();

    void PrintMessage(std::string str);
};


#endif  