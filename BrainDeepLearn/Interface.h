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
    using PrintCB = std::function<void(const char*, std::size_t)>;

    BrainDeepLearnInterface();
    ~BrainDeepLearnInterface();

    bool StartTraining(std::string& root, int epochs = 5, int batchSize = 32, double valSplit = 0.1, int imagineHorizon = 5, bool resume = true);
    bool StopTraining();
    bool PauseTraining();
    bool ResumeTraining();

    bool GetTrainingStatus(StatusMap& status);

    bool TestPerceptionModule();

    bool TestAttentionModule();

    bool TestMemoryModule();

    bool TestDecisionModule();

    bool TestWorldModule();

    bool TestValueEstimationModule();

private:

    PyObject* pModule = nullptr;
    PyObject* pManagerObj = nullptr;

    void Init();
};


#endif  