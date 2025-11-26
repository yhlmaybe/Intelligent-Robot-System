#include "Interface.h"

void BrainDeepLearnInterface::Init() 
{
    if (!Py_IsInitialized()) throw std::runtime_error("Failed to init Python");

    PyGILState_STATE g = PyGILState_Ensure();

    pModule = PyImport_ImportModule("Manager");

    if (!pModule) 
    { 
        PyErr_Print(); 
        throw std::runtime_error("Import Manager failed"); 
    }

    PyObject* pClass = PyObject_GetAttrString(pModule, "ManagerFunction");
    if (!pClass || !PyCallable_Check(pClass)) 
    {
        PyErr_Print(); throw std::runtime_error("ManagerFunction class not callable");
    }
    pManagerObj = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    if (!pManagerObj) 
    {
         PyErr_Print(); throw std::runtime_error("Instantiate ManagerFunction failed"); 
    }
    
    PyGILState_Release(g);
}

BrainDeepLearnInterface::BrainDeepLearnInterface(std::shared_ptr<PythonInteraction::Manager> mag, std::function<void(std::string)> printCallBack) 
{
    Init();
    pyManager = mag;
    printMessageCB = printCallBack;
}

BrainDeepLearnInterface::~BrainDeepLearnInterface() 
{
    if(brThread.joinable())
    {
        Stop();
        brThread.join();
    }

    PyGILState_STATE g = PyGILState_Ensure();
    Py_XDECREF(pManagerObj);
    Py_XDECREF(pModule);
    
    PyGILState_Release(g);
}

void BrainDeepLearnInterface::PrintMessage(std::string str)
{
    if(printMessageCB)
    {
        printMessageCB(str);
    }
    else
    {
        std::fprintf(stderr, "%s\n", str.c_str());
    }
}

bool BrainDeepLearnInterface::TrainModule(bool isOnlineLearning, int epochs, int batchSize, double valSplit, bool resume) 
{
    return RunPythonAsync([this, isOnlineLearning, epochs, batchSize, valSplit, resume]()
    {
        (void)CALL_METHOD_RET_BOOL("TrainModule", "biidi", isOnlineLearning, epochs, batchSize, valSplit, resume?1:0);
    });
}

bool BrainDeepLearnInterface::TrainOCRModule(bool isOnlineLearning, int epochs, int batchSize, double valSplit, bool resume) 
{
    return RunPythonAsync([this, isOnlineLearning, epochs, batchSize, valSplit, resume]()
    {
        (void)CALL_METHOD_RET_BOOL("TrainOCRModule", "biidi", isOnlineLearning, epochs, batchSize, valSplit, resume?1:0);
    });
}

bool BrainDeepLearnInterface::DeployModule(int cameraIndex, bool useHebbian, bool usePlanner)
{
    return RunPythonAsync([this, cameraIndex, useHebbian, usePlanner]()
    {
        (void)CALL_METHOD_RET_BOOL("DeployModule", "ibb", cameraIndex, useHebbian, usePlanner);
    });
}

bool BrainDeepLearnInterface::ExportParmFromCheckpoint(bool isOverride)
{
    return RunPythonAsync([this, isOverride]()
    {
        (void)CALL_METHOD_RET_BOOL("ExportParamsFromCheckpoint", "b", isOverride); 
    });
}

bool BrainDeepLearnInterface::Stop()
{ 
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("Stop"); 
    });
}

bool BrainDeepLearnInterface::Pause() 
{ 
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("Pause"); 
    });
}

bool BrainDeepLearnInterface::Resume() 
{ 
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("Resume"); 
    });
}

bool BrainDeepLearnInterface::ResetHebbianMemory()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("ResetHebbianMemory"); 
    });
}

bool BrainDeepLearnInterface::RunPythonAsync(PyTask task)
{
    if (brThreadRunning.load(std::memory_order_acquire)) 
    {
        PrintMessage("The task is in progress");
        return false;
    }

    brThreadRunning.store(true, std::memory_order_release);

    if (brThread.joinable()) 
    {
        brThread.join();
    }

    brThread = std::thread([this, task = std::move(task)]() mutable 
    {
        PyGILState_STATE g = PyGILState_Ensure();

        pyManager->EnsureStdoutRedirected();

        try 
        {
            task(); 
        } 
        catch (const std::exception& e) 
        {
            fprintf(stderr, "Exception in Python async task: %s\n", e.what());
        } catch (...) 
        {
            fprintf(stderr, "Unknown exception in Python async task\n");
        }

        PyGILState_Release(g);

        brThreadRunning.store(false, std::memory_order_release);

        PrintMessage("The task is complete");
    });

    return true;
}

bool BrainDeepLearnInterface::GetCurrentStatus(StatusMap& status) 
{
    if (!pManagerObj) return false;
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject* r = PyObject_CallMethod(pManagerObj, "GetCurrentStatus", nullptr);
    if (!r || !PyDict_Check(r)) 
    { 
        Py_XDECREF(r); PyGILState_Release(g); return false; 
    }
    status.clear();
    PyObject *key, *val; Py_ssize_t pos=0;
    while (PyDict_Next(r, &pos, &key, &val)) 
    {
        if (PyUnicode_Check(key)) 
        {
            std::string k = PyUnicode_AsUTF8(key);
            if (PyLong_Check(val)) status[k] = (int)PyLong_AsLong(val);
            else if (PyFloat_Check(val)) status[k] = PyFloat_AsDouble(val);
            else if (PyUnicode_Check(val)) status[k] = std::string(PyUnicode_AsUTF8(val));
        }
    }
    Py_DECREF(r);
    PyGILState_Release(g);
    return true;
}

bool BrainDeepLearnInterface::TestPerceptionModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestPerceptionModule");
    });
}

bool BrainDeepLearnInterface::TestAttentionModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestAttentionModule");
    });
}

bool BrainDeepLearnInterface::TestMemoryModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestMemoryModule");
    });
}

bool BrainDeepLearnInterface::TestDecisionModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestDecisionModule");
    });
}

bool BrainDeepLearnInterface::TestWorldModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestWorldModule");
    });
}

bool BrainDeepLearnInterface::TestValueEstimationModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestValueEstimationModule");
    });
}

bool BrainDeepLearnInterface::TestConsciousnessModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestConsciousnessModule");
    });
}

bool BrainDeepLearnInterface::TestIntentionModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestIntentionModule");
    });
}

bool BrainDeepLearnInterface::TestOCRModule()
{
    return RunPythonAsync([this]()
    {
        (void)CALL_METHOD_NOARG("TestOCRModule");
    });
}

bool BrainDeepLearnInterface::TestModuleTrain(bool onlineLearning)
{
    return RunPythonAsync([this, onlineLearning]()
    {
        (void)CALL_METHOD_RET_BOOL("TestModuleTrain", "b", onlineLearning);
    });
}

bool BrainDeepLearnInterface::TestOCRModuleTrain(bool onlineLearning)
{
    return RunPythonAsync([this, onlineLearning]()
    {
        (void)CALL_METHOD_RET_BOOL("TestOCRModuleTrain", "b", onlineLearning);
    });
}