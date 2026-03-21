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

bool BrainDeepLearnInterface::TrainOCRModule(int epochs, int batchSize, double valSplit, bool resume) 
{
    return RunPythonAsync([this, epochs, batchSize, valSplit, resume]()
    {
        (void)CALL_METHOD_RET_BOOL("TrainOCRModule", "iidi", epochs, batchSize, valSplit, resume?1:0);
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
    return CALL_METHOD_RET_STATUSMAP(status, "GetCurrentStatus");
}

bool BrainDeepLearnInterface::SetBasicParameters(std::string name, std::string value)
{
    return CALL_METHOD_RET_BOOL("SetBasicParameters", "ss", name.c_str(), value.c_str());
}

bool BrainDeepLearnInterface::GetBasicParameters(std::string name, StatusValue& value)
{
    return CALL_METHOD_RET_STATUSVALUE(value, "GetBasicParameters", "s", name.c_str());
}

bool BrainDeepLearnInterface::GetBasicParametersDict(StatusMap& parameters)
{
    return CALL_METHOD_RET_STATUSMAP(parameters, "GetBasicParametersDict");
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
    return CALL_METHOD_RET_BOOL("TestModuleTrain", "b", onlineLearning);
    //return RunPythonAsync([this, onlineLearning]()
    //{
    //    (void)CALL_METHOD_RET_BOOL("TestModuleTrain", "b", onlineLearning);
    //});
}

bool BrainDeepLearnInterface::TestOCRModuleTrain()
{
    return CALL_METHOD_NOARG("TestOCRModuleTrain");
    //return RunPythonAsync([this, onlineLearning]()
    //{
    //    (void)CALL_METHOD_RET_BOOL("TestOCRModuleTrain", "b", onlineLearning);
    //});
}

bool BrainDeepLearnInterface::TestOCRRecognitionTrain()
{
    return CALL_METHOD_NOARG("TestOCRRecognitionTrain");
    // return RunPythonAsync([this, onlineLearning]()
    //{
    //     (void)CALL_METHOD_RET_BOOL("TestOCRRecognitionTrain", "b", onlineLearning);
    // });
}
