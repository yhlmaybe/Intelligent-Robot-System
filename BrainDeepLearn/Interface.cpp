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

BrainDeepLearnInterface::BrainDeepLearnInterface() 
{
    Init();
}

BrainDeepLearnInterface::~BrainDeepLearnInterface() 
{
    PyGILState_STATE g = PyGILState_Ensure();
    Py_XDECREF(pManagerObj);
    Py_XDECREF(pModule);
    
    PyGILState_Release(g);
}

bool BrainDeepLearnInterface::StartTraining(std::string& root, int epochs, int batchSize, double valSplit, int imagineHorizon, bool resume) 
{
    return CALL_METHOD_RET_BOOL("StartTraining", "siiidi", root.c_str(), epochs, batchSize, valSplit, imagineHorizon, resume?1:0);
}

bool BrainDeepLearnInterface::StopTraining()
{ 
    return CALL_METHOD_NOARG("StopTraining"); 
}

bool BrainDeepLearnInterface::PauseTraining() 
{ 
    return CALL_METHOD_NOARG("PauseTraining"); 
}

bool BrainDeepLearnInterface::ResumeTraining() 
{ 
    return CALL_METHOD_NOARG("ResumeTraining"); 
}

bool BrainDeepLearnInterface::GetTrainingStatus(StatusMap& status) 
{
    if (!pManagerObj) return false;
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject* r = PyObject_CallMethod(pManagerObj, "GetTrainingStatus", nullptr);
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
            if (PyLong_Check(val))       status[k] = (int)PyLong_AsLong(val);
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
    bool success = CALL_METHOD_NOARG("TestPerceptionModule");
    return success;
}

bool BrainDeepLearnInterface::TestAttentionModule()
{
    bool success = CALL_METHOD_NOARG("TestAttentionModule");
    return success;
}

bool BrainDeepLearnInterface::TestMemoryModule()
{
    bool success = CALL_METHOD_NOARG("TestMemoryModule");
    return success;
}

bool BrainDeepLearnInterface::TestDecisionModule()
{
    bool success = CALL_METHOD_NOARG("TestDecisionModule");
    return success;
}

bool BrainDeepLearnInterface::TestWorldModule()
{
    bool success = CALL_METHOD_NOARG("TestWorldModule");
    return success;
}