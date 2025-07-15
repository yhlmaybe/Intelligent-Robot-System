#include "Interface.h"



struct RedirectObj { PyObject_HEAD; BrainDeepLearnInterface::PrintCB* cb; };

static PyTypeObject RedirectType;

static PyObject* RWrite(RedirectObj* self, PyObject* args) 
{
    const char* buf;
    Py_ssize_t len;
    if (!PyArg_ParseTuple(args, "s#", &buf, &len)) return nullptr;
    if (self->cb && *self->cb) 
    {
        std::string message = std::string(buf, len);
        auto callback = *self->cb;
        if(callback)
        {
            callback(message);
        }
    }
    Py_RETURN_NONE;
}

static PyObject* RFlush(RedirectObj*, PyObject*) { Py_RETURN_NONE; }

static PyTypeObject RedirectType = 
{
    PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyMethodDef RMethods[] = 
{
    {"write", (PyCFunction)RWrite, METH_VARARGS, nullptr},
    {"flush", (PyCFunction)RFlush, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

static bool EnsureRedirectType() 
{
    if (RedirectType.tp_name) 
        return true;

    RedirectType.tp_name = (char *)"cpp.Redirect";
    RedirectType.tp_basicsize = sizeof(RedirectObj);
    RedirectType.tp_flags = Py_TPFLAGS_DEFAULT;
    RedirectType.tp_methods = RMethods;

    return PyType_Ready(&RedirectType) == 0;
}

static PyObject* NewRedirect(BrainDeepLearnInterface::PrintCB* cb) 
{
    RedirectObj* obj = PyObject_New(RedirectObj, &RedirectType);
    obj->cb = cb;
    return reinterpret_cast<PyObject*>(obj);
}

void BrainDeepLearnInterface::Init() 
{
    if (!Py_IsInitialized()) throw std::runtime_error("Failed to init Python");

    PyGILState_STATE g = PyGILState_Ensure();

    pModule = PyImport_ImportModule("ManagerFunction");

    if (!pModule) 
    { 
        PyErr_Print(); 
        throw std::runtime_error("Import ManagerFunction failed"); 
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
    if (pRedirectObj) UninstallRedirect();
    PyGILState_STATE g = PyGILState_Ensure();
    Py_XDECREF(pManagerObj);
    Py_XDECREF(pModule);
    PyGILState_Release(g);
}
bool BrainDeepLearnInterface::SetPrintCallback(PrintCB cb) 
{
    printCb = std::move(cb);
    return InstallRedirect();
}
bool BrainDeepLearnInterface::InstallRedirect() 
{
    if (!EnsureRedirectType()) return false;
    PyGILState_STATE g = PyGILState_Ensure();
    pRedirectObj = NewRedirect(&printCb);
    PyObject* sys = PyImport_ImportModule("sys");

    PySys_SetObject("stdout", pRedirectObj);
    PySys_SetObject("stderr", pRedirectObj);
    Py_DECREF(sys);
    PyGILState_Release(g);
    return true;
}

void BrainDeepLearnInterface::UninstallRedirect() 
{
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject* sys = PyImport_ImportModule("sys");
    PySys_SetObject("stdout", PySys_GetObject("__stdout_orig"));
    PySys_SetObject("stderr", PySys_GetObject("__stderr_orig"));
    Py_DECREF(sys);
    Py_DECREF(pRedirectObj);
    pRedirectObj = nullptr;
    PyGILState_Release(g);
}



bool BrainDeepLearnInterface::StartTraining(std::string& root, int epochs, int batchSize, double valSplit, int imagineHorizon, bool resume) 
{
    return CALL_METHOD_RET_BOOL("StartTraining", "siiidi", root.c_str(), epochs, batchSize, valSplit, imagineHorizon, resume?1:0);
}

bool BrainDeepLearnInterface::StopTraining() { return CALL_METHOD_NOARG("StopTraining"); }

bool BrainDeepLearnInterface::PauseTraining() { return CALL_METHOD_NOARG("PauseTraining"); }

bool BrainDeepLearnInterface::ResumeTraining() { return CALL_METHOD_NOARG("ResumeTraining"); }

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