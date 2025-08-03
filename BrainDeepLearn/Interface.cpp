#include "Interface.h"

static std::mutex print_callback_mutex;

struct RedirectObj 
{ 
    PyObject_HEAD 
    BrainDeepLearnInterface::PrintCB* cb;
    std::atomic<bool> valid{true};  
};

static void Redirect_dealloc(RedirectObj* self) 
{
    self->valid.store(false, std::memory_order_release);
    
    std::lock_guard<std::mutex> lock(print_callback_mutex);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* RWrite(RedirectObj* self, PyObject* args) 
{
    const char* buf;
    Py_ssize_t len;
    if (!PyArg_ParseTuple(args, "s#", &buf, &len)) return nullptr;
    
    if (!self->valid.load(std::memory_order_acquire)) 
        Py_RETURN_NONE;
    
    BrainDeepLearnInterface::PrintCB callback = nullptr;
    {
        std::lock_guard<std::mutex> lock(print_callback_mutex);
        
        if (!self->valid.load(std::memory_order_relaxed) || !self->cb || !*self->cb) 
        {
            Py_RETURN_NONE;
        }
        callback = *self->cb;
    }
    
    if (callback) 
    {
        try 
        {
            callback(std::string(buf, len));
        } 
        catch (...) 
        {
            PySys_WriteStderr("Exception in print callback\n");
        }
    }
    
    Py_RETURN_NONE;
}

static PyObject* RFlush(RedirectObj*, PyObject*) 
{ 
    Py_RETURN_NONE; 
}

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
    RedirectType.tp_dealloc = (destructor)Redirect_dealloc;  
    RedirectType.tp_new = PyType_GenericNew;  

    return PyType_Ready(&RedirectType) == 0;
}


static PyObject* NewRedirect(BrainDeepLearnInterface::PrintCB* cb) 
{
    RedirectObj* obj = PyObject_New(RedirectObj, &RedirectType);
    if (obj) 
    {
        obj->cb = cb;
        obj->valid.store(true, std::memory_order_release);
    }
    return reinterpret_cast<PyObject*>(obj);
}

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
    
    PyObject* sys = PyImport_ImportModule("sys");
    originalStdout = PySys_GetObject("stdout");
    originalStderr = PySys_GetObject("stderr");
    Py_XINCREF(originalStdout); 
    Py_XINCREF(originalStderr);
    Py_DECREF(sys);
    
    PyGILState_Release(g);
}

BrainDeepLearnInterface::BrainDeepLearnInterface() 
{
    Init();
}

BrainDeepLearnInterface::~BrainDeepLearnInterface() 
{
    UninstallRedirect();  
    
    PyGILState_STATE g = PyGILState_Ensure();
    Py_XDECREF(pManagerObj);
    Py_XDECREF(pModule);
    
    Py_XDECREF(originalStdout);
    Py_XDECREF(originalStderr);
    
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
    
    PyObject* newRedirect = NewRedirect(&printCb);
    if (!newRedirect) 
    {
        PyGILState_Release(g);
        return false;
    }
    
    if (pRedirectObj) 
    {
        Py_DECREF(pRedirectObj);
    }
    pRedirectObj = newRedirect;
    
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
    
    if (pRedirectObj) {
        PyObject* sys = PyImport_ImportModule("sys");
        if (originalStdout) 
        {
            PySys_SetObject("stdout", originalStdout);
        }
        if (originalStderr) 
        {
            PySys_SetObject("stderr", originalStderr);
        }
        Py_DECREF(sys);
        
        RedirectObj* robj = reinterpret_cast<RedirectObj*>(pRedirectObj);
        robj->valid.store(false, std::memory_order_release); 
        
        Py_DECREF(pRedirectObj);
        pRedirectObj = nullptr;
    }
    
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
        if (PyErr_Occurred()) {
        // 获取异常详细信息
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        
        // 转换异常为字符串
        PyObject *str = PyObject_Str(value);
        const char *errMsg = PyUnicode_AsUTF8(str);
        printf("Python exception: %s\n", errMsg);
        
        // 清理异常对象
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        Py_XDECREF(str);
    }
    return success;
}