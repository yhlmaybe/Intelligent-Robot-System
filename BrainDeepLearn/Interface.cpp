#include "Interface.h"

namespace
{
    struct JsonQueueObj
    {
        PyObject_HEAD
        IRSThreadTools::ThreadSafeQueue<std::string>* queue;
        bool valid;
    };

    static void JsonQueueDealloc(JsonQueueObj* self)
    {
        if (self)
        {
            self->valid = false;
        }
        Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
    }

    static bool JsonValueToString(PyObject* value, std::string& jsonText)
    {
        if (!value)
        {
            return false;
        }

        if (PyUnicode_Check(value))
        {
            const char* text = PyUnicode_AsUTF8(value);
            if (!text)
            {
                return false;
            }
            jsonText = text;
            return true;
        }

        PyObject* jsonModule = PyImport_ImportModule("json");
        if (!jsonModule)
        {
            return false;
        }

        PyObject* dumped = PyObject_CallMethod(jsonModule, "dumps", "O", value);
        Py_DECREF(jsonModule);
        if (!dumped)
        {
            return false;
        }

        const char* text = PyUnicode_AsUTF8(dumped);
        if (!text)
        {
            Py_DECREF(dumped);
            return false;
        }

        jsonText = text;
        Py_DECREF(dumped);
        return true;
    }

    static PyObject* JsonQueuePush(JsonQueueObj* self, PyObject* args)
    {
        if (!self || !self->valid || !self->queue)
        {
            Py_RETURN_FALSE;
        }

        PyObject* value = nullptr;
        if (!PyArg_ParseTuple(args, "O", &value))
        {
            return nullptr;
        }

        std::string jsonText;
        if (!JsonValueToString(value, jsonText))
        {
            return nullptr;
        }

        self->queue->push(jsonText);
        Py_RETURN_TRUE;
    }

    static PyObject* JsonQueueStop(JsonQueueObj* self, PyObject*)
    {
        if (!self || !self->valid || !self->queue)
        {
            Py_RETURN_FALSE;
        }

        self->queue->stop();
        Py_RETURN_TRUE;
    }

    static PyObject* JsonQueueReset(JsonQueueObj* self, PyObject*)
    {
        if (!self || !self->valid || !self->queue)
        {
            Py_RETURN_FALSE;
        }

        self->queue->reset();
        Py_RETURN_TRUE;
    }

    static PyObject* JsonQueueClearAndPush(JsonQueueObj* self, PyObject* args)
    {
        if (!self || !self->valid || !self->queue)
        {
            Py_RETURN_FALSE;
        }

        PyObject* value = nullptr;
        if (!PyArg_ParseTuple(args, "O", &value))
        {
            return nullptr;
        }

        std::string jsonText;
        if (!JsonValueToString(value, jsonText))
        {
            return nullptr;
        }

        self->queue->clearandpush(jsonText);
        Py_RETURN_TRUE;
    }

    static PyMethodDef JsonQueueMethods[] = {
        {"push", reinterpret_cast<PyCFunction>(JsonQueuePush), METH_VARARGS, nullptr},
        {"clearandpush", reinterpret_cast<PyCFunction>(JsonQueueClearAndPush), METH_VARARGS, nullptr},
        {"stop", reinterpret_cast<PyCFunction>(JsonQueueStop), METH_NOARGS, nullptr},
        {"reset", reinterpret_cast<PyCFunction>(JsonQueueReset), METH_NOARGS, nullptr},
        {nullptr, nullptr, 0, nullptr}
    };

    static PyTypeObject JsonQueueType =
        {
            PyVarObject_HEAD_INIT(NULL, 0) /* ob_base */
            nullptr,                       /* tp_name */
            sizeof(JsonQueueObj),         /* tp_basicsize */
            0,                           /* tp_itemsize */
            0,                           /* tp_dealloc */
            0,                           /* tp_vectorcall_offset / tp_print */
            0,                           /* tp_getattr */
            0,                           /* tp_setattr */
            0,                           /* tp_as_async */
            0,                           /* tp_repr */
            0,                           /* tp_as_number */
            0,                           /* tp_as_sequence */
            0,                           /* tp_as_mapping */
            0,                           /* tp_hash */
            0,                           /* tp_call */
            0,                           /* tp_str */
            0,                           /* tp_getattro */
            0,                           /* tp_setattro */
            0,                           /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT,          /* tp_flags */
            nullptr,                     /* tp_doc */
            0,                           /* tp_traverse */
            0,                           /* tp_clear */
            0,                           /* tp_richcompare */
            0,                           /* tp_weaklistoffset */
            0,                           /* tp_iter */
            0,                           /* tp_iternext */
            0,                           /* tp_methods */
            0,                           /* tp_members */
            0,                           /* tp_getset */
            0,                           /* tp_base */
            0,                           /* tp_dict */
            0,                           /* tp_descr_get */
            0,                           /* tp_descr_set */
            0,                           /* tp_dictoffset */
            0,                           /* tp_init */
            0,                           /* tp_alloc */
            0,                           /* tp_new */
            0,                           /* tp_free (Python 3.9+) */
            0,                           /* tp_is_gc */
            0,                           /* tp_bases */
            0,                           /* tp_mro */
            0,                           /* tp_cache */
            0,                           /* tp_subclasses */
            0,                           /* tp_weaklist */
            0,                           /* tp_del */
            0,                           /* tp_version_tag */
            0,                           /* tp_finalize */
            0,                           /* tp_vectorcall */
            0                            /* tp_print */
    };

    static bool EnsureJsonQueueType()
    {
        if (JsonQueueType.tp_name)
        {
            return true;
        }

        JsonQueueType.tp_name = const_cast<char*>("cpp.JsonQueue");
        JsonQueueType.tp_basicsize = sizeof(JsonQueueObj);
        JsonQueueType.tp_flags = Py_TPFLAGS_DEFAULT;
        JsonQueueType.tp_methods = JsonQueueMethods;
        JsonQueueType.tp_dealloc = reinterpret_cast<destructor>(JsonQueueDealloc);
        JsonQueueType.tp_new = PyType_GenericNew;
        JsonQueueType.tp_getattro = PyObject_GenericGetAttr;
        JsonQueueType.tp_setattro = PyObject_GenericSetAttr;

        return PyType_Ready(&JsonQueueType) == 0;
    }

    static PyObject* NewJsonQueueObject(IRSThreadTools::ThreadSafeQueue<std::string>* queue)
    {
        JsonQueueObj* obj = PyObject_New(JsonQueueObj, &JsonQueueType);
        if (!obj)
        {
            return nullptr;
        }

        obj->queue = queue;
        obj->valid = true;
        return reinterpret_cast<PyObject*>(obj);
    }
}

bool BrainDeepLearnInterface::ExtractUtf8String(PyObject* value, std::string& out)
{
    if (!value || !PyUnicode_Check(value))
    {
        return false;
    }

    const char* text = PyUnicode_AsUTF8(value);
    if (!text)
    {
        PyErr_Clear();
        return false;
    }

    out = text;
    return true;
}

double BrainDeepLearnInterface::ExtractFloatValue(PyObject* value, double defaultValue)
{
    if (!value)
    {
        return defaultValue;
    }

    if (PyFloat_Check(value))
    {
        const double parsed = PyFloat_AsDouble(value);
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return defaultValue;
        }
        return parsed;
    }

    if (PyLong_Check(value))
    {
        const long parsed = PyLong_AsLong(value);
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return defaultValue;
        }
        return static_cast<double>(parsed);
    }

    return defaultValue;
}

bool BrainDeepLearnInterface::ParseBitmapList(PyObject* bitmapObj, VisualStatus& status)
{
    status.width = 0;
    status.height = 0;
    status.bitmapRgb.clear();

    if (!bitmapObj || bitmapObj == Py_None)
    {
        return true;
    }

    if (!PyList_Check(bitmapObj))
    {
        return false;
    }

    const Py_ssize_t height = PyList_Size(bitmapObj);
    if (height <= 0)
    {
        return true;
    }

    PyObject* firstRow = PyList_GetItem(bitmapObj, 0);
    if (!firstRow || !PyList_Check(firstRow))
    {
        return false;
    }

    const Py_ssize_t width = PyList_Size(firstRow);
    if (width <= 0)
    {
        return true;
    }

    status.width = static_cast<int>(width);
    status.height = static_cast<int>(height);
    status.bitmapRgb.resize(static_cast<std::size_t>(width * height * 3), 0);

    for (Py_ssize_t y = 0; y < height; ++y)
    {
        PyObject* row = PyList_GetItem(bitmapObj, y);
        if (!row || !PyList_Check(row) || PyList_Size(row) != width)
        {
            return false;
        }

        for (Py_ssize_t x = 0; x < width; ++x)
        {
            PyObject* pixel = PyList_GetItem(row, x);
            if (!pixel || !PyList_Check(pixel))
            {
                return false;
            }

            const Py_ssize_t channels = PyList_Size(pixel);
            const std::size_t base = static_cast<std::size_t>((y * width + x) * 3);
            for (Py_ssize_t c = 0; c < 3; ++c)
            {
                long value = 0;
                if (c < channels)
                {
                    PyObject* channelValue = PyList_GetItem(pixel, c);
                    if (PyLong_Check(channelValue))
                    {
                        value = PyLong_AsLong(channelValue);
                        if (PyErr_Occurred())
                        {
                            PyErr_Clear();
                            value = 0;
                        }
                    }
                    else if (PyFloat_Check(channelValue))
                    {
                        value = static_cast<long>(PyFloat_AsDouble(channelValue));
                        if (PyErr_Occurred())
                        {
                            PyErr_Clear();
                            value = 0;
                        }
                    }
                }

                if (value < 0)
                {
                    value = 0;
                }
                else if (value > 255)
                {
                    value = 255;
                }

                status.bitmapRgb[base + static_cast<std::size_t>(c)] = static_cast<unsigned char>(value);
            }
        }
    }

    return true;
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

IRSThreadTools::ThreadSafeQueue<std::string> &BrainDeepLearnInterface::GetJsonQueue()
{
    static IRSThreadTools::ThreadSafeQueue<std::string> instance;
    return instance;
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

bool BrainDeepLearnInterface::DeployModule(bool usePlanner)
{
    return RunPythonAsync([this, usePlanner]()
    {
        (void)CALL_METHOD_RET_BOOL(
            "DeployModule", "b", usePlanner);
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

bool BrainDeepLearnInterface::SetJsonQueue()
{
    if (!pManagerObj)
    {
        return false;
    }

    PyGILState_STATE g = PyGILState_Ensure();

    if (!EnsureJsonQueueType())
    {
        PyErr_Print();
        PyGILState_Release(g);
        return false;
    }

    PyObject* queueObj = NewJsonQueueObject(&GetJsonQueue());
    if (!queueObj)
    {
        PyGILState_Release(g);
        return false;
    }

    PyObject* r = PyObject_CallMethod(pManagerObj, "SetJsonQueue", "O", queueObj);
    bool ok = r && PyObject_IsTrue(r);
    if (!r)
    {
        PyErr_Print();
    }

    Py_XDECREF(r);
    Py_DECREF(queueObj);
    PyGILState_Release(g);
    return ok;
}

bool BrainDeepLearnInterface::SetParameterReceiver(std::optional<double> reward, std::optional<double> done, std::optional<std::string> textExt)
{
    if (!pManagerObj)
    {
        return false;
    }

    PyGILState_STATE g = PyGILState_Ensure();

    PyObject* rewardObj = reward.has_value() ? PyFloat_FromDouble(*reward) : Py_None;
    PyObject* doneObj = done.has_value() ? PyFloat_FromDouble(*done) : Py_None;
    PyObject* textObj = textExt.has_value() ? PyUnicode_FromString(textExt->c_str()) : Py_None;

    if (rewardObj == Py_None)
    {
        Py_INCREF(Py_None);
    }
    if (doneObj == Py_None)
    {
        Py_INCREF(Py_None);
    }
    if (textObj == Py_None)
    {
        Py_INCREF(Py_None);
    }

    bool ok = false;
    if (rewardObj && doneObj && textObj)
    {
        PyObject* r = PyObject_CallMethod(pManagerObj, "SetParameterReceiver", "OOO", rewardObj, doneObj, textObj);
        ok = r && PyObject_IsTrue(r);
        if (!r)
        {
            PyErr_Print();
        }
        Py_XDECREF(r);
    }
    else
    {
        PyErr_Print();
    }

    Py_XDECREF(rewardObj);
    Py_XDECREF(doneObj);
    Py_XDECREF(textObj);
    PyGILState_Release(g);
    return ok;
}

bool BrainDeepLearnInterface::InitAgentHandle(bool usePlanner)
{
    return CALL_METHOD_RET_BOOL(
            "InitAgentHandle", "b", usePlanner);
}

bool BrainDeepLearnInterface::AgentHandleForward(
    StatusValue& result,
    const std::string& sensorPacketJson,
    const std::string& feedbackPayloadJson,
    std::optional<double> reward,
    std::optional<double> done)
{
    if (!pManagerObj)
    {
        return false;
    }

    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* rewardObj = nullptr;
    if (reward.has_value())
    {
        rewardObj = PyFloat_FromDouble(reward.value());
    }
    else
    {
        rewardObj = Py_None;
        Py_INCREF(rewardObj);
    }
    PyObject* doneObj = nullptr;
    if (done.has_value())
    {
        doneObj = PyFloat_FromDouble(done.value());
    }
    else
    {
        doneObj = Py_None;
        Py_INCREF(doneObj);
    }
    PyObject* response = PyObject_CallMethod(
        pManagerObj,
        "AgentHandleForwardJson",
        "OOss",
        rewardObj,
        doneObj,
        sensorPacketJson.c_str(),
        feedbackPayloadJson.c_str());
    Py_DECREF(rewardObj);
    Py_DECREF(doneObj);

    bool ok = false;
    if (response && PyUnicode_Check(response))
    {
        const char* text = PyUnicode_AsUTF8(response);
        if (text)
        {
            result = std::string(text);
            ok = true;
        }
    }
    else if (!response)
    {
        PyErr_Print();
    }
    Py_XDECREF(response);
    PyGILState_Release(gil);
    return ok;
}

bool BrainDeepLearnInterface::ResetAgentHandleHebbian()
{
    return CALL_METHOD_NOARG("ResetAgentHandleHebbian");
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

bool BrainDeepLearnInterface::GetCurrentVisualStatus(VisualStatus& status)
{
    if (!pManagerObj)
    {
        return false;
    }

    PyGILState_STATE g = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(pManagerObj, "GetCurrentStatus", nullptr);
    bool ok = false;
    VisualStatus parsed;

    if (result && PyDict_Check(result))
    {
        PyObject* visual = PyDict_GetItemString(result, "visual");
        if (!visual || visual == Py_None)
        {
            status = VisualStatus();
            ok = true;
        }
        else if (PyDict_Check(visual))
        {
            PyObject* textValue = PyDict_GetItemString(visual, "text");
            ExtractUtf8String(textValue, parsed.text);
            parsed.updatedAt = ExtractFloatValue(PyDict_GetItemString(visual, "updated_at"), 0.0);
            ok = ParseBitmapList(PyDict_GetItemString(visual, "bitmap"), parsed);
            if (ok)
            {
                status = std::move(parsed);
            }
        }
    }
    else if (!result)
    {
        PyErr_Print();
    }

    Py_XDECREF(result);
    PyGILState_Release(g);
    return ok;
}

bool BrainDeepLearnInterface::SetVisualStateEnabled(bool enabled)
{
    return CALL_METHOD_RET_BOOL("SetVisualStateEnabled", "i", enabled ? 1 : 0);
}

bool BrainDeepLearnInterface::SetOverrideCheckpointWithModuleParams(bool enabled)
{
    return CALL_METHOD_RET_BOOL("SetOverrideCheckpointWithModuleParams", "i", enabled ? 1 : 0);
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
}

bool BrainDeepLearnInterface::TestOCRModuleTrain()
{
    return CALL_METHOD_NOARG("TestOCRModuleTrain");
}

bool BrainDeepLearnInterface::TestOCRRecognitionTrain()
{
    return CALL_METHOD_NOARG("TestOCRRecognitionTrain");
}
