#include "Interface.h"

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
