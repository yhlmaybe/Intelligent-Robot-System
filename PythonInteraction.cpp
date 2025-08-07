#include "include/PythonInteraction.h"

namespace PythonInteraction
{

    static std::mutex print_callback_mutex;

    struct RedirectObj
    {
        PyObject_HEAD
        Manager::PrintCB *cb;
        std::atomic<bool> valid{true};
    };

    static void Redirect_dealloc(RedirectObj *self)
    {
        self->valid.store(false, std::memory_order_release);

        std::lock_guard<std::mutex> lock(print_callback_mutex);
        Py_TYPE(self)->tp_free((PyObject *)self);
    }

    static PyObject *RWrite(RedirectObj *self, PyObject *args)
    {
        const char *buf;
        Py_ssize_t len;
        if (!PyArg_ParseTuple(args, "s#", &buf, &len))
            return nullptr;

        PyThreadState *tstate = PyThreadState_GET();
        PyFrameObject *frame = tstate->frame;
        while (frame && !frame->f_globals)
            frame = frame->f_back;

        std::string modname = "<unknown>";
        if (frame)
        {
            PyObject *name_obj = PyDict_GetItemString(frame->f_globals, "__name__");
            if (name_obj && PyUnicode_Check(name_obj))
            {
                modname = PyUnicode_AsUTF8(name_obj);
            }
        }

        if (!self->valid.load(std::memory_order_acquire))
            Py_RETURN_NONE;

        Manager::PrintCB cb_copy;
        {
            std::lock_guard<std::mutex> lk(print_callback_mutex);
            if (!self->cb || !*self->cb)
                Py_RETURN_NONE;
            cb_copy = *self->cb;
        }

        try
        {
            cb_copy(buf, static_cast<std::size_t>(len), modname);
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "[PrintCB ex %s] %s\n", modname.c_str(), e.what());
        }
        catch (...)
        {
            fprintf(stderr, "[PrintCB unknown ex %s]\n", modname.c_str());
        }
        Py_RETURN_NONE;
    }

    static PyObject *RFlush(RedirectObj *, PyObject *)
    {
        Py_RETURN_NONE;
    }

    static PyTypeObject RedirectType =
        {
            PyVarObject_HEAD_INIT(NULL, 0)             /* ob_base */
            nullptr,                                    /* tp_name */
            sizeof(RedirectObj),                       /* tp_basicsize */
            0,                                         /* tp_itemsize */
            (destructor)Redirect_dealloc,              /* tp_dealloc */
            0,                                         /* tp_vectorcall_offset / tp_print */
            0,                                         /* tp_getattr */
            0,                                         /* tp_setattr */
            0,                                         /* tp_as_async */
            0,                                         /* tp_repr */
            0,                                         /* tp_as_number */
            0,                                         /* tp_as_sequence */
            0,                                         /* tp_as_mapping */
            0,                                         /* tp_hash */
            0,                                         /* tp_call */
            0,                                         /* tp_str */
            0,                                         /* tp_getattro */
            0,                                         /* tp_setattro */
            0,                                         /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT,                        /* tp_flags */
            "RedirectObj for capturing Python prints", /* tp_doc */
            0,                                         /* tp_traverse */
            0,                                         /* tp_clear */
            0,                                         /* tp_richcompare */
            0,                                         /* tp_weaklistoffset */
            0,                                         /* tp_iter */
            0,                                         /* tp_iternext */
            0,                                         /* tp_methods */
            0,                                         /* tp_members */
            0,                                         /* tp_getset */
            0,                                         /* tp_base */
            0,                                         /* tp_dict */
            0,                                         /* tp_descr_get */
            0,                                         /* tp_descr_set */
            0,                                         /* tp_dictoffset */
            0,                                         /* tp_init */
            0,                                         /* tp_alloc */
            0,                                         /* tp_new */
            0,                                         /* tp_free (Python 3.9+) */
            0,                                         /* tp_is_gc */
            0,                                         /* tp_bases */
            0,                                         /* tp_mro */
            0,                                         /* tp_cache */
            0,                                         /* tp_subclasses */
            0,                                         /* tp_weaklist */
            0,                                         /* tp_del */
            0,                                         /* tp_version_tag */
            0,                                         /* tp_finalize */
            0,                                         /* tp_vectorcall */
            0                                          /* tp_print */
    };

    static PyMethodDef RMethods[] ={
            {"write", (PyCFunction)RWrite, METH_VARARGS, nullptr},
            {"flush", (PyCFunction)RFlush, METH_NOARGS, nullptr},
            {nullptr, nullptr, 0, nullptr}};

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

    static PyObject *NewRedirect(Manager::PrintCB *cb)
    {
        RedirectObj *obj = PyObject_New(RedirectObj, &RedirectType);
        if (obj)
        {
            obj->cb = cb;
            obj->valid.store(true, std::memory_order_release);
        }
        return reinterpret_cast<PyObject *>(obj);
    }

    void Manager::Init()
    {
        PyGILState_STATE g = PyGILState_Ensure();

        PyObject *sys = PyImport_ImportModule("sys");
        originalStdout = PySys_GetObject("stdout");
        Py_XINCREF(originalStdout);
        Py_DECREF(sys);

        PyGILState_Release(g);
    }

    Manager::Manager()
    {
        Py_Initialize();
        PyEval_InitThreads();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./ServoControl')");
        PyRun_SimpleString("sys.path.append('./BrainDeepLearn')");

        Init();
    }

    Manager::~Manager()
    {
        UninstallRedirect();

        PyGILState_STATE g = PyGILState_Ensure();

        Py_XDECREF(originalStdout);

        PyGILState_Release(g);

        Py_Finalize();
    }

    bool Manager::SetPrintCallback(PrintCB cb)
    {
        printCb = std::move(cb);
        return InstallRedirect();
    }

    bool Manager::InstallRedirect()
    {
        if (!EnsureRedirectType())
            return false;

        PyGILState_STATE g = PyGILState_Ensure();

        PyObject *newRedirect = NewRedirect(&printCb);
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

        PyObject *sys = PyImport_ImportModule("sys");
        PySys_SetObject("stdout", pRedirectObj);
        Py_DECREF(sys);

        PyGILState_Release(g);
        return true;
    }

    void Manager::UninstallRedirect()
    {
        PyGILState_STATE g = PyGILState_Ensure();

        if (pRedirectObj)
        {
            PyObject *sys = PyImport_ImportModule("sys");
            if (originalStdout)
            {
                PySys_SetObject("stdout", originalStdout);
            }
            Py_DECREF(sys);

            RedirectObj *robj = reinterpret_cast<RedirectObj *>(pRedirectObj);
            robj->valid.store(false, std::memory_order_release);

            Py_DECREF(pRedirectObj);
            pRedirectObj = nullptr;
        }

        PyGILState_Release(g);
    }
}