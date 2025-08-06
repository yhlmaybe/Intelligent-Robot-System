#ifndef PYTHONINTERACTION_H
#define PYTHONINTERACTION_H

#include <python3.8/Python.h>
#include <python3.8/frameobject.h> 
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


namespace PythonInteraction
{
    class Manager
    {
    public:
        using PrintCB = std::function<void(const char *, std::size_t, const std::string&)>;

        Manager();
        ~Manager();

        bool SetPrintCallback(PrintCB cb);

    private:
        PrintCB printCb;
        
        PyObject *pRedirectObj = nullptr;

        PyObject *originalStdout = nullptr;

        void Init();
        bool InstallRedirect();
        void UninstallRedirect();
    };
}

#endif