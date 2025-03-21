#include "include/IRSFunction.h"

template <typename T>
T* VectorToArray(std::vector<T>& vec) {
    size_t size = vec.size();
    T* arr = new T[size];
    for (size_t i = 0; i < size; ++i) {
        arr[i] = vec[i];
    }
    return arr;
}

template int* VectorToArray<int>(std::vector<int>&);

template double* VectorToArray<double>(std::vector<double>&);

template std::string* VectorToArray<std::string>(std::vector<std::string>&);

IRSThreadBase::IRSThreadBase() : is_running_(false) { }

IRSThreadBase::~IRSThreadBase() 
{
    Stop();
}

void IRSThreadBase::Start() 
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(!is_running_)
    {
        worker_thread_ = std::thread(&IRSThreadBase::Run, this);
        is_running_.store(true);
    }
}

void IRSThreadBase::Stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_running_)
    {
        is_running_.store(false);
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }
    }
}

void IRSThreadBase::Reset() 
{
    std::lock_guard<std::mutex> lock(mutex_);
    Stop();
    Start();
}

void IRSThreadBase::Run() 
{
    while (!is_running_.load()) 
    {
        ExecuteTask();
    }
}