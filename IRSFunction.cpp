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

namespace IRSThreadTools
{
    IRSThreadBase::IRSThreadBase() : is_running_(false), startedTaskCount_(0) {}

    IRSThreadBase::~IRSThreadBase()
    {
        Stop();
    }

    void IRSThreadBase::Start()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        CreateThreadsForNewTasks();

        if (!is_running_.load())
        {
            is_running_ = true;
            lock.unlock();
            cv_.notify_all();
        }
        else
        {
            lock.unlock();
            cv_.notify_all();
        }
    }

    void IRSThreadBase::Stop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (threads_.empty() || !is_running_.load())
        {
            return;
        }

        is_running_.store(false);
        lock.unlock();
        cv_.notify_all();

        for (auto &t : threads_)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
        threads_.clear();
        startedTaskCount_ = 0;
    }

    void IRSThreadBase::RegisterTask(std::function<void()> task)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
    }

    void IRSThreadBase::AddTaskAndStart(std::function<void()> task)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));

        if (is_running_.load())
        {
            CreateThreadsForNewTasks();
            lock.unlock();
            cv_.notify_all();
        }
    }

    void IRSThreadBase::CreateThreadsForNewTasks()
    {
        while (startedTaskCount_ < tasks_.size())
        {
            auto &task = tasks_[startedTaskCount_];
            threads_.emplace_back([this, &task]
                                  {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return is_running_.load();
                        });
                    }
                    while (is_running_.load()) 
                    {
                        task(); 
                    } });

            startedTaskCount_++;
        }
    }
}
