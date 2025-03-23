#ifndef IRSFUNCTION_H
#define IRSFUNCTION_H

#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <mutex>

extern void IRS_MESSAGE(std::string message);

extern void IRS_MESSAGE(const char *format, ...);

template <typename T>
T *VectorToArray(std::vector<T> &vec);

template <typename T>
class ThreadSafeQueue
{
public:
    void push(const T &item);
    T pop();
    void stop();
    void reset();

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    bool stopped_ = false;
};

template <typename T>
void ThreadSafeQueue<T>::push(const T &item)
{
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
    cond_.notify_one();
}

template <typename T>
T ThreadSafeQueue<T>::pop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]()
               { return !queue_.empty() || stopped_; });
    if (stopped_)
    {
        return T{};
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
void ThreadSafeQueue<T>::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cond_.notify_all();
}

template <typename T>
void ThreadSafeQueue<T>::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = false;
    while (!queue_.empty())
        queue_.pop();
}

class IRSThreadBase 
{
public:
    IRSThreadBase();
    virtual ~IRSThreadBase();

    void Start();
    void Stop();
    void Reset();

protected:
    virtual void ExecuteTask() = 0; 

private:
    std::thread worker_thread_;
    std::atomic<bool> is_running_;
    mutable std::mutex mutex_;

    void Run();
};


#endif