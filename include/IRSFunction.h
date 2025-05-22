#ifndef IRSFUNCTION_H
#define IRSFUNCTION_H

#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <map>
#include <type_traits>
#include <tuple>

#include "MsgManager.h"

extern void IRS_MESSAGE(std::string message);

extern void IRS_XML_MESSAGE(std::string message, MessageFunction fun = MessageFunction::Default);

extern void IRS_MESSAGE(const char *format, ...);

template <typename T>
T *VectorToArray(std::vector<T> &vec);

namespace IRSThreadTools
{
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

        void RegisterTask(std::function<void()> task);
        void AddTaskAndStart(std::function<void()> task);

    private:
        std::vector<std::thread> threads_;
        std::atomic<bool> is_running_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::vector<std::function<void()>> tasks_;

        size_t startedTaskCount_;

        void CreateThreadsForNewTasks();
    };
}

namespace XMLSerializer
{
    template <typename Class, typename T>
    struct Member
    {
        const char *name;
        T Class::*ptr;
    };

    template <typename T>
    struct ClassMeta
    {
        static constexpr auto members = std::make_tuple();
        static constexpr const char *xml_root_name = "root";
    };

// 注册宏
#define REGISTER_CLASS(class_type, ...)                               \
    template <>                                                       \
    struct ClassMeta<class_type>                                      \
    {                                                                 \
        static constexpr auto members = std::make_tuple(__VA_ARGS__); \
        static constexpr const char *xml_root_name = #class_type;     \
    };

#define MEMBER(ptr) \
    ::XMLSerializer::Member<std::remove_reference_t<decltype(*this)>, decltype(ptr)> { #ptr, ptr }

    template <typename T>
    void SerializeValue(pugi::xml_node &node, const T &value);

    template <typename T>
    void DeserializeValue(const pugi::xml_node &node, T &value);

    template <typename Obj, size_t I = 0>
    void SerializeImpl(pugi::xml_node &parent, const Obj &obj)
    {
        if constexpr (I < std::tuple_size_v<decltype(ClassMeta<Obj>::members)>)
        {
            const auto &member = std::get<I>(ClassMeta<Obj>::members);
            auto child = parent.append_child(member.name);
            SerializeValue(child, obj.*(member.ptr));
            SerializeImpl<Obj, I + 1>(parent, obj);
        }
    }

    template <typename T>
    void SerializeValue(pugi::xml_node &node, const T &value)
    {
        if constexpr (std::is_arithmetic_v<T>)
        {
            node.text().set(value);
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            node.text().set(value.c_str());
        }
    }

    template <typename T>
    void SerializeValue(pugi::xml_node &node, const std::vector<T> &vec)
    {
        for (const auto &item : vec)
        {
            auto child = node.append_child("item");
            SerializeValue(child, item);
        }
    }

    template <typename K, typename V>
    void SerializeValue(pugi::xml_node &node, const std::map<K, V> &map)
    {
        for (const auto &pair : map)
        {
            auto entry = node.append_child("entry");
            entry.append_attribute("key").set_value(pair.first);
            SerializeValue(entry, pair.second);
        }
    }

    template <typename Obj, size_t I = 0>
    void DeserializeImpl(const pugi::xml_node &parent, Obj &obj)
    {
        if constexpr (I < std::tuple_size_v<decltype(ClassMeta<Obj>::members)>)
        {
            const auto &member = std::get<I>(ClassMeta<Obj>::members);
            auto child = parent.child(member.name);
            if (child)
            {
                DeserializeValue(child, obj.*(member.ptr));
            }
            DeserializeImpl<Obj, I + 1>(parent, obj);
        }
    }

    template <typename T>
    void DeserializeValue(const pugi::xml_node &node, T &value)
    {
        if constexpr (std::is_arithmetic_v<T>)
        {
            if constexpr (std::is_same_v<T, int>)
            {
                value = node.text().as_int();
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                value = node.text().as_double();
            }
            else if constexpr (std::is_same_v<T, bool>)
            {
                value = node.text().as_bool();
            }
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            value = node.text().as_string();
        }
    }

    template <typename T>
    void DeserializeValue(const pugi::xml_node &node, std::vector<T> &vec)
    {
        for (auto child : node.children("item"))
        {
            T item;
            DeserializeValue(child, item);
            vec.push_back(item);
        }
    }

    template <typename K, typename V>
    void DeserializeValue(const pugi::xml_node &node, std::map<K, V> &map)
    {
        for (auto entry : node.children("entry"))
        {
            K key;
            if constexpr (std::is_same_v<K, int>)
            {
                key = entry.attribute("key").as_int();
            }
            else if constexpr (std::is_same_v<K, std::string>)
            {
                key = entry.attribute("key").as_string();
            }
            V val;
            DeserializeValue(entry, val);
            map[key] = val;
        }
    }

    template <typename T>
    std::string ToXml(const T &obj)
    {
        pugi::xml_document doc;
        auto root = doc.append_child(ClassMeta<T>::xml_root_name);
        SerializeImpl(root, obj);

        std::ostringstream oss;
        doc.save(oss, "  ", pugi::format_indent);
        return oss.str();
    }

    template <typename T>
    T FromXml(const std::string &xml)
    {
        pugi::xml_document doc;
        if (!doc.load_string(xml.c_str()))
        {
            throw std::runtime_error("XML parse error");
        }

        auto root = doc.child(ClassMeta<T>::xml_root_name);
        if (!root)
        {
            throw std::runtime_error("Missing root node");
        }

        T obj;
        DeserializeImpl(root, obj);
        return obj;
    }
}

#endif