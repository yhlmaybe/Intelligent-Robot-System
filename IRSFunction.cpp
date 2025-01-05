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

