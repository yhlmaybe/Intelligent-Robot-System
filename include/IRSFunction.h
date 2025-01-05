#ifndef IRSFUNCTION
#define IRSFUNCTION

#include <string>
#include <vector>

extern void IRS_MESSAGE(std::string message);

template <typename T>
T* VectorToArray(std::vector<T>& vec);

#endif