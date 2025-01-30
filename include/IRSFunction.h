#ifndef IRSFUNCTION_H
#define IRSFUNCTION_H

#include <string>
#include <vector>

extern void IRS_MESSAGE(std::string message);

extern void IRS_MESSAGE(const char* format, ...);

template <typename T>
T* VectorToArray(std::vector<T>& vec);

#endif