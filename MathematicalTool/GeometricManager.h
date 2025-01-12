#ifndef GEOMETRICMANAGER_H
#define GEOMETRICMANAGER_H

#include <complex.h>

class Vector3D
{
public:
    double x, y, z;

    Vector3D(double x = 0.0, double y = 0.0, double z = 0.0);
    Vector3D operator+(const Vector3D& vec) const;
    Vector3D operator-(const Vector3D& vec) const;
    double Dot(const Vector3D& vec) const;
    Vector3D Cross(const Vector3D& vec) const;
    double Length() const;
    Vector3D Normalize() const;
};

class Point3D
{
public:
    double x, y, z;

    Point3D(double x = 0.0, double y = 0.0, double z = 0.0);

    double Distance(const Point3D& point) const;
};

#endif