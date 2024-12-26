#include "GeometricManager.h"

Point3D::Point3D(double x, double y, double z) : x(x), y(y), z(z) { }

double Point3D::Distance(const Point3D& point) const 
{
    return std::sqrt(std::pow(point.x - x, 2) + std::pow(point.y - y, 2) + std::pow(point.z - z, 2));
}

Vector3D::Vector3D(double x, double y, double z) : x(x), y(y), z(z) { }

Vector3D Vector3D::operator+(const Vector3D& vec) const 
{
    return Vector3D(x + vec.x, y + vec.y, z + vec.z);
}

Vector3D Vector3D::operator-(const Vector3D& vec) const 
{
    return Vector3D(x - vec.x, y - vec.y, z - vec.z);
}

double Vector3D::Dot(const Vector3D& vec) const 
{
    return x * vec.x + y * vec.y + z * vec.z;
}

Vector3D Vector3D::Cross(const Vector3D& vec) const 
{
    return Vector3D(
        y * vec.z - z * vec.y,
        z * vec.x - x * vec.z,
        x * vec.y - y * vec.x
    );
}

double Vector3D::Length() const 
{
    return std::sqrt(x * x + y * y + z * z);
}

Vector3D Vector3D::Normalize() const 
{
    double len = Length();
    if (len > 0) 
    {
        return Vector3D(x / len, y / len, z / len);
    }
    return *this;
}