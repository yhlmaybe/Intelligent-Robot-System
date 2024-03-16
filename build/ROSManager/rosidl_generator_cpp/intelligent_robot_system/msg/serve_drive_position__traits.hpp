// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#ifndef INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__TRAITS_HPP_
#define INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__TRAITS_HPP_

#include "intelligent_robot_system/msg/serve_drive_position__struct.hpp"
#include <rosidl_generator_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<intelligent_robot_system::msg::ServeDrivePosition>()
{
  return "intelligent_robot_system::msg::ServeDrivePosition";
}

template<>
struct has_fixed_size<intelligent_robot_system::msg::ServeDrivePosition>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<intelligent_robot_system::msg::ServeDrivePosition>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<intelligent_robot_system::msg::ServeDrivePosition>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__TRAITS_HPP_
