// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#ifndef INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_generator_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "intelligent_robot_system/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "intelligent_robot_system/msg/serve_drive_position__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace intelligent_robot_system
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_intelligent_robot_system
cdr_serialize(
  const intelligent_robot_system::msg::ServeDrivePosition & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_intelligent_robot_system
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  intelligent_robot_system::msg::ServeDrivePosition & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_intelligent_robot_system
get_serialized_size(
  const intelligent_robot_system::msg::ServeDrivePosition & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_intelligent_robot_system
max_serialized_size_ServeDrivePosition(
  bool & full_bounded,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace intelligent_robot_system

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_intelligent_robot_system
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, intelligent_robot_system, msg, ServeDrivePosition)();

#ifdef __cplusplus
}
#endif

#endif  // INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
