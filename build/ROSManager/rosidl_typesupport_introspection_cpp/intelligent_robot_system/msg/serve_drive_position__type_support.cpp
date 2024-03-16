// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_generator_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "intelligent_robot_system/msg/serve_drive_position__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace intelligent_robot_system
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ServeDrivePosition_init_function(
  void * message_memory, rosidl_generator_cpp::MessageInitialization _init)
{
  new (message_memory) intelligent_robot_system::msg::ServeDrivePosition(_init);
}

void ServeDrivePosition_fini_function(void * message_memory)
{
  auto typed_message = static_cast<intelligent_robot_system::msg::ServeDrivePosition *>(message_memory);
  typed_message->~ServeDrivePosition();
}

size_t size_function__ServeDrivePosition__id(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ServeDrivePosition__id(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__ServeDrivePosition__id(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void resize_function__ServeDrivePosition__id(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ServeDrivePosition__position(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ServeDrivePosition__position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__ServeDrivePosition__position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void resize_function__ServeDrivePosition__position(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ServeDrivePosition_message_member_array[2] = {
  {
    "id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(intelligent_robot_system::msg::ServeDrivePosition, id),  // bytes offset in struct
    nullptr,  // default value
    size_function__ServeDrivePosition__id,  // size() function pointer
    get_const_function__ServeDrivePosition__id,  // get_const(index) function pointer
    get_function__ServeDrivePosition__id,  // get(index) function pointer
    resize_function__ServeDrivePosition__id  // resize(index) function pointer
  },
  {
    "position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(intelligent_robot_system::msg::ServeDrivePosition, position),  // bytes offset in struct
    nullptr,  // default value
    size_function__ServeDrivePosition__position,  // size() function pointer
    get_const_function__ServeDrivePosition__position,  // get_const(index) function pointer
    get_function__ServeDrivePosition__position,  // get(index) function pointer
    resize_function__ServeDrivePosition__position  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ServeDrivePosition_message_members = {
  "intelligent_robot_system::msg",  // message namespace
  "ServeDrivePosition",  // message name
  2,  // number of fields
  sizeof(intelligent_robot_system::msg::ServeDrivePosition),
  ServeDrivePosition_message_member_array,  // message members
  ServeDrivePosition_init_function,  // function to initialize message memory (memory has to be allocated)
  ServeDrivePosition_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ServeDrivePosition_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ServeDrivePosition_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace intelligent_robot_system


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<intelligent_robot_system::msg::ServeDrivePosition>()
{
  return &::intelligent_robot_system::msg::rosidl_typesupport_introspection_cpp::ServeDrivePosition_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, intelligent_robot_system, msg, ServeDrivePosition)() {
  return &::intelligent_robot_system::msg::rosidl_typesupport_introspection_cpp::ServeDrivePosition_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
