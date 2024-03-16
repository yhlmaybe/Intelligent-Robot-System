// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice
#include "intelligent_robot_system/msg/serve_drive_position__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "intelligent_robot_system/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "intelligent_robot_system/msg/serve_drive_position__struct.h"
#include "intelligent_robot_system/msg/serve_drive_position__functions.h"
#include "fastcdr/Cdr.h"

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

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_generator_c/primitives_sequence.h"  // id, position
#include "rosidl_generator_c/primitives_sequence_functions.h"  // id, position

// forward declare type support functions


using _ServeDrivePosition__ros_msg_type = intelligent_robot_system__msg__ServeDrivePosition;

static bool _ServeDrivePosition__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ServeDrivePosition__ros_msg_type * ros_message = static_cast<const _ServeDrivePosition__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    size_t size = ros_message->id.size;
    auto array_ptr = ros_message->id.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: position
  {
    size_t size = ros_message->position.size;
    auto array_ptr = ros_message->position.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _ServeDrivePosition__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ServeDrivePosition__ros_msg_type * ros_message = static_cast<_ServeDrivePosition__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->id.data) {
      rosidl_generator_c__int32__Sequence__fini(&ros_message->id);
    }
    if (!rosidl_generator_c__int32__Sequence__init(&ros_message->id, size)) {
      return "failed to create array for field 'id'";
    }
    auto array_ptr = ros_message->id.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: position
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->position.data) {
      rosidl_generator_c__int32__Sequence__fini(&ros_message->position);
    }
    if (!rosidl_generator_c__int32__Sequence__init(&ros_message->position, size)) {
      return "failed to create array for field 'position'";
    }
    auto array_ptr = ros_message->position.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_intelligent_robot_system
size_t get_serialized_size_intelligent_robot_system__msg__ServeDrivePosition(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ServeDrivePosition__ros_msg_type * ros_message = static_cast<const _ServeDrivePosition__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name id
  {
    size_t array_size = ros_message->id.size;
    auto array_ptr = ros_message->id.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name position
  {
    size_t array_size = ros_message->position.size;
    auto array_ptr = ros_message->position.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ServeDrivePosition__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_intelligent_robot_system__msg__ServeDrivePosition(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_intelligent_robot_system
size_t max_serialized_size_intelligent_robot_system__msg__ServeDrivePosition(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: id
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: position
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _ServeDrivePosition__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_intelligent_robot_system__msg__ServeDrivePosition(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_ServeDrivePosition = {
  "intelligent_robot_system::msg",
  "ServeDrivePosition",
  _ServeDrivePosition__cdr_serialize,
  _ServeDrivePosition__cdr_deserialize,
  _ServeDrivePosition__get_serialized_size,
  _ServeDrivePosition__max_serialized_size
};

static rosidl_message_type_support_t _ServeDrivePosition__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ServeDrivePosition,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, intelligent_robot_system, msg, ServeDrivePosition)() {
  return &_ServeDrivePosition__type_support;
}

#if defined(__cplusplus)
}
#endif
