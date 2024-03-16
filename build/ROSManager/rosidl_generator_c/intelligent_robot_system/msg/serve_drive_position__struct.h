// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#ifndef INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_H_
#define INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'id'
// Member 'position'
#include "rosidl_generator_c/primitives_sequence.h"

// Struct defined in msg/ServeDrivePosition in the package intelligent_robot_system.
typedef struct intelligent_robot_system__msg__ServeDrivePosition
{
  rosidl_generator_c__int32__Sequence id;
  rosidl_generator_c__int32__Sequence position;
} intelligent_robot_system__msg__ServeDrivePosition;

// Struct for a sequence of intelligent_robot_system__msg__ServeDrivePosition.
typedef struct intelligent_robot_system__msg__ServeDrivePosition__Sequence
{
  intelligent_robot_system__msg__ServeDrivePosition * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} intelligent_robot_system__msg__ServeDrivePosition__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_H_
