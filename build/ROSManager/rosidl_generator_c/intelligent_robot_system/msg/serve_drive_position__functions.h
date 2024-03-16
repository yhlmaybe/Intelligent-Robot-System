// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#ifndef INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__FUNCTIONS_H_
#define INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_generator_c/visibility_control.h"
#include "intelligent_robot_system/msg/rosidl_generator_c__visibility_control.h"

#include "intelligent_robot_system/msg/serve_drive_position__struct.h"

/// Initialize msg/ServeDrivePosition message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * intelligent_robot_system__msg__ServeDrivePosition
 * )) before or use
 * intelligent_robot_system__msg__ServeDrivePosition__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
bool
intelligent_robot_system__msg__ServeDrivePosition__init(intelligent_robot_system__msg__ServeDrivePosition * msg);

/// Finalize msg/ServeDrivePosition message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
void
intelligent_robot_system__msg__ServeDrivePosition__fini(intelligent_robot_system__msg__ServeDrivePosition * msg);

/// Create msg/ServeDrivePosition message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * intelligent_robot_system__msg__ServeDrivePosition__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
intelligent_robot_system__msg__ServeDrivePosition *
intelligent_robot_system__msg__ServeDrivePosition__create();

/// Destroy msg/ServeDrivePosition message.
/**
 * It calls
 * intelligent_robot_system__msg__ServeDrivePosition__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
void
intelligent_robot_system__msg__ServeDrivePosition__destroy(intelligent_robot_system__msg__ServeDrivePosition * msg);


/// Initialize array of msg/ServeDrivePosition messages.
/**
 * It allocates the memory for the number of elements and calls
 * intelligent_robot_system__msg__ServeDrivePosition__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
bool
intelligent_robot_system__msg__ServeDrivePosition__Sequence__init(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array, size_t size);

/// Finalize array of msg/ServeDrivePosition messages.
/**
 * It calls
 * intelligent_robot_system__msg__ServeDrivePosition__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
void
intelligent_robot_system__msg__ServeDrivePosition__Sequence__fini(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array);

/// Create array of msg/ServeDrivePosition messages.
/**
 * It allocates the memory for the array and calls
 * intelligent_robot_system__msg__ServeDrivePosition__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
intelligent_robot_system__msg__ServeDrivePosition__Sequence *
intelligent_robot_system__msg__ServeDrivePosition__Sequence__create(size_t size);

/// Destroy array of msg/ServeDrivePosition messages.
/**
 * It calls
 * intelligent_robot_system__msg__ServeDrivePosition__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_intelligent_robot_system
void
intelligent_robot_system__msg__ServeDrivePosition__Sequence__destroy(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array);

#ifdef __cplusplus
}
#endif

#endif  // INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__FUNCTIONS_H_
