// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice
#include "intelligent_robot_system/msg/serve_drive_position__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


// Include directives for member types
// Member `id`
// Member `position`
#include "rosidl_generator_c/primitives_sequence_functions.h"

bool
intelligent_robot_system__msg__ServeDrivePosition__init(intelligent_robot_system__msg__ServeDrivePosition * msg)
{
  if (!msg) {
    return false;
  }
  // id
  if (!rosidl_generator_c__int32__Sequence__init(&msg->id, 0)) {
    intelligent_robot_system__msg__ServeDrivePosition__fini(msg);
    return false;
  }
  // position
  if (!rosidl_generator_c__int32__Sequence__init(&msg->position, 0)) {
    intelligent_robot_system__msg__ServeDrivePosition__fini(msg);
    return false;
  }
  return true;
}

void
intelligent_robot_system__msg__ServeDrivePosition__fini(intelligent_robot_system__msg__ServeDrivePosition * msg)
{
  if (!msg) {
    return;
  }
  // id
  rosidl_generator_c__int32__Sequence__fini(&msg->id);
  // position
  rosidl_generator_c__int32__Sequence__fini(&msg->position);
}

intelligent_robot_system__msg__ServeDrivePosition *
intelligent_robot_system__msg__ServeDrivePosition__create()
{
  intelligent_robot_system__msg__ServeDrivePosition * msg = (intelligent_robot_system__msg__ServeDrivePosition *)malloc(sizeof(intelligent_robot_system__msg__ServeDrivePosition));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(intelligent_robot_system__msg__ServeDrivePosition));
  bool success = intelligent_robot_system__msg__ServeDrivePosition__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
intelligent_robot_system__msg__ServeDrivePosition__destroy(intelligent_robot_system__msg__ServeDrivePosition * msg)
{
  if (msg) {
    intelligent_robot_system__msg__ServeDrivePosition__fini(msg);
  }
  free(msg);
}


bool
intelligent_robot_system__msg__ServeDrivePosition__Sequence__init(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  intelligent_robot_system__msg__ServeDrivePosition * data = NULL;
  if (size) {
    data = (intelligent_robot_system__msg__ServeDrivePosition *)calloc(size, sizeof(intelligent_robot_system__msg__ServeDrivePosition));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = intelligent_robot_system__msg__ServeDrivePosition__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        intelligent_robot_system__msg__ServeDrivePosition__fini(&data[i - 1]);
      }
      free(data);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
intelligent_robot_system__msg__ServeDrivePosition__Sequence__fini(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      intelligent_robot_system__msg__ServeDrivePosition__fini(&array->data[i]);
    }
    free(array->data);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

intelligent_robot_system__msg__ServeDrivePosition__Sequence *
intelligent_robot_system__msg__ServeDrivePosition__Sequence__create(size_t size)
{
  intelligent_robot_system__msg__ServeDrivePosition__Sequence * array = (intelligent_robot_system__msg__ServeDrivePosition__Sequence *)malloc(sizeof(intelligent_robot_system__msg__ServeDrivePosition__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = intelligent_robot_system__msg__ServeDrivePosition__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
intelligent_robot_system__msg__ServeDrivePosition__Sequence__destroy(intelligent_robot_system__msg__ServeDrivePosition__Sequence * array)
{
  if (array) {
    intelligent_robot_system__msg__ServeDrivePosition__Sequence__fini(array);
  }
  free(array);
}
