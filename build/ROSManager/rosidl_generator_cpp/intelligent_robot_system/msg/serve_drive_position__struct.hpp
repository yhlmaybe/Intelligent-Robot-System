// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from intelligent_robot_system:msg/ServeDrivePosition.idl
// generated code does not contain a copyright notice

#ifndef INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_HPP_
#define INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_HPP_

#include <rosidl_generator_cpp/bounded_vector.hpp>
#include <rosidl_generator_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__intelligent_robot_system__msg__ServeDrivePosition __attribute__((deprecated))
#else
# define DEPRECATED__intelligent_robot_system__msg__ServeDrivePosition __declspec(deprecated)
#endif

namespace intelligent_robot_system
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ServeDrivePosition_
{
  using Type = ServeDrivePosition_<ContainerAllocator>;

  explicit ServeDrivePosition_(rosidl_generator_cpp::MessageInitialization _init = rosidl_generator_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit ServeDrivePosition_(const ContainerAllocator & _alloc, rosidl_generator_cpp::MessageInitialization _init = rosidl_generator_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _id_type =
    std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other>;
  _id_type id;
  using _position_type =
    std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other>;
  _position_type position;

  // setters for named parameter idiom
  Type & set__id(
    const std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other> & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__position(
    const std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other> & _arg)
  {
    this->position = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> *;
  using ConstRawPtr =
    const intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__intelligent_robot_system__msg__ServeDrivePosition
    std::shared_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__intelligent_robot_system__msg__ServeDrivePosition
    std::shared_ptr<intelligent_robot_system::msg::ServeDrivePosition_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ServeDrivePosition_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    return true;
  }
  bool operator!=(const ServeDrivePosition_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ServeDrivePosition_

// alias to use template instance with default allocator
using ServeDrivePosition =
  intelligent_robot_system::msg::ServeDrivePosition_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace intelligent_robot_system

#endif  // INTELLIGENT_ROBOT_SYSTEM__MSG__SERVE_DRIVE_POSITION__STRUCT_HPP_
