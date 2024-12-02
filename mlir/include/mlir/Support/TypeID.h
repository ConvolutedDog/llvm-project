//===- TypeID.h - TypeID RTTI class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a definition of the TypeID class. This provides a non
// RTTI mechanism for producing unique type IDs in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TYPEID_H
#define MLIR_SUPPORT_TYPEID_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TypeName.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// TypeID
//===----------------------------------------------------------------------===//

/// 此类为特定 C++ type 提供高效的唯一标识符。这允许在 opaque context 中比较、散列
/// 和存储 C++ 类型。此类在某些方面与 std::type_index 类似，但可用于任何类型。例如，
/// 此类可用于为类型层次结构实现 LLVM 样式的 isa/dyn_cast 功能：
///
///  struct Base {
///    Base(TypeID typeID) : typeID(typeID) {}
///    TypeID typeID;
///  };
///
///  struct DerivedA : public Base {
///    DerivedA() : Base(TypeID::get<DerivedA>()) {}
///
///    static bool classof(const Base *base) {
///      return base->typeID == TypeID::get<DerivedA>();
///    }
///  };
///
///  void foo(Base *base) {
///    if (DerivedA *a = llvm::dyn_cast<DerivedA>(base))
///       ...
///  }
/// C++ RTTI（Run-Time Type Information）是一个出了名的难题；鉴于共享库的性质，许
/// 多不同的方法在支持方面（即仅支持某些类型的类）或性能方面（例如通过使用字符串比较）
/// 根本上都失败了。此类旨在在性能和启用其使用所需的设置之间取得平衡。
///
/// 假设我们正在添加对某些 Foo 类的支持，下面是支持给定 c++ 类型的一组方式：
///
/// 1. 通过 `MLIR_DECLARE_EXPLICIT_TYPE_ID` 和 `MLIR_DEFINE_EXPLICIT_TYPE_ID`
///    显式定义。
///    - 此方法使用给定的宏明确定义给定类型的类型 ID。这些宏应放置在文件的顶层（即不
///      在任何命名空间或类内）。这是最有效和最有效率的方法，但需要为每种类型提供显式
///      的 annotations。
///      示例：
///      // Foo.h
///      MLIR_DECLARE_EXPLICIT_TYPE_ID(Foo);
///      // Foo.cpp
///      MLIR_DEFINE_EXPLICIT_TYPE_ID(Foo);
///
/// 2. 通过 `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` 显式定义。
///    - 此方法通过直接注释类来明确定义给定类型的类型 ID。这与上述方法具有类似的有效
///      性和效率，但应仅用于内部类；即那些定义受限于特定库的类（通常是匿名命名空间中
///      的类）。
///      示例：
///       namespace {
///       class Foo {
///       public:
///         MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Foo)
///       };
///       } // namespace
///
/// 3. 通过使用类型名称的回退隐式定义。
///    - 此方法通过使用类型名称隐式定义给定类型的类型 ID。此方法不需要用户明确提供任
///      何信息，但需要额外的访问和初始化成本。鉴于此方法使用类型的名称，因此它不能用
///      于在匿名命名空间中定义的类型（当可以检测到时会断言）。字符串名称在这些上下文
///      中不提供任何唯一性的保证。
///
/// This class provides an efficient unique identifier for a specific C++ type.
/// This allows for a C++ type to be compared, hashed, and stored in an opaque
/// context. This class is similar in some ways to std::type_index, but can be
/// used for any type. For example, this class could be used to implement LLVM
/// style isa/dyn_cast functionality for a type hierarchy:
///
///  struct Base {
///    Base(TypeID typeID) : typeID(typeID) {}
///    TypeID typeID;
///  };
///
///  struct DerivedA : public Base {
///    DerivedA() : Base(TypeID::get<DerivedA>()) {}
///
///    static bool classof(const Base *base) {
///      return base->typeID == TypeID::get<DerivedA>();
///    }
///  };
///
///  void foo(Base *base) {
///    if (DerivedA *a = llvm::dyn_cast<DerivedA>(base))
///       ...
///  }
///
/// C++ RTTI is a notoriously difficult topic; given the nature of shared
/// libraries many different approaches fundamentally break down in either the
/// area of support (i.e. only certain types of classes are supported), or in
/// terms of performance (e.g. by using string comparison). This class intends
/// to strike a balance between performance and the setup required to enable its
/// use.
///
/// Assume we are adding support for some class Foo, below are the set of ways
/// in which a given c++ type may be supported:
///
///  * Explicitly via `MLIR_DECLARE_EXPLICIT_TYPE_ID` and
///    `MLIR_DEFINE_EXPLICIT_TYPE_ID`
///
///    - This method explicitly defines the type ID for a given type using the
///      given macros. These should be placed at the top-level of the file (i.e.
///      not within any namespace or class). This is the most effective and
///      efficient method, but requires explicit annotations for each type.
///
///      Example:
///
///        // Foo.h
///        MLIR_DECLARE_EXPLICIT_TYPE_ID(Foo);
///
///        // Foo.cpp
///        MLIR_DEFINE_EXPLICIT_TYPE_ID(Foo);
///
///  * Explicitly via `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`
///   - This method explicitly defines the type ID for a given type by
///     annotating the class directly. This has similar effectiveness and
///     efficiency to the above method, but should only be used on internal
///     classes; i.e. those with definitions constrained to a specific library
///     (generally classes in anonymous namespaces).
///
///     Example:
///
///       namespace {
///       class Foo {
///       public:
///         MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Foo)
///       };
///       } // namespace
///
///  * Implicitly via a fallback using the type name
///   - This method implicitly defines a type ID for a given type by using the
///     type name. This method requires nothing explicitly from the user, but
///     pays additional access and initialization cost. Given that this method
///     uses the name of the type, it may not be used for types defined in
///     anonymous namespaces (which is asserted when it can be detected). String
///     names do not provide any guarantees on uniqueness in these contexts.
///
class TypeID {
  /// TypeID 类表示类型信息对象的存储。
  /// 注意：我们在此指定显式对齐，以允许与 PointerIntPair 和其他需要已知指针对齐的实
  /// 用程序/数据结构一起使用。
  ///
  /// 指定对齐是出于性能和兼容性考虑。对于某些处理器和算法，对齐的数据可以更高效地被访
  /// 问。特别是对于需要已知指针对齐的工具或数据结构（如 PointerIntPair 等），显式对
  /// 齐非常重要。PointerIntPair 是一个能够同时存储一个指针和一个整数的工具，但它要求
  /// 能够确切知道指针的对齐要求，以便高效地存储和访问数据。
  ///
  /// This class represents the storage of a type info object.
  /// Note: We specify an explicit alignment here to allow use with
  /// PointerIntPair and other utilities/data structures that require a known
  /// pointer alignment.
  struct alignas(8) Storage {};

public:
  /// `get<void>()` 是一个静态方法，用于获取 void 类型的类型信息对象（存储在 Storage
  /// 中）。这表示默认构造一个 TypeID 对象，其存储的是 void 类型的类型信息。
  TypeID() : TypeID(get<void>()) {}

  /// Comparison operations.
  inline bool operator==(const TypeID &other) const {
    return storage == other.storage;
  }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Construct a type info object for the given type T.
  template <typename T>
  static TypeID get();
  template <template <typename> class Trait>
  static TypeID get();

  /// 这个方法返回 storage 指针的 void 类型指针，这允许将 TypeID 对象转换为一个通用
  /// 的不透明指针。
  ///
  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(storage);
  }
  static TypeID getFromOpaquePointer(const void *pointer) {
    return TypeID(reinterpret_cast<const Storage *>(pointer));
  }

  /// Enable hashing TypeID.
  friend ::llvm::hash_code hash_value(TypeID id);

private:
  /// 私有构造函数。
  TypeID(const Storage *storage) : storage(storage) {}

  /// 这是 TypeID 类的一个私有成员变量，用于存储类型信息对象的指针。
  ///
  /// The storage of this type info object.
  const Storage *storage;

  friend class TypeIDAllocator;
};

/// Enable hashing TypeID.
inline ::llvm::hash_code hash_value(TypeID id) {
  return DenseMapInfo<const TypeID::Storage *>::getHashValue(id.storage);
}

//===----------------------------------------------------------------------===//
// TypeIDResolver
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides a fallback for resolving TypeIDs. It uses the string
/// name of the type to perform the resolution, and as such does not allow the
/// use of classes defined in "anonymous" contexts.
class FallbackTypeIDResolver {
protected:
  /// Register an implicit type ID for the given type name.
  static TypeID registerImplicitTypeID(StringRef name);
};

/// This class provides a resolver for getting the ID for a given class T. This
/// allows for the derived type to specialize its resolution behavior. The
/// default implementation uses the string name of the type to resolve the ID.
/// This provides a strong definition, but at the cost of performance (we need
/// to do an initial lookup) and is not usable by classes defined in anonymous
/// contexts.
///
/// TODO: The use of the type name is only necessary when building in the
/// presence of shared libraries. We could add a build flag that guarantees
/// "static"-like environments and switch this to a more optimal implementation
/// when that is enabled.
template <typename T, typename Enable = void>
class TypeIDResolver : public FallbackTypeIDResolver {
public:
  /// Trait to check if `U` is fully resolved. We use this to verify that `T` is
  /// fully resolved when trying to resolve a TypeID. We don't technically need
  /// to have the full definition of `T` for the fallback, but it does help
  /// prevent situations where a forward declared type uses this fallback even
  /// though there is a strong definition for the TypeID in the location where
  /// `T` is defined.
  template <typename U>
  using is_fully_resolved_trait = decltype(sizeof(U));
  template <typename U>
  using is_fully_resolved = llvm::is_detected<is_fully_resolved_trait, U>;

  static TypeID resolveTypeID() {
    static_assert(is_fully_resolved<T>::value,
                  "TypeID::get<> requires the complete definition of `T`");
    static TypeID id = registerImplicitTypeID(llvm::getTypeName<T>());
    return id;
  }
};

/// This class provides utilities for resolving the TypeID of a class that
/// provides a `static TypeID resolveTypeID()` method. This allows for
/// simplifying situations when the class can resolve the ID itself. This
/// functionality is separated from the corresponding `TypeIDResolver`
/// specialization below to enable referencing it more easily in different
/// contexts.
struct InlineTypeIDResolver {
  /// Trait to check if `T` provides a static `resolveTypeID` method.
  template <typename T>
  using has_resolve_typeid_trait = decltype(T::resolveTypeID());
  template <typename T>
  using has_resolve_typeid = llvm::is_detected<has_resolve_typeid_trait, T>;

  template <typename T>
  static TypeID resolveTypeID() {
    return T::resolveTypeID();
  }
};
/// This class provides a resolver for getting the ID for a given class T, when
/// the class provides a `static TypeID resolveTypeID()` method. This allows for
/// simplifying situations when the class can resolve the ID itself.
template <typename T>
class TypeIDResolver<
    T, std::enable_if_t<InlineTypeIDResolver::has_resolve_typeid<T>::value>> {
public:
  static TypeID resolveTypeID() {
    return InlineTypeIDResolver::resolveTypeID<T>();
  }
};
} // namespace detail

template <typename T>
TypeID TypeID::get() {
  return detail::TypeIDResolver<T>::resolveTypeID();
}
template <template <typename> class Trait>
TypeID TypeID::get() {
  // An empty class used to simplify the use of Trait types.
  struct Empty {};
  return TypeID::get<Trait<Empty>>();
}

// Declare/define an explicit specialization for TypeID: this forces the
// compiler to emit a strong definition for a class and controls which
// translation unit and shared object will actually have it.
// This can be useful to turn to a link-time failure what would be in other
// circumstances a hard-to-catch runtime bug when a TypeID is hidden in two
// different shared libraries and instances of the same class only gets the same
// TypeID inside a given DSO.
#define MLIR_DECLARE_EXPLICIT_TYPE_ID(CLASS_NAME)                              \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  template <>                                                                  \
  class TypeIDResolver<CLASS_NAME> {                                           \
  public:                                                                      \
    static TypeID resolveTypeID() { return id; }                               \
                                                                               \
  private:                                                                     \
    static SelfOwningTypeID id;                                                \
  };                                                                           \
  } /* namespace detail */                                                     \
  } /* namespace mlir */

#define MLIR_DEFINE_EXPLICIT_TYPE_ID(CLASS_NAME)                               \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  SelfOwningTypeID TypeIDResolver<CLASS_NAME>::id = {};                        \
  } /* namespace detail */                                                     \
  } /* namespace mlir */

// Declare/define an explicit, **internal**, specialization of TypeID for the
// given class. This is useful for providing an explicit specialization of
// TypeID for a class that is known to be internal to a specific library. It
// should be placed within a public section of the declaration of the class.
#define MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CLASS_NAME)               \
  static ::mlir::TypeID resolveTypeID() {                                      \
    static ::mlir::SelfOwningTypeID id;                                        \
    return id;                                                                 \
  }                                                                            \
  static_assert(                                                               \
      ::mlir::detail::InlineTypeIDResolver::has_resolve_typeid<                \
          CLASS_NAME>::value,                                                  \
      "`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` must be placed in a "    \
      "public section of `" #CLASS_NAME "`");

//===----------------------------------------------------------------------===//
// TypeIDAllocator
//===----------------------------------------------------------------------===//

/// 此类提供了一种在运行时定义新 TypeID 的方法。当分配器被析构时，所有分配的 TypeID
/// 都将变为无效，因此不应使用。
///
/// This class provides a way to define new TypeIDs at runtime.
/// When the allocator is destructed, all allocated TypeIDs become invalid and
/// therefore should not be used.
class TypeIDAllocator {
public:
  /// 分配一个新的 TypeID，这被保证在 TypeIDAllocator 的生命期是独一无二的。
  ///
  /// Allocate a new TypeID, that is ensured to be unique for the lifetime
  /// of the TypeIDAllocator.
  TypeID allocate() { return TypeID(ids.Allocate()); }

private:
  /// The TypeIDs allocated are the addresses of the different storages.
  /// Keeping those in memory ensure uniqueness of the TypeIDs.
  llvm::SpecificBumpPtrAllocator<TypeID::Storage> ids;
};

//===----------------------------------------------------------------------===//
// SelfOwningTypeID
//===----------------------------------------------------------------------===//

/// Defines a TypeID for each instance of this class by using a pointer to the
/// instance. Thus, the copy and move constructor are deleted.
/// Note: We align by 8 to match the alignment of TypeID::Storage, as we treat
/// an instance of this class similarly to TypeID::Storage.
class alignas(8) SelfOwningTypeID {
public:
  SelfOwningTypeID() = default;
  SelfOwningTypeID(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID &operator=(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID(SelfOwningTypeID &&) = delete;
  SelfOwningTypeID &operator=(SelfOwningTypeID &&) = delete;

  /// Implicitly converts to the owned TypeID.
  operator TypeID() const { return getTypeID(); }

  /// Return the TypeID owned by this object.
  TypeID getTypeID() const { return TypeID::getFromOpaquePointer(this); }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Builtin TypeIDs
//===----------------------------------------------------------------------===//

/// Explicitly register a set of "builtin" types.
MLIR_DECLARE_EXPLICIT_TYPE_ID(void)

namespace llvm {
template <>
struct DenseMapInfo<mlir::TypeID> {
  static inline mlir::TypeID getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static inline mlir::TypeID getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::TypeID::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::TypeID val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::TypeID lhs, mlir::TypeID rhs) { return lhs == rhs; }
};

/// We align TypeID::Storage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::TypeID> {
  static inline void *getAsVoidPointer(mlir::TypeID info) {
    return const_cast<void *>(info.getAsOpaquePointer());
  }
  static inline mlir::TypeID getFromVoidPointer(void *ptr) {
    return mlir::TypeID::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // namespace llvm

#endif // MLIR_SUPPORT_TYPEID_H
