//===- Types.h - MLIR Type Classes ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPES_H
#define MLIR_IR_TYPES_H

#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class AsmState;

/// Type 类表示 MLIR 中各种数据类型的抽象基类。
/// 
/// Type 类的核心设计理念：
///   - 唯一性 (Uniquing)：Type 实例是唯一的。这意味着即使创建了多个具有相同参数的类
///     型，它们实际上都指向同一个内存位置。唯一性由 `detail::TypeUniquer` 类实现。
///
///   - 不变标识符 (Immutable Identifier)：每个 Type 实例都有一个不变的标识符，用于
///     区分不同类型的实例。这个标识符是类型参数的一部分，一旦创建就无法更改。
///
///   - 可选可变组件 (Optional Mutable Component)：Type 可以包含一个可变组件，但在
///     创建后修改这个组件不会改变 type 的标识符。这意味着可以修改 type 的一些非关键
///     属性，而不会影响类型的唯一性。
///
///   - 基本类型和参数化类型 (Primitives and Parametric Types): 一些类型是基本类型
///    （例如 Index 类型），没有参数。参数化类型则具有附加信息来区分同一类的不同类型
///    （例如 Integer 类型有位宽，i8 和 i16 属于同一类，但它们是不同的 IntegerType
///     实例）。
///
///   - 类型存储 (Type Storage)：Type 实例包装了一个指向由 MLIRContext 拥有的存储
///     对象的指针。TypeStorage 及其派生类存储 type 信息，包括方言、参数和可变组件。
///     DefaultTypeStorage 用于非参数化类型。
///
///   - 派生类型 (Derived Types)：Type 类是基类，具体的类型（如整数类型、浮点类型等）
///     都是它的派生类。派生类需要实现一些方法来处理类型验证和构造。
///
///
/// Type 类的实例是唯一的，具有 immutable identifier 和 an optional mutable com-
/// ponent。它们包装指向 MLIRContext 拥有的存储对象的指针。因此，Type 的实例按值传递。
///
/// 某些类型是 "primitives"，这意味着它们没有任何参数，例如 Index 类型。Parametric
/// types 具有区分同一类类型的附加信息，例如 Integer 类型具有位宽，使 i8 和 i16 属于
/// 同类型，因为它们是 IntegerType 的不同实例。Parametric types 是 unique immutable
/// key 的一部分。类型的 immutable component 可以在创建类型后进行修改，但不能影响类
/// 型的 identity。
///
/// 类型是通过 'detail::TypeUniquer' 类构造和唯一化的。
///
/// Derived type classes 需要实现几个必需的 implementation hooks：
/// * 可选的：
///   - static LogicalResult verifyInvariants(
///                               function_ref<InFlightDiagnostic()> emitError,
///                               Args... args)
///   * 调用 `TypeBase::get/getChecked` 方法时会调用此方法，以确保传入的参数对于构
///     造 type 实例有效。
///   * 如果无法使用 `args` 构造类型，则此方法应返回失败，否则返回成功。
///   * `args` 必须与传递给 `TypeBase::get` 调用的参数相对应。
///
///
/// Type storage objects 从 TypeStorage 继承并包含以下内容：
///    - 定义 type 的方言。
///    - type 的任何参数。
///    - 可选的 mutable component。
/// 对于 non-parametric types，提供了便利的 DefaultTypeStorage。
/// Parametric storage types 必须派生于 TypeStorage 并遵守以下规定：
///    - Define a type alias, KeyTy, to a type that uniquely identifies the
///      instance of the type.
///      * The key type 必须能够从传递到 `detail::TypeUniquer::get` call 的值构造。
///      * 如果 KeyTy 没有 llvm::DenseMapInfo specialization，则 storage class
///        必须定义哈希方法：
///          `static unsigned hashKey(const KeyTy &)`
///
/// - 提供方法 `bool operator==(const KeyTy &) const`，将 storage instance 与
///   an instance of the key type 进行比较。
///
/// - 提供静态构造方法：
///     `DerivedStorage *construct(TypeStorageAllocator &, const KeyTy &key)`
///   用于构建 an instance of the key type。此函数的参数是用于存储上下文中任何唯一
///   数据的分配器和此存储的 key type。
///
/// - 如果它们具有可变组件，则该组件不能是 the key 的一部分。
///
/// Instances of the Type class are uniqued, have an immutable identifier and an
/// optional mutable component.  They wrap a pointer to the storage object owned
/// by MLIRContext.  Therefore, instances of Type are passed around by value.
///
/// Some types are "primitives" meaning they do not have any parameters, for
/// example the Index type.  Parametric types have additional information that
/// differentiates the types of the same class, for example the Integer type has
/// bitwidth, making i8 and i16 belong to the same kind by be different
/// instances of the IntegerType. Type parameters are part of the unique
/// immutable key.  The mutable component of the type can be modified after the
/// type is created, but cannot affect the identity of the type.
///
/// Types are constructed and uniqued via the 'detail::TypeUniquer' class.
///
/// Derived type classes are expected to implement several required
/// implementation hooks:
///  * Optional:
///    - static LogicalResult verifyInvariants(
///                                function_ref<InFlightDiagnostic()> emitError,
///                                Args... args)
///      * This method is invoked when calling the 'TypeBase::get/getChecked'
///        methods to ensure that the arguments passed in are valid to construct
///        a type instance with.
///      * This method is expected to return failure if a type cannot be
///        constructed with 'args', success otherwise.
///      * 'args' must correspond with the arguments passed into the
///        'TypeBase::get' call.
///
///
/// Type storage objects inherit from TypeStorage and contain the following:
///    - The dialect that defined the type.
///    - Any parameters of the type.
///    - An optional mutable component.
/// For non-parametric types, a convenience DefaultTypeStorage is provided.
/// Parametric storage types must derive TypeStorage and respect the following:
///    - Define a type alias, KeyTy, to a type that uniquely identifies the
///      instance of the type.
///      * The key type must be constructible from the values passed into the
///        detail::TypeUniquer::get call.
///      * If the KeyTy does not have an llvm::DenseMapInfo specialization, the
///        storage class must define a hashing method:
///         'static unsigned hashKey(const KeyTy &)'
///
///    - Provide a method, 'bool operator==(const KeyTy &) const', to
///      compare the storage instance against an instance of the key type.
///
///    - Provide a static construction method:
///        'DerivedStorage *construct(TypeStorageAllocator &, const KeyTy &key)'
///      that builds a unique instance of the derived storage. The arguments to
///      this function are an allocator to store any uniqued data within the
///      context and the key type for this storage.
///
///    - If they have a mutable component, this component must not be a part of
///      the key.
class Type {
public:
  /// Example:
  ///   class TestRecursiveType
  ///       : public ::mlir::Type::TypeBase<TestRecursiveType, ::mlir::Type,
  ///                                       TestRecursiveTypeStorage,
  ///                                       ::mlir::TypeTrait::IsMutable> { ... }
  /// Utility class for implementing types.
  template <typename ConcreteType, typename BaseType, typename StorageType,
            template <typename T> class... Traits>
  using TypeBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::TypeUniquer, Traits...>;

  using ImplType = TypeStorage;

  using AbstractTy = AbstractType;

  constexpr Type() = default;
  /* implicit */ Type(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Type(const Type &other) = default;
  Type &operator=(const Type &other) = default;

  bool operator==(Type other) const { return impl == other.impl; }
  bool operator!=(Type other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename... Tys>
  [[deprecated("Use mlir::isa<U>() instead")]]
  bool isa() const;
  template <typename... Tys>
  [[deprecated("Use mlir::isa_and_nonnull<U>() instead")]]
  bool isa_and_nonnull() const;
  template <typename U>
  [[deprecated("Use mlir::dyn_cast<U>() instead")]]
  U dyn_cast() const;
  template <typename U>
  [[deprecated("Use mlir::dyn_cast_or_null<U>() instead")]]
  U dyn_cast_or_null() const;
  template <typename U>
  [[deprecated("Use mlir::cast<U>() instead")]]
  U cast() const;

  /// Return a unique identifier for the concrete type. This is used to support
  /// dynamic type casting.
  TypeID getTypeID() { return impl->getAbstractType().getTypeID(); }

  /// Return the MLIRContext in which this type was uniqued.
  MLIRContext *getContext() const;

  /// Get the dialect this type is registered to.
  Dialect &getDialect() const { return impl->getAbstractType().getDialect(); }

  // Convenience predicates.  This is only for floating point types,
  // derived types should use isa/dyn_cast.
  bool isIndex() const;
  bool isFloat4E2M1FN() const;
  bool isFloat6E2M3FN() const;
  bool isFloat6E3M2FN() const;
  bool isFloat8E5M2() const;
  bool isFloat8E4M3() const;
  bool isFloat8E4M3FN() const;
  bool isFloat8E5M2FNUZ() const;
  bool isFloat8E4M3FNUZ() const;
  bool isFloat8E4M3B11FNUZ() const;
  bool isFloat8E3M4() const;
  bool isFloat8E8M0FNU() const;
  bool isBF16() const;
  bool isF16() const;
  bool isTF32() const;
  bool isF32() const;
  bool isF64() const;
  bool isF80() const;
  bool isF128() const;

  /// Return true if this is an integer type (with the specified width).
  bool isInteger() const;
  bool isInteger(unsigned width) const;
  /// Return true if this is a signless integer type (with the specified width).
  bool isSignlessInteger() const;
  bool isSignlessInteger(unsigned width) const;
  /// Return true if this is a signed integer type (with the specified width).
  bool isSignedInteger() const;
  bool isSignedInteger(unsigned width) const;
  /// Return true if this is an unsigned integer type (with the specified
  /// width).
  bool isUnsignedInteger() const;
  bool isUnsignedInteger(unsigned width) const;

  /// Return the bit width of an integer or a float type, assert failure on
  /// other types.
  unsigned getIntOrFloatBitWidth() const;

  /// Return true if this is a signless integer or index type.
  bool isSignlessIntOrIndex() const;
  /// Return true if this is a signless integer, index, or float type.
  bool isSignlessIntOrIndexOrFloat() const;
  /// Return true of this is a signless integer or a float type.
  bool isSignlessIntOrFloat() const;

  /// Return true if this is an integer (of any signedness) or an index type.
  bool isIntOrIndex() const;
  /// Return true if this is an integer (of any signedness) or a float type.
  bool isIntOrFloat() const;
  /// Return true if this is an integer (of any signedness), index, or float
  /// type.
  bool isIntOrIndexOrFloat() const;

  /// Print the current type.
  void print(raw_ostream &os) const;
  void print(raw_ostream &os, AsmState &state) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(Type arg);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(impl);
  }
  static Type getFromOpaquePointer(const void *pointer) {
    return Type(reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

  /// Returns true if `InterfaceT` has been promised by the dialect or
  /// implemented.
  template <typename InterfaceT>
  bool hasPromiseOrImplementsInterface() {
    return dialect_extension_detail::hasPromisedInterface(
               getDialect(), getTypeID(), InterfaceT::getInterfaceID()) ||
           mlir::isa<InterfaceT>(*this);
  }

  /// Returns true if the type was registered with a particular trait.
  template <template <typename T> class Trait>
  bool hasTrait() {
    return getAbstractType().hasTrait<Trait>();
  }

  /// Return the abstract type descriptor for this type.
  const AbstractTy &getAbstractType() const { return impl->getAbstractType(); }

  /// Return the Type implementation.
  ImplType *getImpl() const { return impl; }

  /// Walk all of the immediately nested sub-attributes and sub-types. This
  /// method does not recurse into sub elements.
  void walkImmediateSubElements(function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) const {
    getAbstractType().walkImmediateSubElements(*this, walkAttrsFn, walkTypesFn);
  }

  /// Replace the immediately nested sub-attributes and sub-types with those
  /// provided. The order of the provided elements is derived from the order of
  /// the elements returned by the callbacks of `walkImmediateSubElements`. The
  /// element at index 0 would replace the very first attribute given by
  /// `walkImmediateSubElements`. On success, the new instance with the values
  /// replaced is returned. If replacement fails, nullptr is returned.
  auto replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                   ArrayRef<Type> replTypes) const {
    return getAbstractType().replaceImmediateSubElements(*this, replAttrs,
                                                         replTypes);
  }

  /// Walk this type and all attibutes/types nested within using the
  /// provided walk functions. See `AttrTypeWalker` for information on the
  /// supported walk function types.
  template <WalkOrder Order = WalkOrder::PostOrder, typename... WalkFns>
  auto walk(WalkFns &&...walkFns) {
    AttrTypeWalker walker;
    (walker.addWalk(std::forward<WalkFns>(walkFns)), ...);
    return walker.walk<Order>(*this);
  }

  /// Recursively replace all of the nested sub-attributes and sub-types using
  /// the provided map functions. Returns nullptr in the case of failure. See
  /// `AttrTypeReplacer` for information on the support replacement function
  /// types.
  template <typename... ReplacementFns>
  auto replace(ReplacementFns &&...replacementFns) {
    AttrTypeReplacer replacer;
    (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)),
     ...);
    return replacer.replace(*this);
  }

protected:
  ImplType *impl{nullptr};
};

inline raw_ostream &operator<<(raw_ostream &os, Type type) {
  type.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// TypeTraitBase
//===----------------------------------------------------------------------===//

namespace TypeTrait {
/// This class represents the base of a type trait.
template <typename ConcreteType, template <typename> class TraitType>
using TraitBase = detail::StorageUserTraitBase<ConcreteType, TraitType>;
} // namespace TypeTrait

//===----------------------------------------------------------------------===//
// TypeInterface
//===----------------------------------------------------------------------===//

/// This class represents the base of a type interface. See the definition  of
/// `detail::Interface` for requirements on the `Traits` type.
template <typename ConcreteType, typename Traits>
class TypeInterface : public detail::Interface<ConcreteType, Type, Traits, Type,
                                               TypeTrait::TraitBase> {
public:
  using Base = TypeInterface<ConcreteType, Traits>;
  using InterfaceBase =
      detail::Interface<ConcreteType, Type, Traits, Type, TypeTrait::TraitBase>;
  using InterfaceBase::InterfaceBase;

protected:
  /// Returns the impl interface instance for the given type.
  static typename InterfaceBase::Concept *getInterfaceFor(Type type) {
#ifndef NDEBUG
    // Check that the current interface isn't an unresolved promise for the
    // given type.
    dialect_extension_detail::handleUseOfUndefinedPromisedInterface(
        type.getDialect(), type.getTypeID(), ConcreteType::getInterfaceID(),
        llvm::getTypeName<ConcreteType>());
#endif

    return type.getAbstractType().getInterface<ConcreteType>();
  }

  /// Allow access to 'getInterfaceFor'.
  friend InterfaceBase;
};

//===----------------------------------------------------------------------===//
// Core TypeTrait
//===----------------------------------------------------------------------===//

/// This trait is used to determine if a type is mutable or not. It is attached
/// on a type if the corresponding ImplType defines a `mutate` function with
/// a proper signature.
namespace TypeTrait {
template <typename ConcreteType>
using IsMutable = detail::StorageUserTrait::IsMutable<ConcreteType>;
} // namespace TypeTrait

//===----------------------------------------------------------------------===//
// Type Utils
//===----------------------------------------------------------------------===//

// Make Type hashable.
inline ::llvm::hash_code hash_value(Type arg) {
  return DenseMapInfo<const Type::ImplType *>::getHashValue(arg.impl);
}

template <typename... Tys>
bool Type::isa() const {
  return llvm::isa<Tys...>(*this);
}

template <typename... Tys>
bool Type::isa_and_nonnull() const {
  return llvm::isa_and_present<Tys...>(*this);
}

template <typename U>
U Type::dyn_cast() const {
  return llvm::dyn_cast<U>(*this);
}

template <typename U>
U Type::dyn_cast_or_null() const {
  return llvm::dyn_cast_or_null<U>(*this);
}

template <typename U>
U Type::cast() const {
  return llvm::cast<U>(*this);
}

} // namespace mlir

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<mlir::Type> {
  static mlir::Type getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static mlir::Type getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Type val) { return mlir::hash_value(val); }
  static bool isEqual(mlir::Type LHS, mlir::Type RHS) { return LHS == RHS; }
};
template <typename T>
struct DenseMapInfo<T, std::enable_if_t<std::is_base_of<mlir::Type, T>::value &&
                                        !mlir::detail::IsInterface<T>::value>>
    : public DenseMapInfo<mlir::Type> {
  static T getEmptyKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return T::getFromOpaquePointer(pointer);
  }
  static T getTombstoneKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return T::getFromOpaquePointer(pointer);
  }
};

/// We align TypeStorage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::Type> {
public:
  static inline void *getAsVoidPointer(mlir::Type I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Type getFromVoidPointer(void *P) {
    return mlir::Type::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

/// Add support for llvm style casts.
/// We provide a cast between To and From if From is mlir::Type or derives from
/// it
template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_same_v<mlir::Type, std::remove_const_t<From>> ||
                     std::is_base_of_v<mlir::Type, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  /// Arguments are taken as mlir::Type here and not as `From`, because when
  /// casting from an intermediate type of the hierarchy to one of its children,
  /// the val.getTypeID() inside T::classof will use the static getTypeID of the
  /// parent instead of the non-static Type::getTypeID that returns the dynamic
  /// ID. This means that T::classof would end up comparing the static TypeID of
  /// the children to the static TypeID of its parent, making it impossible to
  /// downcast from the parent to the child.
  static inline bool isPossible(mlir::Type ty) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    };
  }
  static inline To doCast(mlir::Type ty) { return To(ty.getImpl()); }
};

} // namespace llvm

#endif // MLIR_IR_TYPES_H
