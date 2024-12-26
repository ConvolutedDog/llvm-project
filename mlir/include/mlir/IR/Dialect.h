//===- Dialect.h - IR Dialect Description -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the 'dialect' abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECT_H
#define MLIR_IR_DIALECT_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
class DialectInterface;
class OpBuilder;
class Type;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

/// 方言是 MLIR 操作、类型和属性的组，以及与整个组相关的行为。例如，挂接到其他系统中
/// 以进行常量折叠、接口、用于 asm 打印的默认命名类型等。
///
/// 方言对象的实例加载到特定的 MLIRContext 中。
///
/// Dialects are groups of MLIR operations, types and attributes, as well as
/// behavior associated with the entire group.  For example, hooks into other
/// systems for constant folding, interfaces, default named types for asm
/// printing, etc.
///
/// Instances of the dialect object are loaded in a specific MLIRContext.
///
class Dialect {
public:
  /// 方言提供的 call back 函数类型，用于解析自定义操作。这用于方言提供解析自定
  /// 义操作（包括未注册的操作）的替代方法。
  ///
  /// Type for a callback provided by the dialect to parse a custom operation.
  /// This is used for the dialect to provide an alternative way to parse custom
  /// operations, including unregistered ones.
  using ParseOpHook =
      function_ref<ParseResult(OpAsmParser &parser, OperationState &result)>;

  virtual ~Dialect();

  /// 该实用函数返回是否给定的字符串是有效的方言命名空间。
  ///
  /// Utility function that returns if the given string is a valid dialect
  /// namespace
  static bool isValidNamespace(StringRef str);

  MLIRContext *getContext() const { return context; }

  StringRef getNamespace() const { return name; }

  /// Returns the unique identifier that corresponds to this dialect.
  TypeID getTypeID() const { return dialectID; }

  /// 如果此方言允许未注册的操作，即以方言命名空间为前缀但未使用 addOperation 注册
  /// 的操作，则返回 true。
  ///
  /// Returns true if this dialect allows for unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowsUnknownOperations() const { return unknownOpsAllowed; }

  /// 如果此方言允许未注册类型（即以方言命名空间为前缀但未使用 addType 注册的类型），
  /// 则返回 true。这些类型用 OpaqueType 表示。
  ///
  /// Return true if this dialect allows for unregistered types, i.e., types
  /// prefixed with the dialect namespace but not registered with addType.
  /// These are represented with OpaqueType.
  bool allowsUnknownTypes() const { return unknownTypesAllowed; }

  /// 注册[整个方言范围]的规范化模式。此方法仅应用于注册在[概念上不属于方言中任何单
  /// 个操作]的规范化模式。（在这种情况下，使用 op 的规范化器。）例如，op 接口的规
  /// 范化模式应在此处注册。
  ///
  /// 参考：https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html
  ///
  /// Register dialect-wide canonicalization patterns. This method should only
  /// be used to register canonicalization patterns that do not conceptually
  /// belong to any single operation in the dialect. (In that case, use the op's
  /// canonicalizer.) E.g., canonicalization patterns for op interfaces should
  /// be registered here.
  virtual void getCanonicalizationPatterns(RewritePatternSet &results) const {}

  /// 注册钩子，用于根据给定的属性值实现具有所需结果类型的[单个常量操作]。此方法应使
  /// 用提供的构建器来创建操作而不更改插入位置。生成的操作应为常量，即单个结果、零个
  /// 操作数、无副作用等。成功时，此钩子应返回生成的值以表示常量值。否则，失败时应返
  /// 回 null。
  ///
  /// Registered hook to materialize a single constant operation from a given
  /// attribute value with the desired resultant type. This method should use
  /// the provided builder to create the operation without changing the
  /// insertion position. The generated operation is expected to be constant
  /// like, i.e. single result, zero operands, non side-effecting, etc. On
  /// success, this hook should return the value generated to represent the
  /// constant value. Otherwise, it should return null on failure.
  virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Parsing Hooks
  //===--------------------------------------------------------------------===//

  /// 解析注册到此方言的属性。如果 `type` 非空，则它指的是属性的预期类型。
  ///
  /// Parse an attribute registered to this dialect. If 'type' is nonnull, it
  /// refers to the expected type of the attribute.
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type) const;

  /// 打印已注册到此方言的属性。注意：此方法不需要打印属性的类型，因为它始终由调用
  /// 者打印。
  ///
  /// Print an attribute registered to this dialect. Note: The type of the
  /// attribute need not be printed by this method as it is always printed by
  /// the caller.
  virtual void printAttribute(Attribute, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered attribute printing hook");
  }

  /// 解析注册到此方言的类型。
  ///
  /// Parse a type registered to this dialect.
  virtual Type parseType(DialectAsmParser &parser) const;

  /// 打印已注册到此方言的 type。
  ///
  /// Print a type registered to this dialect.
  virtual void printType(Type, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered type printing hook");
  }

  /// 返回用于解析已注册到此方言的操作的钩子（如果有）。默认情况下，这将查找已注册的操
  /// 作并返回在 RegisteredOperationName 上注册的 `parse()` 方法。方言可以覆盖此行
  /// 为并处理未注册的操作。
  ///
  /// Return the hook to parse an operation registered to this dialect, if any.
  /// By default this will lookup for registered operations and return the
  /// `parse()` method registered on the RegisteredOperationName. Dialects can
  /// override this behavior and handle unregistered operations as well.
  virtual std::optional<ParseOpHook>
  getParseOperationHook(StringRef opName) const;

  /// 打印注册到此方言的操作。此钩子被调用用于已注册的操作，这些操作不会覆盖 `print()`
  /// 方法来定义自己的自定义程序集。
  ///
  /// Print an operation registered to this dialect.
  /// This hook is invoked for registered operation which don't override the
  /// `print()` method to define their own custom assembly.
  virtual llvm::unique_function<void(Operation *, OpAsmPrinter &printer)>
  getOperationPrinter(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Verification Hooks
  //===--------------------------------------------------------------------===//

  /// 根据给定操作中“regionIndex”处区域的“argIndex”处的参数验证此方言的属性。如果
  /// 验证失败，则返回失败，否则返回成功。此钩子可以从任何包含区域的操作中选择性地调
  /// 用。
  ///
  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionArgAttribute(Operation *,
                                                 unsigned regionIndex,
                                                 unsigned argIndex,
                                                 NamedAttribute);

  /// 根据给定操作中“regionIndex”处区域的“resultIndex”处的参数验证此方言的属性。如
  /// 果验证失败，则返回失败，否则返回成功。此钩子可以从任何包含区域的操作中选择性地
  /// 调用。
  ///
  /// Verify an attribute from this dialect on the result at 'resultIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionResultAttribute(Operation *,
                                                    unsigned regionIndex,
                                                    unsigned resultIndex,
                                                    NamedAttribute);

  /// 在给定操作上验证此方言的属性。如果验证失败则返回失败，否则返回成功。
  ///
  /// Verify an attribute from this dialect on the given operation. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute) {
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Interfaces
  //===--------------------------------------------------------------------===//

  /// 如果已注册，则查找给定 interface ID 的 interface，否则为 nullptr。
  ///
  /// Lookup an interface for the given ID if one is registered, otherwise
  /// nullptr.
  DialectInterface *getRegisteredInterface(TypeID interfaceID) {
#ifndef NDEBUG
    handleUseOfUndefinedPromisedInterface(getTypeID(), interfaceID);
#endif

    auto it = registeredInterfaces.find(interfaceID);
    return it != registeredInterfaces.end() ? it->getSecond().get() : nullptr;
  }
  template <typename InterfaceT>
  InterfaceT *getRegisteredInterface() {
#ifndef NDEBUG
    handleUseOfUndefinedPromisedInterface(getTypeID(),
                                          InterfaceT::getInterfaceID(),
                                          llvm::getTypeName<InterfaceT>());
#endif

    return static_cast<InterfaceT *>(
        getRegisteredInterface(InterfaceT::getInterfaceID()));
  }

  /// 如果已注册，则查找给定 ID 的 op interface，否则为 nullptr。
  ///
  /// Lookup an op interface for the given ID if one is registered, otherwise
  /// nullptr.
  virtual void *getRegisteredInterfaceForOp(TypeID interfaceID,
                                            OperationName opName) {
    return nullptr;
  }
  template <typename InterfaceT>
  typename InterfaceT::Concept *
  getRegisteredInterfaceForOp(OperationName opName) {
    return static_cast<typename InterfaceT::Concept *>(
        getRegisteredInterfaceForOp(InterfaceT::getInterfaceID(), opName));
  }

  /// 用这个方言实例注册一个 dialect interface。
  ///
  /// Register a dialect interface with this dialect instance.
  void addInterface(std::unique_ptr<DialectInterface> interface);

  /// 用这个方言实例注册一个集合的 dialect interface。
  ///
  /// Register a set of dialect interfaces with this dialect instance.
  template <typename... Args>
  void addInterfaces() {
    (addInterface(std::make_unique<Args>(this)), ...);
  }
  template <typename InterfaceT, typename... Args>
  InterfaceT &addInterface(Args &&...args) {
    InterfaceT *interface = new InterfaceT(this, std::forward<Args>(args)...);
    addInterface(std::unique_ptr<DialectInterface>(interface));
    return *interface;
  }

  /// 声明将实现给定的接口，但延迟注册。承诺的接口类型可以是任何类型的接口，而不仅仅
  /// 是方言接口，即它也可以是 AttributeInterface/OpInterface/TypeInterface 等。
  ///
  /// Declare that the given interface will be implemented, but has a delayed
  /// registration. The promised interface type can be an interface of any type
  /// not just a dialect interface, i.e. it may also be an
  /// AttributeInterface/OpInterface/TypeInterface/etc.
  template <typename InterfaceT, typename ConcreteT>
  void declarePromisedInterface() {
    unresolvedPromisedInterfaces.insert(
        {TypeID::get<ConcreteT>(), InterfaceT::getInterfaceID()});
  }

  // 为多种类型声明相同的接口。示例：
  // `declaredPromisedInterfaces<FunctionOpInterface, MyFuncType1, MyFuncType2>()`
  //
  // Declare the same interface for multiple types.
  // Example:
  // declarePromisedInterfaces<FunctionOpInterface, MyFuncType1, MyFuncType2>()
  template <typename InterfaceT, typename... ConcreteT>
  void declarePromisedInterfaces() {
    (declarePromisedInterface<InterfaceT, ConcreteT>(), ...);
  }

  /// 检查尝试使用的给定接口是否是此方言的承诺接口，但尚未实现。如果是，则发出致命
  /// 错误。`interfaceName` 是一个可选字符串，其中包含更易于用户阅读的接口名称（
  /// 例如类名）。
  ///
  /// Checks if the given interface, which is attempting to be used, is a
  /// promised interface of this dialect that has yet to be implemented. If so,
  /// emits a fatal error. `interfaceName` is an optional string that contains a
  /// more user readable name for the interface (such as the class name).
  void handleUseOfUndefinedPromisedInterface(TypeID interfaceRequestorID,
                                             TypeID interfaceID,
                                             StringRef interfaceName = "") {
    if (unresolvedPromisedInterfaces.count(
            {interfaceRequestorID, interfaceID})) {
      llvm::report_fatal_error(
          "checking for an interface (`" + interfaceName +
          "`) that was promised by dialect '" + getNamespace() +
          "' but never implemented. This is generally an indication "
          "that the dialect extension implementing the interface was never "
          "registered.");
    }
  }

  /// 检查给定的接口（尝试附加到此方言拥有的构造）是否是此方言尚未实现的承诺接口。如
  /// 果是，则解析接口承诺。
  ///
  /// Checks if the given interface, which is attempting to be attached to a
  /// construct owned by this dialect, is a promised interface of this dialect
  /// that has yet to be implemented. If so, it resolves the interface promise.
  void handleAdditionOfUndefinedPromisedInterface(TypeID interfaceRequestorID,
                                                  TypeID interfaceID) {
    unresolvedPromisedInterfaces.erase({interfaceRequestorID, interfaceID});
  }

  /// 检查是否已针对接口/请求者对做出承诺。
  ///
  /// Checks if a promise has been made for the interface/requestor pair.
  bool hasPromisedInterface(TypeID interfaceRequestorID,
                            TypeID interfaceID) const {
    return unresolvedPromisedInterfaces.count(
        {interfaceRequestorID, interfaceID});
  }

  /// 检查是否已针对接口/请求者对做出承诺。
  ///
  /// Checks if a promise has been made for the interface/requestor pair.
  template <typename ConcreteT, typename InterfaceT>
  bool hasPromisedInterface() const {
    return hasPromisedInterface(TypeID::get<ConcreteT>(),
                                InterfaceT::getInterfaceID());
  }

protected:
  /// 构造函数的参数为该方言采用唯一的命名空间以及要绑定到的上下文。
  /// 注意：命名空间不得包含 '.' 字符。
  /// 注意：属于该方言的所有操作的名称都必须以命名空间开头，后跟 '.' 字符。
  /// 示例：
  ///       - "tf" 表示 TensorFlow 操作，如 "tf.add"。
  ///
  /// The constructor takes a unique namespace for this dialect as well as the
  /// context to bind to.
  /// Note: The namespace must not contain '.' characters.
  /// Note: All operations belonging to this dialect must have names starting
  ///       with the namespace followed by '.'.
  /// Example:
  ///       - "tf" for the TensorFlow ops like "tf.add".
  Dialect(StringRef name, MLIRContext *context, TypeID id);

  /// 派生类使用此方法将其操作添加到集合中。
  ///
  /// This method is used by derived classes to add their operations to the set.
  ///
  template <typename... Args>
  void addOperations() {
    // This initializer_list argument pack expansion is essentially equal to
    // using a fold expression with a comma operator. Clang however, refuses
    // to compile a fold expression with a depth of more than 256 by default.
    // There seem to be no such limitations for initializer_list.
    (void)std::initializer_list<int>{
        0, (RegisteredOperationName::insert<Args>(*this), 0)...};
  }

  /// 为方言注册一组 type classes。
  ///
  /// Register a set of type classes with this dialect.
  template <typename... Args>
  void addTypes() {
    // This initializer_list argument pack expansion is essentially equal to
    // using a fold expression with a comma operator. Clang however, refuses
    // to compile a fold expression with a depth of more than 256 by default.
    // There seem to be no such limitations for initializer_list.
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

  /// 使用此方言注册类型实例。
  /// 一般不建议使用此方法，而建议使用 `addTypes<CustomType>()`。
  ///
  /// Register a type instance with this dialect.
  /// The use of this method is in general discouraged in favor of
  /// 'addTypes<CustomType>()'.
  void addType(TypeID typeID, AbstractType &&typeInfo);

  /// 用这种方言注册一组 attribute classes。
  ///
  /// Register a set of attribute classes with this dialect.
  template <typename... Args>
  void addAttributes() {
    // This initializer_list argument pack expansion is essentially equal to
    // using a fold expression with a comma operator. Clang however, refuses
    // to compile a fold expression with a depth of more than 256 by default.
    // There seem to be no such limitations for initializer_list.
    (void)std::initializer_list<int>{0, (addAttribute<Args>(), 0)...};
  }

  /// 用这种方言注册一个属性实例。
  /// 一般不鼓励使用此方法，而建议使用 `addAttributes<CustomAttr>()`。
  ///
  /// Register an attribute instance with this dialect.
  /// The use of this method is in general discouraged in favor of
  /// 'addAttributes<CustomAttr>()'.
  void addAttribute(TypeID typeID, AbstractAttribute &&attrInfo);

  /// Enable support for unregistered operations.
  void allowUnknownOperations(bool allow = true) { unknownOpsAllowed = allow; }

  /// Enable support for unregistered types.
  void allowUnknownTypes(bool allow = true) { unknownTypesAllowed = allow; }

private:
  Dialect(const Dialect &) = delete;
  void operator=(Dialect &) = delete;

  /// Register an attribute instance with this dialect.
  template <typename T>
  void addAttribute() {
    // Add this attribute to the dialect and register it with the uniquer.
    addAttribute(T::getTypeID(), AbstractAttribute::get<T>(*this));
    detail::AttributeUniquer::registerAttribute<T>(context);
  }

  /// Register a type instance with this dialect.
  template <typename T>
  void addType() {
    // Add this type to the dialect and register it with the uniquer.
    addType(T::getTypeID(), AbstractType::get<T>(*this));
    detail::TypeUniquer::registerType<T>(context);
  }

  /// 该方言的命名空间。
  ///
  /// The namespace of this dialect.
  StringRef name;

  /// 派生的 Op 类的唯一标识符，在上下文中使用以允许多次注册相同的方言。
  ///
  /// The unique identifier of the derived Op class, this is used in the context
  /// to allow registering multiple times the same dialect.
  TypeID dialectID;

  /// 这是拥有此 Dialect 对象的上下文。
  ///
  /// This is the context that owns this Dialect object.
  MLIRContext *context;

  /// 标志指定此方言是否支持未注册的操作，即以方言命名空间为前缀但未使用 addOperation 
  /// 注册的操作。
  ///
  /// Flag that specifies whether this dialect supports unregistered operations,
  /// i.e. operations prefixed with the dialect namespace but not registered
  /// with addOperation.
  bool unknownOpsAllowed = false;

  /// 指定此方言是否允许未注册类型的标志，即以方言命名空间为前缀但未使用 addType 注册
  /// 的类型。这些类型用 OpaqueType 表示。
  ///
  /// Flag that specifies whether this dialect allows unregistered types, i.e.
  /// types prefixed with the dialect namespace but not registered with addType.
  /// These types are represented with OpaqueType.
  bool unknownTypesAllowed = false;

  /// 注册的方言接口的集合。
  ///
  /// A collection of registered dialect interfaces.
  DenseMap<TypeID, std::unique_ptr<DialectInterface>> registeredInterfaces;

  /// 方言（或其构造，即属性/操作/类型/等）承诺实现但尚未提供实现的一组接口。
  ///
  /// A set of interfaces that the dialect (or its constructs, i.e.
  /// Attributes/Operations/Types/etc.) has promised to implement, but has yet
  /// to provide an implementation for.
  DenseSet<std::pair<TypeID, TypeID>> unresolvedPromisedInterfaces;

  friend class DialectRegistry;
  friend void registerDialect();
  friend class MLIRContext;
};

} // namespace mlir

namespace llvm {
/// Provide isa functionality for Dialects.
template <typename T>
struct isa_impl<T, ::mlir::Dialect,
                std::enable_if_t<std::is_base_of<::mlir::Dialect, T>::value>> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::TypeID::get<T>() == dialect.getTypeID();
  }
};
template <typename T>
struct isa_impl<
    T, ::mlir::Dialect,
    std::enable_if_t<std::is_base_of<::mlir::DialectInterface, T>::value>> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return const_cast<::mlir::Dialect &>(dialect).getRegisteredInterface<T>();
  }
};
template <typename T>
struct cast_retty_impl<T, ::mlir::Dialect *> {
  using ret_type = T *;
};
template <typename T>
struct cast_retty_impl<T, ::mlir::Dialect> {
  using ret_type = T &;
};

template <typename T>
struct cast_convert_val<T, ::mlir::Dialect, ::mlir::Dialect> {
  template <typename To>
  static std::enable_if_t<std::is_base_of<::mlir::Dialect, To>::value, To &>
  doitImpl(::mlir::Dialect &dialect) {
    return static_cast<To &>(dialect);
  }
  template <typename To>
  static std::enable_if_t<std::is_base_of<::mlir::DialectInterface, To>::value,
                          To &>
  doitImpl(::mlir::Dialect &dialect) {
    return *dialect.getRegisteredInterface<To>();
  }

  static auto &doit(::mlir::Dialect &dialect) { return doitImpl<T>(dialect); }
};
template <class T>
struct cast_convert_val<T, ::mlir::Dialect *, ::mlir::Dialect *> {
  static auto doit(::mlir::Dialect *dialect) {
    return &cast_convert_val<T, ::mlir::Dialect, ::mlir::Dialect>::doit(
        *dialect);
  }
};

} // namespace llvm

#endif
