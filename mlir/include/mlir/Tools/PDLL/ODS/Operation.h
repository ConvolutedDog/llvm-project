//===- Operation.h - MLIR PDLL ODS Operation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_OPERATION_H_
#define MLIR_TOOLS_PDLL_ODS_OPERATION_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
namespace pdll {
namespace ods {
class AttributeConstraint;
class TypeConstraint;
class Dialect;

//===----------------------------------------------------------------------===//
// VariableLengthKind
//===----------------------------------------------------------------------===//

/// 描述变量的长度特性。具体来说：
/// Single: 变量的数量是固定的，不能多也不能少。
/// Optional: 变量的数量是可选的，即该变量可以出现一次或者不出现。
/// Variadic: 该变量是可变长的，即该变量可以出现多次。
enum VariableLengthKind { Single, Optional, Variadic };

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// 此类提供特定操作属性的 ODS 表示。这包括 name、optionality 等。
///
/// This class provides an ODS representation of a specific operation attribute.
/// This includes the name, optionality, and more.
class Attribute {
public:
  /// 返回当前 attribute 的名称。
  ///
  /// Return the name of this operand.
  StringRef getName() const { return name; }

  /// 返回这个 attribute 是不是可选的。
  ///
  /// Return true if this attribute is optional.
  bool isOptional() const { return optional; }

  /// 返回该属性的 ODS 约束，是一个 `AttributeConstraint` 对象。
  ///
  /// Return the constraint of this attribute.
  const AttributeConstraint &getConstraint() const { return constraint; }

private:
  Attribute(StringRef name, bool optional,
            const AttributeConstraint &constraint)
      : name(name.str()), optional(optional), constraint(constraint) {}

  /// 当前 attribute 的名称。
  ///
  /// The ODS name of the attribute.
  std::string name;

  /// 这个 attribute 是不是可选的。
  ///
  /// A flag indicating if the attribute is optional.
  bool optional;

  /// 该属性的 ODS 约束，是一个 `AttributeConstraint` 对象。
  ///
  /// The ODS constraint of this attribute.
  const AttributeConstraint &constraint;

  /// 允许 `Operation` 类中调用私有的构造函数。
  ///
  /// Allow access to the private constructor.
  friend class Operation;
};

//===----------------------------------------------------------------------===//
// OperandOrResult
//===----------------------------------------------------------------------===//

/// 此类提供特定 Operation 的 `操作数或结果` 的 ODS 表示。其中包括名称、可变长度标志等。
///
/// This class provides an ODS representation of a specific operation operand or
/// result. This includes the name, variable length flags, and more.
class OperandOrResult {
public:
  /// 返回这个 `操作数或结果` 的名称。
  ///
  /// Return the name of this value.
  StringRef getName() const { return name; }

  /// 如果此 `操作数或结果` 是可变长度，即如果它是可变参数或可选的，则返回 true。
  ///
  /// Returns true if this value is variable length, i.e. if it is Variadic or
  /// Optional.
  bool isVariableLength() const {
    return variableLengthKind != VariableLengthKind::Single;
  }

  /// 如果此 `操作数或结果` 是可变长度（不包括 Optional），则返回 true。
  ///
  /// Returns true if this value is variadic (Note this is false if the value is
  /// Optional).
  bool isVariadic() const {
    return variableLengthKind == VariableLengthKind::Variadic;
  }

  /// 返回此变量的可变长度类型，可选的：Single, Optional, Variadic。
  ///
  /// Returns the variable length kind of this value.
  VariableLengthKind getVariableLengthKind() const {
    return variableLengthKind;
  }

  /// 返回该变量的类型约束，是一个 TypeConstraint 对象。
  ///
  /// Return the constraint of this value.
  const TypeConstraint &getConstraint() const { return constraint; }

private:
  OperandOrResult(StringRef name, VariableLengthKind variableLengthKind,
                  const TypeConstraint &constraint)
      : name(name.str()), variableLengthKind(variableLengthKind),
        constraint(constraint) {}

  /// 该变量的名称。
  ///
  /// The ODS name of this value.
  std::string name;

  /// 该变量的可变长度类型，可选的：Single, Optional, Variadic。
  ///
  /// The variable length kind of this value.
  VariableLengthKind variableLengthKind;

  /// 该变量的类型约束。
  ///
  /// The ODS constraint of this value.
  const TypeConstraint &constraint;

  /// Allow access to the private constructor.
  friend class Operation;
};

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

/// 此类提供特定操作的 ODS 表示。这包括 PDL 前端生成模式重写代码所需的所有信息。
///
/// This class provides an ODS representation of a specific operation. This
/// includes all of the information necessary for use by the PDL frontend for
/// generating code for a pattern rewrite.
class Operation {
public:
  /// 返回此操作的源代码位置。
  ///
  /// Return the source location of this operation.
  SMRange getLoc() const { return location; }

  /// 将 attribute 添加到当前 Operation 里。
  ///
  /// Append an attribute to this operation.
  void appendAttribute(StringRef name, bool optional,
                       const AttributeConstraint &constraint) {
    attributes.emplace_back(Attribute(name, optional, constraint));
  }

  /// 向当前 Operation 添加一个操作数，这个操作数存储在 operands 中。
  ///
  /// Append an operand to this operation.
  void appendOperand(StringRef name, VariableLengthKind variableLengthKind,
                     const TypeConstraint &constraint) {
    operands.emplace_back(
        OperandOrResult(name, variableLengthKind, constraint));
  }

  /// 向当前 Operation 添加一个结果，该结果存在 results 中。
  ///
  /// Append a result to this operation.
  void appendResult(StringRef name, VariableLengthKind variableLengthKind,
                    const TypeConstraint &constraint) {
    results.emplace_back(OperandOrResult(name, variableLengthKind, constraint));
  }

  /// 返回当前 Operation 的名称。
  ///
  /// Returns the name of the operation.
  StringRef getName() const { return name; }

  /// 返回当前 Operation 的 summary。
  ///
  /// Returns the summary of the operation.
  StringRef getSummary() const { return summary; }

  /// 返回当前 Operation 的 description。
  ///
  /// Returns the description of the operation.
  StringRef getDescription() const { return description; }

  /// 返回当前 Operation 的 native class name。
  ///
  /// Returns the native class name of the operation.
  StringRef getNativeClassName() const { return nativeClassName; }

  /// 返回当前 Operation 的 attributes。
  ///
  /// Returns the attributes of this operation.
  ArrayRef<Attribute> getAttributes() const { return attributes; }

  /// 返回当前 Operation 的操作数。
  ///
  /// Returns the operands of this operation.
  ArrayRef<OperandOrResult> getOperands() const { return operands; }

  /// 返回当前 Operation 的 结果。
  ///
  /// Returns the results of this operation.
  ArrayRef<OperandOrResult> getResults() const { return results; }

  /// 返回当前 Operation 是否支持类型推断。
  ///
  /// Return if the operation is known to support result type inferrence.
  bool hasResultTypeInferrence() const { return supportsTypeInferrence; }

private:
  Operation(StringRef name, StringRef summary, StringRef desc,
            StringRef nativeClassName, bool supportsTypeInferrence, SMLoc loc);

  /// 该 Operation 的名称。
  ///
  /// The name of the operation.
  std::string name;

  /// 该 Operation 的文档。
  ///
  /// The documentation of the operation.
  std::string summary;
  std::string description;

  /// The native class name of the operation, used when generating native code.
  std::string nativeClassName;

  /// 标志指示操作是否支持类型推断。
  ///
  /// Flag indicating if the operation is known to support type inferrence.
  bool supportsTypeInferrence;

  /// 该操作在源码中的位置。
  ///
  /// The source location of this operation.
  SMRange location;

  /// 当前 Operation 的操作数。
  ///
  /// The operands of the operation.
  SmallVector<OperandOrResult> operands;

  /// 当前 Operation 的结果。
  ///
  /// The results of the operation.
  SmallVector<OperandOrResult> results;

  /// 属于此操作的 attributes 的 SmallVector。
  ///
  /// The attributes of the operation.
  SmallVector<Attribute> attributes;

  /// 允许 `Context` 类中调用私有的构造函数。
  ///
  /// Allow access to the private constructor.
  friend class Dialect;
};
} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_ODS_OPERATION_H_
