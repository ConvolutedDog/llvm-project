//===- Constraint.h - MLIR PDLL ODS Constraints -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a PDLL description of ODS constraints. These are used to
// support the import of constraints defined outside of PDLL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_
#define MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace pdll {
namespace ods {

//===----------------------------------------------------------------------===//
// Constraint
//===----------------------------------------------------------------------===//

/// 此类代表通用的 ODS 约束。
///
/// This class represents a generic ODS constraint.
class Constraint {
public:
  /// Return the unique name of this constraint.
  StringRef getName() const { return name; }

  /// 返回此约束的去重名称。这会尝试删除名称中纯粹为了唯一性而使用的部分，并显示底层名
  /// 称。因此，此名称确实保证了唯一性，并且仅应用于日志记录或其他有损友好的"漂亮"输出。
  ///
  /// Return the demangled name of this constraint. This tries to strip out bits
  /// of the name that are purely for uniquing, and show the underlying name. As
  /// such, this name does guarantee uniqueness and should only be used for
  /// logging or other lossy friendly "pretty" output.
  StringRef getDemangledName() const;

  /// 返回此约束的 summary。
  ///
  /// Return the summary of this constraint.
  StringRef getSummary() const { return summary; }

protected:
  Constraint(StringRef name, StringRef summary)
      : name(name.str()), summary(summary.str()) {}
  Constraint(const Constraint &) = delete;

private:
  /// 该约束的名称。
  ///
  /// The name of the constraint.
  std::string name;
  /// 该约束的 summary。
  /// A summary of the constraint.
  std::string summary;
};

//===----------------------------------------------------------------------===//
// AttributeConstraint
//===----------------------------------------------------------------------===//

/// 此类代表通用的 ODS Attribute 约束。
///
/// This class represents a generic ODS Attribute constraint.
class AttributeConstraint : public Constraint {
public:
  /// 返回此约束的底层 C++ 类的名称。
  ///
  /// Return the name of the underlying c++ class of this constraint.
  StringRef getCppClass() const { return cppClassName; }

private:
  AttributeConstraint(StringRef name, StringRef summary, StringRef cppClassName)
      : Constraint(name, summary), cppClassName(cppClassName.str()) {}

  /// 此约束的底层 C++ 类的名称。
  ///
  /// The c++ class of the constraint.
  std::string cppClassName;

  /// 允许 `Context` 类中调用私有的构造函数。
  ///
  /// Allow access to the constructor.
  friend class Context;
};

//===----------------------------------------------------------------------===//
// TypeConstraint
//===----------------------------------------------------------------------===//

/// 此类代表通用的 ODS Type 约束。
///
/// This class represents a generic ODS Type constraint.
class TypeConstraint : public Constraint {
public:
  /// 返回此约束的底层 C++ 类的名称。
  ///
  /// Return the name of the underlying c++ class of this constraint.
  StringRef getCppClass() const { return cppClassName; }

private:
  TypeConstraint(StringRef name, StringRef summary, StringRef cppClassName)
      : Constraint(name, summary), cppClassName(cppClassName.str()) {}

  /// 此约束的底层 C++ 类的名称。
  ///
  /// The c++ class of the constraint.
  std::string cppClassName;

  /// 允许 `Context` 类中调用私有的构造函数。
  ///
  /// Allow access to the constructor.
  friend class Context;
};

} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_ODS_CONSTRAINT_H_
