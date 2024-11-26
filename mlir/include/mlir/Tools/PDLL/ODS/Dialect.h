//===- Dialect.h - PDLL ODS Dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_DIALECT_H_
#define MLIR_TOOLS_PDLL_ODS_DIALECT_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace pdll {
namespace ods {
class Operation;

/// This class represents an ODS dialect, and contains information on the
/// constructs held within the dialect.
class Dialect {
public:
  ~Dialect();

  /// Return the name of this dialect.
  StringRef getName() const { return name; }

  /// 向当前 dialect 插入新 Operation。返回插入的 Operation，以及指示该 Operation
  /// 是否新插入的布尔值（如果该操作已存在则返回 false）。
  ///
  /// Insert a new operation with the dialect. Returns the inserted operation,
  /// and a boolean indicating if the operation newly inserted (false if the
  /// operation already existed).
  std::pair<Operation *, bool>
  insertOperation(StringRef name, StringRef summary, StringRef desc,
                  StringRef nativeClassName, bool supportsResultTypeInferrence,
                  SMLoc loc);

  /// 查找使用给定名称注册的 Operation，如果没有注册具有该名称的 Operation，则返
  /// 回 nullptr。
  ///
  /// Lookup an operation registered with the given name, or null if no
  /// operation with that name is registered.
  Operation *lookupOperation(StringRef name) const;

  /// 返回所有的当前 dialect 定义的 Operation。
  ///
  /// Return a map of all of the operations registered to this dialect.
  const llvm::StringMap<std::unique_ptr<Operation>> &getOperations() const {
    return operations;
  }

private:
  explicit Dialect(StringRef name);

  /// 当前 dialect 的名称。
  ///
  /// The name of the dialect.
  std::string name;

  /// 所有的当前 dialect 定义的 Operation。
  ///
  /// The operations defined by the dialect.
  llvm::StringMap<std::unique_ptr<Operation>> operations;

  /// 允许 Content 类调用构造函数。
  ///
  /// Allow access to the constructor.
  friend class Context;
};
} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_ODS_DIALECT_H_
