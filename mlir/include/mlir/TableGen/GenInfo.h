//===- GenInfo.h - Generator info -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENINFO_H_
#define MLIR_TABLEGEN_GENINFO_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <utility>

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace mlir {

/// 要调用的生成器函数。
///
/// Generator function to invoke.
using GenFunction =
    std::function<bool(const llvm::RecordKeeper &records, raw_ostream &os)>;

/// 用于对有关生成器的信息进行分组的结构（通过 mlir-tblgen 调用的参数、描述和生成
/// 器函数）。
///
/// Structure to group information about a generator (argument to invoke via
/// mlir-tblgen, description, and generator function).
class GenInfo {
public:
  /// 不应直接调用 GenInfo 构造函数，而应使用 GenRegistration 或 registerGen。
  ///
  /// GenInfo constructor should not be invoked directly, instead use
  /// GenRegistration or registerGen.
  GenInfo(StringRef arg, StringRef description, GenFunction generator)
      : arg(arg), description(description), generator(std::move(generator)) {}

  /// 调用生成器并返回生成器是否失败。
  ///
  /// Invokes the generator and returns whether the generator failed.
  bool invoke(const llvm::RecordKeeper &records, raw_ostream &os) const {
    assert(generator && "Cannot call generator with null generator");
    return generator(records, os);
  }

  /// 返回可传递给“mlir-tblgen”以调用此生成器的命令行选项。
  ///
  /// Returns the command line option that may be passed to 'mlir-tblgen' to
  /// invoke this generator.
  StringRef getGenArgument() const { return arg; }

  /// 返回生成器的描述。
  ///
  /// Returns a description for the generator.
  StringRef getGenDescription() const { return description; }

private:
  // 传递给“mlir-tblgen”以调用此生成器的命令行选项。
  //
  // The argument with which to invoke the generator via mlir-tblgen.
  StringRef arg;

  // 生成器的描述。
  //
  // Description of the generator.
  StringRef description;

  // 生成器函数。
  //
  // Generator function.
  GenFunction generator;
};

/// GenRegistration 提供了一个全局初始化程序，用于注册生成器函数。
///
/// Usage:
///
///   // At namespace scope.
///   static GenRegistration Print("print", "Print records", [](...){...});
///
/// GenRegistration provides a global initializer that registers a generator
/// function.
///
/// Usage:
///
///   // At namespace scope.
///   static GenRegistration Print("print", "Print records", [](...){...});
struct GenRegistration {
  GenRegistration(StringRef arg, StringRef description,
                  const GenFunction &function);
};

} // namespace mlir

#endif // MLIR_TABLEGEN_GENINFO_H_
