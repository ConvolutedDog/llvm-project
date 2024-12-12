//===- GenInfo.cpp - Generator info -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

static llvm::ManagedStatic<std::vector<GenInfo>> generatorRegistry;

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
GenRegistration::GenRegistration(StringRef arg, StringRef description,
                                 const GenFunction &function) {
  // `(arg, description, function)` is implicitly converted to a
  // `GenInfo` object.
  generatorRegistry->emplace_back(arg, description, function);
}

/// 为每个注册的生成器添加命令行选项。
///
/// Adds command line option for each registered generator.
GenNameParser::GenNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const GenInfo *>(opt) {
  for (const auto &kv : *generatorRegistry) {
    addLiteralOption(kv.getGenArgument(), &kv, kv.getGenDescription());
  }
}

void GenNameParser::printOptionInfo(const llvm::cl::Option &o,
                                    size_t globalWidth) const {
  GenNameParser *tp = const_cast<GenNameParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const GenNameParser::OptionInfo *vT1,
                          const GenNameParser::OptionInfo *vT2) {
                         return vT1->Name.compare(vT2->Name);
                       });
  using llvm::cl::parser;
  parser<const GenInfo *>::printOptionInfo(o, globalWidth);
}
