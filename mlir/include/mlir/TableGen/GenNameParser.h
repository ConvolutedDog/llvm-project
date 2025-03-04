//===- GenNameParser.h - Command line parser for generators -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The GenNameParser class adds all passes linked in to the system that are
// creatable to the tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENNAMEPARSER_H_
#define MLIR_TABLEGEN_GENNAMEPARSER_H_

#include "llvm/Support/CommandLine.h"

namespace mlir {
class GenInfo;

/// 为每个注册的生成器添加命令行选项。
///
/// Adds command line option for each registered generator.
struct GenNameParser : public llvm::cl::parser<const GenInfo *> {
  GenNameParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &o,
                       size_t globalWidth) const override;
};
} // namespace mlir

#endif // MLIR_TABLEGEN_GENNAMEPARSER_H_
