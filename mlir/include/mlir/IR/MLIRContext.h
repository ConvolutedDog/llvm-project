//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include <functional>
#include <memory>
#include <vector>

namespace llvm {
class ThreadPoolInterface;
} // namespace llvm

namespace mlir {
namespace tracing {
class Action;
}
class DiagnosticEngine;
class Dialect;
class DialectRegistry;
class DynamicDialect;
class InFlightDiagnostic;
class Location;
class MLIRContextImpl;
class RegisteredOperationName;
class StorageUniquer;
class IRUnit;

/// MLIRContext 是 MLIR operations 的集合的顶级对象。它包含 never dying 的唯一对
/// 象（如类型）以及 tables used to unique them。
///
/// MLIRContext 获得一个多余的 "MLIR" 前缀，因为否则它最终会得到一个非常通用的名称
///（“Context”），并且客户端很少与其交互。
///
/// context 包装了一些多线程设施，特别是默认情况下它将隐式创建一个线程池。如果同时存
/// 在多个 context，或者如果某个进程将长期存在并创建和销毁 contexts，这可能是不可取
/// 的。为了更好地控制线程生成，可以在 context 中注入外部拥有的 ThreadPool。例如：
///
///  llvm::DefaultThreadPool myThreadPool;
///  while (auto *request = nextCompilationRequests()) {
///    MLIRContext ctx(registry, MLIRContext::Threading::DISABLED);
///    ctx.setThreadPool(myThreadPool);
///    processRequest(request, cxt);
///  }
///
/// MLIRContext is the top-level object for a collection of MLIR operations. It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
/// The context wrap some multi-threading facilities, and in particular by
/// default it will implicitly create a thread pool.
/// This can be undesirable if multiple context exists at the same time or if a
/// process will be long-lived and create and destroy contexts.
/// To control better thread spawning, an externally owned ThreadPool can be
/// injected in the context. For example:
///
///  llvm::DefaultThreadPool myThreadPool;
///  while (auto *request = nextCompilationRequests()) {
///    MLIRContext ctx(registry, MLIRContext::Threading::DISABLED);
///    ctx.setThreadPool(myThreadPool);
///    processRequest(request, cxt);
///  }
///
class MLIRContext {
public:
  /// 控制 MLIRContext 对象在多线程环境下的行为：
  /// 1. Single-Threaded（单线程）：在这种模式下，MLIRContext 只会被单个线程使用。
  /// 这种情况下，可以关闭所有的线程安全保护，以避免不必要的性能开销。
  /// 2. Multi-Threaded（多线程）：在这种模式下，MLIRContext 可以被多个线程并发访
  /// 问。必须启用线程安全的机制，例如通过锁(lock)或原子操作(atomic operation)
  /// 来保护对共享资源的访问，以避免数据竞争。
  enum class Threading { DISABLED, ENABLED };
  /// 构造函数。
  ///
  /// Create a new Context.
  explicit MLIRContext(Threading multithreading = Threading::ENABLED);
  explicit MLIRContext(const DialectRegistry &registry,
                       Threading multithreading = Threading::ENABLED);
  ~MLIRContext();

  /// 返回所有 registered IR dialects，作为一个 Dialect * 的向量返回。
  ///
  /// Return information about all IR dialects loaded in the context.
  std::vector<Dialect *> getLoadedDialects();

  /// 返回与此上下文相关的方言注册表。
  ///
  /// Return the dialect registry associated with this context.
  const DialectRegistry &getDialectRegistry();

  /// 将给定方言注册表的内容附加到与此上下文关联的注册表中。
  ///
  /// Append the contents of the given dialect registry to the registry
  /// associated with this context.
  void appendDialectRegistry(const DialectRegistry &registry);

  /// 遍历当前 context 创建的方言列表，返回所有 registered IR dialects 的名字向量。
  ///
  /// Return information about all available dialects in the registry in this
  /// context.
  std::vector<StringRef> getAvailableDialects();

  /// 获取具有给定命名空间的已注册 IR 方言。如果未找到完全匹配，则返回 nullptr。
  ///
  /// Get a registered IR dialect with the given namespace. If an exact match is
  /// not found, then return nullptr.
  Dialect *getLoadedDialect(StringRef name);

  /// 获取给定派生方言类型的已注册 IR 方言。派生类型必须提供静态 `getDialectNamespace`
  /// 方法。
  ///
  /// Get a registered IR dialect for the given derived dialect type. The
  /// derived type must provide a static 'getDialectNamespace' method.
  template <typename T>
  T *getLoadedDialect() {
    return static_cast<T *>(getLoadedDialect(T::getDialectNamespace()));
  }

  /// 获取（或创建）给定派生方言类型的方言。派生类型必须提供静态 `getDialectNamespace` 
  /// 方法。
  ///
  /// Get (or create) a dialect for the given derived dialect type. The derived
  /// type must provide a static 'getDialectNamespace' method.
  template <typename T>
  T *getOrLoadDialect() {
    return static_cast<T *>(
        getOrLoadDialect(T::getDialectNamespace(), TypeID::get<T>(), [this]() {
          std::unique_ptr<T> dialect(new T(this));
          return dialect;
        }));
  }

  /// 在上下文中加载单个方言。
  ///
  /// Load a dialect in the context.
  template <typename Dialect>
  void loadDialect() {
    // Do not load the dialect if it is currently loading. This can happen if a
    // dialect initializer triggers loading the same dialect recursively.
    if (!isDialectLoading(Dialect::getDialectNamespace()))
      getOrLoadDialect<Dialect>();
  }

  /// 在上下文中加载方言列表。方言列表作为模板参数传入。
  ///
  /// Load a list dialects in the context.
  template <typename Dialect, typename OtherDialect, typename... MoreDialects>
  void loadDialect() {
    loadDialect<Dialect>();
    loadDialect<OtherDialect, MoreDialects...>();
  }

  /// Get (or create) a dynamic dialect for the given name.
  DynamicDialect *
  getOrLoadDynamicDialect(StringRef dialectNamespace,
                          function_ref<void(DynamicDialect *)> ctor);

  /// 在此上下文中加载注册表中可用的所有方言。
  ///
  /// Load all dialects available in the registry in this context.
  void loadAllAvailableDialects();

  /// 获取（或创建）给定派生方言名称的方言。如果未找到方言，则将从注册表中加载
  /// 方言。如果未为此名称加载方言且注册表中没有可用的方言，则返回 nullptr。
  ///
  /// Get (or create) a dialect for the given derived dialect name.
  /// The dialect will be loaded from the registry if no dialect is found.
  /// If no dialect is loaded for this name and none is available in the
  /// registry, returns nullptr.
  Dialect *getOrLoadDialect(StringRef name);

  /// 如果我们允许为未注册的方言创建操作，则返回 true。
  ///
  /// Return true if we allow to create operation for unregistered dialects.
  [[nodiscard]] bool allowsUnregisteredDialects();

  /// 允许为未注册的方言创建操作。
  /// **强烈不建议**使用此选项：它在测试期间很方便，但在生产代码中使用它不是一个好
  /// 习惯。一些系统不变量可能会被破坏（例如在创建操作后加载方言），而不会被断言或
  /// 其他方式捕获。
  ///
  /// Enables creating operations in unregistered dialects.
  /// This option is **heavily discouraged**: it is convenient during testing
  /// but it is not a good practice to use it in production code. Some system
  /// invariants can be broken (like loading a dialect after creating
  ///  operations) without being caught by assertions or other means.
  void allowUnregisteredDialects(bool allow = true);

  /// Return true if multi-threading is enabled by the context.
  bool isMultithreadingEnabled();

  /// Set the flag specifying if multi-threading is disabled by the context.
  /// The command line debugging flag `--mlir-disable-threading` is overriding
  /// this call and making it a no-op!
  void disableMultithreading(bool disable = true);
  void enableMultithreading(bool enable = true) {
    disableMultithreading(!enable);
  }

  /// Set a new thread pool to be used in this context. This method requires
  /// that multithreading is disabled for this context prior to the call. This
  /// allows to share a thread pool across multiple contexts, as well as
  /// decoupling the lifetime of the threads from the contexts. The thread pool
  /// must outlive the context. Multi-threading will be enabled as part of this
  /// method.
  /// The command line debugging flag `--mlir-disable-threading` will still
  /// prevent threading from being enabled and threading won't be enabled after
  /// this call in this case.
  void setThreadPool(llvm::ThreadPoolInterface &pool);

  /// Return the number of threads used by the thread pool in this context. The
  /// number of computed hardware threads can change over the lifetime of a
  /// process based on affinity changes, so users should use the number of
  /// threads actually in the thread pool for dispatching work. Returns 1 if
  /// multithreading is disabled.
  unsigned getNumThreads();

  /// Return the thread pool used by this context. This method requires that
  /// multithreading be enabled within the context, and should generally not be
  /// used directly. Users should instead prefer the threading utilities within
  /// Threading.h.
  llvm::ThreadPoolInterface &getThreadPool();

  /// Return true if we should attach the operation to diagnostics emitted via
  /// Operation::emit.
  bool shouldPrintOpOnDiagnostic();

  /// Set the flag specifying if we should attach the operation to diagnostics
  /// emitted via Operation::emit.
  void printOpOnDiagnostic(bool enable);

  /// Return true if we should attach the current stacktrace to diagnostics when
  /// emitted.
  bool shouldPrintStackTraceOnDiagnostic();

  /// Set the flag specifying if we should attach the current stacktrace when
  /// emitting diagnostics.
  void printStackTraceOnDiagnostic(bool enable);

  /// 返回包含有关所有已注册操作的信息的排序数组。
  ///
  /// Return a sorted array containing the information about all registered
  /// operations.
  ArrayRef<RegisteredOperationName> getRegisteredOperations();

  /// 判断操作是否已被注册。
  ///
  /// Return true if this operation name is registered in this context.
  bool isOperationRegistered(StringRef name);

  // 由于只有 MLIRContext.cpp 可以看到 MLIRContextImpl 类型，因此这实际上是私有的。
  //
  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() { return *impl; }

  /// Returns the diagnostic engine for this context.
  DiagnosticEngine &getDiagEngine();

  /// Returns the storage uniquer used for creating affine constructs.
  StorageUniquer &getAffineUniquer();

  /// Returns the storage uniquer used for constructing type storage instances.
  /// This should not be used directly.
  StorageUniquer &getTypeUniquer();

  /// Returns the storage uniquer used for constructing attribute storage
  /// instances. This should not be used directly.
  StorageUniquer &getAttributeUniquer();

  /// These APIs are tracking whether the context will be used in a
  /// multithreading environment: this has no effect other than enabling
  /// assertions on misuses of some APIs.
  void enterMultiThreadedExecution();
  void exitMultiThreadedExecution();

  /// 获取所提供命名空间和 TypeID 的方言：如果此命名空间存在具有不同 TypeID 的方言，
  /// 则中止程序。如果尚未为此 namespace/TypeID 加载方言，使用提供的 ctor 动态创建
  /// 一个并加载它。返回指向上下文所拥有的方言的指针。一般不鼓励使用此方法，而建议使用
  /// `getOrLoadDialect<DialectClass>()`。
  ///
  /// Get a dialect for the provided namespace and TypeID: abort the program if
  /// a dialect exist for this namespace with different TypeID. If a dialect has
  /// not been loaded for this namespace/TypeID yet, use the provided ctor to
  /// create one on the fly and load it. Returns a pointer to the dialect owned
  /// by the context.
  /// The use of this method is in general discouraged in favor of
  /// 'getOrLoadDialect<DialectClass>()'.
  Dialect *getOrLoadDialect(StringRef dialectNamespace, TypeID dialectID,
                            function_ref<std::unique_ptr<Dialect>()> ctor);

  /// 返回上下文注册表的哈希值，可用于粗略指示上下文注册表的状态是否已更改。上下文注册
  /// 表与已加载的方言及其实体（属性、操作、类型等）相关。
  ///
  /// Returns a hash of the registry of the context that may be used to give
  /// a rough indicator of if the state of the context registry has changed. The
  /// context registry correlates to loaded dialects and their entities
  /// (attributes, operations, types, etc.).
  llvm::hash_code getRegistryHash();

  //===--------------------------------------------------------------------===//
  // Action API
  //===--------------------------------------------------------------------===//

  /// Signatures for the action handler that can be registered with the context.
  using HandlerTy =
      std::function<void(function_ref<void()>, const tracing::Action &)>;

  /// Register a handler for handling actions that are dispatched through this
  /// context. A nullptr handler can be set to disable a previously set handler.
  void registerActionHandler(HandlerTy handler);

  /// Return true if a valid ActionHandler is set.
  bool hasActionHandler();

  /// Dispatch the provided action to the handler if any, or just execute it.
  void executeAction(function_ref<void()> actionFn,
                     const tracing::Action &action) {
    if (LLVM_UNLIKELY(hasActionHandler()))
      executeActionInternal(actionFn, action);
    else
      actionFn();
  }

  /// Dispatch the provided action to the handler if any, or just execute it.
  template <typename ActionTy, typename... Args>
  void executeAction(function_ref<void()> actionFn, ArrayRef<IRUnit> irUnits,
                     Args &&...args) {
    if (LLVM_UNLIKELY(hasActionHandler()))
      executeActionInternal<ActionTy, Args...>(actionFn, irUnits,
                                               std::forward<Args>(args)...);
    else
      actionFn();
  }

private:
  /// 如果给定的方言当前正在加载，则返回 true。
  ///
  /// Return true if the given dialect is currently loading.
  bool isDialectLoading(StringRef dialectNamespace);

  /// Internal helper for the dispatch method.
  void executeActionInternal(function_ref<void()> actionFn,
                             const tracing::Action &action);

  /// 调度方法的内部帮助程序。我们在检查是否存在处理程序后到达此处，目的是使此代码保持
  /// 不在线。并避免不必要地调用 Action 的构造函数。
  ///
  /// Internal helper for the dispatch method. We get here after checking that
  /// there is a handler, for the purpose of keeping this code out-of-line. and
  /// avoid calling the ctor for the Action unnecessarily.
  template <typename ActionTy, typename... Args>
  LLVM_ATTRIBUTE_NOINLINE void
  executeActionInternal(function_ref<void()> actionFn, ArrayRef<IRUnit> irUnits,
                        Args &&...args) {
    executeActionInternal(actionFn,
                          ActionTy(irUnits, std::forward<Args>(args)...));
  }

  const std::unique_ptr<MLIRContextImpl> impl;

  MLIRContext(const MLIRContext &) = delete;
  void operator=(const MLIRContext &) = delete;
};

//===----------------------------------------------------------------------===//
// MLIRContext CommandLine Options
//===----------------------------------------------------------------------===//

/// Register a set of useful command-line options that can be used to configure
/// various flags within the MLIRContext. These flags are used when constructing
/// an MLIR context for initialization.
void registerMLIRContextCLOptions();

} // namespace mlir

#endif // MLIR_IR_MLIRCONTEXT_H
