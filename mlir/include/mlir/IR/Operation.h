//===- Operation.h - MLIR Operation Class -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Operation class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATION_H
#define MLIR_IR_OPERATION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/Twine.h"
#include <optional>

namespace mlir {
namespace detail {
/// This is a "tag" used for mapping the properties storage in
/// llvm::TrailingObjects.
enum class OpProperties : char {};
} // namespace detail

/// Operation 是 MLIR 中的基本执行单元。
///
/// 建议阅读以下文档以了解此类：
/// - https://mlir.llvm.org/docs/LangRef/#operations
/// - https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
///
/// Operation 首先由其名称定义，该名称是一个唯一字符串。名称的解释是，如果它
/// 包含 "." 字符，则之前的部分是此 Operation 所属的方言名称，后面的所有内容
/// 都是方言中的此 Operation 名称。
///
/// Operation 定义零个或多个 SSA `Value`，我们将其称为 Operation results。这
/// 个 array of Value 实际以相反的顺序存储在 Operation 之前的 memory 中。对
/// 于具有 3 个结果的 Operation ，我们分配以下内存布局：
///
/// [Result2, Result1, Result0, Operation]
///                             ^ 这是 `Operation*` 指针指向的位置。
///
/// 这样做的结果是，此类必须分配到堆中，由各种 `create` 方法处理。每个 result
/// 包含：
///  - 指向第一次使用的指针（参见 `OpOperand`）
///  - 此结果定义的 SSA 值的类型。
///  - 此结果在数组中的索引。
/// Results 被定义为 `ValueImpl` 的子类，更准确地说是被定义为 `OpResultImpl`
/// 的仅有的两个子类：`InlineOpResult` 和 `OutOfLineOpResult`。前者用于前 5
/// 个结果，后者用于后续结果。它们在存储索引的方式上有所不同：前 5 个结果只
/// 需要 3 位，因此与 Type 指针一起 paced，而后续结果具有额外的 `unsigned`
/// value，因此需要更多空间。
///
/// Operation 也有零个或多个 Operation 数：这些是 SSA Value 的用途，可以是其他 Op
/// 或 Block 参数的 results。这些用途中的每一个都是 `OpOperand` 的一个实例。
/// 这个 optional array 最初与 Operation 类本身一起分配，但可以根据需要在动
/// 态分配中动态 moved out-of-line 。
///
/// Operation 可以包含一个或多个 Regions（可选），存储在尾部分配的 array 中。
/// 每个 `Region` 都是一个 `Block` 列表。每个 `Block` 本身就是一个 Operation
/// 列表。这种结构实际上形成了一棵树。
///
/// 一些 Operation（如 branches）也引用其他 Block，在这种情况下，它们将有一
/// 个 `BlockOperand` array。
///
/// Operation 可以包含可选的 "Properties" 对象：这是一个具有固定大小的预定义
/// C++ 对象。此对象由 Operation 拥有并随 Operation 删除。它可以根据需要转换
/// 为 Attribute，也可以从 Attribute 加载。
///
///
/// 最后，Operation 还包含一个可选的 `DictionaryAttr`、一个 Location 和一个
/// pointer to its parent Block (if any)。
///
/// Operation is the basic unit of execution within MLIR.
///
/// The following documentation are recommended to understand this class:
/// - https://mlir.llvm.org/docs/LangRef/#operations
/// - https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
///
/// An Operation is defined first by its name, which is a unique string. The
/// name is interpreted so that if it contains a '.' character, the part before
/// is the dialect name this operation belongs to, and everything that follows
/// is this operation name within the dialect.
///
/// An Operation defines zero or more SSA `Value` that we refer to as the
/// Operation results. This array of Value is actually stored in memory before
/// the Operation itself in reverse order. That is for an Operation with 3
/// results we allocate the following memory layout:
///
///  [Result2, Result1, Result0, Operation]
///                              ^ this is where `Operation*` pointer points to.
///
/// A consequence of this is that this class must be heap allocated, which is
/// handled by the various `create` methods. Each result contains:
///  - one pointer to the first use (see `OpOperand`)
///  - the type of the SSA Value this result defines.
///  - the index for this result in the array.
/// The results are defined as subclass of `ValueImpl`, and more precisely as
/// the only two subclasses of `OpResultImpl`: `InlineOpResult` and
/// `OutOfLineOpResult`. The former is used for the first 5 results and the
/// latter for the subsequent ones. They differ in how they store their index:
/// the first 5 results only need 3 bits and thus are packed with the Type
/// pointer, while the subsequent one have an extra `unsigned` value and thus
/// need more space.
///
/// An Operation also has zero or more operands: these are uses of SSA Value,
/// which can be the results of other operations or Block arguments. Each of
/// these uses is an instance of `OpOperand`. This optional array is initially
/// tail allocated with the operation class itself, but can be dynamically moved
/// out-of-line in a dynamic allocation as needed.
///
/// An Operation may contain optionally one or multiple Regions, stored in a
/// tail allocated array. Each `Region` is a list of Blocks. Each `Block` is
/// itself a list of Operations. This structure is effectively forming a tree.
///
/// Some operations like branches also refer to other Block, in which case they
/// would have an array of `BlockOperand`.
///
/// An Operation may contain optionally a "Properties" object: this is a
/// pre-defined C++ object with a fixed size. This object is owned by the
/// operation and deleted with the operation. It can be converted to an
/// Attribute on demand, or loaded from an Attribute.
///
///
/// Finally an Operation also contain an optional `DictionaryAttr`, a Location,
/// and a pointer to its parent Block (if any).
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    detail::OpProperties, BlockOperand, Region,
                                    OpOperand> {
public:
  /// 使用特定字段创建新 Operation。如有必要，此构造函数将使用默认属性填充提供的属性
  /// 列表。
  ///
  /// Create a new Operation with the specific fields. This constructor
  /// populates the provided attribute list with default attributes if
  /// necessary.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           NamedAttrList &&attributes,
                           OpaqueProperties properties, BlockRange successors,
                           unsigned numRegions);

  /// 使用特定字段创建新 Operation。此构造函数使用现有属性字典，以避免唯一化属性列表。
  ///
  /// Create a new Operation with the specific fields. This constructor uses an
  /// existing attribute dictionary to avoid uniquing a list of attributes.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           DictionaryAttr attributes,
                           OpaqueProperties properties, BlockRange successors,
                           unsigned numRegions);

  /// 从存储在 `state` 中的字段创建一个新的 Operation。
  ///
  /// Create a new Operation from the fields stored in `state`.
  static Operation *create(const OperationState &state);

  /// 使用特定字段创建新 Operation。
  ///
  /// Create a new Operation with the specific fields.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           NamedAttrList &&attributes,
                           OpaqueProperties properties,
                           BlockRange successors = {},
                           RegionRange regions = {});

  /// Operation 的名称是它的关键标识符。
  ///
  /// The name of an operation is the key identifier for it.
  OperationName getName() { return name; }

  /// 如果此 Operation 具有已注册的 Operation description，则返回它。否则返回
  /// std::nullopt。
  ///
  /// If this operation has a registered operation description, return it.
  /// Otherwise return std::nullopt.
  std::optional<RegisteredOperationName> getRegisteredInfo() {
    return getName().getRegisteredInfo();
  }

  /// 如果此 Operation 具有注册的 Operation 描述，则返回 true，否则返回 false。
  ///
  /// Returns true if this operation has a registered operation description,
  /// otherwise false.
  bool isRegistered() { return getName().isRegistered(); }

  /// 从其 parent block 中移除此 Operation 并将其删除。
  ///
  /// Remove this operation from its parent block and delete it.
  void erase();

  /// 从其 parent block 中移除 Operation ，但不要删除它。
  ///
  /// Remove the operation from its parent block, but don't delete it.
  void remove();

  /// 包含与克隆 Operation 相关的各种 options 的类。此类的用户应将其传递给 Operation
  /// 的 'clone' 方法。
  /// 当前选项包括：
  /// * 克隆是否应递归遍历 Operation 的 regions。
  /// * 克隆是否还应克隆 Operation 的 operands。
  ///
  /// Class encompassing various options related to cloning an operation. Users
  /// of this class should pass it to Operation's 'clone' methods.
  /// Current options include:
  /// * Whether cloning should recursively traverse into the regions of the
  ///   operation or not.
  /// * Whether cloning should also clone the operands of the operation.
  class CloneOptions {
  public:
    /// 默认构造一个选项，所有标志都设置为 false。这意味着 Operation 中可以选择不克隆
    /// 的所有部分都不会被克隆。
    ///
    /// Default constructs an option with all flags set to false. That means all
    /// parts of an operation that may optionally not be cloned, are not cloned.
    CloneOptions();

    /// 构造一个实例，并相应设置克隆 regions 和克隆 operands 标志。
    ///
    /// Constructs an instance with the clone regions and clone operands flags
    /// set accordingly.
    CloneOptions(bool cloneRegions, bool cloneOperands);

    /// 返回一个所有标志都设置为 true 的实例。这是使用 clone 方法时的默认设置，可克隆
    /// Operation 的所有部分。
    ///
    /// Returns an instance with all flags set to true. This is the default
    /// when using the clone method and clones all parts of the operation.
    static CloneOptions all();

    /// 配置克隆是否应遍历 Operation 的任何 regions。如果设置为 true，则 Operation
    /// 的 regions 将被递归克隆。如果设置为 false，克隆的 Operation 将具有相同数量
    /// 的 regions，但它们将为空。
    ///
    /// Operation 的 regions 中嵌套的 Operation 的克隆目前不受其他标志的影响。
    ///
    /// Configures whether cloning should traverse into any of the regions of
    /// the operation. If set to true, the operation's regions are recursively
    /// cloned. If set to false, cloned operations will have the same number of
    /// regions, but they will be empty.
    /// Cloning of nested operations in the operation's regions are currently
    /// unaffected by other flags.
    CloneOptions &cloneRegions(bool enable = true);

    /// 返回 Operation 的 regions 是否也应该被克隆。
    ///
    /// Returns whether regions of the operation should be cloned as well.
    bool shouldCloneRegions() const { return cloneRegionsFlag; }

    /// 配置是否应克隆 Operation 的 operands。否则，生成的克隆将只有零个 operand。
    ///
    /// Configures whether operation' operands should be cloned. Otherwise the
    /// resulting clones will simply have zero operands.
    CloneOptions &cloneOperands(bool enable = true);

    /// 返回 operands 是否也应该被克隆。
    ///
    /// Returns whether operands should be cloned as well.
    bool shouldCloneOperands() const { return cloneOperandsFlag; }

  private:
    /// 是否应该克隆 regions。
    ///
    /// Whether regions should be cloned.
    bool cloneRegionsFlag : 1;
    /// 是否应该克隆 operands。
    ///
    /// Whether operands should be cloned.
    bool cloneOperandsFlag : 1;
  };

  /// 创建此 Operation 的深层副本，使用提供的 map 来 remapping 使用 Operation 之
  /// 外的 values 的任何 operands（如果不存在任何条目，则保留它们）。将对 cloned
  /// sub-Operation 的引用替换为 the corresponding operation that is copied，
  /// 并将这些 mappings 添加到 map 中。
  /// 可选地，可以使用 options 参数配置要克隆 Operation 的哪些部分。
  ///
  /// Create a deep copy of this operation, remapping any operands that use
  /// values outside of the operation using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-operations to the corresponding operation that is copied, and adds
  /// those mappings to the map.
  /// Optionally, one may configure what parts of the operation to clone using
  /// the options parameter.
  ///
  /// Calling this method from multiple threads is generally safe if through the
  /// process of cloning no new uses of 'Value's from outside the operation are
  /// created. Cloning an isolated-from-above operation with no operands, such
  /// as top level function operations, is therefore always safe. Using the
  /// mapper, it is possible to avoid adding uses to outside operands by
  /// remapping them to 'Value's owned by the caller thread.
  Operation *clone(IRMapping &mapper,
                   CloneOptions options = CloneOptions::all());
  Operation *clone(CloneOptions options = CloneOptions::all());

  /// 创建此 operation 的 partial 副本，而无需遍历附加 regions。新 operation 将具
  /// 有与原始 operation 相同数量的 regions，但它们将留空。使用 `mapper`（如果存在）
  /// 重新映射 operands，并更新 `mapper` 以包含结果。
  ///
  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  /// Operands are remapped using `mapper` (if present), and `mapper` is updated
  /// to contain the results.
  Operation *cloneWithoutRegions(IRMapping &mapper);

  /// 创建此 operation 的 partial 副本，而无需遍历附加 regions。新 operation 将具
  /// 有与原始 operation 相同数量的 regions，但它们将留空。
  ///
  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  Operation *cloneWithoutRegions();

  /// 返回包含该 operation 的 operation block。
  ///
  /// Returns the operation block that contains this operation.
  Block *getBlock() { return block; }

  /// 返回与此 operation 关联的上下文。
  ///
  /// Return the context this operation is associated with.
  MLIRContext *getContext() { return location->getContext(); }

  /// 返回与此 operation 相关的方言，如果未加载相关的方言，则返回 nullptr。
  ///
  /// Return the dialect this operation is associated with, or nullptr if the
  /// associated dialect is not loaded.
  Dialect *getDialect() { return getName().getDialect(); }

  /// 定义或派生 operation 的 source location。
  ///
  /// The source location the operation was defined or derived from.
  Location getLoc() { return location; }

  /// 设置定义或派生 operation 的 source location。
  ///
  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// 返回 instruction 所属的 region。如果指令 is unlinked，则返回 nullptr。
  ///
  /// Returns the region to which the instruction belongs. Returns nullptr if
  /// the instruction is unlinked.
  Region *getParentRegion() { return block ? block->getParent() : nullptr; }

  /// 返回包含此 operation 的最近的 surrounding operation；如果这是 top-level
  /// operation，则返回 nullptr。
  ///
  /// Returns the closest surrounding operation that contains this operation
  /// or nullptr if this is a top-level operation.
  Operation *getParentOp() { return block ? block->getParentOp() : nullptr; }

  /// 返回类型为 'OpTy' 的最近的 surrounding parent operation。
  ///
  /// Return the closest surrounding parent operation that is of type 'OpTy'.
  template <typename OpTy>
  OpTy getParentOfType() {
    auto *op = this;
    while ((op = op->getParentOp()))
      if (auto parentOp = dyn_cast<OpTy>(op))
        return parentOp;
    return OpTy();
  }

  /// 返回具有的 trait 为 `Trait` 的最近的 surrounding parent operation。
  ///
  /// Returns the closest surrounding parent operation with trait `Trait`.
  template <template <typename T> class Trait>
  Operation *getParentWithTrait() {
    Operation *op = this;
    while ((op = op->getParentOp()))
      if (op->hasTrait<Trait>())
        return op;
    return nullptr;
  }

  /// 如果此 operation 是 the `other` operation 的 a proper ancestor，则
  /// 返回 true。
  ///
  /// Return true if this operation is a proper ancestor of the `other`
  /// operation.
  bool isProperAncestor(Operation *other);

  /// 如果此 operation 是 the `other` operation 的祖先，则返回 true。 
  /// operation 被视为其自己的祖先，使用 `isProperAncestor` 可避免这种情况。
  ///
  /// Return true if this operation is an ancestor of the `other` operation. An
  /// operation is considered as its own ancestor, use `isProperAncestor` to
  /// avoid this.
  bool isAncestor(Operation *other) {
    return this == other || isProperAncestor(other);
  }

  /// 在此 operation 中，将所有是 'from' 的 operands 替换为 'to'。
  ///
  /// Replace any uses of 'from' with 'to' within this operation.
  void replaceUsesOfWith(Value from, Value to);

  /// 用提供的 `values` 替换此 operation 的所有 results。
  ///
  /// Replace all uses of results of this operation with the provided 'values'.
  template <typename ValuesT>
  void replaceAllUsesWith(ValuesT &&values) {
    getResults().replaceAllUsesWith(std::forward<ValuesT>(values));
  }

  /// 如果给定的 callback（即 `function_ref<bool(OpOperand &)> shouldReplace`）
  /// 返回 true，则用提供的 `values` 替换此 operation 的 results。
  ///
  /// Replace uses of results of this operation with the provided `values` if
  /// the given callback returns true.
  template <typename ValuesT>
  void replaceUsesWithIf(ValuesT &&values,
                         function_ref<bool(OpOperand &)> shouldReplace) {
    getResults().replaceUsesWithIf(std::forward<ValuesT>(values),
                                   shouldReplace);
  }

  /// 销毁此 operation 及其子类数据。
  ///
  /// Destroys this operation and its subclass data.
  void destroy();

  /// 这将从该 operation 中删除所有 operands 的 uses，这是在删除引用时打破引用之
  /// 间的循环依赖关系的重要步骤。
  ///
  /// This drops all operand uses from this operation, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// 删除由此 operation 或其嵌套 regions 定义的所有 values 的使用。
  ///
  /// Drop uses of all values defined by this operation or its nested regions.
  void dropAllDefinedValueUses();

  /// 取消此 operation 与它的 current block 的链接，并将其插入到 `existingOp` 之
  /// 前，该 `existingOp` operation 可能位于同一个 block 中，也可能位于同一函数中
  /// 的另一个 block 中。
  ///
  /// Unlink this operation from its current block and insert it right before
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveBefore(Operation *existingOp);

  /// 取消此 operation 与它的 current block 的链接，并将其将其插入到指定 block 中
  /// 的 `iterator` 之前。
  ///
  /// Unlink this operation from its current block and insert it right before
  /// `iterator` in the specified block.
  void moveBefore(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// 取消此 operation 与它的 current block 的链接，并将其插入到 `existingOp` 之
  /// 后，该 `existingOp` operation 可能位于同一个 block 中，也可能位于同一函数中
  /// 的另一个 block 中。
  ///
  /// Unlink this operation from its current block and insert it right after
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveAfter(Operation *existingOp);

  /// 取消此 operation 与它的 current block 的链接，并将其将其插入到指定 block 中
  /// 的 `iterator` 之后。
  ///
  /// Unlink this operation from its current block and insert it right after
  /// `iterator` in the specified block.
  void moveAfter(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// 给定一个位于同一 parent block 内的 operation `other`，返回当前 operation 是
  /// 否在 parent block 的 operation 列表中位于 `other` 之前。
  /// 注意：此函数的平均复杂度为 O(1)，但最坏情况可能需要 O(N)，其中 N 是 parent
  /// block 内的 operation 总数量。
  ///
  /// Given an operation 'other' that is within the same parent block, return
  /// whether the current operation is before 'other' in the operation list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of operations within the parent block.
  bool isBeforeInBlock(Operation *other);

  void print(raw_ostream &os, const OpPrintingFlags &flags = std::nullopt);
  void print(raw_ostream &os, AsmState &state);
  void dump();

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// 用 'operands' 中提供的 operands 替换此 operation 的当前 operands。
  ///
  /// Replace the current operands of this operation with the ones provided in
  /// 'operands'.
  void setOperands(ValueRange operands);

  /// 用参数 `operands` 中提供的 operands 替换从 'start' 开始到 'start' + 'length'
  /// 结束的 operands。'operands' 可能小于或大于 'start'+'length' 指向的范围。
  ///
  /// Replace the operands beginning at 'start' and ending at 'start' + 'length'
  /// with the ones provided in 'operands'. 'operands' may be smaller or larger
  /// than the range pointed to by 'start'+'length'.
  void setOperands(unsigned start, unsigned length, ValueRange operands);

  /// 将给定的 operands 插入到给定 'index' 处的 operand list 中。
  ///
  /// Insert the given operands into the operand list at the given 'index'.
  void insertOperands(unsigned index, ValueRange operands);

  /// 返回 operands 的数量。
  unsigned getNumOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().size() : 0;
  }

  /// 返回此 operand 当前使用的 the current value。
  Value getOperand(unsigned idx) { return getOpOperand(idx).get(); }
  void setOperand(unsigned idx, Value value) {
    return getOpOperand(idx).set(value);
  }

  /// 擦除位置 `idx` 处的 operand。
  ///
  /// Erase the operand at position `idx`.
  void eraseOperand(unsigned idx) { eraseOperands(idx); }

  /// 擦除从位置 `idx` 开始到位置 `idx`+`length` 结束的 operand。
  ///
  /// Erase the operands starting at position `idx` and ending at position
  /// 'idx'+'length'.
  void eraseOperands(unsigned idx, unsigned length = 1) {
    getOperandStorage().eraseOperands(idx, length);
  }

  /// 擦除在 `eraseIndices` 中设置了相应位的 operands，并将它们从 operand
  /// list 中删除。
  ///
  /// Erases the operands that have their corresponding bit set in
  /// `eraseIndices` and removes them from the operand list.
  void eraseOperands(const BitVector &eraseIndices) {
    getOperandStorage().eraseOperands(eraseIndices);
  }

  // Support operand iteration.
  using operand_range = OperandRange;
  using operand_iterator = operand_range::iterator;

  operand_iterator operand_begin() { return getOperands().begin(); }
  operand_iterator operand_end() { return getOperands().end(); }

  /// 返回 the underlying Value 的 iterator。
  ///
  /// Returns an iterator on the underlying Value's.
  operand_range getOperands() {
    MutableArrayRef<OpOperand> operands = getOpOperands();
    return OperandRange(operands.data(), operands.size());
  }

  /// 返回 storage 中保存的 operands。
  MutableArrayRef<OpOperand> getOpOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().getOperands()
                                          : MutableArrayRef<OpOperand>();
  }

  /// 返回 storage 中保存的第 idx 个 operand。
  OpOperand &getOpOperand(unsigned idx) {
    return getOperandStorage().getOperands()[idx];
  }

  // Support operand type iteration.
  using operand_type_iterator = operand_range::type_iterator;
  using operand_type_range = operand_range::type_range;
  operand_type_iterator operand_type_begin() { return operand_begin(); }
  operand_type_iterator operand_type_end() { return operand_end(); }
  operand_type_range getOperandTypes() { return getOperands().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  /// 返回此 operation 所持有的 results 的数量。
  ///
  /// Return the number of results held by this operation.
  unsigned getNumResults() { return numResults; }

  /// 返回此 operation 所持有的第 `idx` 个 result。
  ///
  /// Get the 'idx'th result of this operation.
  OpResult getResult(unsigned idx) { return OpResult(getOpResultImpl(idx)); }

  /// Support result iteration.
  using result_range = ResultRange;
  using result_iterator = result_range::iterator;

  result_iterator result_begin() { return getResults().begin(); }
  result_iterator result_end() { return getResults().end(); }
  result_range getResults() {
    return numResults == 0 ? result_range(nullptr, 0)
                           : result_range(getInlineOpResult(0), numResults);
  }

  result_range getOpResults() { return getResults(); }
  OpResult getOpResult(unsigned idx) { return getResult(idx); }

  /// Support result type iteration.
  using result_type_iterator = result_range::type_iterator;
  using result_type_range = result_range::type_range;
  result_type_iterator result_type_begin() { return getResultTypes().begin(); }
  result_type_iterator result_type_end() { return getResultTypes().end(); }
  result_type_range getResultTypes() { return getResults().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Operations 可以选择性地携带将 constants 与 names 关联的 attributes 列表。可
  // 以在 operation 的整个生命周期内动态添加和删除 attributes。
  //
  // Operations may optionally carry a list of attributes that associate
  // constants to names.  Attributes may be dynamically added and removed over
  // the lifetime of an operation.

  /// 通过 name 访问 an inherent attribute：如果不存在具有此 name 的 an inherent
  /// attribute，则返回 an empty optional。
  ///
  /// Access an inherent attribute by name: returns an empty optional if there
  /// is no inherent attribute with this name.
  ///
  /// This method is available as a transient facility in the migration process
  /// to use Properties instead.
  std::optional<Attribute> getInherentAttr(StringRef name);

  /// 按 name 设置 an inherent attribute。
  ///
  /// Set an inherent attribute by name.
  ///
  /// This method is available as a transient facility in the migration process
  /// to use Properties instead.
  void setInherentAttr(StringAttr name, Attribute value);

  /// 通过 name 访问可丢弃 attribute，如果可丢弃属性不存在则返回 an null Attribute。
  ///
  /// Access a discardable attribute by name, returns an null Attribute if the
  /// discardable attribute does not exist.
  Attribute getDiscardableAttr(StringRef name) { return attrs.get(name); }

  /// 通过 name 访问可丢弃 attribute，如果可丢弃属性不存在则返回 an null Attribute。
  ///
  /// Access a discardable attribute by name, returns an null Attribute if the
  /// discardable attribute does not exist.
  Attribute getDiscardableAttr(StringAttr name) { return attrs.get(name); }

  /// 通过 name 设置可丢弃的 attribute。
  ///
  /// Set a discardable attribute by name.
  void setDiscardableAttr(StringAttr name, Attribute value) {
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setDiscardableAttr(StringRef name, Attribute value) {
    setDiscardableAttr(StringAttr::get(getContext(), name), value);
  }

  /// 如果存在具有指定 name 的可丢弃 attribute，则将其删除。
  /// 返回上步被删除的 attribute，如果不存在具有该名称的属性，则返回 nullptr。
  ///
  /// Remove the discardable attribute with the specified name if it exists.
  /// Return the attribute that was erased, or nullptr if there was no attribute
  /// with such name.
  Attribute removeDiscardableAttr(StringAttr name) {
    NamedAttrList attributes(attrs);
    Attribute removedAttr = attributes.erase(name);
    if (removedAttr)
      attrs = attributes.getDictionary(getContext());
    return removedAttr;
  }
  Attribute removeDiscardableAttr(StringRef name) {
    return removeDiscardableAttr(StringAttr::get(getContext(), name));
  }

  /// 返回此 operation 中所有可丢弃 attributes 的 range。请注意，对于未将 inherent
  /// attributes 存储为 properties 的 unregistered operations，所有 attributes
  /// 均被视为可丢弃。
  ///
  /// Return a range of all of discardable attributes on this operation. Note
  /// that for unregistered operations that are not storing inherent attributes
  /// as properties, all attributes are considered discardable.
  auto getDiscardableAttrs() {
    std::optional<RegisteredOperationName> opName = getRegisteredInfo();
    ArrayRef<StringAttr> attributeNames =
        opName ? getRegisteredInfo()->getAttributeNames()
               : ArrayRef<StringAttr>();
    return llvm::make_filter_range(
        attrs.getValue(),
        [this, attributeNames](const NamedAttribute attribute) {
          return getPropertiesStorage() ||
                 !llvm::is_contained(attributeNames, attribute.getName());
        });
  }

  /// 将此 operation 上的所有可丢弃 attributes 作为 DictionaryAttr 返回。
  ///
  /// Return all of the discardable attributes on this operation as a
  /// DictionaryAttr.
  DictionaryAttr getDiscardableAttrDictionary() {
    if (getPropertiesStorage())
      return attrs;
    return DictionaryAttr::get(getContext(),
                               llvm::to_vector(getDiscardableAttrs()));
  }

  /// 返回所有未存储为 properties 的 attributes。
  ///
  /// Return all attributes that are not stored as properties.
  DictionaryAttr getRawDictionaryAttrs() { return attrs; }

  /// 返回此 operation 的所有属性。
  ///
  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() { return getAttrDictionary().getValue(); }

  /// 将此 operation 的所有 attributes 作为 DictionaryAttr 返回。
  ///
  /// Return all of the attributes on this operation as a DictionaryAttr.
  DictionaryAttr getAttrDictionary();

  /// 在此 operation 中设置 attributes from a dictionary。
  ///
  /// Set the attributes from a dictionary on this operation.
  /// These methods are expensive: if the dictionnary only contains discardable
  /// attributes, `setDiscardableAttrs` is more efficient.
  void setAttrs(DictionaryAttr newAttrs);
  void setAttrs(ArrayRef<NamedAttribute> newAttrs);
  /// 在此 operation 上设置可丢弃的 attribute dictionary。
  ///
  /// Set the discardable attribute dictionary on this operation.
  void setDiscardableAttrs(DictionaryAttr newAttrs) {
    assert(newAttrs && "expected valid attribute dictionary");
    attrs = newAttrs;
  }
  void setDiscardableAttrs(ArrayRef<NamedAttribute> newAttrs) {
    setDiscardableAttrs(DictionaryAttr::get(getContext(), newAttrs));
  }

  /// 根据 name 获取 attribute。如果存在，则返回指定的 attribute，否则返回 null。
  ///
  /// Return the specified attribute if present, null otherwise.
  /// These methods are expensive: if the dictionnary only contains discardable
  /// attributes, `getDiscardableAttr` is more efficient.
  Attribute getAttr(StringAttr name) {
    if (getPropertiesStorageSize()) {
      if (std::optional<Attribute> inherentAttr = getInherentAttr(name))
        return *inherentAttr;
    }
    return attrs.get(name);
  }
  Attribute getAttr(StringRef name) {
    if (getPropertiesStorageSize()) {
      if (std::optional<Attribute> inherentAttr = getInherentAttr(name))
        return *inherentAttr;
    }
    return attrs.get(name);
  }

  template <typename AttrClass>
  AttrClass getAttrOfType(StringAttr name) {
    return llvm::dyn_cast_or_null<AttrClass>(getAttr(name));
  }
  template <typename AttrClass>
  AttrClass getAttrOfType(StringRef name) {
    return llvm::dyn_cast_or_null<AttrClass>(getAttr(name));
  }

  /// 如果 operation 具有所提供 name 的 attribute，则返回 true，否则返回 false。
  ///
  /// Return true if the operation has an attribute with the provided name,
  /// false otherwise.
  bool hasAttr(StringAttr name) {
    if (getPropertiesStorageSize()) {
      if (std::optional<Attribute> inherentAttr = getInherentAttr(name))
        return (bool)*inherentAttr;
    }
    return attrs.contains(name);
  }
  bool hasAttr(StringRef name) {
    if (getPropertiesStorageSize()) {
      if (std::optional<Attribute> inherentAttr = getInherentAttr(name))
        return (bool)*inherentAttr;
    }
    return attrs.contains(name);
  }
  template <typename AttrClass, typename NameT>
  bool hasAttrOfType(NameT &&name) {
    return static_cast<bool>(
        getAttrOfType<AttrClass>(std::forward<NameT>(name)));
  }

  /// 如果存在具有指定 name 的 attribute，则将其更改为 new value。否则，添加具有指
  /// 定 name 或 value 的 a new attribute。
  ///
  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setAttr(StringAttr name, Attribute value) {
    if (getPropertiesStorageSize()) {
      if (getInherentAttr(name)) {
        setInherentAttr(name, value);
        return;
      }
    }
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(StringAttr::get(getContext(), name), value);
  }

  /// 如果存在具有指定 name 的 attribute，则删除该 attribute。返回被删除的 attribute，
  /// 如果不存在具有该 name 的 attribute，则返回 nullptr。
  ///
  /// Remove the attribute with the specified name if it exists. Return the
  /// attribute that was erased, or nullptr if there was no attribute with such
  /// name.
  Attribute removeAttr(StringAttr name) {
    if (getPropertiesStorageSize()) {
      if (std::optional<Attribute> inherentAttr = getInherentAttr(name)) {
        setInherentAttr(name, {});
        return *inherentAttr;
      }
    }
    NamedAttrList attributes(attrs);
    Attribute removedAttr = attributes.erase(name);
    if (removedAttr)
      attrs = attributes.getDictionary(getContext());
    return removedAttr;
  }
  Attribute removeAttr(StringRef name) {
    return removeAttr(StringAttr::get(getContext(), name));
  }

  /// 过滤 non-dialect attributes 的实用迭代器。
  ///
  /// `bool (*)(NamedAttribute)` 其实是一个可调用的过滤函数，过滤器会在迭代时跳过所
  /// 有该过滤函数返回 false 的元素。
  ///
  /// A utility iterator that filters out non-dialect attributes.
  class dialect_attr_iterator
      : public llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                     bool (*)(NamedAttribute)> {
    static bool filter(NamedAttribute attr) {
      // 过滤条件：Dialect attributes 以 dialect 名称为前缀，就像 operations 一样。
      //
      // Dialect attributes are prefixed by the dialect name, like operations.
      return attr.getName().strref().count('.');
    }

    /// 用 `filter` 函数过滤从 it 到 end 的所有 NamedAttributes。
    explicit dialect_attr_iterator(ArrayRef<NamedAttribute>::iterator it,
                                   ArrayRef<NamedAttribute>::iterator end)
        : llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                bool (*)(NamedAttribute)>(it, end, &filter) {}

    // Allow access to the constructor.
    friend Operation;
  };
  /// iterator_range 类模板充当迭代器对的 range adaptor。简单来说，它将一对迭代器包
  /// 装起来，使其能够像标准库中的 range（例如 std::vector 或 std::array）一样使用，
  /// 例如在基于 range 的 for 循环中。
  ///
  ///  - 模板参数 `IteratorT`: 指定迭代器的类型。 这意味着 iterator_range 可以用于
  ///    各种类型的迭代器。
  ///  - iterator_range 的成员变量 begin_iterator, end_iterator: 分别存储范围的起
  ///    始迭代器和结束迭代器。
  using dialect_attr_range = iterator_range<dialect_attr_iterator>;

  /// 返回与此 operation 的 dialect attributes 相对应的 range。
  ///
  /// Return a range corresponding to the dialect attributes for this operation.
  dialect_attr_range getDialectAttrs() {
    auto attrs = getAttrs();
    return {dialect_attr_iterator(attrs.begin(), attrs.end()),
            dialect_attr_iterator(attrs.end(), attrs.end())};
  }
  dialect_attr_iterator dialect_attr_begin() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.begin(), attrs.end());
  }
  dialect_attr_iterator dialect_attr_end() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.end(), attrs.end());
  }

  /// 为此 operation 设置 dialect attributes，并保留 all inherent。
  ///
  /// Set the dialect attributes for this operation, and preserve all inherent.
  template <typename DialectAttrT>
  void setDialectAttrs(DialectAttrT &&dialectAttrs) {
    NamedAttrList attrs;
    attrs.append(std::begin(dialectAttrs), std::end(dialectAttrs));
    for (auto attr : getAttrs())
      if (!attr.getName().strref().contains('.'))
        attrs.push_back(attr);
    setAttrs(attrs.getDictionary(getContext()));
  }

  /// 对 unset attributes 设置 default attributes。
  ///
  /// Sets default attributes on unset attributes.
  void populateDefaultAttrs() {
    NamedAttrList attrs(getAttrDictionary());
    name.populateDefaultAttrs(attrs);
    setAttrs(attrs.getDictionary(getContext()));
  }

  //===--------------------------------------------------------------------===//
  // Blocks
  //===--------------------------------------------------------------------===//

  /// 返回此 operation 所持有的 regions 总数。
  ///
  /// Returns the number of regions held by this operation.
  unsigned getNumRegions() { return numRegions; }

  /// 返回此 operation 所持有的 regions。
  ///
  /// Returns the regions held by this operation.
  MutableArrayRef<Region> getRegions() {
    // Check the count first, as computing the trailing objects can be slow.
    if (numRegions == 0)
      return MutableArrayRef<Region>();

    auto *regions = getTrailingObjects<Region>();
    return {regions, numRegions};
  }

  /// 返回此 operation 在位置 `index` 处持有的 region。
  ///
  /// Returns the region held by this operation at position 'index'.
  Region &getRegion(unsigned index) {
    assert(index < numRegions && "invalid region index");
    return getRegions()[index];
  }

  //===--------------------------------------------------------------------===//
  // Successors
  //===--------------------------------------------------------------------===//

  // 不同情况下的 Successors：
  //  - 顺序执行: 对于大多数简单的 Operation，它们的 Successor 只是 CFG 中紧随其后的
  //             Operation。 这类似于在普通代码中，语句按顺序执行。
  //  - 条件分支 (if-else, switch): 条件分支 Operation 会有多个 Successors。例如，
  //             if Operation 通常有两个 Successors：一个对应 then 块，另一个对应
  //             else 块（或 else 块为空）。switch 操作符会有多个 successors，对应
  //             于不同的 case。
  //  - 循环 (for, while): 循环 Operation 的 Successors 相对比较复杂。循环体内部的
  //             Operation 的 Successor 可能指向循环体内的下一个 Operation，也可能
  //             指向循环条件检查 Operation。循环结束后的 Successor 是循环后面的
  //             Operation。
  //  - 函数调用: 函数调用 Operation 的 Successor 是函数调用返回后的 Operation。
  //  - 跳转 (goto, branch): 跳转 Operation 会明确指定其 Successor 为 CFG 中的另一
  //             个 Block 或 Operation。

  /// 返回 Block Operand 的列表。
  MutableArrayRef<BlockOperand> getBlockOperands() {
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }

  // Successor iteration.
  using succ_iterator = SuccessorRange::iterator;
  succ_iterator successor_begin() { return getSuccessors().begin(); }
  succ_iterator successor_end() { return getSuccessors().end(); }
  SuccessorRange getSuccessors() { return SuccessorRange(this); }

  bool hasSuccessors() { return numSuccs != 0; }
  unsigned getNumSuccessors() { return numSuccs; }

  /// 获取当前 operation 的第 index 个 Successor。
  Block *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  /// 设置当前 operation 的第 index 个 Successor。
  void setSuccessor(Block *block, unsigned index);

  //===--------------------------------------------------------------------===//
  // Accessors for various properties of operations
  //===--------------------------------------------------------------------===//

  /// 尝试使用 the specified constant operand values 折叠此 operation
  /// - `operands` 中的元素将直接对应于操作的操作数，但如果非常量则可能为空。
  ///
  /// 如果折叠成功，则此函数返回 "success"。
  /// * 如果此操作被就地修改（但未被折叠），则 `results` 为空。
  /// * 否则，`results` 将被 folded results 填充。
  /// 如果折叠不成功，则此函数返回 "failure"。
  ///
  /// Attempt to fold this operation with the specified constant operand values
  /// - the elements in "operands" will correspond directly to the operands of
  /// the operation, but may be null if non-constant.
  ///
  /// If folding was successful, this function returns "success".
  /// * If this operation was modified in-place (but not folded away),
  ///   `results` is empty.
  /// * Otherwise, `results` is filled with the folded results.
  /// If folding was unsuccessful, this function returns "failure".
  LogicalResult fold(ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results);

  /// 尝试折叠此 operation。
  ///
  /// 如果折叠成功，则此函数返回 "success"。
  /// * 如果此操作被就地修改（但未被折叠），则 `results` 为空。
  /// * 否则，`results` 将被 folded results 填充。
  /// 如果折叠不成功，则此函数返回 "failure"。
  ///
  /// Attempt to fold this operation.
  ///
  /// If folding was successful, this function returns "success".
  /// * If this operation was modified in-place (but not folded away),
  ///   `results` is empty.
  /// * Otherwise, `results` is filled with the folded results.
  /// If folding was unsuccessful, this function returns "failure".
  LogicalResult fold(SmallVectorImpl<OpFoldResult> &results);

  /// 如果 `InterfaceT` 已被方言承诺或已实现，则返回 true。
  ///
  /// Returns true if `InterfaceT` has been promised by the dialect or
  /// implemented.
  template <typename InterfaceT>
  bool hasPromiseOrImplementsInterface() const {
    return name.hasPromiseOrImplementsInterface<InterfaceT>();
  }

  /// 如果 operation 已注册 a particular trait，则返回 true，例如
  /// `hasTrait<OperandsAreSignlessIntegerLike>()`。
  ///
  /// Returns true if the operation was registered with a particular trait, e.g.
  /// hasTrait<OperandsAreSignlessIntegerLike>().
  template <template <typename T> class Trait>
  bool hasTrait() {
    return name.hasTrait<Trait>();
  }

  /// 如果 operation *might* 具有提供的 trait，则返回 true。这意味着该 operation 要
  /// 么未注册，要么已使用提供的 trait 进行注册。
  ///
  /// Returns true if the operation *might* have the provided trait. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide trait.
  template <template <typename T> class Trait>
  bool mightHaveTrait() {
    return name.mightHaveTrait<Trait>();
  }

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operation by calling the callback for each nested operation
  /// (including this one), block or region, depending on the callback provided.
  /// The order in which regions, blocks and operations at the same nesting
  /// level are visited (e.g., lexicographical or reverse lexicographical order)
  /// is determined by 'Iterator'. The walk order for enclosing regions, blocks
  /// and operations with respect to their nested ones is specified by 'Order'
  /// (post-order by default). A callback on a block or operation is allowed to
  /// erase that block or operation if either:
  ///   * the walk is in post-order, or
  ///   * the walk is in pre-order and the walk is skipped after the erasure.
  ///
  /// The callback method can take any of the following forms:
  ///   void(Operation*) : Walk all operations opaquely.
  ///     * op->walk([](Operation *nestedOp) { ...});
  ///   void(OpT) : Walk all operations of the given derived type.
  ///     * op->walk([](ReturnOp returnOp) { ...});
  ///   WalkResult(Operation*|OpT) : Walk operations, but allow for
  ///                                interruption/skipping.
  ///     * op->walk([](... op) {
  ///         // Skip the walk of this op based on some invariant.
  ///         if (some_invariant)
  ///           return WalkResult::skip();
  ///         // Interrupt, i.e cancel, the walk based on some invariant.
  ///         if (another_invariant)
  ///           return WalkResult::interrupt();
  ///         return WalkResult::advance();
  ///       });
  template <WalkOrder Order = WalkOrder::PostOrder,
            typename Iterator = ForwardIterator, typename FnT,
            typename RetT = detail::walkResultType<FnT>>
  std::enable_if_t<llvm::function_traits<std::decay_t<FnT>>::num_args == 1,
                   RetT>
  walk(FnT &&callback) {
    return detail::walk<Order, Iterator>(this, std::forward<FnT>(callback));
  }

  /// Generic walker with a stage aware callback. Walk the operation by calling
  /// the callback for each nested operation (including this one) N+1 times,
  /// where N is the number of regions attached to that operation.
  ///
  /// The callback method can take any of the following forms:
  ///   void(Operation *, const WalkStage &) : Walk all operation opaquely
  ///     * op->walk([](Operation *nestedOp, const WalkStage &stage) { ...});
  ///   void(OpT, const WalkStage &) : Walk all operations of the given derived
  ///                                  type.
  ///     * op->walk([](ReturnOp returnOp, const WalkStage &stage) { ...});
  ///   WalkResult(Operation*|OpT, const WalkStage &stage) : Walk operations,
  ///          but allow for interruption/skipping.
  ///     * op->walk([](... op, const WalkStage &stage) {
  ///         // Skip the walk of this op based on some invariant.
  ///         if (some_invariant)
  ///           return WalkResult::skip();
  ///         // Interrupt, i.e cancel, the walk based on some invariant.
  ///         if (another_invariant)
  ///           return WalkResult::interrupt();
  ///         return WalkResult::advance();
  ///       });
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  std::enable_if_t<llvm::function_traits<std::decay_t<FnT>>::num_args == 2,
                   RetT>
  walk(FnT &&callback) {
    return detail::walk(this, std::forward<FnT>(callback));
  }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  /// Drop all uses of results of this operation.
  void dropAllUses() {
    for (OpResult result : getOpResults())
      result.dropAllUses();
  }

  using use_iterator = result_range::use_iterator;
  using use_range = result_range::use_range;

  use_iterator use_begin() { return getResults().use_begin(); }
  use_iterator use_end() { return getResults().use_end(); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() { return getResults().getUses(); }

  /// Returns true if this operation has exactly one use.
  bool hasOneUse() { return llvm::hasSingleElement(getUses()); }

  /// Returns true if this operation has no uses.
  bool use_empty() { return getResults().use_empty(); }

  /// Returns true if the results of this operation are used outside of the
  /// given block.
  bool isUsedOutsideOfBlock(Block *block) {
    return llvm::any_of(getOpResults(), [block](OpResult result) {
      return result.isUsedOutsideOfBlock(block);
    });
  }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() { return user_iterator(use_begin()); }
  user_iterator user_end() { return user_iterator(use_end()); }

  /// Returns a range of all users.
  user_range getUsers() { return {user_begin(), user_end()}; }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.
  InFlightDiagnostic emitOpError(const Twine &message = {});

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.
  InFlightDiagnostic emitError(const Twine &message = {});

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitWarning(const Twine &message = {});

  /// Emit a remark about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitRemark(const Twine &message = {});

  /// Returns the properties storage size.
  int getPropertiesStorageSize() const {
    return ((int)propertiesStorageSize) * 8;
  }
  /// Returns the properties storage.
  OpaqueProperties getPropertiesStorage() {
    if (propertiesStorageSize)
      return getPropertiesStorageUnsafe();
    return {nullptr};
  }
  OpaqueProperties getPropertiesStorage() const {
    if (propertiesStorageSize)
      return {reinterpret_cast<void *>(const_cast<detail::OpProperties *>(
          getTrailingObjects<detail::OpProperties>()))};
    return {nullptr};
  }
  /// Returns the properties storage without checking whether properties are
  /// present.
  OpaqueProperties getPropertiesStorageUnsafe() {
    return {
        reinterpret_cast<void *>(getTrailingObjects<detail::OpProperties>())};
  }

  /// 返回被转换为 an attribute 的 properties。
  /// 这很昂贵，并且在处理 unregistered operation 时非常有用。如果不存在任何
  /// properties，则返回 an empty attribute。
  ///
  /// Return the properties converted to an attribute.
  /// This is expensive, and mostly useful when dealing with unregistered
  /// operation. Returns an empty attribute if no properties are present.
  Attribute getPropertiesAsAttribute();

  /// 从 provided attribute 中设置 properties。
  /// 这是 an expensive operation，如果 attribute 不符合 the expectations
  /// of the properties for this operation，则可能会失败。
  /// 这主要用于 unregistered operation 或在解析通用格式时使用。可以传入可选的
  /// diagnostic emitter 以获得更丰富的错误，如果没有传入，则在错误情况下行为未
  /// 定义。
  ///
  /// Set the properties from the provided attribute.
  /// This is an expensive operation that can fail if the attribute is not
  /// matching the expectations of the properties for this operation. This is
  /// mostly useful for unregistered operations or used when parsing the
  /// generic format. An optional diagnostic emitter can be passed in for richer
  /// errors, if none is passed then behavior is undefined in error case.
  LogicalResult
  setPropertiesFromAttribute(Attribute attr,
                             function_ref<InFlightDiagnostic()> emitError);

  /// 从现有的 other properties object 复制 properties。这两个对象必须是同一类型。
  ///
  /// Copy properties from an existing other properties object. The two objects
  /// must be the same type.
  void copyProperties(OpaqueProperties rhs);

  /// 计算 op properties 的哈希值（如果有）。
  ///
  /// Compute a hash for the op properties (if any).
  llvm::hash_code hashProperties();

private:
  //===--------------------------------------------------------------------===//
  // Ordering
  //===--------------------------------------------------------------------===//

  /// This value represents an invalid index ordering for an operation within a
  /// block.
  static constexpr unsigned kInvalidOrderIdx = -1;

  /// This value represents the stride to use when computing a new order for an
  /// operation.
  static constexpr unsigned kOrderStride = 5;

  /// Update the order index of this operation of this operation if necessary,
  /// potentially recomputing the order of the parent block.
  void updateOrderIfNecessary();

  /// Returns true if this operation has a valid order.
  bool hasValidOrder() { return orderIndex != kInvalidOrderIdx; }

private:
  Operation(Location location, OperationName name, unsigned numResults,
            unsigned numSuccessors, unsigned numRegions,
            int propertiesStorageSize, DictionaryAttr attributes,
            OpaqueProperties properties, bool hasOperandStorage);

  // Operations are deleted through the destroy() member because they are
  // allocated with malloc.
  ~Operation();

  /// Returns the additional size necessary for allocating the given objects
  /// before an Operation in-memory.
  static size_t prefixAllocSize(unsigned numOutOfLineResults,
                                unsigned numInlineResults) {
    return sizeof(detail::OutOfLineOpResult) * numOutOfLineResults +
           sizeof(detail::InlineOpResult) * numInlineResults;
  }
  /// Returns the additional size allocated before this Operation in-memory.
  size_t prefixAllocSize() {
    unsigned numResults = getNumResults();
    unsigned numOutOfLineResults = OpResult::getNumTrailing(numResults);
    unsigned numInlineResults = OpResult::getNumInline(numResults);
    return prefixAllocSize(numOutOfLineResults, numInlineResults);
  }

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    assert(hasOperandStorage && "expected operation to have operand storage");
    return *getTrailingObjects<detail::OperandStorage>();
  }

  /// Returns a pointer to the use list for the given out-of-line result.
  detail::OutOfLineOpResult *getOutOfLineOpResult(unsigned resultNumber) {
    // Out-of-line results are stored in reverse order after (before in memory)
    // the inline results.
    return reinterpret_cast<detail::OutOfLineOpResult *>(getInlineOpResult(
               detail::OpResultImpl::getMaxInlineResults() - 1)) -
           ++resultNumber;
  }

  /// Returns a pointer to the use list for the given inline result.
  detail::InlineOpResult *getInlineOpResult(unsigned resultNumber) {
    // Inline results are stored in reverse order before the operation in
    // memory.
    return reinterpret_cast<detail::InlineOpResult *>(this) - ++resultNumber;
  }

  /// Returns a pointer to the use list for the given result, which may be
  /// either inline or out-of-line.
  detail::OpResultImpl *getOpResultImpl(unsigned resultNumber) {
    assert(resultNumber < getNumResults() &&
           "Result number is out of range for operation");
    unsigned maxInlineResults = detail::OpResultImpl::getMaxInlineResults();
    if (resultNumber < maxInlineResults)
      return getInlineOpResult(resultNumber);
    return getOutOfLineOpResult(resultNumber - maxInlineResults);
  }

  /// Provide a 'getParent' method for ilist_node_with_parent methods.
  /// We mark it as a const function because ilist_node_with_parent specifically
  /// requires a 'getParent() const' method. Once ilist_node removes this
  /// constraint, we should drop the const to fit the rest of the MLIR const
  /// model.
  Block *getParent() const { return block; }

  /// Expose a few methods explicitly for the debugger to call for
  /// visualization.
#ifndef NDEBUG
  LLVM_DUMP_METHOD operand_range debug_getOperands() { return getOperands(); }
  LLVM_DUMP_METHOD result_range debug_getResults() { return getResults(); }
  LLVM_DUMP_METHOD SuccessorRange debug_getSuccessors() {
    return getSuccessors();
  }
  LLVM_DUMP_METHOD MutableArrayRef<Region> debug_getRegions() {
    return getRegions();
  }
#endif

  /// The operation block that contains this operation.
  Block *block = nullptr;

  /// This holds information about the source location the operation was defined
  /// or derived from.
  Location location;

  /// Relative order of this operation in its parent block. Used for
  /// O(1) local dominance checks between operations.
  mutable unsigned orderIndex = 0;

  const unsigned numResults;
  const unsigned numSuccs;
  const unsigned numRegions : 23;

  /// This bit signals whether this operation has an operand storage or not. The
  /// operand storage may be elided for operations that are known to never have
  /// operands.
  bool hasOperandStorage : 1;

  /// The size of the storage for properties (if any), divided by 8: since the
  /// Properties storage will always be rounded up to the next multiple of 8 we
  /// save some bits here.
  unsigned char propertiesStorageSize : 8;
  /// This is the maximum size we support to allocate properties inline with an
  /// operation: this must match the bitwidth above.
  static constexpr int64_t propertiesCapacity = 8 * 256;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  DictionaryAttr attrs;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Operation>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow value to access the 'ResultStorage' methods.
  friend class Value;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Operation, Block>;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Operation, detail::OperandStorage,
                               detail::OpProperties, BlockOperand, Region,
                               OpOperand>;
  size_t numTrailingObjects(OverloadToken<detail::OperandStorage>) const {
    return hasOperandStorage ? 1 : 0;
  }
  size_t numTrailingObjects(OverloadToken<BlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<Region>) const { return numRegions; }
  size_t numTrailingObjects(OverloadToken<detail::OpProperties>) const {
    return getPropertiesStorageSize();
  }
};

inline raw_ostream &operator<<(raw_ostream &os, const Operation &op) {
  const_cast<Operation &>(op).print(os, OpPrintingFlags().useLocalScope());
  return os;
}

} // namespace mlir

namespace llvm {
/// Cast from an (const) Operation * to a derived operation type.
template <typename T>
struct CastInfo<T, ::mlir::Operation *>
    : public ValueFromPointerCast<T, ::mlir::Operation,
                                  CastInfo<T, ::mlir::Operation *>> {
  static bool isPossible(::mlir::Operation *op) { return T::classof(op); }
};
template <typename T>
struct CastInfo<T, const ::mlir::Operation *>
    : public ConstStrippingForwardingCast<T, const ::mlir::Operation *,
                                          CastInfo<T, ::mlir::Operation *>> {};

/// Cast from an (const) Operation & to a derived operation type.
template <typename T>
struct CastInfo<T, ::mlir::Operation>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, ::mlir::Operation &,
                                     CastInfo<T, ::mlir::Operation>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(::mlir::Operation &val) { return T::classof(&val); }
  static T doCast(::mlir::Operation &val) { return T(&val); }
};
template <typename T>
struct CastInfo<T, const ::mlir::Operation>
    : public ConstStrippingForwardingCast<T, const ::mlir::Operation,
                                          CastInfo<T, ::mlir::Operation>> {};

/// Cast (const) Operation * to itself. This is helpful to avoid SFINAE in
/// templated implementations that should work on both base and derived
/// operation types.
template <>
struct CastInfo<::mlir::Operation *, ::mlir::Operation *>
    : public NullableValueCastFailed<::mlir::Operation *>,
      public DefaultDoCastIfPossible<
          ::mlir::Operation *, ::mlir::Operation *,
          CastInfo<::mlir::Operation *, ::mlir::Operation *>> {
  static bool isPossible(::mlir::Operation *op) { return true; }
  static ::mlir::Operation *doCast(::mlir::Operation *op) { return op; }
};
template <>
struct CastInfo<const ::mlir::Operation *, const ::mlir::Operation *>
    : public ConstStrippingForwardingCast<
          const ::mlir::Operation *, const ::mlir::Operation *,
          CastInfo<::mlir::Operation *, ::mlir::Operation *>> {};
} // namespace llvm

#endif // MLIR_IR_OPERATION_H
