//===-- Constraints.td - Constraints definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines constraints/predicates for verifiers.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTRAINTS
#define CONSTRAINTS

include "mlir/IR/Utils.td"


//===----------------------------------------------------------------------===//
// Predicate definitions
//===----------------------------------------------------------------------===//

// Logical 谓词的基类。
//
// 谓词用于组成约束（有关详细信息，请参阅下一节）。谓词有两种类别：
//
// 1. CPred：原始叶谓词。
// 2. 复合谓词：使用谓词组合器（"conjunction", "disjunction", "negation" 或
// "substitution"）由子谓词组成的谓词。
//
// Base class for logical predicates.
//
// Predicates are used to compose constraints (see next section for details).
// There are two categories of predicates:
//
// 1. CPred: the primitive leaf predicate.
// 2. Compound predicate: a predicate composed from child predicates using
//    predicate combiners ("conjunction", "disjunction", "negation" or
//    "substitution").
class Pred;

// 包装任何 C 表达式的 Logical 谓词。
//
// 这是编写更复杂谓词的基础。从 TableGen 的角度来看，它是“原子”谓词，也是 TableGen
// 和 C++ 之间的“接口”。里面的内容已经是 C++ 代码，将被视为不透明字符串，其中有要替
// 换的特殊占位符。
//
// ## 特殊占位符
//
// 特殊占位符可用于引用使用此谓词的上下文中的实体。它们充当封闭环境的“钩子”。操作的
// 约束支持以下特殊占位符：
//
// * `$_builder` 将被 mlir::Builder 实例替换。
// * `$_op` 将被当前 operation 替换。
// * `$_self` 将被此谓词附加到的实体替换。例如，`BoolAttr` 是一个属性约束，它包装了
//   一个 `CPred<"::llvm::isa<BoolAttr>($_self)">`（有关详细信息请参阅以下部分）。
//   那么对于 `F32:$attr`，`$_self` 将被 `$attr` 替换。对于类型约束，这有点特殊，因
//   为我们希望每个类型定义上的约束读起来自然，并且我们希望将类型约束直接附加到操作数
//   结果上，`$_self` 将被操作数/结果的类型替换。例如对于 `F32:$operand` 中的 `F32`，
//   它的 `$_self` 将扩展为 `getOperand(...).getType()`。
//
// 需要注意的一件事是，在 C 表达式中使用这些占位符时，占位符的类型只能保证是基类型。例
// 如，如果您有一个形式为 `CPred<"CheckType($_self)">` 的谓词，则函数 `CheckType`
// 的参数类型应该是 `mlir::Type`。
//
// A logical predicate wrapping any C expression.
//
// This is the basis for composing more complex predicates. It is the "atom"
// predicate from the perspective of TableGen and the "interface" between
// TableGen and C++. What is inside is already C++ code, which will be treated
// as opaque strings with special placeholders to be substituted.
//
// ## Special placeholders
//
// Special placeholders can be used to refer to entities in the context where
// this predicate is used. They serve as "hooks" to the enclosing environment.
// The following special placeholders are supported in constraints for an op:
//
// * `$_builder` will be replaced by a mlir::Builder instance.
// * `$_op` will be replaced by the current operation.
// * `$_self` will be replaced with the entity this predicate is attached to.
//   E.g., `BoolAttr` is an attribute constraint that wraps a
//   `CPred<"::llvm::isa<BoolAttr>($_self)">` (see the following sections for details).
//   Then for `F32:$attr`,`$_self` will be replaced by `$attr`.
//   For type constraints, it's a little bit special since we want the
//   constraints on each type definition reads naturally and we want to attach
//   type constraints directly to an operand/result, $_self will be replaced
//   by the operand/result's type. E.g., for `F32` in `F32:$operand`, its
//   `$_self` will be expanded as `getOperand(...).getType()`.
//
// One thing to be noticed, while using these placeholders in the C expression,
// the type of placeholder is only guaranteed to be the base type. For example,
// if you have a predicate in the form `CPred<"CheckType($_self)">, the argument
// type of the function `CheckType` should be `mlir::Type`.
class CPred<code pred> : Pred {
  code predExpr = "(" # pred # ")";
}

// 谓词组合器的种类。这些必须与 C++ 后端 (tblgen::PredCombinerKind) 实现的谓
// 词紧密匹配。
//
// // Kinds of nodes in a logical predicate tree.
// enum class PredCombinerKind {
//   Leaf,
//   And,
//   Or,
//   Not,
//   SubstLeaves,
//   Concat,
//   // Special kinds that are used in simplification.
//   False,
//   True
// };
//
// Kinds of predicate combiners.  These must closely match the predicates
// implemented by the C++ backend (tblgen::PredCombinerKind).
class PredCombinerKind;
def PredCombinerAnd : PredCombinerKind;
def PredCombinerOr : PredCombinerKind;
def PredCombinerNot : PredCombinerKind;
def PredCombinerSubstLeaves : PredCombinerKind;
def PredCombinerConcat : PredCombinerKind;

// 按照 PredCombinerKind 的定义组合其他谓词的谓词。下面实例化。
//
// A predicate that combines other predicates as defined by PredCombinerKind.
// Instantiated below.
class CombinedPred<PredCombinerKind k, list<Pred> c> : Pred {
  PredCombinerKind kind = k;
  list<Pred> children = c;
}

// 谓词组合器
//
// Predicate combiners

// 如果所有 children 谓词都成立，则该谓词成立。对于零个 children 谓词，谓词始终成立。
//
// A predicate that holds if all of its children hold.  Always holds for zero
// children.
class And<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// 如果其任何 children 谓词成立，则该谓词成立。如果没有任何 children 谓词成立，则该
// 谓词永远不会成立。
//
// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class Or<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// 当其 children 谓词不成立时，该谓词成立。
//
// A predicate that holds if its child does not.
class Neg<Pred child> : CombinedPred<PredCombinerNot, [child]>;

// 在谓词树的叶子的谓词调用中用 "repl" 替换 "pat" 的谓词（即 not CombinedPred）。
//
// 这是简单的字符串替换，没有正则表达式或捕获。
// 如果需要，可以引入具有更复杂逻辑的新谓词。
//
// 例如下面的一段无意义的代码：
// ```
//   class ComplexX1<Type elType>
//       : ConfinedType<AnyComplex, [
//             CPred<"$_self.isInteger(1)">,
//             SubstLeaves<"$_self",
//                         "::llvm::cast<::mlir::ComplexType>($_self).getElementType()",
//              elType.predicate>],
//              "complex type with " # elType.summary # " elements",
//              "::mlir::ComplexType">,
//         SameBuildabilityAs<elType, "::mlir::ComplexType::get($_builder.get" # elType #
//                                  "Type())"> {
//     Type elementType = elType;
//   }
//   def IntType : ComplexX1<I1>;
// ```
//
// ```
// CPred<"$_self.isInteger(1)">
// ```
// 会生成：
// ```
// ((type.isInteger(1)))
// ```
//
// ```
// SubstLeaves<"$_self", 
//   "::llvm::cast<::mlir::ComplexType>($_self).getElementType()", 
//   elType.predicate>
// ```
// 这个 SubstLeaves 会将叶子的谓词调用中的 "$_self" 替换为 
// "::llvm::cast<::mlir::ComplexType>($_self).getElementType()"。 
// 由于在这个例子中，`elType` 是 `I1`，因此它本身就是叶子，它的谓词调用会生成：
// ```
// ((::llvm::cast<::mlir::ComplexType>(type).getElementType().isSignlessInteger(1)))
// ```
// 如果代码更换为：
// ```
// def IntType : ComplexX1<AnyTypeOf<[I1, I8]>>;
// ```
// 在这里，`elType`变成了 `AnyTypeOf<[I1, I8]>`，它的叶子是 `I1` 和 `I8`，则会生成：
// ```
// ((::llvm::cast<::mlir::ComplexType>(type).getElementType().isSignlessInteger(1))) || 
// ((::llvm::cast<::mlir::ComplexType>(type).getElementType().isSignlessInteger(8)))
// ```
//
// A predicate that substitutes "pat" with "repl" in predicate calls of the
// leaves of the predicate tree (i.e., not CombinedPred).
//
// This is plain string substitution without regular expressions or captures.
// New predicates with more complex logical can be introduced should the need
// arise.
class SubstLeaves<string pat, string repl, Pred child>
    : CombinedPred<PredCombinerSubstLeaves, [child]> {
  string pattern = pat;
  string replacement = repl;
}

// 一个谓词，在由 "child" 组成的最终谓词字符串前面添加 "pre" 并将 "suf" 附加到该
// 字符串。这是普通的字符串连接，不会发生 "pre" 和 "suf" 的替换。
//
// A predicate that prepends `pre` and appends `suf` to the final predicate
// string composed from `child`. This is plain string concatenation and there
// will be no substitution happening for `pre` and `suf`.
class Concat<string pre, Pred child, string suf> :
    CombinedPred<PredCombinerConcat, [child]> {
  string prefix = pre;
  string suffix = suf;
}

//===----------------------------------------------------------------------===//
// Constraint definitions
//===----------------------------------------------------------------------===//

// TODO: Merge Constraints into Pred.

// 命名 constraints 的基类。
//
// Op 的 operands/attributes/results 可以有各种要求，例如具有某些 types、具有一定
// 范围内的值等等。此外，对于 graph rewrite rule，用于与现有 graph 匹配的 source
// pattern 具有条件，例如 op 的 operands 必须是更受约束的 subtype、attribute 必须
// 具有某个 value 等等。
//
// 这些要求和条件是使用此类建模的。此类的记录用于在 op verifier 中生成验证代码，并在
// pattern matcher 中匹配代码。
//
// Constraints 是具有 descriptive names 的谓词，以方便执行 inspection、提供良好的
// error messages等。
//
// Base class for named constraints.
//
// An op's operands/attributes/results can have various requirements, e.g.,
// having certain types, having values inside a certain range, and so on.
// Besides, for a graph rewrite rule, the source pattern used to match against
// the existing graph has conditions, like the op's operand must be of a more
// constrained subtype, the attribute must have a certain value, and so on.
//
// These requirements and conditions are modeled using this class. Records of
// this class are used to generate verification code in op verifier, and
// matching code in pattern matcher.
//
// Constraints are predicates with descriptive names, to facilitate inspection,
// provide nice error messages, etc.
class Constraint<Pred pred, string desc = ""> {
  // The predicates that this constraint requires.
  Pred predicate = pred;
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
}

// 用于区分不同约束类型的 subclasses。这些 subclasses 用作 TableGen 后端的标记，以
// 便在需要时以不同方式处理不同的约束类型。不从以下 subclasses 派生的约束被视为未分
// 类的约束。
//
// Subclasses used to differentiate different constraint kinds. These are used
// as markers for the TableGen backend to handle different constraint kinds
// differently if needed. Constraints not deriving from the following subclasses
// are considered as uncategorized constraints.

// Subclass for constraints on a type.
class TypeConstraint<Pred predicate, string summary = "",
                     string cppTypeParam = "::mlir::Type",
                     string cppFunctionNameParam = ""> :
    Constraint<predicate, summary> {
  // The name of the C++ Type class if known, or Type if not.
  string cppType = cppTypeParam;
  // The name of the C++ function that is generated for this type constraint.
  // If empty, no C++ function is generated.
  string cppFunctionName = cppFunctionNameParam;
}

// Subclass for constraints on an attribute.
class AttrConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a region.
class RegionConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a successor.
class SuccessorConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// How to use these constraint categories:
//
// * Use TypeConstraint to specify
//   * Constraints on an op's operand/result definition
//   * Further constraints to match an op's operand/result in source pattern
//
// * Use Attr (a subclass for AttrConstraint) for
//   * Constraints on an op's attribute definition
// * Use AttrConstraint to specify
//   * Further constraints to match an op's attribute in source pattern
//
// * Use uncategorized constraint to specify
//   * Multi-entity constraints in rewrite rules

#endif // CONSTRAINTS
