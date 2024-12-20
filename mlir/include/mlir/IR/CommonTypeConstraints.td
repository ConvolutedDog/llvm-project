//===-- CommonTypeConstraints.td - Common Type Constraints--*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains commonly used type constraints.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_TYPE_CONSTRAINTS_TD
#define COMMON_TYPE_CONSTRAINTS_TD

include "mlir/IR/Constraints.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Common predicates
//===----------------------------------------------------------------------===//

// 类型是否为 VectorType。
// 暂时明确禁止 0-D 向量，直到我们有 good enough coverage。
//
// Whether a type is a VectorType.
// Explicitly disallow 0-D vectors for now until we have good enough coverage.
def IsVectorTypePred : And<[CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
                            CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">]>;

// 允许逐步过渡到 0-D 向量的临时向量类型克隆。
//
// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
def IsVectorOfAnyRankTypePred : CPred<"::llvm::isa<::mlir::VectorType>($_self)">;

// 类型是否为定长VectorType。
//
// Whether a type is a fixed-length VectorType.
def IsFixedVectorTypePred : CPred<[{::llvm::isa<::mlir::VectorType>($_self) &&
                                  !::llvm::cast<VectorType>($_self).isScalable()}]>;

// 类型是否为可扩展的 VectorType。
//
// Whether a type is a scalable VectorType.
def IsVectorTypeWithAnyDimScalablePred
        : CPred<[{::llvm::isa<::mlir::VectorType>($_self) &&
                  ::llvm::cast<VectorType>($_self).isScalable()}]>;

// 类型是否为可扩展的 VectorType，带有 a single trailing scalable dimension。这里
// 是指仅有最后一个维度可扩展，其他维度不可扩展。
//
// Whether a type is a scalable VectorType, with a single trailing scalable dimension.
// Examples:
// Valid:
//   - vector<[4]xf32>, vector<2x3x[2]xi64>, vector<32x[8]xi32>
// Invalid
//   - vector<[4]x8xi32>, vector<[2]x[2]xf64>, vector<2x[8]x4xi32>
def IsVectorTypeWithOnlyTrailingDimScalablePred : And<[
  CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
  CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">,
  CPred<"::llvm::cast<::mlir::VectorType>($_self).getScalableDims().back()">,
  CPred<"!llvm::is_contained(::llvm::cast<::mlir::VectorType>($_self).getScalableDims().drop_back(), true)">
]>;

// 类型是否为 VectorType，并且所有维度是否可扩展。
//
// Whether a type is a VectorType and all dimensions are scalable.
def IsVectorTypeWithAllDimsScalablePred : And<[
  IsVectorTypePred,
  CPred<[{::llvm::cast<::mlir::VectorType>($_self).allDimsScalable()}]>
]>;

// 类型是否为 TensorType。
//
// Whether a type is a TensorType.
def IsTensorTypePred : CPred<"::llvm::isa<::mlir::TensorType>($_self)">;

// 类型是否为 MemRefType。
//
// Whether a type is a MemRefType.
def IsMemRefTypePred : CPred<"::llvm::isa<::mlir::MemRefType>($_self)">;

// 类型是否为 UnrankedMemRefType。
//
// Whether a type is an UnrankedMemRefType
def IsUnrankedMemRefTypePred
        : CPred<"::llvm::isa<::mlir::UnrankedMemRefType>($_self)">;

// 类型是否为 UnrankedTensorType。
//
// Whether a type is an UnrankedTensorType
def IsUnrankedTensorTypePred
        : CPred<"::llvm::isa<::mlir::UnrankedTensorType>($_self)">;

// 类型是否为 RankedTensorType。
//
// Whether a type is a RankedTensorType
def IsRankedTensorTypePred
        : CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">;

// 类型是否为 BaseMemRefType。
//
// Whether a type is a BaseMemRefType
def IsBaseMemRefTypePred
        : CPred<"::llvm::isa<::mlir::BaseMemRefType>($_self)">;

// 类型是否为 ShapedType。
//
// Whether a type is a ShapedType.
def IsShapedTypePred : CPred<"::llvm::isa<::mlir::ShapedType>($_self)">;

// 对于 ShapedType，验证它是否具有静态形状。
//
// For a ShapedType, verify that it has a static shape.
def HasStaticShapePred :
        CPred<"::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()">;

// 类型是否为 TupleType。
//
// Whether a type is a TupleType.
def IsTupleTypePred : CPred<"::llvm::isa<::mlir::TupleType>($_self)">;

// 类型是否具有 ValueSemantics trait。
//
// ValueSemantics trait 用于标记具有值语义的类型。值语义意味着：
//
// 可拷贝性: 该类型的对象可以被拷贝，并且拷贝后的对象与原对象相互独立，修改其中一个不
//          会影响另一个。
// 可赋值性: 该类型的对象可以被赋值，赋值后，被赋值的对象拥有与赋值对象相同的值，
//          但两者仍然独立。
//
// 作用:
// 1. 优化: 编译器可以利用值语义进行优化。例如，如果一个函数的参数具有值语义，编译器
//         可以将参数传递给函数的拷贝，而不是传递引用或指针。这可以避免不必要的内存
//         访问和潜在的副作用。
// 2. 代码生成: 在某些情况下，代码生成器需要知道一个类型是否具有值语义。例如，如果一 
//             个类型具有值语义，代码生成器可以生成更简单的代码来拷贝或赋值该类型的
//             对象。
// 3. 类型检查: 类型检查器可以使用 ValueSemantics 特征来确保某些操作只应用于具有值
//             语义的类型。例如，某些操作可能只对可以安全拷贝的类型有效。
//
//
// Whether a type has a ValueSemantics trait.
def HasValueSemanticsPred : CPred<"$_self.hasTrait<::mlir::ValueSemantics>()">;

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// 一种类型，带有类型约束。
//
// A type, carries type constraints.
class Type<Pred condition, string descr = "",
           string cppType = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppType> {
  string description = "";
  string builderCall = "";
}

// 允许为现有 Type 定义提供备用名称和摘要。
//
// Allows providing an alternative name and summary to an existing type def.
class TypeAlias<Type t, string summary = t.summary> :
    Type<t.predicate, summary, t.cppType> {
  let description = t.description;
  let builderCall = t.builderCall;
}

// 一种特定方言的 Type。
//
// A type of a specific dialect.
class DialectType<Dialect d, Pred condition, string descr = "",
                  string cppType = "::mlir::Type"> :
    Type<condition, descr, cppType> {
  Dialect dialect = d;
}

// 可变 type 约束。它扩展为零个或多个基类型。此类用于支持可变 operands/results。
//
// A variadic type constraint. It expands to zero or more of the base type. This
// class is used for supporting variadic operands/results.
class Variadic<Type type> : TypeConstraint<type.predicate,
                                           "variadic of " # type.summary,
                                           type.cppType> {
  Type baseType = type;
  int minSize = 0;
}

// 嵌套可变 Type 约束。它扩展为 base type 的零个或多个可变范围。此类用于支持可变
// operands/results。`variadicSegmentAttrName` 应对应于提供内部可变操作数组大
// 小的 DenseI32ArrayAttr 参数的名称。
//
// A nested variadic type constraint. It expands to zero or more variadic ranges
// of the base type. This class is used for supporting variadic operands and
// results. `variadicSegmentAttrName` should correspond to the name of an
// DenseI32ArrayAttr argument that provides the sizes of the inner variadic
// operand groups.
class VariadicOfVariadic<Type type, string variadicSegmentAttrName>
    : Variadic<type> {
  string segmentAttrName = variadicSegmentAttrName;
}

// 可选类型约束。它扩展为零个或一个基类型。此类用于支持可选操作数/结果。
//
// An optional type constraint. It expands to either zero or one of the base
// type. This class is used for supporting optional operands/results.
class Optional<Type type> : TypeConstraint<type.predicate, type.summary,
                                           type.cppType> {
  Type baseType = type;
}

// 可以使用 MLIR::Builder 构造的类型。
// 请注意，这不会从 Type “继承”，因为它需要为可构建和不可构建的情况复制 Type 子类，
// 以避免菱形 “继承”。
// TODO：我们可以将其扩展为更通用的 “可构建” 特性，使某些类型和某些属性可构建。
//
// A type that can be constructed using MLIR::Builder.
// Note that this does not "inherit" from Type because it would require
// duplicating Type subclasses for buildable and non-buildable cases to avoid
// diamond "inheritance".
// TODO: we may extend this to a more general 'Buildable' trait, making some
// Types and some Attrs buildable.
class BuildableType<code builder> {
  // 调用构建器调用（如果指定）来构造 BuildableType。
  //
  // The builder call to invoke (if specified) to construct the BuildableType.
  code builderCall = builder;
}

// 当且仅当作为参数传递的类型是可构建的，该类型才是可构建的。
// 这旨在供容器类型之类的类型使用，这些类型只有当其元素的类型可构建时才是可构建的。
//
// A type that's buildable iff the type passed as an argument is buildable.
// This is intended for use by types like container types, which are only
// buildable if the type of their elements is buildable.
class SameBuildabilityAs<Type type, code builder> {
  code builderCall = !if(!empty(type.builderCall), "", builder);
}

// 任何类型均可。
//
// Any type at all.
def AnyType : Type<CPred<"true">, "any type">;

// None type
def NoneType : Type<CPred<"::llvm::isa<::mlir::NoneType>($_self)">, "none type",
                    "::mlir::NoneType">,
      BuildableType<"$_builder.getType<::mlir::NoneType>()">;

// 给定列表中的任何类型
//
// Any type from the given list
class AnyTypeOf<list<Type> allowedTypeList, string summary = "",
                string cppType = "::mlir::Type"> : Type<
    // `Or<!foreach(allowedtype, allowedTypeList, allowedtype.predicate)>` 是 Type
    // 基类的构造函数参数之一。`!foreach` 是一个宏，用于遍历 `allowedTypeList` 中的每个
    // `allowedtype`，并提取它们的 `.predicate`。`Or` 将所有 `predicate` 的结果进行逻
    // 辑或运算。这意味着只要 `allowedTypeList` 中的任何一个类型的 `predicate` 返回值为
    // 真，整个条件就为真。
    //
    // Satisfy any of the allowed types' conditions.
    Or<!foreach(allowedtype, allowedTypeList, allowedtype.predicate)>,
    // `!if(!eq(summary, ""), ...)` 是一个条件宏。它的意思是如果 `summary` 不为空，则
    // 使用 `!interleave` 宏将 `allowedTypeList` 中每个类型的 `summary` 用 " or " 连
    // 接起来。否则，使用提供的 `summary` 参数。这部分用于生成类型的描述字符串。
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypeList, t.summary), " or "),
        summary),
    cppType> {
  list<Type> allowedTypes = allowedTypeList;
}

// 满足所有给定类型约束的类型。
//
// A type that satisfies the constraints of all given types.
class AllOfType<list<Type> allowedTypeList, string summary = "",
                string cppType = "::mlir::Type"> : Type<
    // Satisfy all of the allowed types' conditions.
    And<!foreach(allowedType, allowedTypeList, allowedType.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedTypeList, t.summary), " and "),
        summary),
    cppType> {
  list<Type> allowedTypes = allowedTypeList;
}

// 满足附加谓词的类型。
//
// A type that satisfies additional predicates.
class ConfinedType<Type type, list<Pred> predicates, string summary = "",
                   string cppType = type.cppType> : Type<
    And<!listconcat([type.predicate], !foreach(pred, predicates, pred))>,
    summary, cppType> {
    Type baseType = type;
    list<Pred> predicateList = predicates;
}

// 整数类型。
//
// Integer types.

// 任何整数类型，无论其宽度和符号语义如何。
//
// Any integer type irrespective of its width and signedness semantics.
def AnyInteger : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, "integer",
                      "::mlir::IntegerType">;

// 任何具有特定宽度的整数类型（无论符号语义如何）。
//
// Any integer type (regardless of signedness semantics) of a specific width.
class AnyI<int width>
    : Type<CPred<"$_self.isInteger(" # width # ")">, width # "-bit integer"> {
  int bitwidth = width;
}

// 具有任意给定宽度列表之一的整数类型。
class AnyIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, AnyI<w>),
              !interleave(widths, "/") # "-bit integer",
              "::mlir::IntegerType">;

def AnyI1  : AnyI<1>;
def AnyI8  : AnyI<8>;
def AnyI16 : AnyI<16>;
def AnyI32 : AnyI<32>;
def AnyI64 : AnyI<64>;

// 任何 signless integer type，无论其宽度如何。
//
// Any signless integer type irrespective of its width.
def AnySignlessInteger : Type<
  CPred<"$_self.isSignlessInteger()">, "signless integer",
        "::mlir::IntegerType">;

// 特定宽度的 signless integer type。
//
// Signless integer type of a specific width.
class I<int width>
    : Type<CPred<"$_self.isSignlessInteger(" # width # ")">,
                  width # "-bit signless integer", "::mlir::IntegerType">,
      BuildableType<"$_builder.getIntegerType(" # width # ")"> {
  int bitwidth = width;
}

// 具有任意给定宽度列表之一的 signless integer type。
class SignlessIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, I<w>),
              !interleave(widths, "/") # "-bit signless integer">;

def I1  : I<1>;
def I8  : I<8>;
def I16 : I<16>;
def I32 : I<32>;
def I64 : I<64>;
def I128 : I<128>;

// 任何 signed integer type，无论其宽度如何。
//
// Any signed integer type irrespective of its width.
def AnySignedInteger : Type<
  CPred<"$_self.isSignedInteger()">, "signed integer">;

// 特定宽度的 signed integer type。
//
// Signed integer type of a specific width.
class SI<int width>
    : Type<CPred<"$_self.isSignedInteger(" # width # ")">,
                  width # "-bit signed integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

// 具有任意给定宽度列表之一的 signed integer type。
class SignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, SI<w>),
              !interleave(widths, "/") # "-bit signed integer">;

def SI1  : SI<1>;
def SI8  : SI<8>;
def SI16 : SI<16>;
def SI32 : SI<32>;
def SI64 : SI<64>;

// 任何 unsigned integer type，无论其宽度如何。
//
// Any unsigned integer type irrespective of its width.
def AnyUnsignedInteger : Type<
  CPred<"$_self.isUnsignedInteger()">, "unsigned integer">;

// 特定宽度的 unsigned integer type。
//
// Unsigned integer type of a specific width.
class UI<int width>
    : Type<CPred<"$_self.isUnsignedInteger(" # width # ")">,
                  width # "-bit unsigned integer", "::mlir::IntegerType">,
      BuildableType<
        "$_builder.getIntegerType(" # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

// 具有任意给定宽度列表之一的 unsigned integer type。
class UnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, UI<w>),
              !interleave(widths, "/") # "-bit unsigned integer">;

def UI1  : UI<1>;
def UI8  : UI<8>;
def UI16 : UI<16>;
def UI32 : UI<32>;
def UI64 : UI<64>;

// Index type.
def Index : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index",
                 "::mlir::IndexType">,
            BuildableType<"$_builder.getIndexType()">;

// 任何 signless integer type 或 Index type。
//
// Any signless integer type or index type.
def AnySignlessIntegerOrIndex : Type<CPred<"$_self.isSignlessIntOrIndex()">,
                                     "signless integer or index">;

// 浮点类型。
//
// Floating point types.

// 任何 float type，无论其宽度如何。
//
// Any float type irrespective of its width.
def AnyFloat : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point",
                    "::mlir::FloatType">;

// 特定宽度的 float type。
//
// Float type of a specific width.
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}

// 具有任意给定宽度列表之一的 float type。
class FloatOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, F<w>),
              !interleave(widths, "/") # "-bit float">;

def F16 : F<16>;
def F32 : F<32>;
def F64 : F<64>;
def F80 : F<80>;
def F128 : F<128>;

def BF16 : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
           BuildableType<"$_builder.getBF16Type()">;
def TF32 : Type<CPred<"$_self.isTF32()">, "tf32 type">,
           BuildableType<"$_builder.getTF32Type()">;
def F8E4M3FN : Type<CPred<"$_self.isFloat8E4M3FN()">, "f8E4M3FN type">,
               BuildableType<"$_builder.getFloat8E4M3FNType()">;
def F8E5M2 : Type<CPred<"$_self.isFloat8E5M2()">, "f8E5M2 type">,
             BuildableType<"$_builder.getFloat8E5M2Type()">;
def F8E4M3 : Type<CPred<"$_self.isFloat8E4M3()">, "f8E4M3 type">,
             BuildableType<"$_builder.getFloat8E4M3Type()">;
def F8E4M3FNUZ : Type<CPred<"$_self.isFloat8E4M3FNUZ()">, "f8E4M3FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3FNUZType()">;
def F8E4M3B11FNUZ : Type<CPred<"$_self.isFloat8E4M3B11FNUZ()">, "f8E4M3B11FNUZ type">,
                 BuildableType<"$_builder.getFloat8E4M3B11FNUZType()">;
def F8E5M2FNUZ : Type<CPred<"$_self.isFloat8E5M2FNUZ()">, "f8E5M2FNUZ type">,
                 BuildableType<"$_builder.getFloat8E5M2FNUZType()">;
def F8E3M4 : Type<CPred<"$_self.isFloat8E3M4()">, "f8E3M4 type">,
             BuildableType<"$_builder.getFloat8E3M4Type()">;
def F4E2M1FN : Type<CPred<"$_self.isFloat4E2M1FN()">, "f4E2M1FN type">,
               BuildableType<"$_builder.getFloat4E2M1FNType()">;
def F6E2M3FN : Type<CPred<"$_self.isFloat6E2M3FN()">, "f6E2M3FN type">,
               BuildableType<"$_builder.getFloat6E2M3FNType()">;
def F6E3M2FN : Type<CPred<"$_self.isFloat6E3M2FN()">, "f6E3M2FN type">,
               BuildableType<"$_builder.getFloat6E3M2FNType()">;
def F8E8M0FNU : Type<CPred<"$_self.isFloat8E8M0FNU()">, "f8E8M0FNU type">,
                BuildableType<"$_builder.getFloat8E8M0FNUType()">;

def AnyComplex : Type<CPred<"::llvm::isa<::mlir::ComplexType>($_self)">,
                      "complex-type", "::mlir::ComplexType">;

class Complex<Type elType>
    : ConfinedType<AnyComplex, [
          SubstLeaves<"$_self",
                      "::llvm::cast<::mlir::ComplexType>($_self).getElementType()",
           elType.predicate>],
           "complex type with " # elType.summary # " elements",
           "::mlir::ComplexType">,
      SameBuildabilityAs<elType, "::mlir::ComplexType::get($_builder.get" # elType #
                               "Type())"> {
  Type elementType = elType;
}

class OpaqueType<string dialect, string name, string summary>
  : Type<CPred<"isOpaqueTypeWithName($_self, \""#dialect#"\", \""#name#"\")">,
         summary, "::mlir::OpaqueType">,
    BuildableType<"::mlir::OpaqueType::get("
                  "$_builder.getStringAttr(\"" # dialect # "\"), \""
                  # name # "\")">;

// 函数类型
// 
// Function Type

// 任何函数类型。
//
// Any function type.
def FunctionType : Type<CPred<"::llvm::isa<::mlir::FunctionType>($_self)">,
                              "function type", "::mlir::FunctionType">;

// 容器类型是一种其中嵌入了另一种类型的类型。
//
// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr, string cppType = "::mlir::Type"> :
    // 首先，检查容器谓词。然后将提​​取的元素代入元素类型检查器中。
    //
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<And<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.summary # " values", cppType>;

// 支持不同 Type 的元素组成的容器类型。这个容器接受一个 Type 列表。
// ```
// class ShapedContainerTypeX<list<Type> allowedTypes,
//                           Pred containerPred, string descr,
//                           string cppType = "::mlir::Type"> :
//     Type<And<[containerPred,
//               Concat<"[](::mlir::Type elementType) { return ",
//                 SubstLeaves<"$_self", "elementType",
//                 AnyTypeOf<allowedTypes>.predicate>,
//                 "; }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())">]>,
//          descr # " ofxxx " # AnyTypeOf<allowedTypes>.summary # " valuesxxx", cppType>;
// 
// def IntTypeX : ShapedContainerTypeX<[I1, I8], CPred<"true">, "", "">;
// ```
// 会生成：
// ```
//   ([](::mlir::Type elementType) { 
//     return ((elementType.isSignlessInteger(1))) || 
//            ((elementType.isSignlessInteger(8))); 
//   }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))
// ```
class ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppType = "::mlir::Type"> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppType>;

// Shaped type 是否具有 rank。
//
// Whether a shaped type is ranked.
def HasRankPred : CPred<"::llvm::cast<::mlir::ShapedType>($_self).hasRank()">;

// Shaped type 是否具有指定的 rank 之一。
//
// Whether a shaped type has one of the specified ranks.
class HasAnyRankOfPred<list<int> ranks> : And<[
    HasRankPred,
    Or<!foreach(rank, ranks,
                CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank()
                         == }]
                      # rank>)>]>;

// Shaped type 的 rank 是否大于或等于指定的 rank 。
//
// Whether a shaped type has a rank greater than or equal of the specified rank.
class HasRankGreaterOrEqualPred<int rank> : And<[
    HasRankPred,
    CPred<[{::llvm::cast<::mlir::ShapedType>($_self).getRank() >= }] # rank>
]>;

// 具有值语义的容器。
//
// Container with value semantics.
class ValueSemanticsContainerOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, HasValueSemanticsPred,
  "container with value semantics">;

// 向量类型。
//
// Vector types.

// 类型为 VectorType 的向量 Type，暂时明确禁止 0-D 向量。
class VectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypePred, "vector",
                      "::mlir::VectorType">;

// 类型为 VectorType 的向量 Type，允许 0-D 向量。
//
// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class VectorOfAnyRankOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorOfAnyRankTypePred, "vector",
                      "::mlir::VectorType">;

// 类型为定长 VectorType 的向量 Type。
class FixedVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsFixedVectorTypePred,
          "fixed-length vector", "::mlir::VectorType">;

// 类型为可扩展的 VectorType 的向量 Type。
class ScalableVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypeWithAnyDimScalablePred,
          "scalable vector", "::mlir::VectorType">;

// 类型为可扩展的 VectorType 的向量 Type，带有 a single trailing scalable dimension。
// 这里是指仅有最后一个维度可扩展，其他维度不可扩展。
//
// Any vector with a single trailing scalable dimension, with an element type in
// the `allowedTypes` list.
//
// Note: This Similar to ScalableVectorOf, with the extra requirement that only
// the trailing dim is scalable.
class VectorWithTrailingDimScalableOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypeWithOnlyTrailingDimScalablePred,
          "trailing scalable vector", "::mlir::VectorType">;

// 向量元素的数量是否来自给定的 `allowedRanks` 列表。
//
// Whether the number of elements of a vector is from the given
// `allowedRanks` list
class IsVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// 固定长度向量的元素数量是否来自给定的 `allowedRanks` 列表。
//
// Whether the number of elements of a fixed-length vector is from the given
// `allowedRanks` list
class IsFixedVectorOfRankPred<list<int> allowedRanks> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// 可缩放向量的元素数量是否来自给定的 `allowedRanks` 列表。
//
// Whether the number of elements of a scalable vector is from the given
// `allowedRanks` list
class IsScalableVectorOfRankPred<list<int> allowedRanks> :
  And<[IsVectorTypeWithAnyDimScalablePred,
       Or<!foreach(allowedlength, allowedRanks,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getRank()
                           == }]
                         # allowedlength>)>]>;

// 任何向量，其 rank 来自给定的 `allowedRanks` 列表
//
// Any vector where the rank is from the given `allowedRanks` list
class VectorOfRank<list<int> allowedRanks> : Type<
  IsVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// 任何固定长度的向量，其 rank 来自给定的 `allowedRanks` 列表
//
// Any fixed-length vector where the rank is from the given `allowedRanks` list
class FixedVectorOfRank<list<int> allowedRanks> : Type<
  IsFixedVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// 任何可扩展的向量，其等级来自给定的 `allowedRanks` 列表
//
// Any scalable vector where the rank is from the given `allowedRanks` list
class ScalableVectorOfRank<list<int> allowedRanks> : Type<
  IsScalableVectorOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::VectorType">;

// 任何向量，其 rank 来自给定的 `allowedRanks` 列表，类型来自给定的 `allowedTypes`
// 列表
//
// Any vector where the rank is from the given `allowedRanks` list and the type
// is from the given `allowedTypes` list
class VectorOfRankAndType<list<int> allowedRanks,
                          list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfRank<allowedRanks>],
  VectorOf<allowedTypes>.summary # VectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

// 固定宽度向量，其 rank 来自给定的 `allowedRanks` 列表，类型来自给定的 `allowedTypes`
// 列表
//
// Fixed-width vector where the rank is from the given `allowedRanks` list and
// the type is from the given `allowedTypes` list
class FixedVectorOfRankAndType<list<int> allowedRanks,
                          list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, VectorOfRank<allowedRanks>],
  FixedVectorOf<allowedTypes>.summary # VectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

// 向量的元素数量是否来自给定的 `allowedLengths` 列表。
//
// Whether the number of elements of a vector is from the given
// `allowedLengths` list
class IsVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// 固定长度向量的元素数量是否来自给定的 `allowedLengths` 列表。
//
// Whether the number of elements of a fixed-length vector is from the given
// `allowedLengths` list
class IsFixedVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsFixedVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// 可缩放向量的元素数量是否来自给定的 `allowedLengths` 列表。
//
// Whether the number of elements of a scalable vector is from the given
// `allowedLengths` list
class IsScalableVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypeWithAnyDimScalablePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::VectorType>($_self).getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Normalizes an index so the indices in both directions have the same value.
// For example, when indexing forwards index 2 is the third element. When
// indexing in reverse the third element is -3. This helper would map both of
// these to the "normalized" index of 3. This makes the bounds checking in
// IsNthDimSizeIsOneOfPred simpler (see first CPred).
class NormalizeIndex<int value> {
  int ret = !if(!lt(value, 0),
    !sub(0, value)  /* -value if negative */,
    !add(value, 1)  /* value + 1 if positive*/);
}

// 形状的第 n 个 dim 是否包含在 `allowedSizes` 内。
// `n` 索引的负值代表反向查找的 index。
//
// Whether the n-th dim of the shape is contained within `allowedSizes`.
// Negative values for `n` index in reverse.
//
// Examples:
// IsNthDimSizeIsOneOfPred<0, {2, 3, 4}>
//  - Accepts any shape where the first dim is 2, 3, or 4.
//    * This means shapes like: 2x8x9x5, 4, 3x1, 4x?, etc
// IsNthDimSizeIsOneOfPred<-1, {16}>
//  - Accepts any shape where the last dim is 16.
//    * This means shapes like 2x16, 16, 1x2x3x4x16, etc
// IsNthDimSizeIsOneOfPred<-2, {10, 5}>
//  - Accepts any shape where the second to last dim is 10 or 5.
//    * This means shapes like: 1x10x2, 2x1x4x5x6, 8x10x?, etc
class IsNthDimSizeIsOneOfPred<int n, list<int> allowedSizes>
  : And<[
      CPred<"::llvm::cast<::mlir::ShapedType>($_self).getRank() >= " # NormalizeIndex<n>.ret>,
      CPred<"::llvm::is_contained(ArrayRef<int64_t>({" # !interleave(allowedSizes, ", ") # "}), "
        # "::llvm::cast<::mlir::ShapedType>($_self).getDimSize("
        #   !if(!lt(n, 0),
              "::llvm::cast<::mlir::ShapedType>($_self).getRank() + " # n,
              "" # n)
        # "))">]>;

// 向量的形状是否与给定的 `shape` 列表匹配。
//
// Whether the shape of a vector matches the given `shape` list.
class IsVectorOfShape<list<int> shape>
  : CPred<"::llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>({" # !interleave(shape, ", ") # "})">;

// 任何向量，其元素数量来自给定的 `allowedLengths` 列表。
//
// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfLength<list<int> allowedLengths> : Type<
  IsVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// 任何固定长度的向量，其元素数量来自给定的 `allowedLengths` 列表
//
// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list
class FixedVectorOfLength<list<int> allowedLengths> : Type<
  IsFixedVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// 任何可缩放向量，其元素数量来自给定的 `allowedLengths` 列表
//
// Any scalable vector where the number of elements is from the given
// `allowedLengths` list
class ScalableVectorOfLength<list<int> allowedLengths> : Type<
  IsScalableVectorOfLengthPred<allowedLengths>,
  " of length " # !interleave(allowedLengths, "/"),
  "::mlir::VectorType">;

// 任何向量，其元素数量来自给定的 `allowedLengths` 列表，其 Type 来自给定的
// `allowedTypes` 列表
//
// Any vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes`
// list
class VectorOfLengthAndType<list<int> allowedLengths,
                            list<Type> allowedTypes> : AllOfType<
  [VectorOf<allowedTypes>, VectorOfLength<allowedLengths>],
  VectorOf<allowedTypes>.summary # VectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// 任何固定长度的向量，其中元素的数量来自给定的 `allowedLengths` 列表，类型来自
// `allowedTypes` 列表
//
// Any fixed-length vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class FixedVectorOfLengthAndType<list<int> allowedLengths,
                                 list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, FixedVectorOfLength<allowedLengths>],
  FixedVectorOf<allowedTypes>.summary #
  FixedVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// 任何可缩放向量，其元素数量来自给定的 `allowedLengths` 列表，类型来自
// `allowedTypes` 列表
//
// Any scalable vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes` list
class ScalableVectorOfLengthAndType<list<int> allowedLengths,
                                    list<Type> allowedTypes> : AllOfType<
  [ScalableVectorOf<allowedTypes>, ScalableVectorOfLength<allowedLengths>],
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// 任何可缩放向量，其 rank 来自给定的 `allowedRanks` 列表，其元素数量来自给定的
// `allowedLengths` 列表，类型来自 `allowedTypes` 列表
//
// Any scalable vector where the rank is from the given `allowedRanks` list and
// the number of elements is from the given `allowedLengths` list and the type
// is from the given `allowedTypes` list
class ScalableVectorOfRankAndLengthAndType<list<int> allowedRanks,
                                           list<int> allowedLengths,
                                           list<Type> allowedTypes> : AllOfType<
  [ScalableVectorOfRank<allowedRanks>, ScalableVectorOf<allowedTypes>,
   ScalableVectorOfLength<allowedLengths>],
  ScalableVectorOfRank<allowedRanks>.summary #
  ScalableVectorOf<allowedTypes>.summary #
  ScalableVectorOfLength<allowedLengths>.summary,
  "::mlir::VectorType">;

// 任何 ShapedType，其中第 n 个 dim 的大小包含在 `allowedSizes` 列表中。
// 反向为 `n` 索引的负值。
//
// Any ShapedType where the size of the n-th dim is contained in `allowedSizes`.
// Negative values for `n` index in reverse.
class ShapedTypeWithNthDimOfSize<int n, list<int> allowedSizes> : Type<
  IsNthDimSizeIsOneOfPred<n, allowedSizes>,
  " with dim " # n # " having a size of {" # !interleave(allowedSizes, ", ") # "}",
  "::mlir::ShapedType">;

// 任何具有单个尾随可扩展维度的可扩展向量，其中尾随维度的大小在 `allowedTrailingSizes`
// 列表中，类型在 `allowedTypes` 列表中。
//
// Any scalable vector with a single trailing scalable dimensions, where the
// size of the trailing dimension is in `allowedTrailingSizes` list, and the
// type is in the `allowedTypes` list.
class VectorWithTrailingDimScalableOfSizeAndType<list<int> allowedTrailingSizes,
                                           list<Type> allowedTypes> : AllOfType<
  [VectorWithTrailingDimScalableOf<allowedTypes>,
   ShapedTypeWithNthDimOfSize<-1, allowedTrailingSizes>],
   VectorWithTrailingDimScalableOf<allowedTypes>.summary #
   ShapedTypeWithNthDimOfSize<-1, allowedTrailingSizes>.summary,
  "::mlir::VectorType">;

// 任意类型的向量，不允许零维度
def AnyVector : VectorOf<[AnyType]>;

// 任意类型的向量，允许零维度
// Temporary vector type clone that allows gradual transition to 0-D vectors.
def AnyVectorOfAnyRank : VectorOfAnyRankOf<[AnyType]>;

// 任意定长的向量
def AnyFixedVector : FixedVectorOf<[AnyType]>;

// 任意可扩展的向量
def AnyScalableVector : ScalableVectorOf<[AnyType]>;

// Shaped types.

// 任意 ShapedType
def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped",
                                   "::mlir::ShapedType">;

//===----------------------------------------------------------------------===//
// Tensor types.

// Unranked 的张量类型，其元素类型来自给定的 `allowedTypes` 列表，并且还满足可
// 选的谓词列表。
//
// Unranked tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
class UnrankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                       string summary = "unranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsUnrankedTensorTypePred], preds)>,
      summary, "::mlir::UnrankedTensorType">;

// Ranked 张量类型，其元素类型来自给定的 `allowedTypes` 列表，并且还满足可选的谓词列表。
//
// Ranked tensor type whose element type is from the given `allowedTypes` list,
// and which additionally satisfies an optional list of predicates.
class RankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [],
                     string summary = "ranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsRankedTensorTypePred], preds)>,
      summary, "::mlir::RankedTensorType">;

// 其元素类型来自给定的 `allowedTypes` 列表，并且还满足可选谓词列表的任何张量类型。
//
// Any tensor type whose element type is from the given `allowedTypes`
// list, and which additionally satisfies an optional list of predicates.
//
// TODO: use `Constraint` instead of `Pred`, so we can generate a better
// default summary (a la `ConfinedAttr`).
class TensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tensor">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsTensorTypePred], preds)>,
      summary, "::mlir::TensorType">;

// 任意张量类型，其元素类型可任意。
def AnyTensor  : TensorOf<[AnyType]>;

// 张量类型，其元素类型被指定。
def I1Tensor   : TensorOf<[I1]>;
def I8Tensor   : TensorOf<[I8]>;
def I16Tensor  : TensorOf<[I16]>;
def I32Tensor  : TensorOf<[I32]>;
def I64Tensor  : TensorOf<[I64]>;
def IndexTensor: TensorOf<[Index]>;

def BF16Tensor : TensorOf<[BF16]>;
def F16Tensor  : TensorOf<[F16]>;
def F32Tensor  : TensorOf<[F32]>;
def F64Tensor  : TensorOf<[F64]>;

// 非 0-Rank 的张量类型。
class Non0RankedTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [HasRankGreaterOrEqualPred<1>],
      "non-0-ranked.tensor">;

// 任意 Ranked 张量类型，其元素类型任意。
def AnyRankedTensor : RankedTensorOf<[AnyType]>;
// 任意非 0-Ranked 张量类型，其元素类型任意。
def AnyNon0RankedTensor  : Non0RankedTensorOf<[AnyType]>;
// 任意 Unranked 的张量类型，其元素类型任意。
def AnyUnrankedTensor  : UnrankedTensorOf<[AnyType]>;

// 任意非 0-Ranked 张量类型 或 任意 Unranked 的张量类型，其元素类型任意。
def AnyNon0RankedOrUnrankedTensor
  : AnyTypeOf<[AnyUnrankedTensor, AnyNon0RankedTensor],
              "non-0-ranked or unranked tensor", "::mlir::TensorType">;

// 具有指定类型和 rank 之一的 ranked 张量类型。
//
// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks>
  : RankedTensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensor">;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

// Ranked 静态形状张量类型，其元素类型来自给定的 `allowedTypes` 列表，且具有静
// 态形状的 ShapedType。
class StaticShapeTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [HasStaticShapePred],
                   "statically shaped tensor">;

// 任意 Ranked 静态形状张量类型，，其元素类型任意。
def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// Memref type.

// 任何 unranked 的 memref，其元素类型来自给定的 `allowedTypes` 列表。
//
// Any unranked memref whose element type is from the given `allowedTypes` list.
class UnrankedMemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes,
                        IsUnrankedMemRefTypePred, "unranked.memref",
                        "::mlir::UnrankedMemRefType">;

// 任意 unranked 的 memref，其元素类型任意。
def AnyUnrankedMemRef : UnrankedMemRefOf<[AnyType]>;

// 任何 ranked 的 memref，其元素类型来自给定的 `allowedTypes` 列表。
//
// Any ranked memref whose element type is from the given `allowedTypes` list.
class MemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref",
                        "::mlir::MemRefType">;

// 非 0-Rank 的 MemRef 类型，其元素类型来自给定的 `allowedTypes` 列表。
class Non0RankedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasRankGreaterOrEqualPred<1>],
         "non-0-ranked." # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

// 任意 ranked 的 MemRef，其元素类型任意。
def AnyMemRef : MemRefOf<[AnyType]>;
// 任意非 0-Rank 的 MemRef 类型，其元素类型任意。
def AnyNon0RankedMemRef : Non0RankedMemRefOf<[AnyType]>;

// 任何 memref（ranked 或 unranked），其元素类型来自给定的 `allowedTypes` 列表，
// 并且还满足可选的谓词列表。
//
// Any memref (ranked or unranked) whose element type is from the given
// `allowedTypes` list, and which additionally satisfies an optional list of
// predicates.
class RankedOrUnrankedMemRefOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "ranked or unranked memref">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsBaseMemRefTypePred], preds)>,
      summary, "::mlir::BaseMemRefType">;

// 任何 memref（ranked 或 unranked），其元素类型任意。
def AnyRankedOrUnrankedMemRef  : RankedOrUnrankedMemRefOf<[AnyType]>;
// 任何非 0-Ranked 或 Unranked MemRef，其元素类型任意。
def AnyNon0RankedOrUnrankedMemRef:
    AnyTypeOf<[AnyUnrankedMemRef, AnyNon0RankedMemRef]>;

// Memref 声明可处理任何 memref，与 rank、size（静态或动态）、layout 或 memory
// space 无关。
//
// Memref declarations handle any memref, independent of rank, size, (static or
// dynamic), layout, or memory space.
def I1MemRef  : MemRefOf<[I1]>;
def I8MemRef  : MemRefOf<[I8]>;
def I16MemRef : MemRefOf<[I16]>;
def I32MemRef : MemRefOf<[I32]>;
def I64MemRef : MemRefOf<[I64]>;

def BF16MemRef : MemRefOf<[BF16]>;
def F16MemRef  : MemRefOf<[F16]>;
def F32MemRef  : MemRefOf<[F32]>;
def F64MemRef  : MemRefOf<[F64]>;

// 具有指定类型和 rank 之一的 MemRef类型。
//
// TODO: Have an easy way to add another constraint to a type.
class MemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

// 静态形状的 MemRef 类型，其元素类型来自给定的 `allowedTypes` 列表，且具有静
// 态形状的 ShapedType。
class StaticShapeMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStaticShapePred],
         "statically shaped " # MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

// 静态形状的 MemRef 类型，其元素类型任意，且具有静态形状的 ShapedType。
def AnyStaticShapeMemRef : StaticShapeMemRefOf<[AnyType]>;

// 对于 MemRefType，验证它是否有步幅。
//
// For a MemRefType, verify that it has strides.
def HasStridesPred : CPred<[{ isStrided(::llvm::cast<::mlir::MemRefType>($_self)) }]>;

// 具有步长的 MemRef 类型，其元素类型来自给定的 `allowedTypes` 列表，
// MemRefType 有步幅。
class StridedMemRefOf<list<Type> allowedTypes> :
    ConfinedType<MemRefOf<allowedTypes>, [HasStridesPred],
         "strided " # MemRefOf<allowedTypes>.summary>;

// 具有步长的 MemRef 类型，其元素类型任意，MemRefType 有步幅。
def AnyStridedMemRef : StridedMemRefOf<[AnyType]>;

// 具有步长的、指定 rank 的 MemRef 类型，其元素类型任意，MemRefType 有步幅。
class AnyStridedMemRefOfRank<int rank> :
  AllOfType<[AnyStridedMemRef, MemRefRankOf<[AnyType], [rank]>],
       AnyStridedMemRef.summary # " of rank " # rank>;

// 具有步长的、指定 rank 的 MemRef 类型，其元素类型来自给定的 `allowedTypes`
// 列表，MemRefType 有步幅。
class StridedMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    ConfinedType<MemRefOf<allowedTypes>, [HasAnyRankOfPred<ranks>],
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary>;

// 这表示一个对元素类型没有任何限制的通用元组。
//
// This represents a generic tuple without any constraints on element type.
def AnyTuple : Type<IsTupleTypePred, "tuple", "::mlir::TupleType">;

// 一种容器类型，其中嵌入了其他类型，但（与 ContainerType 不同）可以容纳混合类型
// 的元素。需要调用以生成所有元素类型的列表。
//
// A container type that has other types embedded in it, but (unlike
// ContainerType) can hold elements with a mix of types. Requires a call that
// produces a list of all elements' types.
class MixedContainerType<Type etype, Pred containerPred, code elementTypesCall,
                         string descr> :
    Type<
        And<[
            containerPred,
            Concat<
                "::llvm::all_of(" # elementTypesCall # ", [](::mlir::Type t) { "
                "return t && (",
                SubstLeaves<"$_self", "t", etype.predicate>,
                "); })"
            >
        ]>,
        descr # " with any combination of " # etype.summary # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypesCall = elementTypesCall;
}

// 包含允许类型的混合元素的元组。
//
// A Tuple that holds a mix of elements of the allowed types.
class TupleOf<list<Type> allowedTypes>
    : MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                         "::llvm::cast<::mlir::TupleType>($_self).getTypes()",
                         "tuple">;

// 具有任意嵌套的元组，其中所有元素都是允许类型的混合。
//
// A Tuple with arbitrary nesting, where all elements are a mix of the allowed
// types.
class NestedTupleOf<list<Type> allowedTypes> :
    MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                       "getFlattenedTypes(::llvm::cast<::mlir::TupleType>($_self))",
                       "nested tuple">;

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//
// 对于 "like" 某种类型或类型 T 集合的类型约束，即它们要么是 T，要么是 T 的向量，要么
// 是 T 的张量。
//
// Type constraint for types that are "like" some type or set of types T, that is
// they're either a T, a vector of Ts, or a tensor of Ts.
class TypeOrContainer<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate,
  ValueSemanticsContainerOf<[allowedType]>.predicate]>,
  name>;

// 对于 "like" 某种类型或类型 T 集合的类型约束，即它们要么是 T，要么是可映射的 T 容器。
// 
// Type constraint for types that are "like" some type or set of types T, that is
// they're either a T or a mapable container of Ts.
class TypeOrValueSemanticsContainer<Type allowedType, string name>
    : TypeConstraint<Or<[
  allowedType.predicate,
  ValueSemanticsContainerOf<[allowedType]>.predicate]>,
  name>;

// Temporary constraint to allow gradual transition to supporting 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
class TypeOrContainerOfAnyRank<Type allowedType, string name> : TypeConstraint<Or<[
  allowedType.predicate, VectorOfAnyRankOf<[allowedType]>.predicate,
  TensorOf<[allowedType]>.predicate]>,
  name>;


// Type constraint for bool-like types: bools, vectors of bools, tensors of
// bools.
def BoolLike : TypeOrContainer<I1, "bool-like">;

def BoolLikeOfAnyRank : TypeOrContainerOfAnyRank<I1, "bool-like">;

// Type constraint for signless-integer-like types: signless integers, indices,
// vectors of signless integers or indices, tensors of signless integers.
def SignlessIntegerLike : TypeOrValueSemanticsContainer<
    AnySignlessIntegerOrIndex, "signless-integer-like">;

def SignlessIntegerLikeOfAnyRank : TypeOrContainerOfAnyRank<
    AnySignlessIntegerOrIndex,
    "signless-integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeOrContainer<AnyFloat, "floating-point-like">;

// Type constraint for signless-integer-like or float-like types.
def SignlessIntegerOrFloatLike : TypeConstraint<Or<[
    SignlessIntegerLike.predicate, FloatLike.predicate]>,
    "signless-integer-like or floating-point-like">;

#endif // COMMON_TYPE_CONSTRAINTS_TD
