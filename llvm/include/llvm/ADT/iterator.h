//===- iterator.h - Utilities for using and defining iterators --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_H
#define LLVM_ADT_ITERATOR_H

#include "llvm/ADT/iterator_range.h"
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace llvm {

/// CRTP 基类，以接口的最小子集的形式实现整个标准迭代器 facade。
///
/// 当以核心子集的形式实现大部分迭代器功能是合理时，请使用此方法。如果您需要特殊行为
/// 或这会对性能产生影响，则可能需要改写相关成员。
///
/// 请注意，此方法不提供的一种抽象是通过否定差值来实现加法形式的减法。否定并不总是能
/// 保留信息，我可以看到非常合理的迭代器设计，但这种方法效果不佳。无论如何，它实际上
/// 不会强制添加太多样板。
///
/// 此方法不提供的另一个抽象是通过加一来实现增量。这些对于所有迭代器类别来说并不等同，
/// 并且尊重这一点会增加很多复杂性，但收获甚微。
///
/// 迭代器应该具有类似于指针的 const rules，并具有一个返回 ReferenceT 的、const 限
/// 定的运算符 `*()`。这与以下示例中的第二个和第三个指针相匹配：
/// \code
///   int Value;
///   { int *I = &Value; }             // ReferenceT 'int&'
///   { int *const I = &Value; }       // ReferenceT 'int&'; const
///   { const int *I = &Value; }       // ReferenceT 'const int&'
///   { const int *const I = &Value; } // ReferenceT 'const int&'; const
/// \endcode
/// 若迭代器 facade 返回指向其自身状态的句柄，则 T（以及 PointerT 和 ReferenceT）
/// 通常应为 const 限定的。否则，如果希望客户端修改句柄本身，则可以将字段声明为可变
/// 的或使用 const_cast。
///
/// 希望使用 `iterator_facade_base` 的类应实现以下方法：
///
/// Forward Iterators:
///   (All of the following methods)
///   - DerivedT &operator=(const DerivedT &R);
///   - bool operator==(const DerivedT &R) const;
///   - T &operator*() const;
///   - DerivedT &operator++();
///
/// Bidirectional Iterators:
///   (All methods of forward iterators, plus the following)
///   - DerivedT &operator--();
///
/// Random-access Iterators:
///   (All methods of bidirectional iterators excluding the following)
///   - DerivedT &operator++();
///   - DerivedT &operator--();
///   (and plus the following)
///   - bool operator<(const DerivedT &RHS) const;
///   - DifferenceTypeT operator-(const DerivedT &R) const;
///   - DerivedT &operator+=(DifferenceTypeT N);
///   - DerivedT &operator-=(DifferenceTypeT N);
///
/// CRTP base class which implements the entire standard iterator facade
/// in terms of a minimal subset of the interface.
///
/// Use this when it is reasonable to implement most of the iterator
/// functionality in terms of a core subset. If you need special behavior or
/// there are performance implications for this, you may want to override the
/// relevant members instead.
///
/// Note, one abstraction that this does *not* provide is implementing
/// subtraction in terms of addition by negating the difference. Negation isn't
/// always information preserving, and I can see very reasonable iterator
/// designs where this doesn't work well. It doesn't really force much added
/// boilerplate anyways.
///
/// Another abstraction that this doesn't provide is implementing increment in
/// terms of addition of one. These aren't equivalent for all iterator
/// categories, and respecting that adds a lot of complexity for little gain.
///
/// Iterators are expected to have const rules analogous to pointers, with a
/// single, const-qualified operator*() that returns ReferenceT. This matches
/// the second and third pointers in the following example:
/// \code
///   int Value;
///   { int *I = &Value; }             // ReferenceT 'int&'
///   { int *const I = &Value; }       // ReferenceT 'int&'; const
///   { const int *I = &Value; }       // ReferenceT 'const int&'
///   { const int *const I = &Value; } // ReferenceT 'const int&'; const
/// \endcode
/// If an iterator facade returns a handle to its own state, then T (and
/// PointerT and ReferenceT) should usually be const-qualified. Otherwise, if
/// clients are expected to modify the handle itself, the field can be declared
/// mutable or use const_cast.
///
/// Classes wishing to use `iterator_facade_base` should implement the following
/// methods:
///
/// Forward Iterators:
///   (All of the following methods)
///   - DerivedT &operator=(const DerivedT &R);
///   - bool operator==(const DerivedT &R) const;
///   - T &operator*() const;
///   - DerivedT &operator++();
///
/// Bidirectional Iterators:
///   (All methods of forward iterators, plus the following)
///   - DerivedT &operator--();
///
/// Random-access Iterators:
///   (All methods of bidirectional iterators excluding the following)
///   - DerivedT &operator++();
///   - DerivedT &operator--();
///   (and plus the following)
///   - bool operator<(const DerivedT &RHS) const;
///   - DifferenceTypeT operator-(const DerivedT &R) const;
///   - DerivedT &operator+=(DifferenceTypeT N);
///   - DerivedT &operator-=(DifferenceTypeT N);
///
template <typename DerivedT, typename IteratorCategoryT, typename T,
          typename DifferenceTypeT = std::ptrdiff_t, typename PointerT = T *,
          typename ReferenceT = T &>
class iterator_facade_base {
public:
  // IteratorCategoryT：
  //   std::input_iterator_tag：输入迭代器，只能单向遍历，只能读取元素值。
  //   std::output_iterator_tag：输出迭代器，只能单向遍历，只能写入元素值。
  //   std::forward_iterator_tag：前向迭代器，可以单向遍历，可以读取元素值。
  //                              这是 std::input_iterator_tag 的增强版。
  //   std::bidirectional_iterator_tag：双向迭代器，可以双向遍历，可以读取
  //                                    元素值。
  //   std::random_access_iterator_tag：随机访问迭代器，可以随机访问元素，
  //                                    可以进行算术运算（例如 +, -）。
  using iterator_category = IteratorCategoryT;
  // 迭代器指向的元素的 Type。例如 index_iterator 中的元素 type 是 size_t。
  using value_type = T;
  // 默认值 std::ptrdiff_t 是 C++ 标准库中定义的一个整数类型，用于表示两个
  // 指针之间的差值。它的主要用途是在处理指针算术时，确保能够正确地表示指针之
  // 间的距离，即使指针指向的内存区域非常大。
  using difference_type = DifferenceTypeT;
  using pointer = PointerT;
  using reference = ReferenceT;

protected:
  enum {
    // 当前迭代器是否是随机访问迭代器，可以随机访问元素，可以进行算术运算
    // （例如 +, -）。
    IsRandomAccess = std::is_base_of<std::random_access_iterator_tag,
                                     IteratorCategoryT>::value,
    // 当前迭代器是否是双向迭代器，可以双向遍历，可以读取元素值。
    IsBidirectional = std::is_base_of<std::bidirectional_iterator_tag,
                                      IteratorCategoryT>::value,
  };

  /// 通过间接复制迭代器来计算引用的 proxy object。这用于需要通过间接方式生成引
  /// 用但迭代器对象可能是临时的 API。代理在内部保留迭代器并通过转换运算符公开间
  /// 接引用。
  ///
  /// A proxy object for computing a reference via indirecting a copy of an
  /// iterator. This is used in APIs which need to produce a reference via
  /// indirection but for which the iterator object might be a temporary. The
  /// proxy preserves the iterator internally and exposes the indirected
  /// reference via a conversion operator.
  class ReferenceProxy {
    friend iterator_facade_base;

    DerivedT I;

    // 迭代器对象可能是临时的 API，则可通过 std::move 转移到 ReferenceProxy 内
    // 保留。
    ReferenceProxy(DerivedT I) : I(std::move(I)) {}

  public:
    operator ReferenceT() const { return *I; }
  };

  /// 通过间接复制迭代器来计算指针的 proxy object。这用于需要生成指针但引用可
  /// 能是临时的。代理在内部保留这个引用并通过通过箭头运算符公开指针。
  /// 
  /// A proxy object for computing a pointer via indirecting a copy of a
  /// reference. This is used in APIs which need to produce a pointer but for
  /// which the reference might be a temporary. The proxy preserves the
  /// reference internally and exposes the pointer via a arrow operator.
  class PointerProxy {
    friend iterator_facade_base;

    ReferenceT R;

    template <typename RefT>
    PointerProxy(RefT &&R) : R(std::forward<RefT>(R)) {}

  public:
    PointerT operator->() const { return &R; }
  };

public:
  DerivedT operator+(DifferenceTypeT n) const {
    static_assert(std::is_base_of<iterator_facade_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
    static_assert(
        IsRandomAccess,
        "The '+' operator is only defined for random access iterators.");
    DerivedT tmp = *static_cast<const DerivedT *>(this);
    tmp += n;
    return tmp;
  }
  friend DerivedT operator+(DifferenceTypeT n, const DerivedT &i) {
    static_assert(
        IsRandomAccess,
        "The '+' operator is only defined for random access iterators.");
    return i + n;
  }
  DerivedT operator-(DifferenceTypeT n) const {
    static_assert(
        IsRandomAccess,
        "The '-' operator is only defined for random access iterators.");
    DerivedT tmp = *static_cast<const DerivedT *>(this);
    tmp -= n;
    return tmp;
  }

  DerivedT &operator++() {
    static_assert(std::is_base_of<iterator_facade_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
    return static_cast<DerivedT *>(this)->operator+=(1);
  }
  DerivedT operator++(int) {
    DerivedT tmp = *static_cast<DerivedT *>(this);
    ++*static_cast<DerivedT *>(this);
    return tmp;
  }
  DerivedT &operator--() {
    static_assert(
        IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    return static_cast<DerivedT *>(this)->operator-=(1);
  }
  DerivedT operator--(int) {
    static_assert(
        IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    DerivedT tmp = *static_cast<DerivedT *>(this);
    --*static_cast<DerivedT *>(this);
    return tmp;
  }

#ifndef __cpp_impl_three_way_comparison
  bool operator!=(const DerivedT &RHS) const {
    return !(static_cast<const DerivedT &>(*this) == RHS);
  }
#endif

  bool operator>(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) < RHS) &&
           !(static_cast<const DerivedT &>(*this) == RHS);
  }
  bool operator<=(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) > RHS);
  }
  bool operator>=(const DerivedT &RHS) const {
    static_assert(
        IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return !(static_cast<const DerivedT &>(*this) < RHS);
  }

  /// 返回一个 `PointerProxy` 对象。这很重要，因为它避免了直接返回一个可能为临时的
  /// `DerivedT` 对象的指针。它首先将 `this` 指针强制转换为 `const DerivedT*`，
  /// 然后调用派生类 `DerivedT` 的 `operator*()` 方法。`operator*()` 通常会返回
  /// 一个对迭代器所指向元素的引用 (`ReferenceT`)。这个引用被传递给 `PointerProxy`
  /// 的构造函数，`PointerProxy` 则保存这个引用，并通过其 `operator->()` 提供对
  /// 该元素的指针访问。
  PointerProxy operator->() const {
    return static_cast<const DerivedT *>(this)->operator*();
  }
  /// 返回一个 `ReferenceProxy` 对象。这同样避免了直接返回可能为临时的引用。它首先
  /// 断言当前迭代器是随机访问迭代器 (`IsRandomAccess`)，因为下标运算符只对随机访问
  /// 迭代器有意义。然后，它将 `this` 指针强制转换为 `const DerivedT*`，并调用派生
  /// 类 `DerivedT` 的 `operator+()` 方法，计算出距离当前迭代器 `n` 个元素的迭代器。
  /// 这个新的迭代器会被传递给 `ReferenceProxy` 的构造函数，`ReferenceProxy` 则保
  /// 存这个迭代器，并通过其转换运算符提供对该元素的引用访问。
  ReferenceProxy operator[](DifferenceTypeT n) const {
    static_assert(IsRandomAccess,
                  "Subscripting is only defined for random access iterators.");
    return static_cast<const DerivedT *>(this)->operator+(n);
  }
};

/// CRTP base class 用于将迭代器适配为不同类型的迭代器。
///
/// 此类可通过 CRTP 用于将一个迭代器适配为另一个迭代器。
/// 通常，这是通过在派生类中提供自定义 \c operator* 实现来实现的。其他方法也可以被
/// 覆盖。
///
/// CRTP base class for adapting an iterator to a different type.
///
/// This class can be used through CRTP to adapt one iterator into another.
/// Typically this is done through providing in the derived class a custom \c
/// operator* implementation. Other methods can be overridden as well.
template <
    // `WrappedIteratorT` 表示这个适配器类将要进行适配的原始迭代器类型。它就像一
    // 个容器，包含了实际执行迭代工作的迭代器。`iterator_adaptor_base` 不直接操
    // 作数据，而是通过 `WrappedIteratorT` 来间接访问和操作数据。可能这里之所以
    // 叫 `WrappedIteratorT`，是因为它被 `iterator_adaptor_base` “包装”起来了，
    // 隐藏了底层迭代器的细节，并提供了一个新的接口。
    typename DerivedT, typename WrappedIteratorT,
    // `std::iterator_traits` 提供了一种通用的方法来访问迭代器的属性，例如迭代器
    // 的类别、值类型、指针类型和引用类型，而无需知道迭代器的具体类型。其定义了以下
    // 成员类型：
    //   - iterator_category: 迭代器的类别，例如：
    //       std::input_iterator_tag：输入迭代器，只能单向遍历，只能读取元素值。
    //       std::output_iterator_tag：输出迭代器，只能单向遍历，只能写入元素值。
    //       std::forward_iterator_tag：前向迭代器，可以单向遍历，可以读取元素值。
    //                                  这是 std::input_iterator_tag 的增强版。
    //       std::bidirectional_iterator_tag：双向迭代器，可以双向遍历，可以读取
    //                                        元素值。
    //       std::random_access_iterator_tag：随机访问迭代器，可以随机访问元素，
    //                                        可以进行算术运算（例如 +, -）。
    //     这用于确定迭代器的能力（例如，是否支持双向遍历、随机访问等）。
    //   - value_type: 迭代器指向的值的类型。
    //   - difference_type: 两个迭代器之间的差值的类型，通常是 std::ptrdiff_t。
    //   - pointer: 迭代器的指针类型。
    //   - reference: 迭代器的引用类型。
    typename IteratorCategoryT =
        typename std::iterator_traits<WrappedIteratorT>::iterator_category,
    typename T = typename std::iterator_traits<WrappedIteratorT>::value_type,
    typename DifferenceTypeT =
        typename std::iterator_traits<WrappedIteratorT>::difference_type,
    typename PointerT = std::conditional_t<
        std::is_same<T, typename std::iterator_traits<
                            WrappedIteratorT>::value_type>::value,
        typename std::iterator_traits<WrappedIteratorT>::pointer, T *>,
    typename ReferenceT = std::conditional_t<
        std::is_same<T, typename std::iterator_traits<
                            WrappedIteratorT>::value_type>::value,
        typename std::iterator_traits<WrappedIteratorT>::reference, T &>>
class iterator_adaptor_base
    : public iterator_facade_base<DerivedT, IteratorCategoryT, T,
                                  DifferenceTypeT, PointerT, ReferenceT> {
  using BaseT = typename iterator_adaptor_base::iterator_facade_base;

protected:
  WrappedIteratorT I;

  iterator_adaptor_base() = default;

  explicit iterator_adaptor_base(WrappedIteratorT u) : I(std::move(u)) {
    static_assert(std::is_base_of<iterator_adaptor_base, DerivedT>::value,
                  "Must pass the derived type to this template!");
  }

  const WrappedIteratorT &wrapped() const { return I; }

public:
  using difference_type = DifferenceTypeT;

  DerivedT &operator+=(difference_type n) {
    static_assert(
        BaseT::IsRandomAccess,
        "The '+=' operator is only defined for random access iterators.");
    I += n;
    return *static_cast<DerivedT *>(this);
  }
  DerivedT &operator-=(difference_type n) {
    static_assert(
        BaseT::IsRandomAccess,
        "The '-=' operator is only defined for random access iterators.");
    I -= n;
    return *static_cast<DerivedT *>(this);
  }
  using BaseT::operator-;
  difference_type operator-(const DerivedT &RHS) const {
    static_assert(
        BaseT::IsRandomAccess,
        "The '-' operator is only defined for random access iterators.");
    return I - RHS.I;
  }

  // We have to explicitly provide ++ and -- rather than letting the facade
  // forward to += because WrappedIteratorT might not support +=.
  using BaseT::operator++;
  DerivedT &operator++() {
    ++I;
    return *static_cast<DerivedT *>(this);
  }
  using BaseT::operator--;
  DerivedT &operator--() {
    static_assert(
        BaseT::IsBidirectional,
        "The decrement operator is only defined for bidirectional iterators.");
    --I;
    return *static_cast<DerivedT *>(this);
  }

  friend bool operator==(const iterator_adaptor_base &LHS,
                         const iterator_adaptor_base &RHS) {
    return LHS.I == RHS.I;
  }
  friend bool operator<(const iterator_adaptor_base &LHS,
                        const iterator_adaptor_base &RHS) {
    static_assert(
        BaseT::IsRandomAccess,
        "Relational operators are only defined for random access iterators.");
    return LHS.I < RHS.I;
  }

  ReferenceT operator*() const { return *I; }
};

/// An iterator type that allows iterating over the pointees via some
/// other iterator.
///
/// The typical usage of this is to expose a type that iterates over Ts, but
/// which is implemented with some iterator over T*s:
///
/// \code
///   using iterator = pointee_iterator<SmallVectorImpl<T *>::iterator>;
/// \endcode
template <typename WrappedIteratorT,
          typename T = std::remove_reference_t<decltype(
              **std::declval<WrappedIteratorT>())>>
struct pointee_iterator
    : iterator_adaptor_base<
          pointee_iterator<WrappedIteratorT, T>, WrappedIteratorT,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category,
          T> {
  pointee_iterator() = default;
  template <typename U>
  pointee_iterator(U &&u)
      : pointee_iterator::iterator_adaptor_base(std::forward<U &&>(u)) {}

  T &operator*() const { return **this->I; }
};

template <typename RangeT, typename WrappedIteratorT =
                               decltype(std::begin(std::declval<RangeT>()))>
iterator_range<pointee_iterator<WrappedIteratorT>>
make_pointee_range(RangeT &&Range) {
  using PointeeIteratorT = pointee_iterator<WrappedIteratorT>;
  return make_range(PointeeIteratorT(std::begin(std::forward<RangeT>(Range))),
                    PointeeIteratorT(std::end(std::forward<RangeT>(Range))));
}

template <typename WrappedIteratorT,
          typename T = decltype(&*std::declval<WrappedIteratorT>())>
class pointer_iterator
    : public iterator_adaptor_base<
          pointer_iterator<WrappedIteratorT, T>, WrappedIteratorT,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category,
          T> {
  mutable T Ptr;

public:
  pointer_iterator() = default;

  explicit pointer_iterator(WrappedIteratorT u)
      : pointer_iterator::iterator_adaptor_base(std::move(u)) {}

  T &operator*() const { return Ptr = &*this->I; }
};

template <typename RangeT, typename WrappedIteratorT =
                               decltype(std::begin(std::declval<RangeT>()))>
iterator_range<pointer_iterator<WrappedIteratorT>>
make_pointer_range(RangeT &&Range) {
  using PointerIteratorT = pointer_iterator<WrappedIteratorT>;
  return make_range(PointerIteratorT(std::begin(std::forward<RangeT>(Range))),
                    PointerIteratorT(std::end(std::forward<RangeT>(Range))));
}

template <typename WrappedIteratorT,
          typename T1 = std::remove_reference_t<decltype(
              **std::declval<WrappedIteratorT>())>,
          typename T2 = std::add_pointer_t<T1>>
using raw_pointer_iterator =
    pointer_iterator<pointee_iterator<WrappedIteratorT, T1>, T2>;

} // end namespace llvm

#endif // LLVM_ADT_ITERATOR_H
