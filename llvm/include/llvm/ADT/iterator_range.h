//===- iterator_range.h - A range adaptor for iterators ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This provides a very simple, boring adaptor for a begin and end iterator
/// into a range type. This should be used to build range views that work well
/// with range based for loops and range based constructors.
///
/// Note that code here follows more standards-based coding conventions as it
/// is mirroring proposed interfaces for standardization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_RANGE_H
#define LLVM_ADT_ITERATOR_RANGE_H

#include "llvm/ADT/ADL.h"
#include <type_traits>
#include <utility>

namespace llvm {

template <typename From, typename To, typename = void>
struct explicitly_convertible : std::false_type {};

template <typename From, typename To>
struct explicitly_convertible<
    From, To,
    std::void_t<decltype(static_cast<To>(
        std::declval<std::add_rvalue_reference_t<From>>()))>> : std::true_type {
};

/// 定义了一个名为 iterator_range 的类模板，它充当迭代器对的 range adaptor。简单
/// 来说，它将一对迭代器包装起来，使其能够像标准库中的 range（例如 std::vector 或
/// std::array）一样使用，例如在基于 range 的 for 循环中。
///
///  - 模板参数 `IteratorT`: 指定迭代器的类型。 这意味着 iterator_range 可以用于各
///    种类型的迭代器。
///  - 成员变量 begin_iterator, end_iterator: 分别存储范围的起始迭代器和结束迭代器。
///
/// A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
#if __GNUC__ == 7 || (__GNUC__ == 8 && __GNUC_MINOR__ < 4)
  // Be careful no to break gcc-7 and gcc-8 < 8.4 on the mlir target.
  // See https://github.com/llvm/llvm-project/issues/63843
  template <typename Container>
#else
  template <
      typename Container,
      // 如果 `llvm::detail::IterOfRange<Container>` 可以显式转换为 `IteratorT`，
      // 那么 explicitly_convertible<...>::value 为 true。这时候才能够启用使用
      // `typename Container` 的这个 `iterator_range` 构造函数的特定模板实例化。
      //
      // `std::enable_if_t` 决定了启用使用 `typename Container` 的这个构造函数的
      // 特定模板实例化。`std::enable_if_t` 和 `explicitly_convertible` 的组合起
      // 到了一个筛选器的作用。它们只允许在满足特定条件下，`iterator_range` 的构造
      // 函数模板使用特定的 `Container` 类型进行实例化。如果不满足条件，编译器会默
      // 默地忽略这个实例化，不会报错。所以，它不是启用 `typename Container` 本身
      // （`Container` 始终是一个模板参数），而是启用使用该 `Container` 类型调用
      // `iterator_range` 构造函数的可能性。换句话说，`std::enable_if_t` 控制的
      // 是模板实例化，而不是模板参数本身的启用或禁用。`Container` 始终是一个有效
      // 的模板参数，在不满足条件时，使用该 `Container` 的特定 `iterator_range`
      // 构造函数实例化会被编译器忽略。
      //
      // 若 `explicitly_convertible<...>::value` 为 false，`std::enable_if_t`
      // 会产生一个未定义的类型。这在函数参数列表中是不允许的。编译器会报错，因为函
      // 数参数列表中不能出现未定义的类型。通过在 `std::enable_if_t<...>` 后面添
      // 加 `* = nullptr`，实际上把 `std::enable_if_t<...>` 的结果变成了一个指
      // 向 `nullptr` 的指针。即使 `std::enable_if_t` 产生一个未定义的类型，后面
      // `nullptr` 仍然是一个有效的类型（`void*`），所以编译器不会报错。但是，因为
      // 这个参数是默认参数，且值为 `nullptr`，它不会影响函数的行为。
      std::enable_if_t<explicitly_convertible<
          llvm::detail::IterOfRange<Container>, IteratorT>::value> * = nullptr>
#endif
  iterator_range(Container &&c)
      : begin_iterator(adl_begin(c)), end_iterator(adl_end(c)) {
  }
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
  bool empty() const { return begin_iterator == end_iterator; }
};

template <typename Container>
iterator_range(Container &&)
    -> iterator_range<llvm::detail::IterOfRange<Container>>;

/// 用于迭代 sub-ranges 的便捷函数。
///
/// 这提供了一些语法糖，使在 for 循环中使用 sub-ranges 变得更容易一些。
/// 类似于 std::make_pair()。
///
/// Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T> iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

}

#endif
