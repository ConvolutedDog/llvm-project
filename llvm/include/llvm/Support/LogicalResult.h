//===- LogicalResult.h - Utilities for handling success/failure -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LOGICALRESULT_H
#define LLVM_SUPPORT_LOGICALRESULT_H

#include <cassert>
#include <optional>

namespace llvm {
/// 此类表示一种表示成功或失败的有效方法。在适当的情况下，它应该优于使用 `bool`，因为
/// 它避免了在解释 bool 结果时出现的所有歧义。
///
/// 此类标记为 NODISCARD 以确保结果得到处理。用户可以使用 `(void)` 明确丢弃结果，例
/// 如 `(void)functionThatReturnsALogicalResult();`。
///
/// 鉴于此类的预期性质，通常不应将其用作经常忽略结果的函数的结果。此类旨在与下面的实用
/// 函数结合使用。
///
/// This class represents an efficient way to signal success or failure. It
/// should be preferred over the use of `bool` when appropriate, as it avoids
/// all of the ambiguity that arises in interpreting a boolean result. This
/// class is marked as NODISCARD to ensure that the result is processed. Users
/// may explicitly discard a result by using `(void)`, e.g.
/// `(void)functionThatReturnsALogicalResult();`. Given the intended nature of
/// this class, it generally shouldn't be used as the result of functions that
/// very frequently have the result ignored. This class is intended to be used
/// in conjunction with the utility functions below.
struct [[nodiscard]] LogicalResult {
public:
  /// If isSuccess is true a `success` result is generated, otherwise a
  /// 'failure' result is generated.
  static LogicalResult success(bool IsSuccess = true) {
    return LogicalResult(IsSuccess);
  }

  /// If isFailure is true a `failure` result is generated, otherwise a
  /// 'success' result is generated.
  static LogicalResult failure(bool IsFailure = true) {
    return LogicalResult(!IsFailure);
  }

  /// Returns true if the provided LogicalResult corresponds to a success value.
  constexpr bool succeeded() const { return IsSuccess; }

  /// Returns true if the provided LogicalResult corresponds to a failure value.
  constexpr bool failed() const { return !IsSuccess; }

private:
  LogicalResult(bool IsSuccess) : IsSuccess(IsSuccess) {}

  /// Boolean indicating if this is a success result, if false this is a
  /// failure result.
  bool IsSuccess;
};

/// Utility function to generate a LogicalResult. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline LogicalResult success(bool IsSuccess = true) {
  return LogicalResult::success(IsSuccess);
}

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
inline LogicalResult failure(bool IsFailure = true) {
  return LogicalResult::failure(IsFailure);
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a success value.
inline bool succeeded(LogicalResult Result) { return Result.succeeded(); }

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a failure value.
inline bool failed(LogicalResult Result) { return Result.failed(); }

/// 此类支持表示失败结果或类型为 `T` 的有效值。这允许与 LogicalResult 集成，同时在成
/// 功路径上提供值。
///
/// This class provides support for representing a failure result, or a valid
/// value of type `T`. This allows for integrating with LogicalResult, while
/// also providing a value on the success path.
template <typename T> class [[nodiscard]] FailureOr : public std::optional<T> {
public:
  /// 允许从 LogicalResult 构造。结果必须失败。
  /// 成功结果应使用类型 `T` 的适当实例。
  ///
  /// Allow constructing from a LogicalResult. The result *must* be a failure.
  /// Success results should use a proper instance of type `T`.
  FailureOr(LogicalResult Result) {
    assert(failed(Result) &&
           "success should be constructed with an instance of 'T'");
  }
  FailureOr() : FailureOr(failure()) {}
  FailureOr(T &&Y) : std::optional<T>(std::forward<T>(Y)) {}
  FailureOr(const T &Y) : std::optional<T>(Y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  FailureOr(const FailureOr<U> &Other)
      : std::optional<T>(failed(Other) ? std::optional<T>()
                                       : std::optional<T>(*Other)) {}

  operator LogicalResult() const { return success(has_value()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using std::optional<T>::operator bool;
  using std::optional<T>::has_value;
};

/// Wrap a value on the success path in a FailureOr of the same value type.
template <typename T,
          typename = std::enable_if_t<!std::is_convertible_v<T, bool>>>
inline auto success(T &&Y) {
  return FailureOr<std::decay_t<T>>(std::forward<T>(Y));
}

/// 此类表示解析类操作的成功/失败，这些操作认为使用 `||` 将可失败操作链接在一起很重要。
/// 这是 `LogicalResult` 的扩展版本，允许显式转换为 bool。
///
/// 此类不应用于一般错误处理情况 - 我们更倾向于使用 `succeeded`/`failed` 谓词保持逻
/// 辑明确。但是，如果没有这个，传统的 monadic 样式解析逻辑有时会被样板所吞没，因此我
/// 们为重要的狭隘情况提供这个。
///
/// This class represents success/failure for parsing-like operations that find
/// it important to chain together failable operations with `||`.  This is an
/// extended version of `LogicalResult` that allows for explicit conversion to
/// bool.
///
/// This class should not be used for general error handling cases - we prefer
/// to keep the logic explicit with the `succeeded`/`failed` predicates.
/// However, traditional monadic-style parsing logic can sometimes get
/// swallowed up in boilerplate without this, so we provide this for narrow
/// cases where it is important.
///
class [[nodiscard]] ParseResult : public LogicalResult {
public:
  ParseResult(LogicalResult Result = success()) : LogicalResult(Result) {}

  /// Failure is true in a boolean context.
  constexpr explicit operator bool() const { return failed(); }
};
} // namespace llvm

#endif // LLVM_SUPPORT_LOGICALRESULT_H
