//===- Lexer.h - Lexer for the Toy language -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple Lexer for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_LEXER_H
#define TOY_LEXER_H

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace toy {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purposes.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  Lexer(std::string filename)
      : lastLocation(
            {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n"
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    // `.front()` returns a reference to the first character of the `StringRef`
    // object.
    auto nextchar = curLineBuffer.front();
    // `.drop_front()` returns a new `StringRef` object that excludes the first 
    // character of the original `StringRef` object.
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char)lastChar;

      if (identifierStr == "return")
        return tok_return;
      if (identifierStr == "def")
        return tok_def;
      if (identifierStr == "var")
        return tok_var;
      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifierStr;

  /// If the current Token is a number, this contains the value.
  double numVal = 0;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    // It will read one line at a time and return an empty string when the 
    // end of the buffer is reached. `begin` holds the current position in 
    // the buffer (the starting point for reading the next line).
    auto *begin = current;
    // Loop that increments `current` while it is within the bounds of the 
    // buffer and hasn't encountered a newline character ('\n'). This loop 
    // continues until it finds either the end of the buffer or a newline 
    // character.
    while (current <= end && *current && *current != '\n')
      ++current;
    // Checks if `current` is still within the bounds of the buffer and if 
    // the character at `current` is not null. If true, it increments the 
    // `current` to move past the newline character. This effectively all-
    // ows the reading of the next line to begin.
    if (current <= end && *current)
      ++current;
    // Creates a `llvm::StringRef` object called `result`, initializing it 
    // with a substring from the `begin` pointer to `current`. The length 
    // of the substring is calculated as `current - begin`, representing 
    // the last line read.
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }
  /// Declares two private member variables: `current` (a pointer to the 
  /// current position being read in the buffer) and `end` (a pointer to 
  /// the end of the buffer). These are used to track the reading posi-
  /// tion within the buffer.
  const char *current, *end;
};
} // namespace toy

#endif // TOY_LEXER_H
