#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
// [0-255] means the ascii value of an unknown character like '+'.
enum Token {
  tok_eof = -1,        // EOF: In C language, you can use functions such as fgetc, 
                       // fgets, or fread to read files, and use the return value 
                       // to determine whether EOF has been reached.

  // commands
  tok_def = -2,        // The def command.
  tok_extern = -3,     // The extern command.

  // primary
  tok_identifier = -4, // The identifier identified by the Lexer is usually used 
                       // to naming the symbols of the name variables, functions, 
                       // classes or other entities.
  tok_number = -5      // The numeric value.
};

// If the current token is an identifier, the `IdentifierStr` global variable ho-
// lds the name of the identifier.
static std::string IdentifierStr; // Filled in if tok_identifier
// If the current token is a numeric literal (like 1.0), the `NumVal` global vari-
// able holds its value.
static double NumVal;             // Filled in if tok_number

/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  // `isalpha()` is a standard library function that checks if a character is a 
  // letter.
  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;

    // `isalnum()` is a standard library function that checks whether a character 
    // is a letter or a number. If the current IdentifierStr starts from a letter
    // and is followed by letters or numbers, then this is a complete Identifier.
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")     // def
      return tok_def;
    if (IdentifierStr == "extern")  // extern
      return tok_extern;
    return tok_identifier;          // The symbols of the name variables, functions
                                    // classes or other entities.
  }

  // `isdigit()` is a standard library function that checks whether a character is 
  // a digit character (i.e., the characters '0' to '9').
  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    // `strtod()` is a standard library function that converts a string to a double-
    // precision floating point number.
    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') { // // Skip any comments lines.
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

namespace {

/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  // `NumberExprAST` class captures the numeric value of the literal as an instance 
  // variable. This allows later phases of the compiler to know what the stored nu-
  // meric value is.
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  // A virtual method to pretty print the code.
  VariableExprAST(const std::string &Name) : Name(Name) {}
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  // `LHS` and `RHS` point to each `ExprAST` object.
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  // `Args` points to a list of `ExprAST` object.
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  // `calls` capture a function name as well as a list of any argument expressions.
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}
};

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}

  const std::string &getName() const { return Name; }
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;  // Point to an item of `enum Token`.
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  // `isaSCII()` is a standard library function used to check whether the given 
  // character is an ASCII character.
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  // This takes the current number value, creates a NumberExprAST node.
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  // This advances the lexer to the next token.
  getNextToken(); // consume the number
  // Returns the NumberExprAST node.
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  // Use recursion by calling `ParseExpression()`. It allows us to handle recur-
  // sive grammars, and keeps each production very simple. Note that parentheses 
  // do not cause construction of AST nodes themselves. While we could do it this 
  // way, the most important role of parentheses are to guide the parser and pro-
  // vide grouping. Once the parser constructs the AST, parentheses are not needed.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");
  getNextToken(); // eat ).
  return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  // Define a helper function to wrap all the simple expression-parsing logic to-
  // gether into one entry point. Call this class of expressions "primary" expre-
  // ssions. In order to parse an arbitrary primary expression, we need to deter-
  // mine what sort of expression it is.
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  }
}

/// binoprhs
///   ::= ('+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // The basic idea of operator precedence parsing is to break down an expression 
  // with potentially ambiguous binary operators into pieces. Consider, for exam-
  // ple, the expression "a+b+(c+d)*e*f+g". Operator precedence parsing considers 
  // this as a stream of primary expressions separated by binary operators. As su-
  // ch, it will first parse the leading primary expression "a", then it will see 
  // the pairs [+, b] [+, (c+d)] [*, e] [*, f] and [+, g]. 
  
  // Note that because parentheses are primary expressions, the binary expression 
  // parser doesn’t need to worry about nested subexpressions like (c+d) at all.
  
  // To start, an expression is a primary expression potentially followed by a se-
  // quence of [binop,primaryexpr] pairs.
  
  // `ParseBinOpRHS()` is the function that parses the sequence of pairs. It takes 
  // a precedence and a pointer to an expression for the part that has been parsed 
  // so far. Note that "x" is a perfectly valid expression: As such, "binoprhs" is 
  // allowed to be empty, in which case it returns the expression that is passed 
  // into it.

  // Like the expression "a+b+(c+d)*e*f+g", Primary == "a" and BinOpRHS is the pa-
  // irs [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  // Like the expression "1+b+(c+d)*e*f+g", Primary == 1 and BinOpRHS is the pairs 
  // [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  // Like the expression "(c+d*5)+b+(c+d)*e*f+g", Primary == "(c+d*5)" and BinOpRHS 
  // is the pairs [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  
  
  /*
            +
          /   \
         +     g
       /   \
      +     * 
     / \   / \
    a   b *   f
         / \
        e   +
           / \
          c   d
  */

  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    // The priority of the currently obtained operator is not close enough to that 
    // of the left-hand expression, so the parsed left-hand expression LHS is re-
    // turned and parsing ends.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    getNextToken(); // eat binop

    // Parse the primary expression after the binary operator.
    auto RHS = ParsePrimary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    // If the priority of the current operator (TokPrec) is lower than NextPrec, 
    // it means that the following operator needs to be combined with the current 
    // RHS. By recursively calling ParseBinOpRHS, RHS is continued to be parsed 
    // as the left subexpression, and the priority is increased (TokPrec + 1).
    if (TokPrec < NextPrec) {
      // Like "1+1+1", passing TokPrec + 1 means we want to the first "1+1" to be 
      // combined together.
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS =
        std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  }
}

/// expression
///   ::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  // Like the expression "a+b+(c+d)*e*f+g", Primary == "a" and BinOpRHS is the pa-
  // irs [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  // Like the expression "1+b+(c+d)*e*f+g", Primary == 1 and BinOpRHS is the pairs 
  // [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  // Like the expression "(c+d*5)+b+(c+d)*e*f+g", Primary == "(c+d*5)" and BinOpRHS 
  // is the pairs [+, b] [+, (c+d)] [*, e] [*, f] and [+, g].
  // So, `ParsePrimary()` needs to parse the expressions whose first token is iden-
  // tifier, number, or '('.
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != tok_identifier)
    return LogErrorP("Expected function name in prototype");

  std::string FnName = IdentifierStr;
  getNextToken();

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Top-Level parsing
//===----------------------------------------------------------------------===//

static void HandleDefinition() {
  if (ParseDefinition()) {
    fprintf(stderr, "Parsed a function definition.\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (ParseExtern()) {
    fprintf(stderr, "Parsed an extern\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (ParseTopLevelExpr()) {
    fprintf(stderr, "Parsed a top-level expr\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    fprintf(stderr, "ready> ");
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
