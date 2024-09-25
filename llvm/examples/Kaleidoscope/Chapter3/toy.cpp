#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
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

  // The `codegen()` method says to emit IR for that AST node along with all the 
  // things it depends on, and they all return an LLVM Value object. "Value" is 
  // the class used to represent a "Static Single Assignment (SSA) register" or 
  // "SSA value" in LLVM.
  virtual Value *codegen() = 0;
  // "= 0" means that it is a pure virtual function, it cannot be instantiated.
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}

  Value *codegen() override;
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}

  Value *codegen() override;
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

  Value *codegen() override;
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

  Function *codegen();
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

  Function *codegen();
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
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
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
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
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
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
    if (TokPrec < NextPrec) {
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
// Code Generation
//===----------------------------------------------------------------------===//

// `TheContext` is an opaque object that owns a lot of core LLVM data structures, 
// such as the type and constant value tables. We don't need to understand it in 
// detail, we just need a single instance to pass into APIs that require it.
static std::unique_ptr<LLVMContext> TheContext;
// `TheModule` is an LLVM construct that contains functions and global variables. 
// In many ways, it is the top-level structure that the LLVM IR uses to contain 
// code. It will own the memory for all of the IR that we generate, which is why 
// the codegen() method returns a raw Value*, rather than a unique_ptr<Value>.
static std::unique_ptr<Module> TheModule;
// The `Builder` object is a helper object that makes it easy to generate LLVM in-
// structions. Instances of the `IRBuilder` class template keep track of the curr-
// ent place to insert instructions and has methods to create new instructions.
static std::unique_ptr<IRBuilder<>> Builder;
// The `NamedValues` map keeps track of which values are defined in the current 
// scope and what their LLVM representation is. (In other words, it is a symbol 
// table for the code). In this form of Kaleidoscope, the only things that can be 
// referenced are function parameters. As such, function parameters will be in 
// this map when generating code for their function body.
static std::map<std::string, Value *> NamedValues;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

// 将一个浮点数值（Val）转换为 LLVM IR 中的浮点常量表示，返回一个指向该浮点常量的指针（
// Value*），以便在后续的代码生成过程中使用。Value* 是返回类型，表示返回一个指向 Value 
// 对象的指针。Value 是 LLVM 中所有值的基类。
Value *NumberExprAST::codegen() {
  // ConstantFP 是 LLVM 中的一个类，用于表示浮点常量。get 是 ConstantFP 类的一个静态方
  // 法，用于创建一个新的浮点常量。*TheContext 是一个指向 LLVM 上下文的指针。LLVMContext 
  // 是 LLVM 中保存所有全局数据的容器。APFloat(Val) 是一个 APFloat 对象的构造函数调用，
  // APFloat 是 LLVM 中表示任意精度浮点数的类。
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  // Assume that the variable has already been emitted somewhere and its value 
  // is available. In practice, the only values that can be in the NamedValues 
  // map are function arguments. This code simply checks to see that the speci-
  // fied name is in the map (if not, an unknown variable is being referenced) 
  // and returns the value for it.
  Value *V = NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");
  return V;
}

Value *BinaryExprAST::codegen() {
  // We recursively emit code for the left-hand side of the expression, then the 
  // right-hand side, then we compute the result of the binary expression. In th-
  // is code, we do a simple switch on the opcode to create the right LLVM ins-
  // truction.
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;

  switch (Op) {
  case '+':
    // `IRBuilder` knows where to insert the newly created instruction, all you 
    // have to do is specify what instruction to create (e.g. with CreateFAdd), 
    // which operands to use (L and R here) and optionally provide a name for 
    // the generated instruction.
    // If the code above emits multiple "addtmp" variables, LLVM will automati-
    // cally provide each one with an increasing, unique numeric suffix. Local 
    // value names for instructions are purely optional, but it makes it much 
    // easier to read the IR dumps.
    return Builder->CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder->CreateFSub(L, R, "subtmp");
  case '*':
    return Builder->CreateFMul(L, R, "multmp");
  case '<':
    L = Builder->CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    // `uitofp` instruction converts its input integer into a floating point 
    // value by treating the input as an unsigned value. If we used the sitofp 
    // instruction, the '<' operator would return 0.0 and -1.0, depending on 
    // the input value.
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
  default:
    return LogErrorV("invalid binary operator");
  }
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  // This code initially does a function name lookup in the LLVM Module's sy-
  // mbol table. Recall that the LLVM Module is the container that holds the 
  // functions we are JIT'ing. By giving each function the same name as what 
  // the user specifies, we can use the LLVM symbol table to resolve function 
  // names for us.
  Function *CalleeF = TheModule->getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  // Once we have the function to call, we recursively codegen each argument 
  // that is to be passed in, and create an LLVM call instruction. Note that 
  // LLVM uses the native C calling conventions by default, allowing these 
  // calls to also call into standard library functions like "sin" and "cos", 
  // with no additional effort.
  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

// Note first that this function returns a "Function*" instead of a "Value*". 
// Because a "prototype" really talks about the external interface for a fun-
// ction (not the value computed by an expression), it makes sense for it to 
// return the LLVM Function it corresponds to when codegen'd.
Function *PrototypeAST::codegen() {
  // Make the function type:  double(double,double) etc.
  // The call to `FunctionType::get` creates the FunctionType that should be 
  // used for a given Prototype. Since all function arguments in Kaleidoscope 
  // are of type double, the first line creates a vector of "N" LLVM double 
  // types. 
  std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
  // It then uses the Functiontype::get method to create a function type that 
  // takes "N" doubles as arguments, returns one double as a result, and that 
  // is not vararg (the false parameter indicates this). Note that Types in 
  // LLVM are uniqued just like Constants are, so you don't "new" a type, you 
  // "get" it.
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
  // The line bellow actually creates the IR Function corresponding to the 
  // Prototype. This indicates the type, linkage and name to use, as well as 
  // which module to insert into. "external linkage" means that the function 
  // may be defined outside the current module and/or that it is callable by 
  // functions outside the module. The Name passed in is the name the user 
  // specified: since "TheModule" is specified, this name is registered in 
  // "TheModule"s symbol table.
  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  // Finally, we set the name of each of the function's arguments according 
  // to the names given in the Prototype. This step isn't strictly necessary, 
  // but keeping the names consistent makes the IR more readable, and allows 
  // subsequent code to refer directly to the arguments for their names, ra-
  // ther than having to look up them up in the Prototype AST.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}

Function *FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' declaration.
  Function *TheFunction = TheModule->getFunction(Proto->getName());

  // 首先检查当前模块中是否已经存在一个与给定函数原型（Proto）同名的函数。
  // For function definitions, we start by searching TheModule's symbol table 
  // for an existing version of this function, in case one has already been 
  // created using an 'extern' statement. If Module::getFunction returns null 
  // then no previous version exists, so we'll codegen one from the Prototype. 
  if (!TheFunction)
    // 如果不存在，则调用 Proto->codegen() 生成函数原型。
    TheFunction = Proto->codegen();

  // 如果函数原型生成失败，返回 nullptr。
  if (!TheFunction)
    return nullptr;

  // Create a new basic block to start insertion into.
  // Now we get to the point where the Builder is set up. The first line cre-
  // ates a new basic block (named "entry"), which is inserted into TheFunc-
  // tion. The second line then tells the builder that new instructions should 
  // be inserted into the end of the new basic block. Basic blocks in LLVM are 
  // an important part of functions that define the Control Flow Graph. Since 
  // we don't have any control flow, our functions will only contain one block 
  // at this point. We'll fix this in Chapter 5 :).
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  // Next we add the function arguments to the NamedValues map (after first 
  // clearing it out) so that they"re accessible to VariableExprAST nodes.
  NamedValues.clear();
  for (auto &Arg : TheFunction->args())
    NamedValues[std::string(Arg.getName())] = &Arg;

  // Once the insertion point has been set up and the NamedValues map populated, 
  // we call the codegen() method for the root expression of the function. If no 
  // error happens, this emits code to compute the expression into the entry block 
  // and returns the value that was computed. Assuming no error, we then create 
  // an LLVM ret instruction, which completes the function. Once the function is 
  // built, we call verifyFunction, which is provided by LLVM. This function does 
  // a variety of consistency checks on the generated code, to determine if our 
  // compiler is doing everything right. Using this is important: it can catch a 
  // lot of bugs. Once the function is finished and validated, we return it.
  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
    Builder->CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  // Handling of the error case. For simplicity, we handle this by merely deleting 
  // the function we produced with the `eraseFromParent` method. This allows the 
  // user to redefine a function that they incorrectly typed in before: if we did
  // not delete it, it would live in the symbol table, with a body, preventing fu-
  // ture redefinition.
  TheFunction->eraseFromParent();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModule() {
  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition:");
      FnIR->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read top-level expression:");
      FnIR->print(errs());
      fprintf(stderr, "\n");

      // Remove the anonymous expression.
      FnIR->eraseFromParent();
    }
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

  // Make the module, which holds all the code.
  InitializeModule();

  // Run the main "interpreter loop" now.
  MainLoop();

  // Print out all of the generated code.
  TheModule->print(errs(), nullptr);

  return 0;
}
