import ast
import inspect
import astor
import textwrap

########
## IR ##
########
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
     | FuncDef(Str name, Str* args, Stmt body)
"""

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']

class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']

class UnOp(ast.AST):
    _fields = ['op', 'e']

class Ref(ast.AST):
    _fields = ['name', 'index']

    def __init__(self, name, index=None):
        super().__init__(name, index)

class IntConst(ast.AST):
    _fields = ['val',]

class FloatConst(ast.AST):
    _fields = ['val',]

## Stmts ##
class Assign(ast.AST):
    _fields = ['ref', 'val']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']
    
    def __init__(self, cond, body, elseBody=None):
        return super().__init__(cond, body, elseBody)

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']

class PythonToSimple(ast.NodeVisitor):
    """
    Translate a Python AST to our simplified IR.
    
    TODO: Your first job is to complete this implementation.
    You only need to translate Python constructs which are actually 
    representable in our simplified IR.
    
    As a bonus, try implementing logic to catch non-representable 
    Python AST constructs and raise a `NotImplementedError` when you
    do. Without this, the compiler will just explode with an 
    arbitrary error or generate malformed results. Carefully catching
    and reporting errors is one of the most challenging parts of 
    building a user-friendly compiler.

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)

    """
    def visit_block(self, body):
        return Block([self.visit(line) for line in body])

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        elif isinstance(node.n, float):
            return FloatConst(val=node.n)
        else:
            raise NotImplementedError("TODO: implement me!")

    def visit_BinOp(self, node):
        return BinOp(op=node.op,
                     left=self.visit(node.left),
                     right=self.visit(node.right))

    def visit_BoolOp(self, node):
        left = self.visit(node.values[0])
        for right in node.values[1:]:
            left = BinOp(op=node.op,
                         left=left,
                         right=self.visit(right))
        return left

    def visit_Compare(self, node):
        left = self.visit(node.left)
        expr = None
        for op, right in zip(node.ops, node.comparators):
            right = self.visit(right)
            compare = CmpOp(op=op, left=left, right=right)
            if expr is None:
                expr = compare
            else:
                expr = BinOp(op=ast.And(),
                             left=expr,
                             right=compare)
            left = right
        return expr

    def visit_UnaryOp(self, node):
        return UnOp(op=node.op, e=self.visit(node.operand))

    def visit_Name(self, node):
        return Ref(name=node.id)

    def visit_Subscript(self, node):
        return Ref(name=self.visit(node.value),
                   index=self.visit(node.slice))

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise NotImplementedError("TODO: add support for tuple assignment")
        return Assign(ref=self.visit(node.targets[0]),
                      val=self.visit(node.value))

    def visit_If(self, node):
        return If(cond=self.visit(node.test),
                  body=self.visit_block(node.body),
                  elseBody=self.visit_block(node.orelse))

    def visit_For(self, node):
        _fields = ['var', 'min', 'max', 'body']
        if not isinstance(node.iter, ast.Call) or node.iter.func.id != 'range':
            raise NotImplementedError("TODO: Add support for for-loops besides range")
        args = node.iter.args
        if len(args) > 2:
            raise NotImplementedError("TODO: Add support for range() with more than 2 arguments")

        if len(args) == 1:
            min_ = IntConst(val=0)
        else:
            min_ = self.visit(args[0])
        max_ = self.visit(args[-1])

        return For(var=node.target.id,
                   min=min_,
                   max=max_,
                   body=self.visit_block(node.body))

    def visit_Return(self, node):
        return Return(self.visit(node.value))
    
    def visit_FunctionDef(self, func):
        assert isinstance(func.body, list)
        body = self.visit_block(func.body)
        return FuncDef(func.name, [self.visit(arg) for arg in func.args.args], body)

    def generic_visit(self, node):
        print("Visiting unsupported node", node.__class__.__name__)
        return node

def Interpret(ir, *args):
    assert isinstance(ir, FuncDef)
    assert len(args) == len(ir.args) # TODO: you should handle functions with arguments
    
    # Initialize a symbol table, to store variable => value bindings
    # TODO: fill this with the function arguments to start
    syms = {}
    for arg, val in zip(ir.args, args):
        syms[arg.arg] = val
    
    # Build a visitor to evaluate Exprs, using the symbol table to look up
    # variable definitions
    class EvalExpr(ast.NodeVisitor):
        def __init__(self, symbolTable):
            self.syms = symbolTable

        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.And):
                return left and right
            elif isinstance(node.op, ast.Or):
                return left or right
            else:
                raise NotImplementedError("TODO: unsupported BinOp", node.op)

        def visit_CmpOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Eq):
                return left == right
            elif isinstance(node.op, ast.NotEq):
                return left != right
            elif isinstance(node.op, ast.Lt):
                return left < right
            elif isinstance(node.op, ast.Gt):
                return left > right
            elif isinstance(node.op, ast.LtE):
                return left <= right
            elif isinstance(node.op, ast.GtE):
                return left >= right
            else:
                raise NotImplementedError("TODO: unsupported CmpOp", node.op)

        def visit_UnOp(self, node):
            e = self.visit(node.e)
            if isinstance(node.op, ast.USub):
                return -1 * e
            elif isinstance(node.op, ast.Not):
                return not e
            else:
                raise NotImplementedError("TODO: unsupported UnOp", node.op)

        def visit_Ref(self, node):
            val = self.syms[node.name]
            if node.index is None:
                return val
            else:
                return val[self.visit(node.index)]

        def visit_FloatConst(self, node):
            return node.val
        
        def visit_IntConst(self, node):
            return node.val

        def generic_visit(self, node):
            if isinstance(node, int):
                return node
            raise Exception("Visited unsupported node", node)

    
    evaluator = EvalExpr(syms)
    
    # TODO: you will probably need to track more than just a single current
    #       statement to deal with Blocks and nesting.
    """
    Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
     | FuncDef(Str name, Str* args, Stmt body)
    """
    stack = [ir.body]
    while stack:
        stmt = stack.pop()
        assert isinstance(stmt, ast.AST)
        if isinstance(stmt, Return):
            return evaluator.visit(stmt.val)
        elif isinstance(stmt, Assign):
            if stmt.ref.index is None:
                syms[stmt.ref.name] = evaluator.visit(stmt.val)
            else:
                syms[stmt.ref.name][stmt.ref.index] = evaluator.visit(stmt.val)
        elif isinstance(stmt, Block):
            for line in reversed(stmt.body):
                stack.append(line)
        elif isinstance(stmt, If):
            cond = evaluator.visit(stmt.cond)
            if cond:
                stack.append(stmt.body)
            else:
                stack.append(stmt.elseBody)
        elif isinstance(stmt, For):
            min_ = evaluator.visit(stmt.min)
            max_ = evaluator.visit(stmt.max)
            if not isinstance(min_, int):
                raise TypeError("object cannot be interpreted as an integer")
            if not isinstance(max_, int):
                raise TypeError("object cannot be interpreted as an integer")

            if min_ < max_:
                # Append the continuation.
                stmt.min = min_ + 1
                stack.append(stmt)
                # Append the body for this iteration.
                stack.append(stmt.body)
                syms[stmt.var] = min_
        elif isinstance(stmt, FuncDef):
            syms[stmt.name] = stmt
        else:
            raise NotImplementedError("TODO: add support for the full IR")

def Compile(f):
    """'Compile' the function f"""
    # Parse and extract the function definition AST
    fun = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
    print("Python AST:\n{}\n".format(astor.dump(fun)))
    
    simpleFun = PythonToSimple().visit(fun)
    
    print("Simple IR:\n{}\n".format(astor.dump(simpleFun)))
    
    # package up our generated simple IR in a 
    def run(*args):
        return Interpret(simpleFun, *args)
    
    return run


#############
## TEST IT ##
#############

# Define a trivial test program to start
def trivial(x) -> int:
    return x

def operations(arg) -> int:
    x = 0 < 1 < arg
    y = 1 > 3
    z = not (x and y and (x or y))

    a = 1
    b = 2.0
    c = a / b + a
    return c

def if_else(a, b) -> int:
    if a > b:
        a = a + b
    else:
        a = a - b
    return a

def for_loop(c) -> int:
    for i in range(1, 10):
        c = c + i

    return c

"""
Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
"""

def test_it():
    trivialInterpreted = Compile(trivial)
    operationsInterpreted = Compile(operations)
    if_elseInterpreted = Compile(if_else)
    for_loopInterpreted = Compile(for_loop)

    # run the original and our version, checking that their output matches:
    assert trivial(1) == trivialInterpreted(1)
    
    # TODO: add more of your own tests which exercise the functionality
    #       of your completed implementation
    assert operations(2) == operationsInterpreted(2)

    assert if_else(1, 2) == if_elseInterpreted(1, 2)
    assert for_loop(0) == for_loopInterpreted(0)
    
if __name__ == '__main__':
    test_it()
