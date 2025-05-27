import sys
import os
from collections import defaultdict

def parse_grammar(grammar_file):
    """
    解析文法文件，将显式的 ε 产生式转换为空列表
    返回：productions, 非终结符列表, 终结符列表, 起始符号
    """
    productions = []
    nonterminals = set()
    terminals = set()
    start_symbol = None
    with open(grammar_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text or '->' not in text:
                continue
            head, body = text.split('->')
            head = head.strip()
            if start_symbol is None:
                start_symbol = head
            nonterminals.add(head)
            for alt in body.split('|'):
                symbols = [sym.strip().strip('"') for sym in alt.split()]
                if symbols == ['ε']:
                    symbols = []  # 把 ε 当作空产生式
                productions.append((head, symbols))
    # 收集终结符
    for head, body in productions:
        for sym in body:
            if sym not in nonterminals:
                terminals.add(sym)
    terminals.add('$')
    return productions, list(nonterminals), list(terminals), start_symbol


def parse_tokens(token_file):
    """
    解析 token 文件，返回 [(symbol, line_no), ...]，末尾加上 ('$', last_line)
    """
    tokens = []
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            ln = int(parts[0])
            tok_type, tok_val = parts[1], parts[2]
            if tok_type in ('keyword', 'operator', 'limiter'):
                sym = tok_val
            elif tok_type == 'identifier':
                sym = 'identifier'
            elif tok_type == 'const':
                sym = 'const'
            else:
                sym = tok_type
            tokens.append((sym, ln))
    # 添加结束符
    last_line = tokens[-1][1] if tokens else 0
    tokens.append(('$', last_line))
    return tokens


def compute_first(productions, nonterminals, terminals):
    """
    计算 FIRST 集合，支持空产生式（body=[]）
    返回：{非终结符: set([...])}
    """
    first = {X: set() for X in nonterminals}
    changed = True
    while changed:
        changed = False
        for head, body in productions:
            before = len(first[head])
            if not body:
                first[head].add('ε')
            else:
                for sym in body:
                    if sym in terminals:
                        first[head].add(sym)
                        break
                    first[head] |= (first.get(sym, {sym}) - {'ε'})
                    if 'ε' not in first.get(sym, set()):
                        break
                else:
                    first[head].add('ε')
            if len(first[head]) > before:
                changed = True
    return first


def compute_follow(productions, nonterminals, start_symbol, first):
    """
    计算 FOLLOW 集合，起始符号的 FOLLOW 包含 '$'
    返回：{非终结符: set([...])}
    """
    follow = {X: set() for X in nonterminals}
    follow[start_symbol].add('$')
    changed = True
    while changed:
        changed = False
        for head, body in productions:
            trailer = follow[head].copy()
            for sym in reversed(body):
                if sym in nonterminals:
                    before = len(follow[sym])
                    follow[sym] |= trailer
                    if 'ε' in first[sym]:
                        trailer |= (first[sym] - {'ε'})
                    else:
                        trailer = first[sym].copy()
                    if len(follow[sym]) > before:
                        changed = True
                else:
                    trailer = {sym}
    return follow


class Item:
    """
    项目 (A -> α·β, a)，用于 LR(1) 分析
    """
    def __init__(self, head, body, dot, lookahead):
        self.head = head
        self.body = body
        self.dot = dot
        self.lookahead = lookahead
    def __eq__(self, other):
        return (self.head, tuple(self.body), self.dot, self.lookahead) == \
               (other.head, tuple(other.body), other.dot, other.lookahead)
    def __hash__(self):
        return hash((self.head, tuple(self.body), self.dot, self.lookahead))


def compute_first_seq(seq, first):
    """
    计算序列 seq 的 FIRST 集合，用于 closure 中的 lookahead
    """
    res = set()
    for sym in seq:
        if sym in first:
            res |= (first[sym] - {'ε'})
            if 'ε' not in first[sym]:
                break
        else:
            res.add(sym)
            break
    else:
        res.add('ε')
    return res


def closure(items, productions, nonterminals, first):
    """
    计算 LR(1) 项目集闭包
    """
    C = set(items)
    changed = True
    while changed:
        changed = False
        for it in list(C):
            if it.dot < len(it.body):
                B = it.body[it.dot]
                if B in nonterminals:
                    beta = it.body[it.dot+1:]
                    for la in compute_first_seq(beta + [it.lookahead], first):
                        for ph, pb in productions:
                            if ph == B:
                                new_item = Item(B, pb, 0, la)
                                if new_item not in C:
                                    C.add(new_item)
                                    changed = True
    return C


def goto(items, X, productions, nonterminals, first):
    """
    项目集 I 在符号 X 下的 GOTO，返回新的项目集
    """
    moved = [Item(it.head, it.body, it.dot+1, it.lookahead)
             for it in items
             if it.dot < len(it.body) and it.body[it.dot] == X]
    return closure(moved, productions, nonterminals, first)


def build_table(productions, nonterminals, terminals, start_symbol, first, follow):
    """
    构造 LR(1) 的 ACTION 和 GOTO 表
    返回：action, goto_tbl
    """
    # 增广文法
    aug_sym = start_symbol + "'"
    productions.insert(0, (aug_sym, [start_symbol]))
    nonterminals.append(aug_sym)
    C = [closure([Item(aug_sym, [start_symbol], 0, '$')], productions, nonterminals, first)]
    # 构造规范族
    changed = True
    while changed:
        changed = False
        for I in list(C):
            for X in nonterminals + terminals:
                J = goto(I, X, productions, nonterminals, first)
                if J and J not in C:
                    C.append(J)
                    changed = True
    # 初始化表
    action = defaultdict(dict)
    goto_tbl = defaultdict(dict)
    for i, I in enumerate(C):
        for it in I:
            # shift
            if it.dot < len(it.body) and it.body[it.dot] in terminals:
                a = it.body[it.dot]
                j = C.index(goto(I, a, productions, nonterminals, first))
                action[i][a] = ('s', j)
            # reduce or accept
            elif it.dot == len(it.body):
                if it.head == aug_sym:
                    action[i]['$'] = ('acc', '')
                else:
                    prod_idx = productions.index((it.head, it.body))
                    for a in follow[it.head]:
                        action[i][a] = ('r', prod_idx)
        # GOTO 部分
        for A in nonterminals:
            J = goto(I, A, productions, nonterminals, first)
            if J:
                goto_tbl[i][A] = C.index(J)
    return action, goto_tbl


def parse(tokens, action, goto_tbl, productions):
    """
    用 ACTION/GOTO 表对 token 列表进行 LR(1) 语法分析
    返回：(True/False, 错误信息)
    """
    stack = [0]
    idx = 0
    while True:
        state = stack[-1]
        symbol, ln = tokens[idx]
        entry = action.get(state, {}).get(symbol)
        if not entry:
            expected = list(action.get(state, {}).keys())
            return False, f"第 {ln} 行遇到意外符号 '{symbol}'，期望: {expected}"
        typ, target = entry
        if typ == 's':
            stack.extend([symbol, target])
            idx += 1
        elif typ == 'r':
            hd, bd = productions[target]
            if bd:
                stack = stack[:-2*len(bd)]
            prev_state = stack[-1]
            stack.extend([hd, goto_tbl[prev_state][hd]])
        else:  # 'acc'
            return True, ''


def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python task2.py <grammar.txt> [token_file]")
        sys.exit(1)
    grammar_file = sys.argv[1]
    token_file = sys.argv[2] if len(sys.argv) > 2 else 'token'

    # 如果未指定 token_file，但存在默认文件 'token'，则使用它
    if not os.path.isfile(token_file) and os.path.isfile('token'):
        token_file = 'token'

    # 解析文法和 token
    prods, nonterms, terms, start_sym = parse_grammar(grammar_file)
    toks = parse_tokens(token_file)

    # 计算 FIRST 和 FOLLOW
    first = compute_first(prods, nonterms, terms)
    follow = compute_follow(prods, nonterms, start_sym, first)

    # 构建 ACTION/GOTO 表
    action_tbl, goto_tbl = build_table(prods, nonterms, terms, start_sym, first, follow)

    # 打印 ACTION 表
    print("\nACTION 表：")
    for state in sorted(action_tbl.keys()):
        print(f"状态 {state}: ", end='')
        for sym, act in action_tbl[state].items():
            print(f"{sym}:{act} ", end='')
        print()

    # # 打印 GOTO 表
    print("\nGOTO 表：")
    for state in sorted(goto_tbl.keys()):
        print(f"状态 {state}: ", end='')
        for sym, tgt in goto_tbl[state].items():
            print(f"{sym}:{tgt} ", end='')
        print()

    # 执行语法分析
    ok, err = parse(toks, action_tbl, goto_tbl, prods)
    print("YES" if ok else "NO")
    if not ok:
        print(err)


if __name__ == '__main__':
    main()
