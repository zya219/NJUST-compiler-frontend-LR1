import sys
import argparse
import re
from collections import defaultdict, deque, namedtuple

# Token 数据结构：存储行号、类型和值
Token = namedtuple('Token', ['line', 'type', 'value'])

# ------- 解析正规文法文件 --------
def parse_grammar(filename):
    """
    读取并解析 grammar.txt 文件，将每行的 A -> B | C 分割为字典格式
    返回值：grammar，dict: 非终结符 -> 产生式列表
    """
    grammar = defaultdict(list)
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()           # 去除首尾空白
                if not line or '->' not in line:
                    continue                 # 跳过空行或无效行
                head, prods = line.split('->', 1)
                head = head.strip()         # 非终结符名称
                for prod in prods.split('|'):
                    symbols = prod.strip().split()
                    grammar[head].append(symbols)
    except FileNotFoundError:
        print(f"Error: 文法文件 '{filename}' 未找到。", file=sys.stderr)
        sys.exit(1)
    return grammar

# ------- NFA 状态类 --------
class State:
    """
    NFA 状态节点，包含 transitions（字典）和是否为接受态标志
    transitions: dict symbol -> list of 下一个 State
    is_final: 是否为接受态
    token_type: 如果是接受态，对应的 token 类型
    """
    def __init__(self):
        self.transitions = defaultdict(list)
        self.is_final = False
        self.token_type = None

# ------- 构造子 NFA --------
def build_nfa(symbols, grammar, token_type, visited=None):
    """
    根据 symbols 递归构造 NFA。
    grammar: 文法字典，token_type: 当前 token 类型
    visited: 用于防止左递归的集合
    返回值：start_state, end_state
    """
    if visited is None:
        visited = set()

    # 如果没有符号，创建一个接受态
    if not symbols:
        s = State()
        s.is_final = True
        s.token_type = token_type
        return s, s

    sym = symbols[0]
    rest = symbols[1:]

    # 处理 ε（空串）
    if sym == 'ε':
        return build_nfa(rest, grammar, token_type, visited)

    # 非终结符，需要展开
    if sym in grammar:
        if sym in visited:
            # 已经访问过，避免无限递归
            return build_nfa(rest, grammar, token_type, visited)
        visited.add(sym)
        start = State()
        end = State()
        # 对每个产生式递归构造子 NFA
        for prod in grammar[sym]:
            sub_s, sub_e = build_nfa(prod, grammar, token_type, visited.copy())
            start.transitions['ε'].append(sub_s)
            sub_e.transitions['ε'].append(end)
        # 连接后续符号
        next_s, next_e = build_nfa(rest, grammar, token_type, visited)
        end.transitions['ε'].append(next_s)
        return start, next_e

    # 终结符：直接创建一个转移
    start = State()
    mid = State()
    start.transitions[sym].append(mid)
    next_s, next_e = build_nfa(rest, grammar, token_type, visited)
    mid.transitions['ε'].append(next_s)
    return start, next_e

# ------- 合并 NFA --------
def merge_nfas(nfa_map):
    """
    将多个类型的 NFA 起始态合并到一个超级起始态，使用 ε 转移
    nfa_map: token_type -> start_state
    返回值：super_start
    """
    super_start = State()
    for tk, st in nfa_map.items():
        super_start.transitions['ε'].append(st)
    return super_start

# ------- ε-闭包 --------
def epsilon_closure(states):
    """
    计算给定状态集合的 ε-闭包，返回包含所有可通过 ε 到达的状态集合
    """
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for nxt in s.transitions.get('ε', []):
            if nxt not in closure:
                closure.add(nxt)
                stack.append(nxt)
    return closure

# ------- move 集合 --------
def move(states, symbol):
    """
    对状态集合执行符号转移，返回新的状态集合
    """
    result = set()
    for s in states:
        for nxt in s.transitions.get(symbol, []):
            result.add(nxt)
    return result

# ------- NFA → DFA --------
def nfa_to_dfa(start):
    """
    子集构造法，将 NFA 转换为 DFA
    返回：dfa_states（dict: state_set-> transitions）、起始 DFA 状态、dfa_final 接受态映射
    """
    start_cl = frozenset(epsilon_closure([start]))
    dfa_states = {start_cl: {}}
    dfa_final = {}
    queue = deque([start_cl])
    while queue:
        curr = queue.popleft()
        # 收集所有可能的符号
        syms = {sym for st in curr for sym in st.transitions if sym != 'ε'}
        for sym in syms:
            nxt_set = frozenset(epsilon_closure(move(curr, sym)))
            if nxt_set not in dfa_states:
                dfa_states[nxt_set] = {}
                queue.append(nxt_set)
            dfa_states[curr][sym] = nxt_set
    # 标记接受态及其类型
    for st_set in dfa_states:
        for s in st_set:
            if s.is_final:
                dfa_final[st_set] = s.token_type
                break
    return dfa_states, start_cl, dfa_final

# ------- 使用 DFA 识别并最长匹配 --------
def match_with_dfa(dfa, start, finals, text):
    """
    在文本前缀尝试最长匹配，返回 (匹配串, 长度, token_type)
    """
    pos = 0
    curr = start
    last_final = None
    last_pos = 0
    while pos < len(text) and text[pos] in dfa[curr]:
        curr = dfa[curr][text[pos]]
        pos += 1
        if curr in finals:
            last_final = curr
            last_pos = pos
    if last_final:
        return text[:last_pos], last_pos, finals[last_final]
    return None, 0, None

# ------- 正则表达式，用于匹配完整复数常量与普通常量 --------
COMPLEX_REGEX = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[+-]\d+(?:\.\d+)?(?:[eE][+-]?\d+)?i")
CONST_REGEX   = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?i?")

STRING_REGEX = re.compile(r'^"(?:\\.|[^"\\])*"')
# ------- 主函数 --------
def main():
    # 解析命令行参数：grammar, source, output
    parser = argparse.ArgumentParser(description="基于正规文法的词法分析器")
    parser.add_argument('grammar', help='正规文法文件路径')
    parser.add_argument('source', help='源代码文件路径')
    parser.add_argument('-o', '--output', default='token.txt', help='输出 token 文件名')
    args = parser.parse_args()

    # 读取文法并构建 NFA-DFA
    grammar = parse_grammar(args.grammar)
    token_types = ['keyword', 'identifier', 'operator', 'limiter']
    nfa_map = {}
    for tt in token_types:
        if tt not in grammar:
            print(f"Error: 文法中未定义 token '{tt}'", file=sys.stderr)
            sys.exit(1)
        start_state = State()
        for prod in grammar[tt]:
            sub_s, _ = build_nfa(prod, grammar, tt)
            start_state.transitions['ε'].append(sub_s)
        nfa_map[tt] = start_state
    # 合并 NFA，生成 DFA
    merged_nfa = merge_nfas(nfa_map)
    dfa, dfa_start, dfa_finals = nfa_to_dfa(merged_nfa)

    # 开始词法分析
    tokens = []
    try:
        with open(args.source, 'r', encoding='utf-8') as sf:
            for ln, line in enumerate(sf, 1):
                pos = 0
                while pos < len(line):
                    # 跳过空白
                    if line[pos].isspace():
                        pos += 1
                        continue
                    # 优先匹配复数常量
                    m_c = COMPLEX_REGEX.match(line[pos:])
                    if m_c:
                        val = m_c.group(0)
                        tokens.append(Token(ln, 'const', val))
                        pos += len(val)
                        continue
                    # 再匹配普通常量
                    m_n = CONST_REGEX.match(line[pos:])
                    if m_n:
                        val = m_n.group(0)
                        tokens.append(Token(ln, 'const', val))
                        pos += len(val)
                        continue
                    # 优先正则匹配字符串常量
                    m_str = STRING_REGEX.match(line[pos:])
                    if m_str:
                        val = m_str.group(0)
                        tokens.append(Token(ln, 'const', val))
                        pos += len(val)
                        continue
                    # 其他 token 用 DFA
                    tok, length, tt = match_with_dfa(dfa, dfa_start, dfa_finals, line[pos:])
                    if tok:
                        tokens.append(Token(ln, tt, tok))
                        pos += length
                    else:
                        # 无法识别的字符
                        tokens.append(Token(ln, 'unrecognized', line[pos]))
                        pos += 1
    except FileNotFoundError:
        print(f"Error: 源文件 '{args.source}' 未找到。", file=sys.stderr)
        sys.exit(1)

    # 输出到控制台
    print(f"{'Line':<6}{'Type':<12}{'Value'}")
    print('-' * 40)
    for t in tokens:
        print(f"{t.line:<6}{t.type:<12}{t.value}")

    # 写入 token 文件
    try:
        with open(args.output, 'w', encoding='utf-8') as outf:
            for t in tokens:
                outf.write(f"{t.line}\t{t.type}\t{t.value}\n")
        print(f"Tokens 已写入 '{args.output}'")
    except IOError as e:
        print(f"Error: 无法写入 '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
