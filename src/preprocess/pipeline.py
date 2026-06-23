import pickle
from pathlib import Path
from queue import Queue
from tqdm import tqdm
from tree_sitter import Language, Parser
from . import java_python, java_cpp, python_cpp, python_cs, cpp_cs, java_cs


class Pipeline:
    def __init__(self, source_data_dir, target_data_dir, dataset_name):
        self.source_data_dir = Path(source_data_dir)
        self.target_data_dir = Path(target_data_dir)
        self.dataset_name = dataset_name

    def run(self):
        # 步骤 1：生成跨语言代码对
        print('1. Generating pairs...')
        self.generate_pairs(self.source_data_dir, self.target_data_dir)

        # 第二步：加载代码文件
        print('2. Loading codes...')
        codes = self.load_codes()

        # 第三步：将代码解析为抽象语法树（AST）
        print('3. Parsing codes into ASTs...')
        asts = self.parse_and_unify_codes(codes)

        # 第四步：从AST中提取子树
        print('4. Extracting subtrees...')
        subtrees = self.extract_subtrees(asts)

        # 第五步：保存提取的子树到目标目录
        print('5. Saving subtrees...')
        self.save_subtrees(subtrees)

    def generate_pairs(self, source_data_dir, target_data_dir):
        if self.dataset_name == "Java_Python":
            java_python.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == "Java_Cpp":
            java_cpp.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == "Python_Cpp":
            python_cpp.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == "Python_CSharp":
            python_cs.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == "Cpp_CSharp":
            cpp_cs.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == "Java_CSharp":
            java_cs.generate_pairs(source_data_dir, target_data_dir)
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

    def load_codes(self):
        code_file = self.target_data_dir / 'codes.pkl'
        lang_map_file = self.target_data_dir / 'code_language_map.pkl'

        codes = []
        with open(code_file, 'rb') as f:
            while True:
                try:
                    codes.append(pickle.load(f))
                except EOFError:
                    break

        with open(lang_map_file, 'rb') as f:
            code_language_map = pickle.load(f)

        if len(codes) != len(code_language_map):
            raise ValueError("codes.pkl 和 code_language_map.pkl 的数量不一致")

        codes_with_lang = [(code, code_language_map[code]) for code in codes]
        return codes_with_lang

    def parse_and_unify_codes(self, codes_with_lang):
        java_parser = Parser()
        python_parser = Parser()
        cpp_parser = Parser()
        csharp_parser = Parser()

        JAVA_LANGUAGE = Language('./build/my-languages.so', 'java')
        PYTHON_LANGUAGE = Language('./build/my-languages.so', 'python')
        CPP_LANGUAGE = Language('./build/my-languages.so', 'cpp')
        CSHARP_LANGUAGE = Language('./build/my-languages.so', 'c_sharp')

        java_parser.set_language(JAVA_LANGUAGE)
        python_parser.set_language(PYTHON_LANGUAGE)
        cpp_parser.set_language(CPP_LANGUAGE)
        csharp_parser.set_language(CSHARP_LANGUAGE)

        asts = []

        with tqdm(total=len(codes_with_lang), desc="Parsing and Unifying ASTs", dynamic_ncols=True, leave=True) as pbar:
            for code, lang in codes_with_lang:
                try:
                    if lang == 'Java':
                        tree = java_parser.parse(bytes(code, 'utf8'))
                        root_node = tree.root_node
                        ast = self._convert_and_optimize_tree_to_ast(root_node, 'Java')
                        asts.append(ast)
                    elif lang == 'Python':
                        tree = python_parser.parse(bytes(code, 'utf8'))
                        root_node = tree.root_node
                        ast = self._convert_and_optimize_tree_to_ast(root_node, 'Python')
                        asts.append(ast)
                    elif lang == 'C++':
                        tree = cpp_parser.parse(bytes(code, 'utf8'))
                        root_node = tree.root_node
                        ast = self._convert_and_optimize_tree_to_ast(root_node, 'C++')
                        asts.append(ast)
                    elif lang == 'C#':
                        tree = csharp_parser.parse(bytes(code, 'utf8'))
                        root_node = tree.root_node
                        ast = self._convert_and_optimize_tree_to_ast(root_node, 'C#')
                        asts.append(ast)
                    else:
                        print(f"Unsupported language: {lang}")
                        asts.append(None)
                except Exception as e:
                    print(f"Error parsing {lang} code: {e}")
                    asts.append(None)
                finally:
                    pbar.update(1)

        return asts

    def _convert_and_optimize_tree_to_ast(self, root_node, lang):
        from networkx import DiGraph

        def traverse(node, graph, parent_id=None):
            node_type = self._map_node_type(node.type, lang)

            if node_type in {'comment', 'whitespace', 'unknown'}:
                return

            node_id = len(graph)
            graph.add_node(node_id, type=node_type)

            if parent_id is not None:
                graph.add_edge(parent_id, node_id)

            for child in node.children:
                traverse(child, graph, node_id)

        ast_graph = DiGraph()
        traverse(root_node, ast_graph)
        self._ensure_root_connectivity(ast_graph)
        return ast_graph

    def _ensure_root_connectivity(self, graph):
        if 0 not in graph:
            graph.add_node(0, type='root')
            for node_id in list(graph.nodes):
                if node_id != 0 and graph.in_degree(node_id) == 0:
                    graph.add_edge(0, node_id)

    def _map_node_type(self, node_type, lang):
        if lang == 'Java':
            mapping = {
                # 核心控制结构 - 这些是unified_delimiter中的关键类型
                'method_declaration': 'function',  # 方法声明 -> 函数
                'constructor_declaration': 'function',  # 构造函数也视为function
                'for_statement': 'for_loop',  # for循环
                'enhanced_for_statement': 'for_loop',  # foreach也映射为for_loop
                'while_statement': 'while_loop',  # while循环
                'do_statement': 'do_while_loop',  # do-while循环
                'if_statement': 'condition',  # if语句
                'else_statement': 'condition',  # else也应视为condition
                'switch_statement': 'switch',  # switch语句
                'return_statement': 'return',  # return语句
                'try_statement': 'try_block',  # try语句块
                'catch_clause': 'catch_block',  # catch语句块
                'throw_statement': 'exception',  # throw语句视为exception

                # 辅助结构 - 帮助构建完整的AST
                'class_declaration': 'class',  # 类声明
                'interface_declaration': 'interface',  # 接口声明
                'variable_declarator': 'variable',  # 变量声明
                'field_declaration': 'field',  # 字段声明
                'method_invocation': 'function_call',  # 方法调用
                'array_creation_expression': 'array_creation',  # 数组创建
                'array_access': 'array_access',  # 数组访问
                'binary_expression': 'binary',  # 二元表达式
                'assignment_expression': 'assignment',  # 赋值表达式
                'break_statement': 'break',  # break语句
                'continue_statement': 'continue',  # continue语句
                'synchronized_statement': 'synchronized',  # synchronized语句
                'block': 'block',  # 代码块
                'expression_statement': 'expression',  # 表达式语句
                'finally_clause': 'finally_block',  # finally块
            }

        elif lang == 'Python':
            mapping = {
                # 核心控制结构
                'function_definition': 'function',  # 函数定义
                'method_definition': 'function',  # 方法定义也视为function
                'for_statement': 'for_loop',  # for循环
                'while_statement': 'while_loop',  # while循环
                'if_statement': 'condition',  # if语句
                'elif_clause': 'condition',  # elif从句
                'else_clause': 'condition',  # else从句
                'return_statement': 'return',  # return语句
                'try_statement': 'try_block',  # try语句
                'except_clause': 'catch_block',  # except从句(Python中的catch)
                'raise_statement': 'exception',  # raise语句(Python中的throw)

                # 辅助结构
                'class_definition': 'class',  # 类定义
                'call': 'function_call',  # 函数调用
                'assignment': 'assignment',  # 赋值
                'binary_operator': 'binary',  # 二元运算符
                'break_statement': 'break',  # break语句
                'continue_statement': 'continue',  # continue语句
                'with_statement': 'with_block',  # with语句
                'finally_clause': 'finally_block',  # finally块
                'block': 'block',  # 代码块
                'expression_statement': 'expression',  # 表达式语句
                'subscript': 'array_access',  # 下标访问(对应数组访问)
            }

        elif lang == 'C++':
            mapping = {
                # 核心控制结构
                'function_definition': 'function',  # 函数定义
                'for_statement': 'for_loop',  # for循环
                'for_range_declaration': 'for_loop',  # 范围for循环(C++11)
                'while_statement': 'while_loop',  # while循环
                'do_statement': 'do_while_loop',  # do-while循环
                'if_statement': 'condition',  # if语句
                'else_clause': 'condition',  # else从句
                'switch_statement': 'switch',  # switch语句
                'return_statement': 'return',  # return语句
                'try_statement': 'try_block',  # try语句
                'catch_clause': 'catch_block',  # catch从句
                'throw_statement': 'exception',  # throw语句

                # 辅助结构
                'class_specifier': 'class',  # 类说明符
                'struct_specifier': 'struct',  # 结构体
                'call_expression': 'function_call',  # 函数调用
                'declaration': 'declaration',  # 声明
                'variable_declarator': 'variable',  # 变量声明
                'binary_expression': 'binary',  # 二元表达式
                'assignment_expression': 'assignment',  # 赋值表达式
                'break_statement': 'break',  # break语句
                'continue_statement': 'continue',  # continue语句
                'compound_statement': 'block',  # 复合语句(代码块)
                'expression_statement': 'expression',  # 表达式语句
                'subscript_expression': 'array_access',  # 下标表达式(数组访问)
            }

        elif lang == 'C#':
            mapping = {
                # 核心控制结构
                'method_declaration': 'function',  # 方法声明
                'constructor_declaration': 'function',  # 构造函数也视为function
                'for_statement': 'for_loop',  # for循环
                'foreach_statement': 'for_loop',  # foreach循环也映射为for_loop
                'while_statement': 'while_loop',  # while循环
                'do_statement': 'do_while_loop',  # do-while循环
                'if_statement': 'condition',  # if语句
                'else_clause': 'condition',  # else从句
                'switch_statement': 'switch',  # switch语句
                'return_statement': 'return',  # return语句
                'try_statement': 'try_block',  # try语句
                'catch_clause': 'catch_block',  # catch从句
                'throw_statement': 'exception',  # throw语句

                # 辅助结构
                'class_declaration': 'class',  # 类声明
                'interface_declaration': 'interface',  # 接口声明
                'invocation_expression': 'function_call',  # 方法调用
                'variable_declaration': 'variable_declaration',  # 变量声明
                'variable_declarator': 'variable',  # 变量声明符
                'binary_expression': 'binary',  # 二元表达式
                'assignment_expression': 'assignment',  # 赋值表达式
                'break_statement': 'break',  # break语句
                'continue_statement': 'continue',  # continue语句
                'block': 'block',  # 代码块
                'expression_statement': 'expression',  # 表达式语句
                'element_access_expression': 'array_access',  # 元素访问(数组访问)
                'finally_clause': 'finally_block',  # finally块
            }
        else:
            mapping = {}

        return mapping.get(node_type, node_type)

    def extract_subtrees(self, asts):
        # 根据数据集名称选择最合适的 unified_delimiter
        if self.dataset_name == 'Java_Python':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'condition', 'return', 'try_block',
                                 'catch_block', 'exception']
        elif self.dataset_name == 'Python_Cpp':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'condition', 'return', 'try_block',
                                 'catch_block', 'exception']
        elif self.dataset_name == 'Python_CSharp':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'condition', 'return', 'try_block',
                                 'catch_block', 'exception']
        elif self.dataset_name == 'Java_Cpp':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'do_while_loop', 'condition', 'switch', 'return',
                                 'try_block', 'catch_block', 'exception']
        elif self.dataset_name == 'Cpp_CSharp':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'do_while_loop', 'condition', 'switch', 'return',
                                 'try_block', 'catch_block', 'exception']
        elif self.dataset_name == 'Java_CSharp':
            unified_delimiter = ['function', 'for_loop', 'while_loop', 'do_while_loop', 'condition', 'switch', 'return',
                                 'try_block', 'catch_block', 'exception']
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

        def node_type(node_id, graph):
            return graph.nodes[node_id]['type']

        def dfs(root, graph, queue, flag):
            subtree = [node_type(root, graph)]
            if node_type(root, graph) in unified_delimiter and flag:
                queue.put(root)
            else:
                for child in graph.successors(root):
                    subtree.append(dfs(child, graph, queue, True))
            return subtree

        def extract_subtrees_from_ast(root, graph, queue):
            subtrees = []
            if node_type(root, graph) in unified_delimiter:
                subtree = dfs(root, graph, queue, False)
                subtrees.append(subtree)

            if queue.empty():
                for child in graph.successors(root):
                    subtrees.extend(extract_subtrees_from_ast(child, graph, queue))
            else:
                subtrees.extend(extract_subtrees_from_ast(queue.get(), graph, queue))

            return subtrees

        all_subtrees = []
        for ast in tqdm(asts, desc="Extracting Subtrees"):
            if ast is None:
                all_subtrees.append([])  # 如果没有解析到 AST，返回空子树
                continue

            subtrees = extract_subtrees_from_ast(0, ast, Queue())
            if not subtrees:
                subtrees = [dfs(0, ast, Queue(), False)]

            all_subtrees.append(subtrees)

        return all_subtrees

    def save_subtrees(self, subtrees):
        subtree_file = self.target_data_dir / 'subtrees.pkl'

        with open(subtree_file, 'wb') as f:
            for subtree in subtrees:
                pickle.dump(subtree, f)

        print(f"子树已保存到文件：{subtree_file}")
