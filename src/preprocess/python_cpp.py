# 导入必要的模块
import os
import pickle
import random
import json
import re  # 新增: 用于正则表达式处理
from tqdm import tqdm  # 用于显示进度条，方便观察数据处理的进度


def generate_pairs(source_data_dir, target_data_dir):
    """
    从JSONL文件生成跨语言代码对（Python和C++）并保存为.pkl文件。
    同时保存每段代码的语言信息到code_language_map.pkl文件。
    确保codes.pkl包含每个克隆对中的两个代码。

    :param source_data_dir: 包含JSONL文件的源目录，字符串类型
    :param target_data_dir: 目标数据存储目录，用于保存生成的 .pkl 文件，字符串类型
    """

    # 查找JSONL文件路径
    train_jsonl = os.path.join(source_data_dir, "pair_train.jsonl")  # 训练集 JSONL 文件路径
    valid_jsonl = os.path.join(source_data_dir, "pair_valid.jsonl")  # 验证集 JSONL 文件路径
    test_jsonl = os.path.join(source_data_dir, "pair_test.jsonl")  # 测试集 JSONL 文件路径

    # 检查文件是否存在，若不存在则抛出异常
    for file_path in [train_jsonl, valid_jsonl, test_jsonl]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

    def clean_python_code(code):
        """
        清理Python代码，针对tree-sitter优化。
        保留代码结构，移除装饰性内容。

        :param code: 输入的 Python 代码，字符串类型
        :return: 清理后的代码，字符串类型；若出错则返回 None
        """
        try:
            # 标准化行尾符
            code = code.replace('\r\n', '\n')

            # 移除Python注释
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

            # 移除三引号文档字符串
            code = re.sub(r'[\'\"]{3}[\s\S]*?[\'\"]{3}', '', code)

            # 移除导入语句（保留原有功能）
            lines = []
            for line in code.split('\n'):
                # 跳过以 'import' 或 'from' 开头的行
                if not (line.strip().startswith('import') or line.strip().startswith('from')):
                    lines.append(line)

            # 移除装饰器但保留函数定义
            code = '\n'.join(lines)
            code = re.sub(r'@\w+(\(.*?\))?\n', '\n', code)

            # 移除空行但保留代码结构
            lines = [line for line in code.split('\n') if line.strip() or line.startswith(' ')]

            return '\n'.join(lines)
        except:
            return None  # 异常情况下返回 None

    def clean_cpp_code(code):
        """
        清理C++代码，针对tree-sitter优化。
        保留代码结构，移除装饰性内容。

        :param code: 输入的 C++ 代码，字符串类型
        :return: 清理后的代码，字符串类型；若出错则返回 None
        """
        try:
            # 标准化行尾符
            code = code.replace('\r\n', '\n')

            # 移除C++单行注释
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)

            # 移除C++多行注释
            code = re.sub(r'/\*[\s\S]*?\*/', '', code)

            lines = []
            for line in code.split('\n'):
                # 保留条件编译指令，移除其他预处理指令（包括#include，保留原有功能）
                if line.strip().startswith('#'):
                    if any(x in line for x in ['#if', '#ifdef', '#ifndef', '#else', '#elif', '#endif']):
                        lines.append(line)
                else:
                    lines.append(line)

            # 移除空行但保留代码结构
            lines = [line for line in lines if line.strip() or line.startswith(' ')]

            return '\n'.join(lines)
        except:
            return None  # 异常情况下返回 None

    def read_jsonl(file_path):
        """
        读取 JSONL 文件并返回 JSON 对象列表。

        :param file_path: JSONL 文件路径，字符串类型
        :return: JSON 对象列表，每个元素为解析后的字典
        """
        data = []  # 存储解析后的 JSON 对象
        with open(file_path, 'r', encoding='utf-8') as f:  # f: 文件对象
            for line in f:  # line: 文件中的每一行
                try:
                    data.append(json.loads(line.strip()))  # 解析每行 JSON 并添加到列表
                except:
                    continue  # 跳过格式错误的行
        return data

    # 如果目标目录不存在，则创建
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)  # 创建目标目录

    # 读取JSONL文件
    print(f"正在读取JSONL文件...")
    train_data = read_jsonl(train_jsonl)  # 训练集数据，列表形式，每个元素为 JSON 对象
    valid_data = read_jsonl(valid_jsonl)  # 验证集数据，列表形式
    test_data = read_jsonl(test_jsonl)  # 测试集数据，列表形式

    # 用于存储代码和映射
    all_codes = []  # 所有独特代码的列表
    code_to_idx = {}  # 代码到索引的映射
    all_language_map = {}  # 代码到语言的映射
    code_tasks = []  # 存储每个代码对应的任务ID

    # 存储各数据集的样本对
    dataset_pairs = {
        "train": [],
        "valid": [],
        "test": []
    }

    # 按数据集存储代码索引和语言信息
    dataset_code_info = {
        "train": {"py": [], "cpp": [], "task_to_indices": {}},
        "valid": {"py": [], "cpp": [], "task_to_indices": {}},
        "test": {"py": [], "cpp": [], "task_to_indices": {}}
    }

    # 处理每个数据集，提取代码并构建正样本
    print("提取代码和构建正样本...")
    for dataset_name, dataset, ds_key in [
        ("训练集", train_data, "train"),
        ("验证集", valid_data, "valid"),
        ("测试集", test_data, "test")
    ]:
        print(f"处理{dataset_name}...")

        # 使用tqdm添加进度条
        for item in tqdm(dataset, desc=f"处理{dataset_name}数据"):
            # 提取任务ID
            task_id = item["Task"]

            # 提取语言信息
            lang1 = item["Category1"].lower()
            lang2 = item["Category2"].lower()

            # 只处理Python和C++代码对
            if (lang1 == "py" and lang2 == "cpp") or (lang1 == "cpp" and lang2 == "py"):
                code1 = None
                code2 = None

                # 清理代码
                if lang1 == "py":
                    code1 = clean_python_code(item["Code1"])
                    code2 = clean_cpp_code(item["Code2"])
                    lang1_name = "Python"
                    lang2_name = "C++"
                else:
                    code1 = clean_cpp_code(item["Code1"])
                    code2 = clean_python_code(item["Code2"])
                    lang1_name = "C++"
                    lang2_name = "Python"

                # 如果两段代码都有效，则添加为正样本
                if code1 and code2:
                    # 检查代码是否已存在，若不存在则添加
                    if code1 not in code_to_idx:
                        idx1 = len(all_codes)
                        code_to_idx[code1] = idx1
                        all_codes.append(code1)
                        all_language_map[code1] = lang1_name
                        code_tasks.append(task_id)

                        # 按语言类型添加到相应数据集的列表中
                        if lang1_name == "Python":
                            dataset_code_info[ds_key]["py"].append(idx1)
                        else:
                            dataset_code_info[ds_key]["cpp"].append(idx1)

                        # 记录任务ID到代码索引的映射
                        if task_id not in dataset_code_info[ds_key]["task_to_indices"]:
                            dataset_code_info[ds_key]["task_to_indices"][task_id] = []
                        dataset_code_info[ds_key]["task_to_indices"][task_id].append(idx1)
                    else:
                        idx1 = code_to_idx[code1]

                    if code2 not in code_to_idx:
                        idx2 = len(all_codes)
                        code_to_idx[code2] = idx2
                        all_codes.append(code2)
                        all_language_map[code2] = lang2_name
                        code_tasks.append(task_id)

                        # 按语言类型添加到相应数据集的列表中
                        if lang2_name == "Python":
                            dataset_code_info[ds_key]["py"].append(idx2)
                        else:
                            dataset_code_info[ds_key]["cpp"].append(idx2)

                        # 记录任务ID到代码索引的映射
                        if task_id not in dataset_code_info[ds_key]["task_to_indices"]:
                            dataset_code_info[ds_key]["task_to_indices"][task_id] = []
                        dataset_code_info[ds_key]["task_to_indices"][task_id].append(idx2)
                    else:
                        idx2 = code_to_idx[code2]

                    # 添加为正样本对
                    dataset_pairs[ds_key].append((idx1, idx2, 1))

    # 生成负样本
    print("生成负样本...")
    for ds_key in ["train", "valid", "test"]:
        positive_count = len(dataset_pairs[ds_key])
        if positive_count == 0:
            continue

        print(f"{ds_key}数据集有{positive_count}个正样本")

        # 构建正样本对集合，用于避免重复
        positive_pairs = set()
        for idx1, idx2, _ in dataset_pairs[ds_key]:
            positive_pairs.add((idx1, idx2))
            positive_pairs.add((idx2, idx1))  # 正样本对是双向的

        # 获取该数据集的Python和C++代码索引
        py_indices = dataset_code_info[ds_key]["py"]
        cpp_indices = dataset_code_info[ds_key]["cpp"]

        # 准备不同类型的负样本穷举
        neg_types = [
            ("py-cpp", py_indices, cpp_indices),  # Python和C++
            ("py-py", py_indices, py_indices),  # Python和Python
            ("cpp-cpp", cpp_indices, cpp_indices)  # C++和C++
        ]

        all_negative_pairs = []

        # 穷举所有可能的负样本对
        for neg_type, indices1, indices2 in neg_types:
            if not indices1 or not indices2:  # 如果某种语言的代码不存在，则跳过
                continue

            print(f"为{ds_key}穷举{neg_type}类型负样本...")

            # 双重循环穷举所有可能的组合
            for i in tqdm(range(len(indices1)), desc=f"穷举{neg_type}负样本"):
                idx1 = indices1[i]
                task1 = code_tasks[idx1]

                for j in range(len(indices2)):
                    if indices1 is indices2 and i == j:  # 对于相同类型的语言，避免自身匹配
                        continue

                    idx2 = indices2[j]
                    task2 = code_tasks[idx2]

                    # 确保来自不同任务且不是正样本
                    if task1 != task2 and (idx1, idx2) not in positive_pairs:
                        all_negative_pairs.append((idx1, idx2, 0))

        print(f"{ds_key}数据集可能的负样本总数: {len(all_negative_pairs)}")

        # 选择负样本，确保正负样本平衡
        if len(all_negative_pairs) > positive_count * 10:
            random.shuffle(all_negative_pairs)
            negative_pairs = all_negative_pairs[:positive_count * 10]
            print(f"{ds_key}数据集随机选择了{positive_count * 10}个负样本以保持平衡")
        else:
            negative_pairs = all_negative_pairs
            print(f"{ds_key}数据集使用全部{len(all_negative_pairs)}个可能的负样本")

        # 添加负样本并打乱
        dataset_pairs[ds_key].extend(negative_pairs)
        random.shuffle(dataset_pairs[ds_key])

        print(
            f"{ds_key}数据集: 正样本{positive_count}个, 负样本{len(negative_pairs)}个, 总计{len(dataset_pairs[ds_key])}个样本对")

    # 保存语言映射
    print("保存数据...")
    with open(os.path.join(target_data_dir, 'code_language_map.pkl'), 'wb') as lang_file:
        pickle.dump(all_language_map, lang_file)

    # 保存代码并添加进度条
    with open(os.path.join(target_data_dir, 'codes.pkl'), 'wb') as fc:
        for code in tqdm(all_codes, desc="保存代码到codes.pkl"):
            pickle.dump(code, fc)

    # 保存代码对
    for role, pairs in zip(['train', 'val', 'test'],
                           [dataset_pairs["train"], dataset_pairs["valid"], dataset_pairs["test"]]):
        with open(os.path.join(target_data_dir, f'{role}.pkl'), 'wb') as fp:
            for idx1, idx2, label in tqdm(pairs, desc=f"保存{role}数据集"):
                pair = ((idx1, idx2), label)
                pickle.dump(pair, fp)

    # 验证数量一致性
    print("验证保存的文件...")
    codes = []
    with open(os.path.join(target_data_dir, 'codes.pkl'), 'rb') as fc:
        while True:
            try:
                codes.append(pickle.load(fc))
            except EOFError:
                break

    print(f"codes.pkl 中保存了 {len(codes)} 个代码")
    print(f"code_language_map.pkl 中有 {len(all_language_map)} 个代码的语言映射")

    if len(codes) != len(all_language_map):
        print("警告：保存的代码数量与语言映射数量不一致！")
    else:
        print("验证通过！codes.pkl 和 code_language_map.pkl 数量一致")

    for ds_key in ["train", "valid", "test"]:
        pos_count = sum(1 for _, _, label in dataset_pairs[ds_key] if label == 1)
        neg_count = sum(1 for _, _, label in dataset_pairs[ds_key] if label == 0)
        print(
            f"{ds_key}数据集: 正样本 {pos_count}对, 负样本 {neg_count}对, 正负比例 {pos_count / (neg_count if neg_count else 1):.2f}")

    print(
        f"训练集: {len(dataset_pairs['train'])}对, 验证集: {len(dataset_pairs['valid'])}对, 测试集: {len(dataset_pairs['test'])}对")
