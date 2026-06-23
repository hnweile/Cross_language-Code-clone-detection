import os
import pickle
import random
from tqdm import tqdm  # 引入进度条库


def generate_pairs(source_data_dir, target_data_dir):
    """
    生成跨语言代码对并保存为 .pkl 文件。
    同时保存每段代码的语言信息到 code_language_map.pkl 文件。
    :param source_data_dir: 源数据目录，包含多个问题的解决方案（Python 和 C#）。
    :param target_data_dir: 目标数据存储目录。
    """

    def read_code_python(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = []
                for line in f.readlines():
                    if any([line.strip().startswith('import'), line.strip().startswith('from')]):
                        continue
                    text.append(line)
                text = ''.join(text)
            return text
        except:
            return None

    def read_code_csharp(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = []
                for line in f.readlines():
                    # 删除掉C#的头文件引用等无关代码
                    if line.strip().startswith('using'):
                        continue
                    text.append(line)
                text = ''.join(text)
            return text
        except:
            return None

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    total_code_categories, unique_codes = [], set()
    code_language_map = {}

    # 使用进度条对整个目录进行遍历
    for problem_dir in tqdm(os.listdir(source_data_dir), desc="Processing problem directories"):
        problem_path = os.path.join(source_data_dir, problem_dir)
        if not os.path.isdir(problem_path):
            continue

        python_dir = os.path.join(problem_path, "Python")
        csharp_dir = os.path.join(problem_path, "C#")
        if not (os.path.exists(python_dir) and os.path.exists(csharp_dir)):
            continue

        # 处理 Python 文件夹
        if os.path.exists(python_dir):
            for python_file in os.listdir(python_dir):
                code = read_code_python(os.path.join(python_dir, python_file))
                if code and code not in unique_codes:
                    total_code_categories.append((code, problem_dir))
                    unique_codes.add(code)
                    code_language_map[code] = 'Python'

        # 处理 C# 文件夹
        if os.path.exists(csharp_dir):
            for csharp_file in os.listdir(csharp_dir):
                code = read_code_csharp(os.path.join(csharp_dir, csharp_file))
                if code and code not in unique_codes:
                    total_code_categories.append((code, problem_dir))
                    unique_codes.add(code)
                    code_language_map[code] = 'C#'

    # 确保 codes.pkl 和 code_language_map.pkl 的数量一致
    # 即保证每个代码片段的语言信息和代码保持一致，并且没有重复的片段
    assert len(total_code_categories) == len(code_language_map), "Mismatch between code and language map lengths."

    random.seed(1)
    random.shuffle(total_code_categories)

    # 生成正负样本对，确保正负样本比例为1:1
    positive_pairs = []
    negative_pairs = []
    for i in tqdm(range(len(total_code_categories)), desc="Generating code pairs"):
        for j in range(i + 1, len(total_code_categories)):
            code_i = total_code_categories[i][0]
            code_j = total_code_categories[j][0]
            problem_i = total_code_categories[i][1]
            problem_j = total_code_categories[j][1]

            # 获取代码的语言
            lang_i = code_language_map[code_i]
            lang_j = code_language_map[code_j]

            if problem_i == problem_j and lang_i != lang_j:
                positive_pairs.append((i, j, 1))  # 相同问题编号且不同语言为正样本
            elif problem_i != problem_j:
                negative_pairs.append((i, j, 0))  # 不同问题编号为负样本
            # 相同问题相同语言的情况不列入样本

    # 如果负样本大于正样本，进行负样本采样，确保正负样本数量一致
    if len(negative_pairs) > len(positive_pairs):
        negative_pairs = random.sample(negative_pairs, len(positive_pairs))

    # 合并正负样本
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    split_point_1 = int(len(all_pairs) * 0.8)  # 80% 用于训练
    split_point_2 = int(len(all_pairs) * 0.9)  # 10% 用于验证
    train_pairs = all_pairs[:split_point_1]
    val_pairs = all_pairs[split_point_1:split_point_2]
    test_pairs = all_pairs[split_point_2:]

    code2idx = {}

    print(f"codes.pkl 的数量为{len(total_code_categories)} 和 code_language_map.pkl 的数量为{len(code_language_map)}")

    with open(os.path.join(target_data_dir, 'code_language_map.pkl'), 'wb') as lang_file:
        pickle.dump(code_language_map, lang_file)

    with open(os.path.join(target_data_dir, 'codes.pkl'), 'wb') as fc:
        # 对训练、验证、测试集分别保存数据
        for role, pairs in zip(['train', 'val', 'test'], [train_pairs, val_pairs, test_pairs]):
            with open(os.path.join(target_data_dir, f'{role}.pkl'), 'wb') as fp:
                # 使用 tqdm 为保存数据对部分添加进度条
                for idx1, idx2, label in tqdm(pairs, desc=f"Saving {role} data", total=len(pairs)):
                    for idx in [idx1, idx2]:
                        if total_code_categories[idx][0] not in code2idx:
                            code2idx[total_code_categories[idx][0]] = len(code2idx)
                            pickle.dump(total_code_categories[idx][0], fc)

                    pair = (
                        (code2idx[total_code_categories[idx1][0]], code2idx[total_code_categories[idx2][0]]),
                        label
                    )
                    pickle.dump(pair, fp)
