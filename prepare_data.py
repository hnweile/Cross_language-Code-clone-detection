# from preprocess import Pipeline
#
# if __name__ == '__main__':
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser()
#     parser.add_argument('--source_data_dir')
#     parser.add_argument('--target_data_dir')
#     parser.add_argument('--dataset_name')
#     args = parser.parse_args()
#
#     pipeline = Pipeline(args.source_data_dir, args.target_data_dir, args.dataset_name)
#     pipeline.run()


from preprocess import Pipeline
from preprocess import Pipeline
import warnings

# 忽略 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # 指定参数值
    source_data_dir = "D:/Develop/code_clone/DSFM/source_data"
    target_data_dir = "D:/Develop/code_clone/DSFM/datasets/python_cpp"
    # 可选
    dataset_name = "Python_Cpp"

    # 创建 Pipeline 对象
    pipeline = Pipeline(source_data_dir, target_data_dir, dataset_name)

    # 运行 Pipeline
    pipeline.run()

