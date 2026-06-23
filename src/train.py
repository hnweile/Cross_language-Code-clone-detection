import os
import time

import torch
from torch import nn
from tqdm import tqdm

from model.data import Data
from model.metric import Metric
from model.model import Model

import warnings

warnings.filterwarnings("ignore")


# 添加自定义混合分类-相似性损失
class EnhancedContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.temp = temp

    def forward(self, outputs, labels):
        # 标准BCE损失
        bce_loss = self.bce(outputs, labels)

        # 对比学习损失
        similarity = torch.sigmoid(outputs / self.temp)
        pos_loss = -(labels * torch.log(similarity + 1e-7))
        neg_loss = -((1 - labels) * torch.log(1 - similarity + 1e-7))

        contrast_loss = pos_loss + neg_loss

        # 组合损失
        return bce_loss + 0.4 * contrast_loss.mean()


# 自定义学习率调度器回调函数，用于输出当前学习率
class LRLogger:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr

    def update_lr(self):
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return self._last_lr


# 设置默认数据目录和保存目录
# data_dir = "../datasets/java_cs_1"  # Java和C#
# data_dir = "../datasets/cpp_cs_1"  # C++和C#
# data_dir = "../datasets/java_python"  # Java和python, 数据集路径
# data_dir = "../datasets/python_cs_3"  # python和C#, 数据集路径
# data_dir = "../datasets/Java_cpp_1"  # Java和C++, 数据集路径
data_dir = "../datasets/python_cpp"  # python和C++, 数据集路径
save_dir = "../results"  # 结果保存路径

# 如果保存目录不存在，则创建该目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 确定使用的设备是 CUDA 还是 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')  # 打印当前使用的设备

# 初始化数据处理，batch_size 是每次训练/验证的样本数量，num_workers 是加载数据时的线程数
data = Data(data_dir, batch_size=128, num_workers=0)
# 获取训练集、验证集和测试集的 DataLoader
train_loader, val_loader, test_loader = data.train_loader(), data.val_loader(), data.test_loader()

# 初始化模型，len(data.vocab.token_to_idx) 表示输入特征的维度，128 是隐藏层的大小，200 是输出的维度
model = Model(len(data.vocab.token_to_idx), 128, 200).to(device)
print(len(data.vocab.token_to_idx))

# 使用增强的对比学习损失函数替代原来的二元交叉熵损失
criterion = EnhancedContrastiveLoss()
# criterion = nn.BCEWithLogitsLoss()
# 初始化优化器，使用 Adam 优化器，学习率为 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 使用学习率调度器，根据验证集的损失调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# mode='min' 表示当验证损失不再下降时，减少学习率；factor=0.5 表示每次降低学习率为原来的一半；patience=1 表示当损失没有改善时等待 1 次 epoch

# 创建学习率日志记录器
lr_logger = LRLogger(optimizer)

# 添加一个标志变量，用于追踪是否已经执行过特殊的学习率减半操作
reduced_once = False

# 初始化训练和评估的指标计算器
train_metric, eval_metric = Metric(), Metric()

best = 0.  # 初始化最佳评估指标为 0
# 进行训练 12 轮
for epoch in range(1, 31):
    # 在每个epoch开始时打印当前学习率
    current_lr = lr_logger.get_last_lr()[0]
    print(f'epoch: {epoch}, 当前学习率: {current_lr:.6f}')  # 打印当前的 epoch 和学习率

    model.train()  # 设置模型为训练模式
    train_metric.reset()  # 重置训练时的指标
    loss, forward_time = 0., 0.  # 初始化损失和前向计算时间
    for (features1, features2), labels in tqdm(train_loader, desc='train'):  # 遍历训练集
        labels = labels.to(device)  # 将标签数据移动到指定设备（GPU/CPU）

        start_time = time.time()  # 记录开始时间
        outputs = model(features1, features2)  # 获取模型的输出
        end_time = time.time()  # 记录结束时间

        # 计算损失
        ls = criterion(outputs, labels)
        # 更新训练时的评估指标
        train_metric.update(torch.sigmoid(outputs), labels)

        optimizer.zero_grad()  # 清空之前的梯度
        ls.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        loss += ls.item() * labels.size(0)  # 累加损失
        forward_time += end_time - start_time  # 累加前向计算的时间
    print(f'train loss: {(loss / len(train_loader.dataset)):.4f}\t'  # 打印训练损失和前向时间
          f'forward_time: {(1e6 * forward_time / len(train_loader.dataset)):.4f}e-6\t', end='')
    train_metric.compute()  # 计算并输出训练时的评估指标

    # 验证集评估
    model.eval()  # 设置模型为评估模式
    eval_metric.reset()  # 重置评估时的指标
    loss, forward_time = 0., 0.  # 初始化损失和前向计算时间
    for (features1, features2), labels in tqdm(val_loader, desc='val'):  # 遍历验证集
        labels = labels.to(device)  # 将标签数据移动到指定设备

        start_time = time.time()  # 记录开始时间
        outputs = model(features1, features2)  # 获取模型的输出
        end_time = time.time()  # 记录结束时间

        # 计算损失
        ls = criterion(outputs, labels)
        # 更新验证集的评估指标
        eval_metric.update(torch.sigmoid(outputs), labels)

        loss += ls.item() * labels.size(0)  # 累加损失
        forward_time += end_time - start_time  # 累加前向计算的时间
    print(f'val loss: {(loss / len(val_loader.dataset)):.4f}\t'  # 打印验证损失和前向时间
          f'forward_time: {(1e6 * forward_time / len(val_loader.dataset)):.4f}e-6\t', end='')
    cur = eval_metric.compute()  # 计算并输出验证时的评估指标

    # 计算验证损失
    val_loss = loss / len(val_loader.dataset)

    # 特殊条件：当验证损失低于0.165时，学习率减半，只执行一次
    if val_loss < 0.165 and not reduced_once:
        prev_lr = lr_logger.get_last_lr()[0]
        # 手动将学习率减半
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        new_lr = [group['lr'] for group in optimizer.param_groups][0]
        print(f"\n特殊条件触发: 验证损失 {val_loss:.4f} < 0.165, 学习率从 {prev_lr:.6f} 减半为 {new_lr:.6f}")
        reduced_once = True  # 标记为已执行，确保只执行一次
        lr_logger.update_lr()  # 更新学习率记录
    else:
        # 正常的学习率调整
        prev_lr = lr_logger.get_last_lr()[0]
        scheduler.step(val_loss)  # 使用验证集损失来调整学习率
        new_lr = [group['lr'] for group in optimizer.param_groups][0]

        # 检查学习率是否发生变化并更新记录器
        if prev_lr != new_lr:
            print(f"\n学习率已更新: {prev_lr:.6f} -> {new_lr:.6f}")
        lr_logger.update_lr()

    if cur > best:  # 如果当前模型在验证集上的指标更好
        # 保存当前最好的模型
        # torch.save(model, os.path.join(save_dir, 'best_cpp_cs.pt'))  # C++和C#
        # torch.save(model, os.path.join(save_dir, 'best_java_cs.pt'))  # Java和C#
        # torch.save(model, os.path.join(save_dir, 'best_java_python.pt'))  # Java和python
        # torch.save(model, os.path.join(save_dir, 'best_python_cs.pt'))  # python和C#
        # torch.save(model, os.path.join(save_dir, 'best_java_cpp.pt'))  # Java和C++
        torch.save(model, os.path.join(save_dir, 'best_python_cpp.pt'))  # python和C++
        best = cur  # 更新最佳指标

# 加载最佳模型并进行测试
# model = torch.load(os.path.join(save_dir, 'best_cpp_cs.pt'), map_location=device)  # C++和C#
# model = torch.load(os.path.join(save_dir, 'best_java_cs.pt'), map_location=device)  # Java和C#
# model = torch.load(os.path.join(save_dir, 'best_java_python.pt'), map_location=device)  # Java和python
# model = torch.load(os.path.join(save_dir, 'best_python_cs.pt'), map_location=device)  # python和C#
# model = torch.load(os.path.join(save_dir, 'best_java_cpp.pt'), map_location=device)  # Java和C++
model = torch.load(os.path.join(save_dir, 'best_python_cpp.pt'), map_location=device)  # python和C++
model.eval()  # 设置模型为评估模式
eval_metric.reset()  # 重置评估时的指标
loss, forward_time = 0., 0.  # 初始化损失和前向计算时间
for (features1, features2), labels in tqdm(test_loader, desc='test'):  # 遍历测试集
    labels = labels.to(device)  # 将标签数据移动到指定设备

    start_time = time.time()  # 记录开始时间
    outputs = model(features1, features2)  # 获取模型的输出
    end_time = time.time()  # 记录结束时间

    # 计算损失
    ls = criterion(outputs, labels)
    # 更新测试集的评估指标
    eval_metric.update(torch.sigmoid(outputs), labels)

    loss += ls.item() * labels.size(0)  # 累加损失
    forward_time += end_time - start_time  # 累加前向计算的时间
print(f'test loss: {(loss / len(test_loader.dataset)):.4f}\t'  # 打印测试损失和前向时间
      f'forward_time: {(1e6 * forward_time / len(test_loader.dataset)):.4f}e-6\t', end='')
eval_metric.compute()  # 计算并输出测试时的评估指标
# 打印模型的参数数量
print(f'number_of_parameters: {(sum(param.nelement() for param in model.parameters()) / 1e6):.2f}M')