import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, token_vocab_size, embedding_dim, hidden_dim):
        """
        初始化模型的各个层和超参数。
        :param token_vocab_size: 词汇表大小，模型处理的词汇总数。
        :param embedding_dim: 嵌入层的维度，即每个词汇的嵌入向量大小。
        :param hidden_dim: 隐藏层的维度，用于控制 RNN 和其他全连接层的大小。
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_gpu = torch.cuda.is_available()
        self.embedding_dim = embedding_dim

        # 嵌入层，将词汇的整数索引映射到一个固定维度的嵌入向量
        self.embed = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=0)

        # 学术深度点1: 词嵌入丰富层 - 增强模型捕获代码语义的能力
        self.embedding_enrichment = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 主要RNN层：使用双向GRU
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)

        # 学术深度点2: 注意力机制 - 关注重要的子树特征
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 线性变换层：将嵌入维度映射到相同维度
        self.linear = nn.Linear(embedding_dim, embedding_dim)

        # 合并层：将双向GRU输出的2倍hidden_dim降为hidden_dim
        self.combine = nn.Linear(2 * hidden_dim, hidden_dim)

        # 学术深度点3: 交叉特征交互模块 - 捕获两段代码之间的深层交互
        self.cross_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 简化的相似度权重(可学习参数)
        self.sim_weight = nn.Parameter(torch.ones(2))

        # 两个线性层，用于输出相似度计算
        self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(hidden_dim, 1)

        # 用于存储节点的列表
        self.node_list = []

        # 用于存储当前批次的节点
        self.batch_node = None

    def forward(self, x1, x2):
        """
        定义前向传播函数，计算模型的输出。
        :param x1: 输入的第一个子树
        :param x2: 输入的第二个子树
        :return: 计算的输出
        """

        def encode_subtree_roots(subtree_roots):
            """
            对子树根节点进行编码并遍历子树。
            :param subtree_roots: 输入的子树根节点
            :return: 批次中的所有子树根节点的嵌入表示
            """

            def traverse(node, batch_index):
                """
                对单棵子树进行遍历并计算其嵌入表示。
                :param node: 当前子树的节点
                :param batch_index: 当前子树的索引
                :return: 该子树的嵌入表示
                """
                size = len(node)
                if not size:
                    return None

                index, children_index = [], []
                current_node, children = [], []
                for i in range(size):
                    index.append(i)
                    current_node.append(node[i][0])
                    temp = node[i][1:]
                    c_num = len(temp)
                    for j in range(c_num):
                        children_index.append(i)
                        children.append(temp[j])

                th = torch.cuda if self.use_gpu else torch

                # 应用词嵌入丰富层 - 学术深度点1的应用
                node_embeddings = self.embed(th.LongTensor(current_node))
                node_embeddings = self.embedding_enrichment(node_embeddings)
                batch_current = self.linear(node_embeddings)

                # 子树遍历处理
                if children_index:
                    tree = traverse(children, [batch_index[i] for i in children_index])
                    batch_current.index_add_(0, th.LongTensor(children_index), tree)

                # 更新当前批次的节点表示
                self.batch_node = self.batch_node.index_reduce(0, th.LongTensor(batch_index), batch_current, 'amax')
                return batch_current

            batch_size = len(subtree_roots)
            self.batch_node = torch.zeros(batch_size, self.embedding_dim,
                                          requires_grad=True).cuda() if self.use_gpu else torch.zeros(batch_size,
                                                                                                      self.embedding_dim,
                                                                                                      requires_grad=True)
            traverse(subtree_roots, list(range(batch_size)))
            return self.batch_node

        def encode(x):
            """
            对输入的子树进行编码，返回每个子树的嵌入表示。
            :param x: 输入的子树数据
            :return: 编码后的子树嵌入表示
            """
            batch_subtrees = x
            batch_size, n_subtrees_list = len(batch_subtrees), [len(subtrees) for subtrees in batch_subtrees]

            # 展开所有子树为一个列表
            all_subtrees = [batch_subtrees[i][j] for i in range(batch_size) for j in range(n_subtrees_list[i])]
            batch_subtree_embeddings = encode_subtree_roots(all_subtrees)

            # 将子树嵌入表示拆分为原来的子树数量
            batch_subtree_embeddings = torch.split(batch_subtree_embeddings, n_subtrees_list)

            # 对齐子树嵌入的序列长度
            batch_subtree_embeddings = nn.utils.rnn.pad_sequence(batch_subtree_embeddings)
            batch_subtree_embeddings = nn.utils.rnn.pack_padded_sequence(
                batch_subtree_embeddings, torch.tensor(n_subtrees_list), enforce_sorted=False)

            # 使用 RNN 处理子树嵌入
            output, _ = self.rnn(batch_subtree_embeddings)

            # 解包序列
            batch_subtree_embeddings, lengths = nn.utils.rnn.pad_packed_sequence(output, padding_value=0)

            # 应用注意力机制 - 学术深度点2的应用
            attention_scores = self.attention(batch_subtree_embeddings)  # [seq_len, batch_size, 1]

            # 对于掩码的处理，确保padding不参与注意力计算
            mask = torch.arange(batch_subtree_embeddings.size(0)).unsqueeze(1) < torch.tensor(
                n_subtrees_list).unsqueeze(0)
            mask = mask.to(batch_subtree_embeddings.device).float().unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))

            attention_weights = F.softmax(attention_scores, dim=0)

            # 应用注意力权重进行加权求和
            context = (batch_subtree_embeddings * attention_weights).sum(dim=0)

            # 通过 combine 层来降维
            batch_subtree_embeddings = self.combine(batch_subtree_embeddings)

            # 对输出进行转置和拆分
            batch_subtree_embeddings = torch.transpose(batch_subtree_embeddings, 0, 1)
            batch_subtree_embeddings = [subtree_embeddings[:n_subtrees] for n_subtrees, subtree_embeddings in
                                        zip(n_subtrees_list, batch_subtree_embeddings)]
            return batch_subtree_embeddings

        # 对输入的两个子树进行编码
        h1, h2 = encode(x1), encode(x2)

        # 获取代码表示
        z11, z12 = [torch.max(hi1, dim=0).values for hi1 in h1], [torch.max(hi2, dim=0).values for hi2 in h2]
        z11, z12 = torch.stack(z11), torch.stack(z12)

        # 交叉特征交互 - 学术深度点3的应用
        cross_features = self.cross_interaction(torch.cat([z11, z12], dim=1))

        # 计算第一种相似度(基础点乘相似度)
        z1 = torch.sum(z11 * z12, dim=1)

        # 计算第二种相似度(考虑跨子树交互)
        z2 = torch.stack(
            [(hi1.unsqueeze(0) * hi2.unsqueeze(1)).sum(dim=0).sum(dim=0) for hi1, hi2 in zip(h1, h2)]
        )
        z2 = self.linear2(z2).squeeze()

        # 使用可学习权重组合两种相似度
        sim_weights = F.softmax(self.sim_weight, dim=0)
        z_combined = z1 * sim_weights[0] + z2 * sim_weights[1]

        # 融合交叉特征和相似度得分
        final_output = self.linear1(cross_features).squeeze() + z_combined

        return final_output
