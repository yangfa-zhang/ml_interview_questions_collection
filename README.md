### 八股问题
#### **xgboost与传统决策树的不同**
- 传统GBDT的核心是拟合残差（你可以理解为损失函数的一阶导数，尤其是在均方误差损失下）。
XGBoost则更进一步，利用损失函数的二阶泰勒展开，这意味着它不仅使用了一阶导数（梯度，g_i），还使用了二阶导数（海森，h_i）。这使得XGBoost的优化更加精确，能够支持各种可微的损失函数，并且在寻找最佳分裂点和计算叶子节点输出值时，都充分利用了这些二阶信息，从而实现更快的收敛和更好的性能。
- 计算基础：XGBoost基于**一阶和二阶梯度**，而传统决策树基于不纯度指标（基尼、信息增益）或MSE。
- 正则化：XGBoost的增益计算内置了L2正则项(λ)和最小分裂增益(γ)，直接影响了分裂的决策，有助于防止过拟合。传统决策树通常是独立的剪枝步骤。
- 灵活性：基于梯度和海森使得XGBoost能够支持任何可微的损失函数，而传统决策树的增益公式通常与特定的损失函数（如分类的交叉熵、回归的MSE）绑定。

#### **xgboost如何处理缺失值**
- 分别尝试将缺失值归到左右子树：在寻找最佳分裂点时，**把所有缺失值的样本都归到左/右子树**，计算分裂后带来的增益,选择增益最大的方向。
- 无需手动填充：省去了数据预处理中缺失值填充的复杂步骤，避免了因不当填充引入偏差。
- 自适应性：XGBoost能够根据数据自身的模式，学习出最优的缺失值处理策略，而不是简单地用一个固定值替代。这意味着对于不同的特征和数据集，它能找到最适合该特征的缺失值“去向”。
- 鲁棒性：模型对含有缺失值的数据更加鲁棒。

#### **为什么同时使用一阶导数和二阶导数比只使用一阶导数更好？**
- 步长问题：梯度只告诉我们方向，没有告诉我们应该迈多大一步。如果步长（学习率）过大，可能会跳过最小值；如果过小，收敛速度会很慢。
- 局部震荡：在平坦区域收敛慢，在陡峭区域可能会来回震荡。不考虑曲率：它假设函数是线性的，没有考虑损失函数的“弯曲”程度。在一个平坦的区域，梯度可能很小，但在一个凹凸不平的区域，梯度可能很大，但这些信息不能很好地指导我们如何调整。
- 在GBDT中的体现：GBDT拟合残差，可以看作是L2损失函数下，拟合负梯度。它在每次迭代中都朝着梯度下降的方向努力，但没有一个更精细的“导航系统”来**指导它走多大的步子**。
- 同时使用一阶导数和二阶导数时，这类似于牛顿法（或者说，它是牛顿法的一个近似）。
一阶导数（梯度 g i）：告诉我们函数值在当前点的变化方向;二阶导数（海森 h i）：告诉我们函数在当前点的曲率（即梯度的变化率）。它描述了损失函数在当前点附近是凸的还是凹的，以及凸凹的程度。
- 更精确的优化方向和步长：通过结合曲率信息，牛顿法可以找到一个更优化的步长，更直接地跳向最小值。它能更好地“预判”前方损失函数的变化趋势。
- 收敛速度更快：**通常比纯粹的梯度下降收敛得更快**，尤其是在损失函数是凸函数的情况下。
- 对非二次损失函数的处理能力：对于复杂的、非二次的损失函数，二阶信息能够更好地捕捉其局部形状，从而提供更准确的下降方向。
统一框架：XGBoost通过二阶泰勒展开，将各种损失函数都统一到一个框架下进行优化。无论你的损失函数是均方误差、对数损失还是其他，只要它是可二次微分的，XGBoost就能使用相同的优化过程。

#### **lightgbm和xgboost的不同点**
- 基于 GBDT 的决策树生长策略：
XGBoost： 默认采用按层生长 (Level-wise) 的策略。它会同时分裂同一层的所有叶子节点，这样可以并行计算，但可能会产生一些不必要的计算，因为它不区分当前层哪些节点对提升模型性能贡献大。
LightGBM： 采用按叶子生长 (Leaf-wise) 的策略。它每次从当前所有叶子节点中，找到分裂增益最大的那个叶子节点进行分裂。这种策略在分裂次数相同的情况下，通常能降低更多的误差，得到更好的精度。但它也可能导致树的深度更深，更容易过拟合（可以通过限制最大深度来缓解）。
- 特征并行与数据并行：
XGBoost： 主要支持数据并行。
LightGBM： 支持特征并行 (Feature Parallel) 和数据并行 (Data Parallel)。
特征并行： 不同的机器学习不同的特征子集。当数据量很大，但特征维度不高时，这种方式效果显著。
数据并行： 不同的机器学习不同的数据子集。当特征维度很高，但数据量适中时，这种方式效果更好。
- **直方图算法 (Histogram Algorithm)**：
XGBoost： 在分裂节点时，需要遍历所有样本，并对每个特征的每个可能分裂点计算增益。
LightGBM： 使用直方图算法。它将连续的特征值离散化成 K 个离散的桶（bins）。在寻找最佳分裂点时，只需要遍历这 K 个桶，而不是原始的 N 个数据点，大大减少了计算量。这不仅加快了训练速度，也减少了内存消耗。
- 互斥特征捆绑 (Exclusive Feature Bundling, EFB)：
这个是 LightGBM 独有的优化。现实中很多高维数据是稀疏的，即很多特征是稀疏的，并且很多特征都是互斥的（例如，同一个样本不可能同时属于两个完全不相交的类别）。EFB 就是将这些互斥特征捆绑在一起，形成一个新的特征，从而减少特征的数量，进一步提升训练速度。
- **单边梯度抽样 (Gradient-based One-Side Sampling, GOSS)**：
传统 GBDT 在计算梯度时，会使用所有样本。
GOSS 的核心思想是，梯度小的样本对模型训练的贡献较小，梯度大的样本则相反。因此，GOSS 在每次迭代中，会保留梯度较大的样本，并随机采样梯度较小的样本。这样既能保证训练的准确性，又能减少样本数量，从而加快训练速度。

#### **LightGBM 的 Leaf-wise 策略 vs. XGBoost 的 Level-wise 策略**
我们来想象一下决策树的生长过程，就像修剪一棵植物。

- XGBoost 的 Level-wise (按层生长) 策略：
想象你有一棵植物，你每次都从树顶往下，同时修剪同一层的所有枝叶。
优点： 这种方式的优点是，你可以很容易地并行处理，因为所有同一层的节点都可以同时考虑。这使得在计算资源充足时，训练速度可能很快。
缺点： 它的问题在于，有些枝叶可能对植物的整体生长（模型性能提升）没什么帮助，但你还是修剪了它们。这意味着可能会做一些不必要的计算，导致在达到相同精度的前提下，它可能需要更多的分裂次数，或者树的深度更深，模型更复杂。
- LightGBM 的 Leaf-wise (按叶子生长) 策略：
现在想象你是一个更精明的园丁。你每次修剪时，都会查看植物上所有的叶子（代表模型当前的叶子节点），然后找到那个最需要修剪（分裂增益最大）的叶子进行修剪。修剪完这片叶子后，它会变成两个新的叶子，你再从所有叶子中选择下一个最需要修剪的。
优点：
效率更高： LightGBM 每次都选择“最有价值”的分裂点，这意味着在相同次数的分裂下，它通常能更快地降低模型的误差，达到更高的精度。因为它专注于对模型提升最大的部分。
收敛速度快： 由于每次分裂都是“最优”的，所以模型通常能更快地收敛到较好的性能。
缺点：
容易过拟合 (Overfitting)： 因为它总是选择最佳分裂点，这可能导致树变得非常深且不平衡。一棵特别深的树，就像一个记忆力超强但缺乏泛化能力的学生，它可能把训练数据中的噪音都记住了，导致在新数据上表现不好。
并行化难度： 每次分裂都会改变叶子节点的集合，这使得并行处理（同时找多个最佳分裂点）变得更复杂。不过，LightGBM 在其他方面（如直方图算法）弥补了这一并行化劣势。

#### **直方图算法对模型性能的影响**
直方图算法是 LightGBM 提速的关键之一。
- 工作原理回顾： 它不是直接在原始的连续特征值上寻找最佳分裂点，而是将这些连续值“分桶”到有限的几个离散区间（像直方图的柱子）。然后，它只需要遍历这些“桶”的边界，而不是所有原始数据点，来找到最佳分裂点。
- 对模型性能（准确性）的影响：
理论上： 这种离散化可能会导致轻微的精度损失。因为你把连续值近似了，就像你把精确的数字 3.14159 近似为 3.14 一样，丢失了一些信息。这意味着，理论上，最佳分裂点可能落在两个桶的中间，而直方图算法只能在桶的边界选择。
实际上： 在绝大多数情况下，这种精度损失可以忽略不计，甚至因为模型训练速度加快，你可以尝试更多的迭代次数或更复杂的模型结构，反而可能获得更高的精度。
- 原因：
分桶数量足够多： LightGBM 默认的分桶数量通常是 256 个，对于大多数数据集来说，这已经足够细致，不会损失太多信息。
模型是集成学习： GBDT 模型是通过多棵弱学习器（决策树）的组合来提升性能的。即使单棵树由于离散化导致略有不精确，多棵树的组合效果会弥补这个缺陷。
噪音鲁棒性： 适度的离散化甚至可以帮助模型对噪音更鲁棒，因为它不会对微小的特征值变化过于敏感。
总结： 直方图算法是为了提升速度和减少内存而设计的，它在牺牲极小部分理论精度的前提下，换来了巨大的效率提升。在实际应用中，这种微小的精度损失通常是完全可以接受的，甚至不容易被察觉。

#### **何时选择 LightGBM 而不是 XGBoost**
- LightGBM 的优势场景：
大数据集： 当你的数据集非常大（百万甚至上亿条记录）时，LightGBM 的直方图算法、EFB 和 GOSS 优化能显著提升训练速度和降低内存消耗。它通常比 XGBoost 更快地完成训练。
追求极致速度： 如果训练时间是你的首要考虑因素，LightGBM 往往是更好的选择。
特征维度较高且稀疏的数据： EFB 在这种情况下效果显著，能有效减少特征数量。
模型性能已达到瓶颈，希望微调： 即使精度上 XGBoost 也能做得很好，但 LightGBM 可能会在训练速度上带来惊喜，让你能更快地迭代模型。
- XGBoost 的优势场景：
数据集不是特别大，或者对精度要求极高，且时间不是最关键因素： 虽然 LightGBM 精度通常也很高，但 XGBoost 的 Level-wise 策略在某些特殊情况下可能探索到更全局的最优解。
需要更强的正则化控制： XGBoost 在参数控制上可能提供更细粒度的正则化选项，对于一些复杂的过拟合问题，可能更容易进行调优。
对并行化支持要求更高： 在某些特定硬件配置下，XGBoost 的 Level-wise 并行可能表现更优。
对稳定性要求很高： XGBoost 在工业界应用时间更长，社区更成熟，可能在一些极端边缘案例下表现更稳定。

#### **GOSS 和 EFB 的区别**
- GOSS (Gradient-based One-Side Sampling)：
如何提升效率： 它的核心是减少了参与梯度计算的样本数量。它保留了所有梯度大的样本（这些样本误差大，对模型提升贡献大），然后随机抽样了一部分梯度小的样本（这些样本误差小，对模型提升贡献小）。这样，每次迭代计算梯度和寻找最佳分裂点时，处理的样本总量就减少了，从而加速了训练。
不是“减少训练的样本”： 严格来说，GOSS 不是减少了原始训练集中的样本数量，而是在每次迭代时，减少了参与当前这棵树学习的样本数量。整个模型的训练仍然会遍历所有样本，只是在每一步迭代中，对样本的关注度不同。
- EFB (Exclusive Feature Bundling)：
如何提升效率： 它的核心是减少了特征的数量。它识别出那些“互斥”的特征（即它们不可能同时非零），然后将它们捆绑成一个新的特征。例如，在一个 One-Hot 编码的类别特征中，一个样本在同一时间只能属于一个类别，那么这些 One-Hot 编码的特征就是互斥的。将它们捆绑后，在遍历特征、寻找分裂点时，需要处理的特征数量就变少了，从而加速了训练。
不是“减少训练的样本”： EFB 不会改变样本的数量，它改变的是特征的数量。
- 总结：
GOSS 优化的是样本维度上的计算（减少了每次迭代中参与计算的样本）。
EFB 优化的是特征维度上的计算（减少了实际处理的特征数量）。 两者都是为了提升训练效率，但侧重点不同。

#### **当 LightGBM 过拟合时，除了 max_depth，你更应该优先考虑调整？**
减小 num_leaves
增大 min_data_in_leaf

#### **xgboost的超参数**
**[Model complexity parameters]>**
- lambda: 	L2 regularization. Smoother than L1. Better for sparse data. Prevents overfitting.
- reg_lambda/alpha: 	Regularization. Control model complexity. Prevents overfitting.
- gamma: 	TREE ONLY. Minimum loss reduction for split. Prevents overfitting.
- max_depth: 	Higher = more complex model. Prevents overfitting.
- subsample: 	Number of samples per tree. Prevents overfitting.
- colsample_bytree: 	Fraction of features used per tree. Prevents overfitting.
- min_child_weight: 	Minimum sum of instance weight in a child. Prevents overfitting.

**[Training and Optimization Parameters]>**
- eta: 	Learning rate.
- booster: 	"gbtree" for nonlinear features. "gblinear" for linear features
- grow_policy: 	Controls how new nodes are added to the tree. "lossguide" for best split. "depthwise" for best depth.

### 岗位相关问题


### 手撕代码
#### **transformer**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, embed_dim)
        self.l2 = nn.Linear(embed_dim, output_dim)
        self.layer = nn.TransformerEncoderLayer(d_model= embed_dim, nhead= num_heads, dim_feedforward= ff_dim)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers= num_layers)


    def forward(self, x):
        x = self.l1(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.l2(x)

torch.manual_seed(42)
seq_length = 10
num_samples = 100
input_dim = 1
X = torch.rand(num_samples, seq_length, input_dim)  # Random sequences
y = torch.sum(X, dim=1)  # Target is the sum of each sequence

# Initialize the model, loss function, and optimizer
input_dim = 1
embed_dim = 16
num_heads = 2
num_layers = 2
ff_dim = 64
output_dim = 1

model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class EarlyStopping():
    def __init__(self, model, patience=10):
        self.model = model
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.earlystopping = False

    def __call__(self, val_loss):
        if self.val_loss < self.best_loss: 
            self.best_loss = self.val_loss
            self.count = 0
            torch.save(self.model.state_dict(), 'best_model.pt')
        else:
            self.count+=1
        if self.count >= self.patience:
            self.earlystopping = True
        return self.earlystopping

# Training loop
epochs = 1000
es = EarlyStopping(model)
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if es(loss):
        print(f'EarlStopping At Epoch {epoch+1}')

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
```
#### **GCN**
```python
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(input_dim,output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        if not adj.is_sparse:
            adj = adj.to_sparse()
        x = torch.sparse.mm(adj, x)

        if self.bias is not None:
            return x + self.bias 
        else:
            return x 

class fafa_GCN(nn.Module):
    def __init__(self, n_feat, n_hidd, n_class, dropout=0.5):
        super().__init__()
        self.gc1 = GCNConv(n_feat, n_hidd)
        self.gc2 = GCNConv(n_hidd, n_class)
        self.dropout = dropout

    def forward(self, data: torch_geometric.data.Data):
        x = data.x
        adj = data.edge_index
        adj = to_dense_adj(adj, max_num_nodes=x.size(0))[0]

        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# 构造模拟数据
x = torch.randn(4, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 1],
    [1, 0, 2, 1, 3, 2, 1, 3]
], dtype=torch.long)
y = torch.tensor([0, 1, 0, 1])
data = Data(x=x, edge_index=edge_index, y=y)


model = fafa_GCN(n_feat= 3, n_hidd= 4, n_class= 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

class EarlyStopping():
    def __init__(self, model, patience=10):
        self.model = model
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.earlystopping = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss: 
            self.best_loss = val_loss
            self.count = 0
            torch.save(self.model.state_dict(), 'best_model.pt')
        else:
            self.count+=1
        if self.count >= self.patience:
            self.earlystopping = True
        return self.earlystopping


epochs = 10000
es = EarlyStopping(model)
for epoch in range(epochs):
    predictions = model(data)
    loss = criterion(predictions, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if es(val_loss=loss):
        print(f'EarlyStopping At Epoch {epoch+1}')
        break

    if (epoch+1) % 10 == 0:
        pred = predictions.argmax(dim=1)
        acc = (pred == data.y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
```