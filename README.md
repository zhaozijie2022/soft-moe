# Light Soft Mixture of Experts

一个[Soft Mixture of Experts](https://arxiv.org/abs/2308.00951v1)的轻量化实现。
不同于其他与ViT耦合的soft-moe实现，**本代码没有与任何领域的特定方法耦合，也不依赖于PyTorch以外的任何框架**，可以适用于任何向量化序列，移植和嵌入更加方便。

## Sparse MoE

In [Sparse MoE](https://arxiv.org/abs/1701.06538), the model processes input tokens in a sequential manner.
Let's assume that the input consists of a sequence of length  $n$, represented as $X\in\mathbb{R}^{n\times d}$, where each token is denoted as $x\in\mathbb{R}^d$.
The model comprises multiple experts, where each expert is a small neural network (e.g., MLP).
Since different experts excel at processing different aspects, it is necessary to determine which experts should be invoked to process the current token.
The component responsible for selecting the experts based on the input is called a **router**, denoted as $G(x) \in \mathbb{R}^e$.
The output of the router can be considered as the scores assigned to the $e$ experts.
To choose the experts for processing the current token, we select the $k$ experts with the highest scores.
Then the weights assigned to each expert are $w=SoftMax(TopK(G(x)))\in R^{k}$.
Aggregate the outputs of the selected experts, the final result can be expressed by

$$
y = \sum_{i=1}^k w_i \  Expert_i(x)
$$

在Sparse MoE中，tokens以序列的模式进入模型，假设长度为 $n$ 的序列作为输入 $X\in\mathbb{R}^{n\times d}, x\in\mathbb{R}^d$。MoE模型对应 $e$ 个experts，每个expert为一个小的神经网络(e.g. MLP)。由于不同的expert擅长处理的领域不同，我们需要根据输入来选择调用哪些experts来处理当前token。选择专家的模块被称为router，记作 $G(x) \in \mathbb{R}^e$，它的输出可以被看做这e个专家的得分。从中选择得分最高的k个来处理当前token，则每个experts的权重为 $w=SoftMax(TopK(G(x)))\in R^{k}$。
根据这些权重将Expert的输出进行加权得到最终结果。

## Soft MoE

### The Shortcoming of SparseMoE
<ol>
  <li>The model does not consider the relationship between tokens, it is just a replacement for the FFN;</li>
  <li>The model faces the problem of load balancing, that is, it is easy to converge to a small number of experts handling the vast majority of inputs, while most experts are lazy;</li>
  <li>Token Dropping: For tokens that have not been seen, there may be a performance crash;</li>
</ol>

<ol start="1">
  <li>没有考虑token与token之间的关联，只是一个ffn的替代品；</li>
  <li>面临负载均衡问题，即容易收敛到一小部分的experts处理了绝大部分输入，而大部分的experts在偷懒；</li>
  <li>Token Dropping：对于没有见过的token，可能面临着性能崩溃；</li>
</ol>


### Design of SoftMoE
According to the design of [Soft-MoE](https://arxiv.org/abs/2308.00951v1), a model has $e$ experts, each expert corresponds to $s$ slots, that is, each expert processes $s$ tokens. <br />
依照 oft-MoE的设计, 一个SofMoE拥有 $e$ 个Experts, 每个Experts对应$s$个slots，即每个expert处理$s$个tokens。

```python
self.experts = nn.ModuleList([
    Mlp(d_model, d_model, d_model,
        hidden_activation=F.relu, output_activation=F.relu,
        layer_norm=True, out_layer_norm=True, use_residual=False)
    for _ in range(num_experts)
])
```
The original router is replaced by a parameter $\Phi \in \mathbb{R}^{d \times (e \times s)}$. <br />
原有的router被替换为了参数 $\Phi \in \mathbb{R}^{d \times (e \times s)}$。

```python
self.phi = nn.Parameter(torch.randn(d_model, num_experts, num_slots))
```

For the input $X\in\mathbb{R}^{n \times d}$, we calculate the weights as follows: <br />
对于输入 $X\in\mathbb{R}^{n \times d}$，计算权重 <br />

$$
W=X \Phi \in \mathbb{R}^{n \times (e \times s)}
$$

```python
# compute weights, which are used both in dispatch and combine
weights = torch.einsum("b n d , d e s -> b n e s", x, self.phi)
```

The weight matrix has two dimensions: the sequence length $n$ and the number of processable tokens $e \times s$ (where $e$ is the number of experts, each processing $s$ tokens). 
This matrix allows for the conversion between dimensions $n$ and $e \times s$. To normalize the dimension $n$, we use softmax and multiply it with the input: <br />
这个权重矩阵有两个维度, 序列长度 $n$ 和可处理的tokens数量 $e \times s$ ($e$个专家，每个处理$s$个token)。这个矩阵对其他矩阵相乘可以实现 $n$ 和 $e \times s$ 之间的相互转化。
对维度 $n$ 使用softmax归一化，与输入相乘

$$
In_E = SoftMax(W^T) X \in \mathbb{R}^{(e\times s) \times d}
$$

$In_E$ can be seen as applying a soft pooling operation on the tokens, transforming the length from $n$ to $e \times s$. <br />
$In_E$ 相当于对tokens进行了一次soft pooling，将长度 $n$变为 $e \times s$，

```python
# dispatch tokens to experts
dispatch_weights = F.softmax(weights, dim=1)
experts_inputs = torch.einsum("b n e s, b n d -> b e s d", dispatch_weights, x)
```

We then dispatch $In_E$ to $e$ experts, with each expert handling $s$ tokens. This step doesn't change the dimensions. <br />
将 $In_E$ 送入 $e$ 个experts，每个expert对应 $s$ 个tokens，这一步不改变维度。

$$
Out_E = Experts(In_E) \in \mathbb{R}^{(e\times s) \times d}
$$

```python
expert_outputs = torch.stack([self.experts[i](experts_inputs[:, i]) for i in range(self.num_experts)])
expert_outputs = einops.rearrange(expert_outputs, "e b s d -> b (e s) d")
```

Next, we aggregate $Out_E$: <br />
将 $Out_E$ 进行聚合，

$$
y = SoftMax(W) Out_E \in \mathbb{R}^{n \times d}
$$


```python
# combine expert outputs
combine_weights = einops.rearrange(weights, "b n e s -> b n (e s)")
combine_weights = F.softmax(combine_weights, dim=-1)
out = torch.einsum("b n z, b z d -> b n d", combine_weights, expert_outputs)
```

