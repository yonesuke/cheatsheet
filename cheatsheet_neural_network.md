# Neural Network Cheatsheet

A comprehensive guide to the fundamental concepts, architectures, and theoretical foundations of Neural Networks and Deep Learning.

---

## 1. Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Without them, a multi-layer neural network would be mathematically equivalent to a single linear transformation (a generalized linear model).

### Common Functions

*   **Sigmoid**:
    $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
    *   **Range**: $(0, 1)$
    *   **Derivative**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
    *   **Pros**: Interpretable as probability.
    *   **Cons**:
        *   **Vanishing Gradient**: For large $|x|$, $\sigma'(x) \approx 0$.
        *   **Non-zero Centered**: Outputs are always positive, which can introduce zig-zagging dynamics in gradient updates.

*   **Tanh (Hyperbolic Tangent)**:
    $$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
    *   **Range**: $(-1, 1)$
    *   **Derivative**: $\tanh'(x) = 1 - \tanh^2(x)$
    *   **Pros**: Zero-centered outputs solve the zig-zagging problem.
    *   **Cons**: Still suffers from vanishing gradient problem as $|x| \to \infty$.

*   **ReLU (Rectified Linear Unit)**:
    $$ f(x) = \max(0, x) $$
    *   **Derivative**: $f'(x) = 1 \text{ if } x > 0 \text{ else } 0$
    *   **Pros**: **Sparsity** and **No Vanishing Gradient** in the positive domain.
    *   **Cons**: "Dying ReLU" problem (dead neurons).

*   **GELU (Gaussian Error Linear Unit)**:
    $$ \text{GELU}(x) = x \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$
    *   **Approximation**: $0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$
    *   **Theory**: Can be viewed as a smooth approximation to ReLU, or as "Dropout" applied to the input where the drop probability depends on the input value itself.
    *   **Pros**: Smoothness allows for better optimization landscapes compared to ReLU. Used in BERT, GPT-2/3, ViT.

*   **Swish / SiLU (Sigmoid Linear Unit)**:
    $$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$
    *   **SiLU**: Special case where $\beta=1$.
    *   **Derivative**: $f'(x) = f(x) + \sigma(x)(1 - f(x))$.
    *   **Pros**: Non-monotonic bump for $x < 0$ allows small negative gradients to flow, potentially helping "unstick" dead neurons. Discovered via Neural Architecture Search.

*   **Softmax**:
    $$ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}} $$
    *   **Usage**: Output layer for multi-class classification.

---

## 2. MLP (Multi-Layer Perceptron)

### Architecture
An MLP consists of an input layer, one or more hidden layers, and an output layer.
$$ h^{(l)} = \phi(W^{(l)}h^{(l-1)} + b^{(l)}) $$

### Theoretical Background: Universal Approximation
*   **Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1991)**:
    A feedforward network with a **single hidden layer** of finite width can approximate any continuous function $f: [0, 1]^n \to \mathbb{R}$ to arbitrary precision $\epsilon$, given a non-polynomial activation function $\sigma$.
    $$ F(x) = \sum_{i=1}^N v_i \sigma(w_i^T x + b_i) $$
    *   **Note**: Describes *representational capacity*, not *learnability*. The required width $N$ may be exponential in dimension $n$.

### Regularization: Dropout (Srivastava et al., 2014)
A technique to prevent overfitting by randomly dropping units during training.

*   **Logic (Inverted Dropout)**:
    *   **Training**: Randomly set neurons to zero with probability $p$. Mask vector $r \sim \text{Bernoulli}(1-p)$.
        $$ h^{(l)} = \text{Activation}(W h^{(l-1)} + b) \odot \frac{r}{1-p} $$
        scaling by $\frac{1}{1-p}$ maintains the expected magnitude of the activations.
    *   **Inference**: Use the full network without dropout. The scaling during training ensures no scaling is needed at test time.

*   **Theoretical Background**:
    1.  **Ensemble Interpretation**: Dropout can be viewed as training an ensemble of $2^N$ exponentially many thinned subnetworks that share weights. Test time prediction approximates averaging the predictions of these exponentially many models.
    2.  **Bayesian Approximation (Gal & Ghahramani, 2016)**: Dropout training is mathematically equivalent to approximate Variational Inference in deep Gaussian Processes. Enabling dropout at inference time (**MC Dropout**) allows sampling from the approximate posterior distribution, providing a measure of **epistemic uncertainty** without changing the architecture.
    3.  **Co-adaptation**: Prevents complex co-adaptations in which a feature detector is only helpful in the context of several other specific feature detectors.

---

---

## 3. Normalization Layers

### Batch Normalization (BatchNorm)
Standard in CNNs. Normalizes activations across the **mini-batch** dimension.
$$ \hat{x}_{i} = \frac{x_{i} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} $$
$$ y_{i} = \gamma \hat{x}_{i} + \beta $$
*   **Stats**: $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}$ computed per channel across batch $(N, H, W)$.
*   **Inference**: Uses running averages of $\mu, \sigma$ tracked during training.
*   **Theory**:
    *   **Internal Covariate Shift**: Originally claimed to reduce the shifting distribution of internal layers.
    *   **Landscape Smoothing**: Later shown to make the optimization landscape significantly smoother (Lipschitzness), allowing larger learning rates (Santurkar et al., 2018).
*   **Cons**: Breaks independence between samples; performance degrades with small batch sizes; problematic for RNNs.

### Layer Normalization (LayerNorm)
Standard in NLP/Transformers. Normalizes across the **feature** dimension for a single sample.
$$ \mu_L = \frac{1}{H} \sum_{j=1}^H x_{ij}, \quad \sigma_L^2 = \frac{1}{H} \sum_{j=1}^H (x_{ij} - \mu_L)^2 $$
$$ \hat{x}_{ij} = \frac{x_{ij} - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} $$
*   **Independence**: Computation is independent of batch size and other samples.
*   **Theory**: Invariance to re-scaling of weight matrix and input signal.

### RMS Normalization (RMSNorm)
Simplified LayerNorm used in modern LLMs (e.g., Llama, Gopher).
$$ \bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{H} \sum_{j=1}^H x_j^2}} \cdot \gamma_i $$
*   **Difference**: Removed mean centering ($\mu_L$). Only re-scales based on Root Mean Square.
*   **Theory**: Hypothesizes that the re-centering invariance of LayerNorm is less important than re-scaling invariance. Reducing operations speeds up training without performance loss.

---

## 4. CNN (Convolutional Neural Networks)

### Architecture
*   **Convolution**: Cross-correlation between input $I$ and kernel $K$.
    $$ (I * K)(x) = \int I(x')K(x-x') dx' $$
*   **Pooling**: Subsampling operation (Max, Average) for dimensionality reduction and invariance.

### Theoretical Background: Equivariance & Inductive Bias
*   **Translation Equivariance**:
    A map $f$ is equivariant to group action $g$ if $f(g \cdot x) = g \cdot f(x)$. Convolutions are equivariant to translation: shifting the input shifts the feature map.
    $$ [\text{Shift}_s(f) * g](t) = \text{Shift}_s(f * g)(t) $$
*   **Priors**: CNNs impose strong priors:
    1.  **Locality**: Features are locally correlated.
    2.  **Stationarity**: Features (edges, textures) are useful everywhere in the image (parameter sharing).

---

## 5. ResNet (Residual Networks)

### Architecture
Introduced to allow training of extremely deep networks (hundreds/thousands of layers) by learning residual functions.
*   **Residual Block**:
    $$ x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l) $$
    where $x_l$ is the input and $\mathcal{F}$ is the residual mapping (e.g., 2 or 3 conv layers).

### Mathematical Theory: The Gradient Highway
ResNet solves the **Vanishing Gradient** degradation problem not just by initialization, but by structure.

1.  **Forward Propagation**:
    By unrolling the recursion $x_{l+1} = x_l + \mathcal{F}(x_l)$, we can express $x_L$ as the sum of $x_l$ plus all intermediate residuals:
    $$ x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) $$
    This contrasts with plain networks $x_L = \prod W_i x_0$, where multiplicative dynamics cause instability.

2.  **Backward Propagation (Gradient Flow)**:
    Consider the gradient of the loss $\mathcal{L}$ with respect to the input of the $l$-th layer $x_l$. Using the chain rule:
    $$ \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} $$
    From the additive formulation above, the term $\frac{\partial x_L}{\partial x_l}$ becomes:
    $$ \frac{\partial x_L}{\partial x_l} = \frac{\partial}{\partial x_l} \left( x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right) = I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) $$
    Combining these:
    $$ \frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_L}}_{\text{Direct Term}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_L} \left( \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)}_{\text{Residual Term}} $$

3.  **Implication**:
    *   The **"Direct Term"** ($\frac{\partial \mathcal{L}}{\partial x_L}$) propagates information directly from the loss to any layer $l$ without passing through the weights of the intermediate layers. The gradient flow is theoretically unimpeded (it does not vanish even if weights are small).
    *   This structure acts as a **Gradient Highway**, allowing gradients to flow effortlessly to earlier layers.
    *   Even if the weights in $\mathcal{F}$ are initialized to be small, the total gradient is dominated by the identity term $I$, ensuring learning can start effectively.

---

## 6. Neural ODEs

### Concept
Taking the layer depth limit $L \to \infty$ and step size $\Delta t \to 0$ of a ResNet leads to a continuous-depth model defined by an ODE.
$$ \frac{dz(t)}{dt} = f(z(t), t, \theta) $$
The output is the solution to the IVP (Initial Value Problem) at time $T$:
$$ z(T) = z(0) + \int_0^T f(z(t), t, \theta) dt $$

### Theoretical Background: Adjoint Sensitivity Method
Standard backpropagation would require storing all intermediate activations, leading to high memory cost. Neural ODEs use the **Adjoint Sensitivity Method** to compute gradients with constant memory cost $O(1)$.
*   **Adjoint State**: Define $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$. Its dynamics are given by another ODE running backwards in time:
    $$ \frac{da(t)}{dt} = -a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial z} $$
*   **Gradient Computation**:
    $$ \frac{d\mathcal{L}}{d\theta} = -\int_T^0 a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt $$

---

## 7. RNN, LSTM, GRU & Seq2Seq

### RNN (Recurrent Neural Networks)
Processes sequence data $x_1, \dots, x_T$ by maintaining a hidden state $h_t$ that acts as memory.

*   **Forward Dynamics**:
    $$ h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$
    $$ y_t = W_{hy}h_t + b_y $$

*   **BPTT (Backpropagation Through Time)**:
    Training involves unrolling the network over $T$ time steps. The total loss is $L = \sum_{t=1}^T L_t$.
    Gradients are accumulated backward from $t=T$ to $1$.

*   **Theoretical Issue: Vanishing/Exploding Gradients**:
    Detailed analysis of the gradient flow:
    $$ \frac{\partial h_T}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^T W_{hh}^T \text{diag}(\tanh'(z_t)) $$
    Since $\|\tanh'\| \le 1$, if the largest singular value of $W_{hh}$ is $< 1$, the gradient shrinks exponentially ($a^T \to 0$), making it impossible to learn long-term dependencies. If $> 1$, it explodes ($a^T \to \infty$).

*   **Turing Completeness**: RNNs are **Turing Complete** (Siegelmann & Sontag, 1995). With infinite precision and time, they can simulate any algorithm.

### LSTM (Long Short-Term Memory)
Designed to solve the vanishing gradient problem by introducing a **Cell State** $C_t$ that allows gradients to flow unchanged.

*   **Architecture (Gates)**:
    1.  **Forget Gate**: What to discard from old memory?
        $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
    2.  **Input Gate**: What new information to store?
        $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
        $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad (\text{Candidate update}) $$
    3.  **Cell Update**: Additive interaction (Key feature!).
        $$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
    4.  **Output Gate**: Determine next hidden state.
        $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
        $$ h_t = o_t \odot \tanh(C_t) $$

*   **Gradient Highway (Constant Error Carousel)**:
    Looking at the gradient of the cell state:
    $$ \frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots $$
    If $f_t \approx 1$ (remember), the gradient passes through as identity, preserving error signals over long distances.

### GRU (Gated Recurrent Unit)
A simplified variant of LSTM that merges cell and hidden states.
*   **Equations**:
    $$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad (\text{Update Gate}) $$
    $$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad (\text{Reset Gate}) $$
    $$ \tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) $$
    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$
*   **Pros**: Fewer parameters than LSTM, often comparable performance.

### Seq2Seq (Sequence-to-Sequence) / Encoder-Decoder
Framework for mapping sequence $X$ to sequence $Y$ (e.g., Translation).

*   **Encoder**: Processes input sequence $x_1, \dots, x_N$ into a fixed-length **Context Vector** $v$ (usually the final hidden state $h_N$).
*   **Decoder**: Generates output sequence $y_1, \dots, y_M$ from $v$.
    $$ h^{dec}_t = \text{RNN}(h^{dec}_{t-1}, y_{t-1}, v) $$
*   **Teacher Forcing**:
    During training, the *ground truth* token $y_{t-1}^*$ is fed as input to step $t$, rather than the model's own prediction $\hat{y}_{t-1}$. This stabilizes training but can lead to "Exposure Bias" (model relies on perfect past during training but fails with its own errors during inference).
*   **Bottleneck Problem**: Compressing the entire source sentence info into a single vector $v$ limits performance for long sentences. This motivated the **Attention Mechanism**.

---

## 8. Attention & Transformer

### Scaled Dot-Product Attention
The core mechanism that maps a query and a set of key-value pairs to an output.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
*   **Inputs**: $Q$ (Query), $K$ (Key), $V$ (Value) are matrices where rows correspond to sequence tokens.
*   **Scaling Factor $\frac{1}{\sqrt{d_k}}$**: Prevents the dot products $QK^T$ from growing too large in magnitude. Large values push the Softmax function into regions with extremely small gradients (vanishing gradients), scaling helps maintain variance around 1.

### Multi-Head Attention (MHA)
Instead of a single attention function, we project queries, keys, and values $h$ times with different, learned linear projections to $d_k, d_k, d_v$ dimensions.
$$ \text{Head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_h)W^O $$
*   **Benefit**: Allows the model to jointly attend to information from different representation subspaces at different positions.

### Transformer Block Architecture
A Transformer is composed of a stack of identical layers. Each layer has two sub-layers:
1.  **Multi-Head Self-Attention**:
    $$ y_1 = \text{LayerNorm}(x + \text{MultiHead}(x, x, x)) $$
2.  **Position-wise Feed-Forward Network (FFN)**:
    Applied to each position separately and identically. Usually two linear transformations with a ReLU/GELU activation in between.
    $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
    $$ y_2 = \text{LayerNorm}(y_1 + \text{FFN}(y_1)) $$
*   **Residual Connections & LayerNorm**: Crucial for training deep transformers. $\text{LayerNorm}(x + \text{Sublayer}(x))$.
*   **Positional Encoding**: Injected at the input (via addition) to provide order information, since Attention is permutation invariant.

### FlashAttention (IO-Aware Exact Attention)
Standard Attention implementation requires $O(N^2)$ memory to store the attention matrix $A = (QK^T)$, which often exceeds GPU high-bandwidth memory (HBM).
*   **Problem**: Materializing the $N \times N$ matrix $S = QK^T$ and $P = \text{softmax}(S)$ in HBM is slow due to memory bandwidth constraints.
*   **FlashAttention (Dao et al., 2022)**:
    *   **Tiling**: Computes attention in blocks (tiles) using on-chip SRAM (fast cache), never writing the full $N \times N$ matrix to HBM.
    *   **Recomputation**: Does not store the attention matrix for the backward pass; instead, it recomputes it on-the-fly.
    *   **Result**: Exact same output as standard attention, providing significant speedup (2-4x) and linear memory complexity with respect to sequence length in HBM, enabling much longer context windows.

---

## 9. GNN (Graph Neural Networks)

### Definition & Framework (MPNN)
Operates on data represented as graphs $G=(V, E)$. The core mechanism is **Message Passing**.
$$ h_v^{(k)} = \text{UPDATE}^{(k)} \left( h_v^{(k-1)}, \text{AGGREGATE}^{(k)}\left( \{ h_u^{(k-1)} \mid u \in \mathcal{N}(v) \} \right) \right) $$
*   **Permutation Equivariance**: The output $f(PAP^T, PX) = P f(A, X)$ ensures predictions heavily depend on graph structure, not node ordering.
*   **Inductive Bias**: Relational inductive bias, assuming dependencies exist defined by edges.

### Common Architectures

1.  **GCN (Graph Convolutional Network)** (Kipf & Welling, 2017)
    *   **Mechanism**: Approximates spectral graph convolutions (Chebyshev polynomials) to a simple first-order neighborhood average.
    *   **Update Rule**:
        $$ H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) $$
        where $\tilde{A} = A + I$ (self-loops) and $\tilde{D}$ is the degree matrix of $\tilde{A}$.
    *   **Insight**: Effectively performs a weighted average of neighbor features (smoothing), acting as a low-pass filter on graph signals.

2.  **GAT (Graph Attention Network)** (Veličković et al., 2018)
    *   **Mechanism**: Assigns different importance weights $\alpha_{ij}$ to different neighbors using Self-Attention.
    *   **Update Rule**:
        $$ h_v' = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u \right), \quad \alpha_{vu} = \text{softmax}(\text{LeakyReLU}(a^T [Wh_v \| Wh_u])) $$
    *   **Insight**: Allows the model to focus on relevant neighbors, improving expressivity over fixed GCN weights. Supports heterogeneous graphs implicitly.

3.  **GIN (Graph Isomorphism Network)** (Xu et al., 2019)
    *   **Mechanism**: Uses **Sum** aggregation instead of Mean/Max to preserve structural information.
    *   **Update Rule**:
        $$ h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right) $$
    *   **Theory**: Proven to be as powerful as the **Weisfeiler-Lehman (1-WL) test** for graph isomorphism, whereas GCN/GraphSAGE are strictly less powerful.

### Challenges
*   **Oversmoothing**: In Deep GCNs, repeated averaging causes node representations to converge to the same value (indistinguishable), limiting depth.
*   **Over-squashing**: Information from distant nodes must be compressed into fixed-size vectors through a "bottleneck", losing information (critical in graphs with exponential expansion like trees).

### Inference Examples
1.  **Node Classification**: Classifying documents in a citation network or fraud detection in transaction graphs.
    *   *Input*: Features of all nodes + Partial labels. *Output*: Labels for unlabeled nodes.
2.  **Graph Classification**: Predicting molecular properties (toxicity, solubility).
    *   *Mechanism*: Use a **Readout Function** (Global Pooling) like Sum or Mean over all node embeddings to get a graph vector $h_G$.
3.  **Link Prediction**: Recommender systems (User-Item Bipartite Graph).
    *   *Mechanism*: Predict if an edge exists between user $u$ and item $i$ by computing score $s = h_u^T h_i$.
4.  **Heterogeneous/Multi-Graph Inference**: Knowledge Graphs with multiple edge types.
    *   *Mechanism*: Use Relational GCN (R-GCN) where weights $W_r$ are specific to edge type $r$.

---

## 10. Optimal Transport (OT)

### Concept
Provides a geometrically meaningful distance defining the "cost" to transform one probability distribution into another.

### Theoretical Definitions
*   **Wasserstein Distance (Earth Mover's Distance)**:
    $$ W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|] $$
    Provides a weak topology where distributions with disjoint supports (common in high dimensions) still have meaningful distances and gradients, unlike Kullback-Leibler.
*   **Kantorovich-Rubinstein Duality**:
    $$ W_1(P, Q) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)] $$

---

## 11. KAN (Kolmogorov-Arnold Networks)

### Theoretical Foundation: Kolmogorov-Arnold Representation Theorem
The theorem (1957) states that any multivariate continuous function $f(x_1, \dots, x_n)$ on a bounded domain can be represented as:
$$ f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^n \psi_{p,q}(x_p) \right) $$
where $\Phi_q$ and $\psi_{p,q}$ are continuous single-variable functions.

### KAN Architecture (Liu et al., 2024)
Inspired by this theorem, KANs differ fundamentally from MLPs.
*   **MLP**: Activate($\sum$ Weight $\times$ Input). (Fixed Activation, Learnable Weights).
*   **KAN**: $\sum$ LearnableFunction(Input). (Learnable Activation, Fixed Weights/Sum).

### Key Differences & Mechanism
1.  **Edges as Functions**: In KANs, the "activation functions" are placed on the **edges** (arrows) of the graph, not the nodes.
    $$ x^{(l+1)}_j = \sum_{i=1}^{N_l} \phi_{l, j, i}(x^{(l)}_i) $$
    where $\phi$ is a learnable 1D function parametrized by **B-splines**.
2.  **No Linear Weights**: There are no traditional weight matrices $W$. The "weights" are the coefficients of the spline basis functions.
3.  **Interpretability**: KANs can often be symbolically pruned to reveal the exact mathematical formula governing the data (e.g., discovering $f(x,y) = \sin(x) + y^2$).
4.  **Scaling Laws**: KANs have shown faster scaling properties (Accuracy vs Parameters) than MLPs for certain scientific / PDE solving tasks.
