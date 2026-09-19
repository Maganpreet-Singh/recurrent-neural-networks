# 🔄 Recurrent Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-RNN-blue?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.20.0-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

<p align="center">
  <strong>A hands-on exploration of Recurrent Neural Networks, from mathematical forward propagation to a practical TensorFlow/Keras sentiment-classification model.</strong>
</p>

---

## 📌 Overview

This repository is a focused deep-learning study of **Recurrent Neural Networks (RNNs)**.

The current implementation demonstrates how an RNN processes a sequence one time step at a time, maintains a hidden state, and transforms the final hidden representation into a binary prediction.

The notebook begins with a small, interpretable vocabulary and a toy set of movie-review sequences. Instead of immediately hiding the mechanics behind a framework abstraction, it first builds the core computation manually with **NumPy** and then connects those concepts to **TensorFlow/Keras**.

The result is a compact but highly educational progression:

```
Words
  ↓
Integer / One-Hot Encoding
  ↓
Sequence Representation
  ↓
RNN Hidden-State Updates
  ↓
Final Hidden State
  ↓
Output Layer
  ↓
Sigmoid Probability
  ↓
Binary Prediction
```

---

## ✨ What This Repository Covers

### 🧠 Core RNN Concepts

- Sequential data representation
- Vocabulary creation
- One-hot encoding
- Time steps
- Hidden states
- Initial hidden state
- Input-to-hidden weights
- Hidden-to-hidden recurrent weights
- Bias terms
- Hidden-state recurrence
- `tanh` activation
- Output projection
- Sigmoid activation
- Binary cross-entropy
- Parameter counting

### ⚙️ Framework Implementation

- `tf.keras.layers.SimpleRNN`
- `tf.keras.layers.Dense`
- `tf.keras.layers.Embedding`
- Adam optimizer
- Binary cross-entropy loss
- Accuracy metric
- Training and evaluation
- Sequence-level prediction
- Model saving with TensorFlow SavedModel

### 🔍 Practical Learning

The notebook explicitly compares the conceptual RNN equations with a working neural-network implementation, making it useful for understanding not only **how to use RNNs**, but also **what happens inside them**.

---

## 📁 Repository Structure

```text
recurrent-neural-networks/
│
├── RNN_Forward_Propagation.ipynb
│   └── Manual RNN mathematics + TensorFlow/Keras implementation
│
└── README.md
```

---

## 📓 Main Notebook

### `RNN_Forward_Propagation.ipynb`

This notebook is the core of the repository.

It walks through a complete miniature RNN pipeline using a six-example toy sentiment dataset.

### Main stages

```text
1. Import dependencies
2. Set reproducible random seeds
3. Define a small vocabulary
4. One-hot encode vocabulary words
5. Build labeled sequences
6. Inspect sequence dimensions
7. Define RNN parameters
8. Calculate parameter count
9. Perform manual forward propagation
10. Track hidden states
11. Compute output probability
12. Calculate binary cross-entropy
13. Build a Keras SimpleRNN
14. Train the network
15. Inspect sequence outputs
16. Introduce Embedding
17. Train an embedding + RNN model
18. Evaluate predictions
19. Save the trained TensorFlow model
```

---

# 🧩 1. Problem Setup

The notebook uses a deliberately small vocabulary:

```text
movie
actor
good
bad
not
```

The vocabulary contains:

```text
Vocabulary size = 5
Time steps = 3
Input features = 5
Hidden units = 3
```

Each review contains exactly three tokens.

Example:

```text
["movie", "good", "actor"]
```

The toy dataset contains six labeled sequences:

| Sequence | Label |
|---|---:|
| movie good actor | 1 |
| movie bad actor | 0 |
| not good movie | 1 |
| movie not bad | 0 |
| good movie actor | 1 |
| bad movie actor | 0 |

Here:

- `1` represents the positive class.
- `0` represents the negative class.

> **Important:** This is a deliberately tiny educational dataset. The reported training performance demonstrates that the model can fit this toy dataset; it should not be interpreted as a general sentiment-analysis benchmark.

---

# 🔢 2. One-Hot Encoding

Each word is represented as a vector of length 5.

For example:

```text
movie → [1, 0, 0, 0, 0]
actor → [0, 1, 0, 0, 0]
good  → [0, 0, 1, 0, 0]
bad   → [0, 0, 0, 1, 0]
not   → [0, 0, 0, 0, 1]
```

A three-word sequence therefore becomes a matrix with shape:

```text
(3, 5)
```

Across all six training examples:

```text
X.shape = (6, 3, 5)
```

This introduces the central sequence-learning structure:

[
	ext{samples} 	imes 	ext{timesteps} 	imes 	ext{features}
]

---

# 🔁 3. Understanding the RNN State

The recurrent computation is represented by:

[
h_t = 	anh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
]

Where:

| Symbol | Meaning |
|---|---|
| (x_t) | Input at time step (t) |
| (h_{t-1}) | Previous hidden state |
| (h_t) | Current hidden state |
| (W_{xh}) | Input-to-hidden weights |
| (W_{hh}) | Hidden-to-hidden recurrent weights |
| (b_h) | Hidden-layer bias |

The final output is:

[
y = h_T W_{hy} + b_y
]

followed by sigmoid:

[
hat{y} = sigma(y)
]

The key idea is simple:

> **The current hidden state is influenced by both the current input and information carried from previous time steps.**

---

# 🧮 4. Manual RNN Forward Propagation

One of the strongest parts of this repository is the explicit implementation of the RNN equations.

The notebook defines:

```text
W_xh
W_hh
b_h
W_hy
b_y
```

For the current configuration, the manual recurrent component contains **31 parameters**.

The parameter breakdown is:

| Parameter group | Count |
|---|---:|
| Input → hidden | 15 |
| Hidden → hidden | 9 |
| Hidden bias | 3 |
| Hidden → output | 3 |
| Output bias | 1 |
| **Total** | **31** |

The general binary-output parameter formula used here is:

[
D H + H^2 + H + H + 1
]

where:

- (D) = input feature dimension
- (H) = hidden-unit count

or equivalently:

[
D H + H^2 + 2H + 1
]

---

# 🧠 5. Hidden-State Evolution

For the example sequence:

```text
movie → good → actor
```

the notebook initializes:

[
h_0 = [0,0,0]
]

Then computes:

[
h_1 = f(x_1,h_0)
]

[
h_2 = f(x_2,h_1)
]

[
h_3 = f(x_3,h_2)
]

The sequence is therefore processed sequentially rather than as three independent inputs.

This is the defining characteristic of recurrent computation.

---

# 📤 6. Output Prediction

After processing the complete sequence, the notebook uses the final hidden state to produce an output score.

Conceptually:

[
z = h_3 W_{hy} + b_y
]

and:

[
hat{y} = rac{1}{1+e^{-z}}
]

The resulting value lies between 0 and 1 and can be interpreted as the model's estimated probability for the positive class.

A threshold of 0.5 is used for binary prediction:

```text
prediction >= 0.5 → class 1
prediction < 0.5  → class 0
```

---

# 📉 7. Binary Cross-Entropy

The notebook also implements the binary cross-entropy loss directly:

[
L = -[ylog(hat{y}) + (1-y)log(1-hat{y})]
]

The implementation clips predictions before taking logarithms to avoid numerical problems near 0 and 1.

This makes the notebook useful for connecting:

```text
prediction
   ↓
probability
   ↓
loss
   ↓
training objective
```

---

# 🏗️ 8. TensorFlow/Keras SimpleRNN

After the manual implementation, the notebook uses TensorFlow/Keras to construct an equivalent recurrent architecture.

Conceptually:

```text
Input
  ↓
SimpleRNN
  ↓
Dense
  ↓
Sigmoid
  ↓
Binary output
```

The model uses:

```python
tf.keras.layers.SimpleRNN
```

with a `tanh` activation and a dense sigmoid output.

This gives a practical bridge between the mathematics and a production-style deep-learning API.

---

# 📊 9. Sequence Output

The notebook also demonstrates `return_sequences=True`.

Instead of returning only the final hidden state, the model produces an output for every time step:

```text
time step 1 → output
time step 2 → output
time step 3 → output
```

For a three-step sequence, the resulting shape is:

```text
(batch_size, 3, 1)
```

This distinction is essential when working with:

- sequence classification
- sequence labeling
- many-to-many architectures
- stacked recurrent layers

---

# 🔤 10. Embeddings

The notebook then moves from manually created one-hot vectors to integer token representations.

Example representation:

```text
movie → 1
actor → 2
good  → 3
bad   → 4
not   → 5
```

An embedding layer converts these integer IDs into dense vectors.

The architecture becomes:

```text
Integer Token IDs
        ↓
Embedding
        ↓
SimpleRNN
        ↓
Dense
        ↓
Sigmoid
```

The practical model in the notebook uses:

- vocabulary size + padding index
- embedding dimension = 16
- SimpleRNN hidden size = 8
- Dense binary output

The resulting model contains **305 trainable parameters** under the demonstrated configuration.

---

# 🚀 11. Training

The practical Keras model is compiled with:

```text
Optimizer: Adam
Loss: Binary Cross-Entropy
Metric: Accuracy
```

Training is performed directly on the toy sequence dataset.

The notebook trains for:

```text
100 epochs
batch size = 2
```

Because the dataset is intentionally tiny, the model can fit it very easily.

The reported evaluation result reaches:

```text
Accuracy = 1.0
```

Again, this reflects fitting the **six-example toy dataset**, not performance on a real-world sentiment corpus.

---

# 🔍 12. Example Predictions

The trained practical model is tested on sequences such as:

```text
movie good actor
movie bad actor
not good movie
movie not bad
```

The notebook reports probabilities and binary predictions for these sequences.

Example behavior demonstrated by the saved run:

| Review | Probability | Prediction |
|---|---:|---:|
| movie good actor | 0.9177 | 1 |
| movie bad actor | 0.0696 | 0 |
| not good movie | 0.9583 | 1 |
| movie not bad | 0.0609 | 0 |

These values show that the trained model separates the toy positive and negative examples in this controlled setup.

---

# 💾 13. Model Saving

The trained model is exported using TensorFlow's SavedModel mechanism.

The notebook saves the model under:

```text
rnn_sentiment_model
```

This is a useful first step toward separating:

```text
training
   ↓
serialization
   ↓
deployment / inference
```

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations and manual RNN computation |
| TensorFlow | Deep-learning framework |
| Keras | High-level neural-network API |
| Matplotlib | Visualization support |
| Google Colab | Notebook execution environment |
| T4 GPU | Runtime accelerator configured in Colab |

The notebook records TensorFlow version **2.20.0** in its execution environment.

---

# 🧪 Learning Objectives

After working through this repository, you should be able to explain:

### Fundamentals

- What sequential data is
- Why ordinary dense networks do not naturally model order
- What a hidden state represents
- Why recurrent weights are reused across time

### Mathematics

- RNN recurrence equations
- Input-to-hidden transformation
- Hidden-to-hidden transformation
- Output projection
- `tanh` activation
- Sigmoid probabilities
- Binary cross-entropy

### Implementation

- Build sequence tensors
- Encode tokens
- Implement an RNN step manually
- Track hidden states
- Count trainable parameters
- Build a Keras `SimpleRNN`
- Use `Embedding`
- Train and evaluate a recurrent model
- Save a trained TensorFlow model

---

# 📐 Tensor Shapes

Understanding shapes is critical when working with RNNs.

For the manual one-hot pipeline:

```text
X
└── (6, 3, 5)

6  = samples
3  = time steps
5  = input features
```

For a recurrent hidden representation:

```text
hidden state
└── (hidden_units,)
```

For the sequence-returning Keras model:

```text
output
└── (batch_size, time_steps, features)
```

These dimensions become even more important when moving to LSTM, GRU, bidirectional networks, and attention-based architectures.

---

# 🧭 Conceptual Roadmap

This repository currently establishes the fundamentals of vanilla RNNs.

A natural progression from here is:

```text
Vanilla RNN
   ↓
Backpropagation Through Time
   ↓
Vanishing / Exploding Gradients
   ↓
LSTM
   ↓
GRU
   ↓
Bidirectional RNNs
   ↓
Sequence-to-Sequence Models
   ↓
Attention
   ↓
Transformers
```

---

# ⚠️ Important Limitations

This repository is primarily an educational implementation.

The current notebook uses:

- a tiny hand-crafted vocabulary
- six toy training examples
- fixed-length sequences of three tokens
- a binary sentiment-style label
- manually chosen recurrent weights for the forward-pass demonstration

Therefore, it is **not** intended to represent a production sentiment-analysis system.

For a more realistic benchmark, the project could later use a larger dataset with:

- train/validation/test splits
- padding and masking
- larger vocabularies
- unknown-token handling
- regularization
- dropout
- hyperparameter tuning
- confusion matrices
- precision, recall, and F1
- systematic error analysis

---

# 💡 Why Start With a Manual RNN?

Framework APIs are powerful, but they can hide the mechanics.

A manual forward pass exposes the internal flow:

```text
x₁ ──┐
     ↓
    h₁ ──┐
         ↓
x₂ ────> h₂ ──┐
              ↓
x₃ ─────────> h₃
               ↓
             output
```

That makes it much easier to understand why an RNN is called **recurrent**.

The hidden state is carried forward through the sequence, allowing the network to maintain a learned representation of previous inputs.

---

# 🔥 Key Takeaways

> **An RNN does not process every time step in isolation.**

The model repeatedly applies the same recurrent transformation while updating its hidden state.

The essential recurrence is:

[
h_t = 	anh(x_tW_{xh} + h_{t-1}W_{hh}+b_h)
]

The final hidden state can then feed an output layer for classification.

The practical notebook demonstrates the complete journey:

```text
Token
  ↓
Encoding
  ↓
Sequence
  ↓
Hidden State
  ↓
Recurrent Computation
  ↓
Output Probability
  ↓
Loss
  ↓
Training
  ↓
Prediction
```

That progression is the real purpose of this repository: **understand the mechanism before abstracting it behind a library.**

---

# ▶️ How to Run

## Option 1 — Google Colab

Open:

[`RNN_Forward_Propagation.ipynb`](./RNN_Forward_Propagation.ipynb)

Then open it in Google Colab and run the notebook from top to bottom.

The notebook was executed in a Colab environment with a T4 GPU configuration.

## Option 2 — Local Jupyter Environment

Clone the repository:

```bash
git clone https://github.com/Maganpreet-Singh/recurrent-neural-networks.git
cd recurrent-neural-networks
```

Install dependencies:

```bash
pip install numpy tensorflow matplotlib
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
RNN_Forward_Propagation.ipynb
```

and execute the cells sequentially.

---

# 🗂️ Suggested Future Repository Structure

As this learning project grows, the repository can evolve into:

```text
recurrent-neural-networks/
│
├── notebooks/
│   ├── 01_rnn_forward_propagation.ipynb
│   ├── 02_rnn_backpropagation.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_gru.ipynb
│   └── 05_text_classification.ipynb
│
├── src/
│   ├── rnn.py
│   ├── lstm.py
│   ├── gru.py
│   └── preprocessing.py
│
├── models/
│
├── data/
│
├── README.md
└── requirements.txt
```

This would make the project easier to navigate as more architectures are added.

---

# 📚 Recommended Next Topics

Once vanilla RNN forward propagation is comfortable, study:

### 1. Backpropagation Through Time

Understand how gradients flow across multiple time steps.

### 2. Vanishing and Exploding Gradients

Understand one of the major limitations of vanilla recurrent networks.

### 3. LSTM

Study:

- forget gate
- input gate
- output gate
- cell state

### 4. GRU

Compare GRU's simplified gating mechanism with LSTM.

### 5. Bidirectional RNNs

Learn how information can be processed from both directions.

### 6. Sequence-to-Sequence Models

Move from single-output classification to sequence generation and transformation.

### 7. Attention and Transformers

Understand why modern sequence architectures increasingly rely on attention mechanisms rather than recurrence.

---

# 🤝 Contributions

This repository is primarily a personal learning and experimentation project.

Suggestions, corrections, improvements, and educational extensions are welcome.

A useful contribution should ideally improve one of the following:

- mathematical clarity
- implementation quality
- reproducibility
- documentation
- visualization
- model experimentation
- dataset realism

---

# 📜 License

No explicit license file is currently included in the repository.

Until a license is added, the repository should be treated according to the default copyright rules that apply to the published source.

---

# 👨‍💻 Author

**Maganpreet Singh**

Computer Science & Engineering student focused on:

```text
Python
Data Science
Machine Learning
Deep Learning
Computer Vision
Natural Language Processing
```

GitHub:  
https://github.com/Maganpreet-Singh

---

# ⭐ Repository Philosophy

This project follows a simple principle:

> **Learn the mathematics → implement the idea → use the framework → build something real.**

Deep learning becomes much less mysterious when the tensors, equations, activations, hidden states, and gradients stop being black boxes.

This repository is one step in that progression.

---

<p align="center">
  <strong>🔄 From Sequence → State → Prediction</strong>
</p>

<p align="center">
  Built with Python, NumPy, TensorFlow, and curiosity.
</p>
