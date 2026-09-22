# 🔄 Recurrent Neural Networks Lab

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Recurrent%20Neural%20Networks-7c3aed?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p align="center">
  <strong>From recurrent equations to GRU classification and LSTM language modeling.</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-repository-structure">Structure</a> •
  <a href="#-learning-path">Learning Path</a> •
  <a href="#-experiments">Experiments</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🧭 Overview

This repository is a hands-on study of sequence modeling with recurrent neural networks.

It is organized as a progression rather than a pile of unrelated notebooks:

~~~text
RNN fundamentals
      ↓
Forward propagation
      ↓
Token representations
      ↓
Embeddings
      ↓
Bidirectional recurrence
      ↓
Deep recurrent architectures
      ↓
GRU sentiment analysis
      ↓
LSTM language modeling
      ↓
Next-word prediction
      ↓
Autoregressive text generation
      ↓
Attention / Transformers
~~~

The central idea is simple:

> A sequence model must represent not just the current input, but the information that has arrived before it.

A vanilla recurrent neural network does this with a hidden state:

h_t = φ(W_xh x_t + W_hh h_(t-1) + b_h)

The exact notation can vary, but the concept is stable: the current state depends on the current input and the previous state.

This repository deliberately starts close to the mathematics. Later notebooks then move into practical TensorFlow/Keras implementations and NLP applications.

The result is intended to be useful in three ways:

1. as a learning curriculum,
2. as a reference for recurrent-network implementation,
3. as a portfolio project showing progression from fundamentals to applied deep learning.

---

## 🎯 Project Objectives

The repository has several connected objectives.

### Understand recurrent computation

Learn why recurrence exists, how state is updated, how information flows through time steps, and how recurrent parameters are reused.

### Understand representations

Move from symbolic tokens and one-hot vectors to integer token IDs and learned embeddings.

### Understand architectural families

Study the relationship between vanilla RNNs, bidirectional RNNs, LSTMs, and GRUs.

### Build NLP systems

Use recurrent architectures for sequence classification and next-token prediction.

### Learn evaluation

Understand why training loss alone is not enough. Inspect validation behavior, accuracy, top-k accuracy, perplexity, prediction distributions, and generated text where appropriate.

### Learn model engineering

Save trained models, preserve preprocessing mappings, manage dependencies, and organize experiments so another person can understand what was done.

---

# 📁 Repository Structure

The repository is organized around the following portfolio structure:

~~~text
recurrent-neural-networks/
│
├── notebooks/
│   ├── 01_rnn_forward_propagation.ipynb
│   ├── 02_rnn_embedding.ipynb
│   ├── 03_bidirectional_rnn.ipynb
│   ├── 04_deep_rnn_lstm_gru.ipynb
│   ├── 05_gru_sentiment_analysis.ipynb
│   └── 06_lstm_next_word_predictor.ipynb
│
├── models/
│   └── gru_sentiment_model.keras
│
├── docs/
│   └── learning-guide.md
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── LICENSE
~~~

The notebook files are numbered intentionally.

The number is not a claim that one architecture is universally more important than another. It simply establishes a recommended reading order.

The project can later grow into a larger machine-learning engineering layout:

~~~text
recurrent-neural-networks/
│
├── notebooks/
│   ├── fundamentals/
│   ├── architectures/
│   ├── sentiment/
│   └── language_modeling/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── configs/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
├── scripts/
├── docs/
├── requirements.txt
├── README.md
└── .github/
~~~

That second structure is the long-term target for turning the learning project into a reusable ML codebase.

---

# 🧹 File Migration Map

The cleaned repository maps the original files as follows:

| Original file | New location |
|---|---|
| RNN_Forward_Propagation.ipynb | notebooks/01_rnn_forward_propagation.ipynb |
| rnn_embedding.ipynb | notebooks/02_rnn_embedding.ipynb |
| bidirectional_rnn.ipynb | notebooks/03_bidirectional_rnn.ipynb |
| Deep_RNN_LSTM_GRU.ipynb | notebooks/04_deep_rnn_lstm_gru.ipynb |
| GRU_Sentiment_Analysis.ipynb | notebooks/05_gru_sentiment_analysis.ipynb |
| LSTM_Next_Word_Predictor.ipynb | notebooks/06_lstm_next_word_predictor.ipynb |
| gru_sentiment_model.keras | models/gru_sentiment_model.keras |

The goal is to make the repository understandable without opening any notebook first.

A visitor should be able to answer three questions immediately:

- What is this project?
- What experiments exist?
- Where do I start?

---

# 🗺️ Learning Path

The recommended path is:

~~~text
01 → 02 → 03 → 04 → 05 → 06
~~~

### Stage 01 — RNN Forward Propagation

Start with equations, hidden states, recurrent weights, output computation, and parameter counting.

### Stage 02 — Embeddings

Replace sparse token representations with trainable dense vectors.

### Stage 03 — Bidirectional RNNs

Explore information flow in both temporal directions.

### Stage 04 — Deep RNN / LSTM / GRU

Move from the basic recurrent cell to gated and stacked recurrent architectures.

### Stage 05 — GRU Sentiment Analysis

Apply recurrent modeling to an NLP classification problem.

### Stage 06 — LSTM Next-Word Prediction

Work on a larger language-modeling pipeline with WikiText-103, vocabulary construction, context windows, next-token prediction, sampling, and text generation.

---

# 🔁 1. Recurrent Neural Network Fundamentals

A recurrent neural network processes a sequence one step at a time.

For a sequence:

~~~text
x₁, x₂, x₃, ..., x_T
~~~

the model repeatedly updates a state:

~~~text
h₀
 ↓
x₁ → h₁
       ↓
x₂ → h₂
       ↓
x₃ → h₃
       ↓
...
       ↓
x_T → h_T
~~~

The same recurrent transformation is reused at every step.

This gives the architecture a useful property: the model can process sequences whose temporal structure is not naturally represented by a single independent feature vector.

The hidden state is not a perfect memory.

It is a learned representation with limited capacity.

That distinction matters.

An RNN does not literally store a clean transcript of everything it has seen. It repeatedly compresses information into its current state representation.

---

## 🧮 Vanilla RNN Equation

A common formulation is:

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

Here:

| Component | Meaning |
|---|---|
| x_t | Current input |
| h_(t-1) | Previous hidden state |
| h_t | Current hidden state |
| W_xh | Input-to-hidden weights |
| W_hh | Recurrent weights |
| b_h | Bias |
| tanh | Nonlinear activation |

The recurring term is:

W_hh h_(t-1)

That term is what carries information from one step into the next.

---

# 🧩 2. Forward Propagation

The first notebook is designed to make the recurrence visible.

It uses a deliberately small vocabulary and a tiny sequence dataset so the reader can inspect the data directly.

The conceptual pipeline is:

~~~text
tokens
  ↓
encoding
  ↓
sequence tensor
  ↓
initial hidden state
  ↓
recurrent updates
  ↓
final hidden state
  ↓
output projection
  ↓
sigmoid
  ↓
binary prediction
~~~

The notebook demonstrates the mechanics with a small vocabulary containing:

~~~text
movie
actor
good
bad
not
~~~

and a compact collection of fixed-length examples.

This is an educational setup, not a realistic sentiment benchmark.

Its value comes from transparency.

When the number of examples is tiny, every tensor and equation can be inspected.

---

## 🔢 One-Hot Encoding

With a vocabulary size of five, each token can be represented with five values.

Example:

~~~text
movie → [1, 0, 0, 0, 0]
actor → [0, 1, 0, 0, 0]
good  → [0, 0, 1, 0, 0]
bad   → [0, 0, 0, 1, 0]
not   → [0, 0, 0, 0, 1]
~~~

A three-token sequence therefore has shape:

~~~text
(3, 5)
~~~

A batch of six samples has:

~~~text
(6, 3, 5)
~~~

The dimensions represent:

~~~text
samples × time steps × features
~~~

Shape literacy becomes increasingly important as the project moves into embeddings, stacked LSTMs, and language modeling.

---

## 🧮 Parameter Counting

For a simple binary-output RNN with input dimension D and hidden dimension H, the parameter count can be expressed as:

D·H + H² + 2H + 1

The terms correspond to:

- input-to-hidden weights
- hidden-to-hidden recurrent weights
- hidden bias
- hidden-to-output weights
- output bias

For the small configuration used by the original educational notebook:

| Parameter group | Count |
|---|---:|
| Input → hidden | 15 |
| Hidden → hidden | 9 |
| Hidden bias | 3 |
| Hidden → output | 3 |
| Output bias | 1 |
| **Total** | **31** |

Parameter counting is useful because it gives a quick connection between architecture design and computational cost.

---

# 📤 3. From Hidden State to Prediction

Once the final hidden state has been computed, a dense layer can transform it into an output score.

The binary pipeline is:

~~~text
final hidden state
       ↓
linear output
       ↓
sigmoid probability
       ↓
class threshold
~~~

The sigmoid function is:

σ(z) = 1 / (1 + e^(-z))

which maps a real-valued score into the interval:

~~~text
0 → 1
~~~

Binary cross-entropy then connects the prediction to the training objective.

The important conceptual sequence is:

~~~text
model parameters
      ↓
hidden-state computation
      ↓
output score
      ↓
probability
      ↓
loss
      ↓
parameter update
~~~

---

# 🔤 4. Embeddings

One-hot encoding is excellent for teaching.

It is usually not the representation you want to build an NLP system around at scale.

The next stage therefore introduces embedding vectors.

The pipeline becomes:

~~~text
token
 ↓
integer ID
 ↓
embedding lookup
 ↓
dense vector
 ↓
RNN
~~~

An embedding layer can be thought of as a trainable matrix.

If the vocabulary contains V tokens and the embedding dimension is E, the embedding matrix has a conceptual shape of:

~~~text
(V, E)
~~~

Each token ID selects one row.

This is a simple operation, but it is a foundational idea in modern NLP.

Embeddings provide a compact continuous representation that a sequence model can process more efficiently than a very large one-hot vector.

---

## 🧠 Representation Learning

A crucial lesson is that embeddings are learned representations.

The model does not begin with a perfect dictionary saying which words are synonyms.

Instead, training updates the embedding vectors so that they become useful for the objective.

That means the resulting space depends on:

- data
- preprocessing
- architecture
- objective
- optimization
- regularization
- random initialization

Embeddings should therefore be understood as learned statistical representations rather than magical containers of human meaning.

---

# ↔️ 5. Bidirectional RNNs

A regular recurrent network processes a sequence in a single direction.

For a four-token sequence:

~~~text
t₁ → t₂ → t₃ → t₄
~~~

a bidirectional architecture also processes:

~~~text
t₄ → t₃ → t₂ → t₁
~~~

The two information streams can then be combined.

This creates representations informed by:

- left context
- right context

That can be useful for tasks where the entire sequence is already available.

Examples include:

- sequence classification
- sequence labeling
- contextual token representations
- certain NLP tagging tasks

However, bidirectionality is not automatically valid everywhere.

For causal forecasting, future observations may not be available when a prediction must be produced.

Architecture selection has to follow the information available at inference time.

---

# 🏗️ 6. Deep Recurrent Architectures

A single recurrent layer is not the end of recurrent modeling.

Multiple recurrent layers can be stacked:

~~~text
input
  ↓
RNN / LSTM / GRU
  ↓
RNN / LSTM / GRU
  ↓
output
~~~

This lets later layers operate on representations produced by earlier recurrent layers.

The benefit is additional representational capacity.

The cost is additional parameters, memory, and compute.

This is one of the recurring engineering trade-offs throughout deep learning:

~~~text
capacity ↑
compute ↑
memory ↑
risk of overfitting can ↑
~~~

More depth must earn its place through better validation behavior or useful capability.

---

# 🚪 7. LSTM

Long Short-Term Memory adds explicit mechanisms for controlling information flow.

A useful conceptual view is:

~~~text
previous cell state
        │
        ▼
   ┌──────────┐
   │   LSTM   │
   └──────────┘
     ▲      ▲
     │      │
 previous   input
 hidden
~~~

The architecture maintains:

- a cell state
- a hidden state

and uses gates to regulate information.

---

## Forget Gate

The forget gate controls how much previous cell-state information should be retained.

Conceptually:

~~~text
old memory
   ↓
keep some
discard some
~~~

## Input Gate

The input gate controls how much new information should be written into the cell state.

## Candidate State

A candidate update provides new information that may be integrated into memory.

## Output Gate

The output gate controls how much of the internal state becomes the new hidden representation.

The broad intuition is:

~~~text
remember
write
expose
~~~

This is more structured than the single-state update in a vanilla RNN.

---

# ⚡ 8. GRU

A Gated Recurrent Unit provides another gated recurrent design.

Its architecture is often described through update and reset mechanisms.

A conceptual view is:

~~~text
previous hidden state
        +
      input
        ↓
 gates control information
        ↓
 new hidden state
~~~

Compared with LSTM, GRU uses a simpler state structure.

That can reduce architectural complexity while still giving the network explicit mechanisms for controlling information flow.

Again, there is no universal law that one gated recurrent architecture must dominate another.

The correct comparison requires:

- the same task
- compatible preprocessing
- sensible validation
- comparable compute budgets
- meaningful metrics

---

# ❤️ 9. GRU Sentiment Analysis

The sentiment notebook moves the repository from concept demonstrations into a recognizable NLP application.

The high-level workflow is:

~~~text
raw text
  ↓
tokenization
  ↓
integer sequence
  ↓
padding / batching
  ↓
embedding
  ↓
GRU
  ↓
classification layer
  ↓
sentiment probability
~~~

The repository also stores a trained Keras model artifact:

~~~text
models/gru_sentiment_model.keras
~~~

Keeping the trained artifact separate from the notebook creates a useful project boundary:

~~~text
Notebook
    = how the model was built

Model artifact
    = trained result that can be loaded elsewhere
~~~

For a mature version of the project, the sentiment experiment should additionally report:

- train split size
- validation split size
- test split size
- class balance
- confusion matrix
- precision
- recall
- F1 score
- representative errors
- inference examples

A model that achieves high training accuracy can still generalize poorly.

Evaluation must therefore be separated from fitting.

---

# 📝 10. LSTM Next-Word Prediction

The language-modeling notebook is the most substantial experiment in the current repository.

It uses the WikiText-103 configuration:

~~~text
Salesforce/wikitext
wikitext-103-v1
~~~

The workflow covers far more than simply attaching an LSTM layer to some text.

It includes:

- dataset loading
- text cleaning
- split inspection
- line-length analysis
- token frequency analysis
- vocabulary construction
- PAD and UNK token handling
- token-to-ID conversion
- fixed-length context windows
- TensorFlow data pipelines
- embedding
- stacked LSTM layers
- normalization
- dropout
- AdamW optimization
- mixed precision when a GPU is available
- callbacks
- training history
- top-k accuracy
- validation and test evaluation
- perplexity
- next-word probability inspection
- top-k sampling
- temperature sampling
- autoregressive generation
- vocabulary serialization
- model serialization

This is a meaningful progression from the tiny hand-built RNN example.

---

# 📚 11. WikiText-103 Pipeline

The notebook begins with raw text lines.

Those lines are cleaned and transformed into token sequences.

The conceptual data pipeline is:

~~~text
WikiText lines
      ↓
clean text
      ↓
tokens
      ↓
frequency analysis
      ↓
vocabulary
      ↓
integer IDs
      ↓
context windows
      ↓
TensorFlow Dataset
~~~

The notebook uses a bounded token budget instead of blindly consuming all available text.

The demonstrated configuration includes:

| Setting | Value |
|---|---:|
| Maximum vocabulary | 30,000 |
| Context length | 32 |
| Embedding dimension | 256 |
| LSTM layer 1 | 384 units |
| LSTM layer 2 | 256 units |
| Dropout | 0.25 |
| Batch size | 256 |
| Epoch budget | 20 |
| Train token limit | 12,000,000 |
| Validation token limit | 600,000 |
| Test token limit | 600,000 |
| Minimum token frequency | 2 |

These settings are experimental configuration, not universal recommendations.

A different dataset, hardware budget, or objective can justify very different choices.

---

# 🪜 12. Context Windows

A language model trained for next-word prediction needs training examples of the form:

~~~text
input:
w₁ w₂ w₃ ... w_T

target:
w_(T+1)
~~~

Sliding this window across a token stream generates many supervised examples.

For context length T:

~~~text
tokens [0 : T]       → target token [T]
tokens [1 : T + 1]   → target token [T + 1]
tokens [2 : T + 2]   → target token [T + 2]
...
~~~

This converts an unsupervised-looking text stream into a supervised next-token objective.

That transformation is one of the most important ideas in practical language modeling.

---

# 🧱 13. Stacked LSTM Architecture

The WikiText notebook constructs a model conceptually similar to:

~~~text
Token IDs
    ↓
Embedding
    ↓
Layer Normalization
    ↓
LSTM, return sequences
    ↓
Layer Normalization
    ↓
LSTM, final state
    ↓
Layer Normalization
    ↓
Dropout
    ↓
Dense vocabulary projection
    ↓
Next-token logits
~~~

The first LSTM returns a sequence so that the second recurrent layer can continue processing temporal information.

The second LSTM returns a final representation.

The dense layer maps that representation into a vocabulary-sized output.

The output is a vector of logits rather than a single binary probability.

---

# 🎯 14. Next-Token Prediction

For a vocabulary of size V, the model produces V logits.

Those logits are converted into a probability distribution with softmax:

p_i = exp(z_i) / sum_j exp(z_j)

The result represents the model's belief over possible next tokens.

For example:

~~~text
candidate token     probability
--------------------------------
word A                 0.41
word B                 0.23
word C                 0.12
word D                 0.08
...
~~~

The model is not required to believe only one token is possible.

It represents a distribution.

That distribution becomes especially important during generation.

---

# 📉 15. Cross-Entropy

For next-token prediction, the model is trained to place probability mass on the observed target token.

Sparse categorical cross-entropy can be used when the target is represented as an integer class ID.

The general learning loop is:

~~~text
predict distribution
        ↓
compare with target token
        ↓
calculate loss
        ↓
backpropagate
        ↓
update weights
~~~

As training proceeds, the model attempts to reduce expected prediction error on the training distribution.

---

# 📊 16. Accuracy Metrics

The language-model notebook tracks:

- top-1 accuracy
- top-3 accuracy
- top-5 accuracy

### Top-1

The correct token must be the most probable token.

### Top-3

The correct token can appear anywhere among the three highest-probability candidates.

### Top-5

The correct token can appear anywhere among the five highest-probability candidates.

Top-k metrics are particularly informative for language models because natural language can be ambiguous.

Multiple words may be plausible continuations even when only one token appears in the evaluation corpus.

---

# 📐 17. Perplexity

The language model also calculates perplexity from cross-entropy loss.

With natural-log cross-entropy:

PPL = exp(loss)

Perplexity can be interpreted as an approximate measure of how uncertain the model is over the observed distribution.

Lower perplexity generally corresponds to assigning higher probability to observed targets, assuming the evaluation setups are comparable.

Important:

Perplexity is not directly comparable across arbitrary experiments.

Changes in:

- tokenizer
- vocabulary
- dataset
- preprocessing
- target construction
- evaluation split

can change the metric substantially.

---

# 🌡️ 18. Temperature

The notebook exposes temperature during sampling.

The conceptual transformation is:

p_i = softmax(z_i / T)

At lower temperature:

~~~text
distribution becomes sharper
high-probability choices dominate
generation becomes more conservative
~~~

At higher temperature:

~~~text
distribution becomes flatter
more candidates become viable
generation becomes more varied
~~~

Temperature changes decoding.

It does not retrain the network.

This distinction is important when debugging generative behavior.

---

# 🔝 19. Top-k Sampling

Top-k sampling limits the candidate vocabulary to the k highest-scoring tokens.

The general process is:

~~~text
model logits
     ↓
rank tokens
     ↓
keep top-k
     ↓
discard the rest
     ↓
renormalize
     ↓
sample
~~~

This prevents extremely low-probability tokens from entering the candidate set.

Top-k does not fix a badly trained model.

It simply changes the inference distribution.

---

# ✍️ 20. Autoregressive Generation

Generation is performed repeatedly.

Starting with a seed prompt:

~~~text
the future of science
~~~

the model predicts the next token.

That token is appended.

The updated context is then used to make the next prediction.

The loop becomes:

~~~text
seed
 ↓
predict
 ↓
append
 ↓
truncate context
 ↓
predict
 ↓
append
 ↓
repeat
~~~

This is called autoregressive generation because each new prediction depends on the sequence produced so far.

The process can continue for an arbitrary number of generation steps, subject to the context-window logic implemented by the notebook.

---

# 🧪 21. Reproducibility

Reproducibility is more than adding a random seed.

A useful experiment record should include:

~~~text
code version
dataset version
preprocessing
hyperparameters
random seeds
hardware
software versions
training procedure
evaluation procedure
model artifacts
~~~

The current language-model notebook explicitly sets seeds and records key configuration values.

For future versions, the project should move recurring configuration into reusable files rather than duplicating constants across notebooks.

---

# 💾 22. Model Artifacts

The repository currently includes a trained GRU model:

~~~text
models/gru_sentiment_model.keras
~~~

The language-model notebook also contains serialization logic for:

~~~text
wikitext103_lstm_next_word_predictor.keras
wikitext103_word_to_id.json
wikitext103_id_to_word.json
training_history.csv
~~~

The recommended repository architecture separates these generated artifacts from notebooks.

That makes it easier to answer:

~~~text
What is source?
What is generated?
What is trained?
What is reusable?
~~~

---

# 📈 23. Visualization

The notebooks include visual inspection of data and training behavior.

Strong sequence-model visualizations include:

- dataset split sizes
- token frequency
- text-length distribution
- training loss
- validation loss
- top-k accuracy
- perplexity
- prediction probabilities
- confidence values
- vocabulary statistics
- embedding statistics

Visualization should answer a question.

For example:

### Is the model overfitting?

Compare training and validation loss.

### Are predictions uncertain?

Inspect probability distributions.

### Is a small subset of words dominating?

Inspect frequency plots.

### Is generation overly conservative?

Compare samples at different temperatures.

### Does model capacity justify itself?

Compare validation quality against parameter count and runtime.

---

# 🧠 24. Common Failure Modes

## Shape mismatches

Sequence models create three-dimensional tensors frequently.

Always know whether you have:

~~~text
(batch, time)
~~~

or:

~~~text
(batch, time, features)
~~~

or:

~~~text
(batch, features)
~~~

Blindly guessing shapes is a debugging tax.

---

## Token-indexing mistakes

Keep token-to-ID and ID-to-token mappings consistent.

Use explicit special tokens such as:

~~~text
<PAD>
<UNK>
~~~

when the task requires them.

---

## Padding mistakes

Padding is not real language.

The model must either learn to ignore padded positions or use masking where appropriate.

---

## Data leakage

Do not allow test information to leak into vocabulary construction or preprocessing statistics when the experiment is intended to simulate a clean train/test workflow.

---

## Overfitting

Tiny datasets can be memorized very quickly.

A model that performs brilliantly on the training samples can still be useless outside them.

---

## Generation loops

Poorly trained language models can repeat tokens, fall into degenerate loops, or produce incoherent sequences.

This can arise from:

- insufficient training
- limited data
- model capacity
- decoding strategy
- vocabulary constraints
- context limitations

---

# 🧪 25. Controlled Experiments To Add

The strongest future version of this project should add ablations.

Instead of changing five things at once, change one factor while holding the others fixed.

### Embedding dimension

Compare:

~~~text
64
128
256
~~~

Measure:

- parameter count
- validation loss
- perplexity
- runtime

### Context length

Compare:

~~~text
16
32
64
128
~~~

Ask whether additional context produces measurable benefit.

### Hidden dimension

Increase recurrent width and observe the compute/performance trade-off.

### Dropout

Compare regularization settings while watching validation behavior.

### Optimizer

Compare Adam and AdamW under a controlled setup.

### RNN family

Compare:

~~~text
SimpleRNN
GRU
LSTM
~~~

using the same data pipeline and evaluation protocol.

That last condition matters enormously.

A fair comparison requires a fair experimental design.

---

# 🔬 26. Recommended Evaluation Framework

For classification experiments:

| Area | Recommended measurement |
|---|---|
| Overall performance | Accuracy |
| Positive prediction quality | Precision |
| Positive detection | Recall |
| Balanced metric | F1 |
| Error structure | Confusion matrix |
| Threshold behavior | ROC-AUC where appropriate |

For language modeling:

| Area | Recommended measurement |
|---|---|
| Training objective | Cross-entropy |
| Exact prediction | Top-1 |
| Useful candidate coverage | Top-3 / Top-5 |
| Uncertainty | Perplexity |
| Generation behavior | Human inspection + sample diversity |

Metrics do not replace inspection.

Especially for generated text, a number can only capture part of the story.

---

# 🧠 27. RNN vs LSTM vs GRU

At the conceptual level:

| Architecture | State design | Main idea |
|---|---|---|
| Vanilla RNN | Hidden state | Simple recurrence |
| LSTM | Hidden + cell state | Explicit gated memory |
| GRU | Gated hidden state | Simpler gated recurrence |
| Bidirectional variant | Forward + backward states | Both directional contexts |

The useful takeaway is not that one row is always superior.

The architecture should match the problem and constraints.

Think in terms of:

~~~text
task
+
data
+
sequence length
+
latency
+
memory
+
training budget
+
deployment constraints
~~~

---

# 🧭 28. From RNNs to Transformers

Transformers changed sequence modeling, but learning recurrent networks is still valuable.

RNNs teach:

- hidden state
- state transitions
- temporal dependence
- parameter sharing
- gradient flow
- autoregressive prediction
- sequence-to-sequence thinking

Those ideas form useful mental foundations for later attention and Transformer study.

A logical continuation is:

~~~text
RNN
 ↓
LSTM / GRU
 ↓
Seq2Seq
 ↓
Attention
 ↓
Self-Attention
 ↓
Transformer
 ↓
Pretrained Transformers
 ↓
Fine-Tuning / Inference
~~~

---

# 🛠️ Setup

## 1. Clone the repository

~~~text
git clone https://github.com/Maganpreet-Singh/recurrent-neural-networks.git
cd recurrent-neural-networks
~~~

## 2. Create a virtual environment

### Windows

~~~text
python -m venv .venv
.venv\Scripts\activate
~~~

### Linux / macOS

~~~text
python3 -m venv .venv
source .venv/bin/activate
~~~

## 3. Install dependencies

The repository includes a baseline <code>requirements.txt</code>.

~~~text
pip install -r requirements.txt
~~~

The current baseline includes:

~~~text
numpy
pandas
matplotlib
tensorflow
datasets
jupyter
~~~

TensorFlow availability can depend on the Python version and operating system, so the exact environment should be validated locally before training the larger experiments.

## 4. Launch Jupyter

~~~text
jupyter notebook
~~~

Open the desired notebook from:

~~~text
notebooks/
~~~

---

# ☁️ Google Colab

Google Colab is a convenient runtime for the heavier notebooks.

A practical flow is:

~~~text
open notebook
    ↓
connect runtime
    ↓
select hardware
    ↓
install missing packages
    ↓
run from top to bottom
~~~

The larger WikiText language-modeling experiment can benefit from GPU access.

The notebook checks for GPU availability and can enable mixed precision when suitable hardware is present.

---

# 📦 Dependency Strategy

A growing machine-learning repository benefits from deterministic environments.

The next step beyond a baseline requirements file is to pin tested versions.

For example:

~~~text
package
==
exact version
~~~

The right versions should be based on the environment that has actually been tested.

Do not blindly freeze a notebook session and assume that the resulting file is automatically portable.

Compatibility between:

- Python
- TensorFlow
- CUDA
- cuDNN
- NumPy

can matter for local GPU workflows.

---

# 🧱 29. Production-Oriented Repository Evolution

A notebook-only project is excellent for learning.

A production-oriented project needs stronger separation of responsibilities.

The next structural evolution is:

~~~text
notebooks
    = experiments and explanations

src
    = reusable implementation

configs
    = experiment settings

tests
    = correctness checks

models
    = trained artifacts

reports
    = metrics and figures

scripts
    = repeatable commands
~~~

This architecture enables a transition from:

~~~text
"Here is my notebook"
~~~

to:

~~~text
"Here is a reproducible machine-learning project."
~~~

---

# 🧪 30. Suggested Source Layout

A mature implementation could use:

~~~text
src/
├── data/
│   ├── preprocessing.py
│   ├── vocabulary.py
│   └── datasets.py
│
├── models/
│   ├── rnn.py
│   ├── gru.py
│   ├── lstm.py
│   └── embedding.py
│
├── training/
│   ├── train.py
│   ├── callbacks.py
│   └── seed.py
│
├── evaluation/
│   ├── classification.py
│   └── language_modeling.py
│
└── inference/
    ├── sentiment.py
    └── text_generation.py
~~~

The notebooks can then import the reusable logic.

That eliminates copy-paste code and makes experiments easier to reproduce.

---

# 🧪 31. Testing Strategy

Even deep-learning repositories need tests.

Useful tests include:

### Vocabulary round-trip

~~~text
word → id → word
~~~

### Padding

Verify expected sequence length.

### Context-window generation

Verify that inputs and targets are aligned.

### Output shape

Verify expected model tensor dimensions.

### Serialization

Save a model, reload it, and run the same input through both versions.

### Determinism

Where practical, verify seeded preprocessing behavior.

Testing should focus on things that can fail silently and contaminate experiments.

---

# 🔐 32. Security and Data Hygiene

Never commit:

~~~text
API keys
passwords
tokens
private datasets
credentials
~~~

Add environment-specific artifacts to <code>.gitignore</code>.

Generated files should also be managed intentionally.

Do not commit every intermediate checkpoint simply because Git can technically store it.

For larger artifacts, use an appropriate model/data registry or Git LFS strategy.

---

# 📋 33. Recommended .gitignore

A useful starting point is:

~~~text
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
logs/
artifacts/
data/raw/
~~~

The exact contents should evolve with the project.

---

# 📄 34. Documentation Strategy

A 50,000-word README is usually counterproductive.

A better documentation architecture is:

~~~text
README.md
    ↓
project overview
    ↓
docs/
    ├── learning-guide.md
    ├── fundamentals.md
    ├── architectures.md
    ├── sentiment-analysis.md
    ├── language-modeling.md
    └── experiments.md
~~~

This gives the repository two layers:

### Fast layer

README for visitors, reviewers, recruiters, and collaborators.

### Deep layer

Documentation for the person who wants the full technical story.

That is the scalable way to reach very deep documentation without turning the repository landing page into an enormous wall of text.

---

# 📚 35. Study Guide

Use the project as an active learning checklist.

## RNN fundamentals

- [ ] Explain sequence data
- [ ] Explain time steps
- [ ] Explain hidden state
- [ ] Derive the recurrence equation
- [ ] Count parameters
- [ ] Explain tanh
- [ ] Explain sigmoid
- [ ] Explain binary cross-entropy

## Representation

- [ ] One-hot encoding
- [ ] Integer token IDs
- [ ] Embedding lookup
- [ ] Padding
- [ ] Masking
- [ ] Tensor shapes

## Architecture

- [ ] SimpleRNN
- [ ] Deep RNN
- [ ] Bidirectional RNN
- [ ] LSTM
- [ ] GRU

## NLP

- [ ] Sentiment classification
- [ ] Next-token prediction
- [ ] Vocabulary construction
- [ ] Context windows
- [ ] Top-k accuracy
- [ ] Perplexity
- [ ] Temperature sampling
- [ ] Autoregressive generation

## Engineering

- [ ] Save model
- [ ] Save vocabulary
- [ ] Record hyperparameters
- [ ] Add reproducible dependencies
- [ ] Add tests
- [ ] Add CI
- [ ] Build inference API

---

# 🚀 36. Roadmap

## Phase 1 — RNN Foundations

Add:

- manual backpropagation through time
- gradient-flow demonstrations
- vanishing-gradient experiments
- exploding-gradient experiments
- hidden-state visualizations

## Phase 2 — Architecture Comparisons

Create controlled comparisons among:

- SimpleRNN
- GRU
- LSTM
- Bidirectional GRU
- Bidirectional LSTM

## Phase 3 — Stronger Sentiment Analysis

Add:

- explicit test set
- confusion matrix
- precision
- recall
- F1
- error analysis
- threshold analysis

## Phase 4 — Better Language Modeling

Add experiments for:

- context length
- vocabulary size
- embedding size
- hidden size
- dropout
- optimizer
- learning rate
- recurrent depth

## Phase 5 — Deployment

Build an inference service:

~~~text
HTTP request
     ↓
tokenization
     ↓
trained model
     ↓
prediction
     ↓
JSON response
~~~

Possible frameworks:

~~~text
FastAPI
Flask
~~~

## Phase 6 — Interactive Demo

Potential UI options:

~~~text
Streamlit
Gradio
HTML + JavaScript
~~~

Possible demos:

- sentiment analyzer
- next-word predictor
- temperature comparison
- top-k comparison
- model confidence viewer

## Phase 7 — Attention and Transformers

Move into:

- attention from first principles
- self-attention
- positional encoding
- encoder architecture
- decoder architecture
- Transformer training
- pretrained model fine-tuning

---

# 🧠 37. Deeper Conceptual Connections

RNNs teach a broader lesson about machine learning.

A model is fundamentally an information-processing system.

For an RNN:

~~~text
input
 ↓
state transition
 ↓
new state
 ↓
future computation
~~~

For an LSTM:

~~~text
input + hidden state + cell state
 ↓
gated update
 ↓
new hidden state + new cell state
~~~

For an autoregressive language model:

~~~text
context
 ↓
probability distribution
 ↓
sample token
 ↓
new context
 ↓
repeat
~~~

Different architectures may look radically different in code, but they all solve the same broad engineering problem:

> How should useful information be represented and transformed so that the next prediction becomes better?

---

# 💡 38. Why Manual Mathematics Matters

Framework APIs can make difficult architectures feel easy.

That is both useful and dangerous.

Useful because engineers can build quickly.

Dangerous because abstraction can hide the mechanism.

A manual RNN forward pass forces you to see:

~~~text
x_t
 ↓
W_xh x_t
 +
W_hh h_(t-1)
 +
b_h
 ↓
tanh
 ↓
h_t
~~~

Once that is clear, a <code>SimpleRNN</code> layer becomes less mysterious.

The same principle scales to LSTM and GRU.

Learn the mechanism.

Then use the abstraction.

---

# ⚠️ 39. What This Repository Does Not Claim

This repository is a learning and experimentation project.

It does not claim to provide:

- state-of-the-art NLP performance
- a production-grade sentiment benchmark
- a production-grade language model
- a universal architecture recommendation
- a complete Transformer implementation

The experiments demonstrate concepts and engineering patterns.

They are stepping stones toward larger projects.

---

# 🏁 40. Final Takeaway

The real purpose of this repository is not to collect six notebooks.

It is to document a progression:

~~~text
mathematics
    ↓
tensor representation
    ↓
recurrent state
    ↓
embeddings
    ↓
gated recurrence
    ↓
NLP classification
    ↓
language modeling
    ↓
generation
    ↓
deployment
~~~

That progression gives the project a coherent identity.

The repository begins with:

> “What exactly happens during one recurrent update?”

and eventually reaches:

> “How can a recurrent model assign probabilities to the next token and use those probabilities to generate text?”

That is a substantial conceptual journey.

---

# ⭐ Repository Philosophy

~~~text
Understand the equation.
        ↓
Understand the tensor.
        ↓
Implement the mechanism.
        ↓
Use the framework.
        ↓
Run the experiment.
        ↓
Measure the result.
        ↓
Save the artifact.
        ↓
Document the limits.
        ↓
Build something real.
~~~

A strong ML project is not defined by how many notebooks it contains.

It is defined by whether another person can understand:

~~~text
what was built
why it was built
how it was trained
how it was evaluated
what the results mean
what the limitations are
what comes next
~~~

That is the standard this repository should continue to follow.

---

# 👨‍💻 Author

**Maganpreet Singh**

Computer Science & Engineering student focused on:

~~~text
Python
Data Science
Machine Learning
Deep Learning
Natural Language Processing
Computer Vision
~~~

GitHub:  
https://github.com/Maganpreet-Singh

Repository:  
https://github.com/Maganpreet-Singh/recurrent-neural-networks

---

# 📄 License

Add an explicit license file before describing this repository as openly reusable.

Public visibility does not automatically grant broad reuse rights.

---

<p align="center">
  <strong>🔄 Sequence → State → Context → Prediction</strong>
</p>

<p align="center">
  Built with Python, TensorFlow, Keras, NumPy, and curiosity.
</p>