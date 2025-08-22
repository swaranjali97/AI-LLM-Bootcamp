## BERT Attention Visualization

This project shows how a Transformer (BERT) processes a sentence end-to-end and visualizes its attention heads with BertViz.

### What are tensors?
- **Tensors** are multi-dimensional arrays used to store data in deep learning. Shapes describe their dimensions, e.g., `(batch_size, seq_len)` or `(batch_size, seq_len, hidden_size)`.
- In PyTorch, tensors back the model inputs, intermediate states, and outputs.

### Data flow (high level)
1. **Sentence** → a string: "Transformers are amazing!".
2. **Tokenization** → BERT WordPiece tokens; mapped to integer `input_ids` (plus special tokens `[CLS]`, `[SEP]`).
3. **Embeddings** → `input_ids` are converted to continuous vectors (token, position, and segment embeddings).
4. **Transformer layers** → stacked self-attention + feed-forward blocks produce:
   - **hidden_states**: a tuple with one tensor per layer (including the embedding output at index 0), each of shape `(batch, seq_len, hidden_size)`.
   - **attentions**: a tuple with one tensor per layer, each of shape `(batch, num_heads, seq_len, seq_len)`.
5. **Last hidden state** → final layer representation of each token: `(batch, seq_len, hidden_size)`.
6. **Visualization** → BertViz `head_view` renders attention patterns per head/layer.

### Files
- `bert_visualize.ipynb`: Main notebook to run the demo and visualization.
- `requirements.txt`: Dependencies list.

### How to run

#### Option A: Google Colab
1. Upload this repository or open the notebook directly in Colab.
2. Run all cells. The first cell installs needed packages in the Colab runtime.

#### Option B: Local Jupyter
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```
4. Open `bert_visualize.ipynb` and run the cells top to bottom.

### Notes
- The notebook loads `bert-base-uncased` with `output_hidden_states=True` and `output_attentions=True`.
- It prints shapes of `input_ids`, `last_hidden_state`, `hidden_states`, and `attentions`.
- BertViz `head_view` renders an interactive attention visualization for the example sentence.