import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Create Sentences and Labels
sentences = [
    'I am not happy',
    'I had so much fun',
    'she ruined the whole trip',
    'Food was so delicious'
]
labels = [0, 1, 0, 1]


# Create Word Counter
word_counter = Counter(' '.join(sentences).split())
vocab = {word.lower(): i for i, word in enumerate(word_counter.keys(), start=2)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

# Tokenization and Vectorization
sequences = [[vocab.get(word.lower(), vocab['<UNK>']) for word in sentence.split()] for sentence in sentences]

# Generate Padded Sequences
padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=vocab['<PAD>'])

# Convert labels into tensor
labels = torch.tensor(labels, dtype=torch.long)

# Create the Sentiment RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        # Take the output of the last time step
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        return out

# Define hyperparameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 2  # Binary classification: 0 or 1

# Instantiate the model
model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

input_seq = padded_sequences[0].unsqueeze(0)
output = model(input_seq)
print(output) # Output

print(torch.max(output)) # Max Output