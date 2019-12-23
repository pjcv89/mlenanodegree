import torch
import torch.nn as nn

class AdamNetV2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=32, n_layers=2,
              is_bidirectional=False, dropout=0.5, output_dim=1, padding_idx=None, txt_field=None, label_field=None):
        super().__init__()
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                                  padding_idx=padding_idx)
    
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                        num_layers=n_layers, bidirectional=is_bidirectional,
                       dropout=dropout)
    
        self.fc = nn.Linear((is_bidirectional+1)*hidden_dim, output_dim)
    
        self.is_bidirectional = is_bidirectional
        
        self.txt_field = txt_field
        self.label_field = label_field
        
    def forward(self, input_sequence, sequence_length):
        embeddings = self.embedding(input_sequence)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, 
                                                        sequence_length)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_embeddings)
        
        if self.is_bidirectional:
            output = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        else:
            output = hidden_state[-1,:,:]

        scores = self.fc(output)

        return scores