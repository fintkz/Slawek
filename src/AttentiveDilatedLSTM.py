import torch
import torch.nn as nn


class AttentiveDilatedLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        attention_size,
        dilations,
        batch_first=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.dilations = dilations
        self.batch_first = batch_first
        self.num_layers = len(dilations)

        # Input projection for first layer
        self.input_projection = nn.Linear(input_size, hidden_size)

        # LSTM Cell parameters for each layer
        self.lstm_cells = nn.ModuleList(
            [
                nn.LSTMCell(hidden_size, hidden_size)
                for _ in range(self.num_layers)
            ]
        )

        # Attention mechanism parameters
        self.attention_weights = nn.ModuleList(
            [
                nn.Linear(hidden_size, attention_size)
                for _ in range(self.num_layers)
            ]
        )
        self.attention_context = nn.ModuleList(
            [nn.Linear(attention_size, 1) for _ in range(self.num_layers)]
        )

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters using the same method regardless of batch_first setting"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_attention(self, hidden_states, layer_idx):
        """Apply attention mechanism to hidden states"""
        attention_energies = torch.tanh(
            self.attention_weights[layer_idx](hidden_states)
        )
        attention_weights = torch.softmax(
            self.attention_context[layer_idx](attention_energies), dim=1
        )
        context = torch.bmm(attention_weights.transpose(1, 2), hidden_states)
        return context.squeeze(1)

    def _process_sequence(self, x, h_states, c_states):
        """Process the sequence in batch-first format"""
        batch_size, seq_len, _ = x.size()
        outputs = []
        hidden_states_history = [[] for _ in range(self.num_layers)]

        for t in range(seq_len):
            layer_input = self.input_projection(x[:, t, :])

            for layer in range(self.num_layers):
                if t % self.dilations[layer] == 0:
                    hidden_states_history[layer].append(h_states[layer])

                    if len(hidden_states_history[layer]) > 1:
                        historical_states = torch.stack(
                            hidden_states_history[layer], dim=1
                        )
                        attn_context = self._apply_attention(
                            historical_states, layer
                        )
                        layer_input = layer_input + attn_context

                    h_states[layer], c_states[layer] = self.lstm_cells[layer](
                        layer_input, (h_states[layer], c_states[layer])
                    )
                    layer_input = h_states[layer]

            outputs.append(h_states[-1])

        return torch.stack(outputs, dim=1), (h_states, c_states)

    def forward(self, x, initial_states=None):
        # Ensure input is in the correct format
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first

        # Initialize or format states
        if initial_states is None:
            h_states = [
                torch.zeros(x.size(0), self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
            c_states = [
                torch.zeros(x.size(0), self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_states = [state.clone() for state in initial_states[0]]
            c_states = [state.clone() for state in initial_states[1]]

        # Process sequence
        outputs, (final_h, final_c) = self._process_sequence(
            x, h_states, c_states
        )

        # Convert back to sequence_first if needed
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, (final_h, final_c)
