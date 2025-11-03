import torch
import torch.nn as nn
    
class OptimizedRegression(nn.Module):
    def __init__(self, base_model, hidden_size, dropout_rate, num_layers=2, num_outputs=1):
        """
        num_outputs: number of regression outputs (1 for single label, 2 for valence+arousal)
        """
        super().__init__()
        self.wav2vec2 = base_model
        self.num_outputs = num_outputs

        layers = []
        input_size = self.wav2vec2.config.hidden_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size

        # Final output layer is adjustable (1 or 2)
        layers.append(nn.Linear(input_size, num_outputs))

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        out = self.regressor(pooled)
        return out.squeeze(1) if self.num_outputs == 1 else out
    
class ComplexRegression(nn.Module):
    def __init__(self, base_model, hidden_size, dropout_rate, num_layers=3,
                 activation='gelu', use_batch_norm=True, pooling_method='mean',
                 use_residual=True, num_outputs=1):
        """
        num_outputs: number of regression outputs (1 for single label, 2 for valence+arousal)
        """
        super().__init__()
        self.wav2vec2 = base_model
        self.pooling_method = pooling_method
        self.use_residual = use_residual
        self.num_outputs = num_outputs

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'swish': nn.SiLU()
        }
        act_fn = activations.get(activation, nn.ReLU())

        layers = []
        input_size = self.wav2vec2.config.hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.regressor = nn.Sequential(*layers)

        # Toggleable output dimension
        self.output_layer = nn.Linear(input_size, num_outputs)

        if use_residual:
            self.residual_projection = nn.Linear(self.wav2vec2.config.hidden_size, num_outputs)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = (
            outputs.last_hidden_state.mean(dim=1)
            if self.pooling_method == 'mean'
            else outputs.last_hidden_state[:, 0, :]
        )

        x = self.regressor(pooled)
        out = self.output_layer(x)
        if self.use_residual:
            out += self.residual_projection(pooled)

        return out.squeeze(1) if self.num_outputs == 1 else out