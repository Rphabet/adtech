import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import time_x_embed, rnn_x_attn


class DJHAN(nn.Module):
    def __init__(self, event_vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int,
                 attention_dim1: int, hidden_dim2: int, attention_dim2: int,
                 batch_first: bool = True, bidirectional: bool = True) -> None:
        super(DJHAN, self).__init__()

        self.hidden_dim2 = hidden_dim2
        self.action_net = ActionBlock(event_vocab_size=event_vocab_size,
                                      embedding_dim=embedding_dim,
                                      hidden_dim=hidden_dim,
                                      num_layers=num_layers,
                                      attention_dim=attention_dim1,
                                      batch_first=batch_first,
                                      bidirectional=bidirectional
                                      )
        self.journey_net = JourneyBlock(input_dim=hidden_dim * 2,  # bi-directional
                                        hidden_dim=hidden_dim2,
                                        attention_dim=attention_dim2,
                                        num_layers=num_layers,
                                        batch_first=batch_first,
                                        bidirectional=bidirectional
                                        )

        self.ln = nn.LayerNorm((hidden_dim * 2) + ((hidden_dim2 * 2) * 5))  # LN after concat

        self.fc1 = nn.Linear((hidden_dim * 2) + ((hidden_dim2 * 2) * 5), hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim2 // 2)
        self.fc3 = nn.Linear(hidden_dim2 // 2, 1)

    def forward(self, action, tau, j):
        action_output_list = self.action_net(action, tau)

        final_journey = action_output_list[-1]

        journey_input = torch.stack(action_output_list, 1).unsqueeze(-2)
        journey_output, j_scores = self.journey_net(journey_input, j)
        journey_output = journey_output.reshape(-1, (self.hidden_dim2 * 2) * 5)

        # add and norm (final journey)
        output = torch.cat((journey_output, final_journey), dim=1)
        output = self.ln(output)

        # fc classification
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = torch.sigmoid(output)

        return output


class ActionEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(ActionEmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.embedding(x)

        return output


class TimeAttnNet(nn.Module):
    def __init__(self) -> None:
        super(TimeAttnNet, self).__init__()

        self.initial_theta = nn.Parameter(torch.FloatTensor([1.0]))
        self.decay_mu = nn.Parameter(torch.FloatTensor([-0.1]))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        init = self.initial_theta.expand_as(tau)
        mu = self.decay_mu.expand_as(tau)
        sigma = torch.sigmoid((init - mu * tau))

        return sigma


# New version of RNN Block
class ActionRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 batch_first: bool = True,
                 bidirectional: bool = True) -> None:
        super(ActionRNN, self).__init__()

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, hidden=None) -> tuple:
        """
        :param x: Action-embedded inputs with shape (batch_size, seq_length, input_dim)
        :param hidden: previous hidden states. In first layer, it might be None
        :return: hidden representations with shape of (batch_size, seq_length, hidden_dim * num_directions)
        """
        output, hidden = self.gru(x, hidden)
        # if self.dropout_rate:
        #     output = self.dropout(output)

        return output, hidden


class ActionAttnNet(nn.Module):
    def __init__(self, input_dim: int, attention_size: int) -> None:
        super(ActionAttnNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, attention_size)
        self.fc2 = nn.Linear(attention_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Weighted sequence representation of shape (batch_size, input_dim)
        """

        batch_size, seq_length, input_dim = x.shape

        # Flatten the batch and sequence dimension
        x_flattened = x.reshape(batch_size * seq_length, input_dim)

        # Two Linear Layers
        attention_weights = self.fc1(x_flattened)
        attention_weights = F.relu(attention_weights)
        attention_weights = self.fc2(attention_weights)

        # Reshape and Softmax
        attention_weights = attention_weights.reshape(batch_size, seq_length)
        attention_weights = F.softmax(attention_weights, dim=1)

        return attention_weights


class ActionBlock(nn.Module):
    def __init__(self, event_vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int,
                 attention_dim: int, batch_first: bool = True,
                 bidirectional: bool = True) -> None:
        super(ActionBlock, self).__init__()
        self.embedding_block = ActionEmbeddingLayer(event_vocab_size, embedding_dim)
        self.time_attn_block = TimeAttnNet()
        self.journeys = nn.ModuleList(
            [ActionRNN(embedding_dim, hidden_dim, num_layers, batch_first, bidirectional) for _ in
             range(5)])
        if bidirectional:
            self.attn_block = ActionRNN(hidden_dim * 2, attention_dim)
        else:
            self.attn_block = ActionRNN(hidden_dim, attention_dim)

    def forward(self, action: torch.Tensor, tau: torch.Tensor):
        output_list = []

        embedded = self.embedding_block(action)
        time_attn = self.time_attn_block(tau)
        weighted_embedding = time_x_embed(embedded, time_attn)

        hidden = None  # for first journey, hidden either be None or zero tensor

        for i, journey in enumerate(self.journeys):
            output, hidden = journey(weighted_embedding[:, i, :, :], hidden)
            attn_weights = self.attn_block(output)
            weighted_sum_attn = rnn_x_attn(output, attn_weights)
            output_list.append(weighted_sum_attn)

        return output_list


class JourneyRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 batch_first: bool = True, bidirectional: bool = True) -> None:
        super(JourneyRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=batch_first, bidirectional=bidirectional)

    def init_hidden(self, batch_size: int) -> torch.Tensor:

        num_layers = 2 if self.gru.bidirectional else 1
        num_layers *= self.gru.num_layers

        return torch.zeros(num_layers, batch_size, self.gru.hidden_size)

    def forward(self, x, hidden: torch.Tensor = None, test: bool = False) -> tuple:
        if test:
            device = "cpu"
        else:
            device = "cuda:0"

        if hidden is None:
            batch_size = x.shape[0]
            hidden = self.init_hidden(batch_size)
            hidden = hidden.to(device)

        gru_out, hidden = self.gru(x, hidden)
        # if self.dropout_rate:
        #     gru_out = self.dropout(gru_out)

        return gru_out, hidden


class JourneyAttnNet(nn.Module):
    def __init__(self, input_dim: int, attention_size: int) -> None:
        super(JourneyAttnNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, attention_size)
        self.fc2 = nn.Linear(attention_size, 1)

    def forward(self, x: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        batch_size, journey_length, input_dim = x.shape

        # Attn
        x_flattened = x.reshape(batch_size * journey_length, input_dim)
        attention_weights = self.fc1(x_flattened)
        attention_weights = F.relu(attention_weights)
        attention_weights = self.fc2(attention_weights)

        # Reshape and Softmax
        attention_weights = attention_weights.reshape(batch_size, journey_length)
        attention_weights = F.softmax(attention_weights, dim=1)

        # if necessary
        # attn weights for non-existent journeys
        # attention_weights = attention_weights * (1 - j.squeeze(-1))

        return attention_weights


class JourneyBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 attention_dim: int, batch_first: bool = True,
                 bidirectional: bool = True) -> None:
        super(JourneyBlock, self).__init__()
        self.journeys = nn.ModuleList(
            [JourneyRNN(input_dim, hidden_dim, num_layers, batch_first, bidirectional) for _ in range(5)])
        if bidirectional:
            self.attn_block = JourneyAttnNet(hidden_dim * 2, attention_dim)
        else:
            self.attn_block = JourneyAttnNet(hidden_dim, attention_dim)

    def forward(self, x: torch.Tensor, j: torch.Tensor, test=False):
        journey_outputs = []

        hidden = None  # for first journey, hidden either be None or zero tensor

        for i, journey in enumerate(self.journeys):
            output, hidden = journey(x[:, i, :, :], hidden, test=test)
            journey_outputs.append(output)

        combined_output = torch.cat(journey_outputs, dim=1)

        journey_attn_score = self.attn_block(combined_output, j)
        weighted_sum_attn = combined_output * (journey_attn_score.unsqueeze(-1))

        return weighted_sum_attn, journey_attn_score
