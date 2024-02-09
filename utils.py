import torch


# Time Attention Learning Block Operation
def time_x_embed(embed: torch.Tensor, time_attn: torch.Tensor) -> torch.Tensor:
    nu = embed * time_attn.unsqueeze(-1)

    return nu


def rnn_x_attn(rnn_out: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    weighted_sum = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)

    return weighted_sum