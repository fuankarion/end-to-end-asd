import torch
import torch.nn as nn

class assignation_loss_audio(torch.nn.Module):
    def __init__(self, graph_size):
        super(assignation_loss_audio, self).__init__()
        self.graph_size = graph_size
        self.softmax_layer = torch.nn.Softmax(dim=1)

    def forward(self, outputs, audio_targets):
        pred = self.softmax_layer(outputs)[:, 1] # Positive predictions

        pred = pred.view((-1, self.graph_size))
        pred = pred[:, 1:]
        max_pred, _ = torch.max(pred, dim=1)

        audio_gt = audio_targets
        no_assig_penalty = audio_gt*(audio_gt - max_pred)
        bad_assig_penalty = (1-audio_gt)*max_pred
        total_penalty = no_assig_penalty + bad_assig_penalty

        return torch.mean(total_penalty)