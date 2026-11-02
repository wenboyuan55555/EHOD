import logging
import numpy as np
import torch
from torch import nn
from modules import ExpMemory, ExpLambsEmbedding, ExpMemoryUpdater, ExpMessageAggregator, MLPMessageFunction


class CTDG_Encoder(nn.Module):
    def __init__(self, device, n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64, lamb=None,
                 output=30, init_lamb=0.5):
        super(CTDG_Encoder, self).__init__()
        if lamb is None:
            lamb = 1
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.n_nodes = n_nodes
        self.node_features = torch.Tensor(node_features).to(device)
        self.device = device
        self.memory_dimension = memory_dimension
        self.output = output

        self.lamb = torch.Tensor([lamb * output]).unsqueeze(0).to(device)

        self.raw_message_dimension = memory_dimension + n_nodes
        self.memory = ExpMemory(n_nodes=n_nodes, memory_dimension=memory_dimension, device=device)
        self.memory_updater = ExpMemoryUpdater(memory=self.memory, lamb=self.lamb, device=device)
        self.message_aggregator = ExpMessageAggregator(device, self.raw_message_dimension, self.lamb)
        self.message_function = MLPMessageFunction(self.raw_message_dimension, message_dimension)

        self.exp_embedding = ExpLambsEmbedding()
        self.embedding_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * 2, memory_dimension, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(memory_dimension, memory_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.e_lamb = nn.Parameter(torch.Tensor([init_lamb]), requires_grad=True)
        self.static_road_embedding = nn.Embedding(self.n_nodes, memory_dimension)

    def compute_embeddings(self, source_nodes, now_time, unique_sources, unique_messages, unique_timestamps):
        if len(source_nodes) > 0:
            updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_sources,
                                                                                         unique_messages,
                                                                                         unique_timestamps)
        else:
            updated_memory = self.memory.get_memory()
            updated_last_update = self.memory.last_update
            print(0)

        recent_embeddings = self.exp_embedding(updated_memory, list(range(self.n_nodes)), self.memory_dimension)
        recent_embeddings = self.embedding_transform(torch.cat(recent_embeddings, dim=-1))
        recent_decayed_embeddings = recent_embeddings * torch.exp(
            (updated_last_update - now_time) / self.output).reshape([-1, 1])
        embeddings = self.e_lamb * self.static_road_embedding.weight + (1 - self.e_lamb) * recent_decayed_embeddings
        return embeddings

    def compute_message(self, source_nodes, destination_nodes, timestamps):
        if len(source_nodes) > 0:
            memory = self.memory.get_memory()
            last_embeddings = self.exp_embedding(memory, list(range(self.n_nodes)), self.memory_dimension)[0]
            last_decayed_embeddings = last_embeddings[destination_nodes] * torch.exp(
                (self.memory.last_update[destination_nodes] - timestamps) / self.output).reshape([-1, 1])
            last_decayed_embeddings = torch.cat([last_decayed_embeddings, self.node_features[destination_nodes]], dim=1)
            raw_messages = self.get_raw_messages(source_nodes, last_decayed_embeddings, timestamps)
            unique_sources, unique_raw_messages, unique_timestamps = self.message_aggregator(source_nodes, raw_messages)
            unique_messages = torch.cat([self.message_function(unique_raw_messages[:, :self.raw_message_dimension]),
                                         unique_raw_messages[:, self.raw_message_dimension:]], dim=-1)
        else:
            print("it is None len=0")
            unique_sources, unique_messages, unique_timestamps = None, None, None

        return unique_sources, unique_messages, unique_timestamps

    def get_raw_messages(self, source_nodes, destination_embeddings, edge_times):
        source_message = torch.cat(
            [destination_embeddings, torch.ones([destination_embeddings.shape[0], 1]).to(self.device)], dim=1)
        messages = dict()
        unique_nodes = np.unique(source_nodes)
        a_ind = np.arange(source_message.shape[0])
        for node_i in unique_nodes:
            ind = a_ind[source_nodes == node_i]
            messages[node_i] = [source_message[ind], edge_times[ind]]
        return messages

    def init_memory(self):
        self.memory.__init_memory__()

    def backup_memory(self):
        return [self.memory.backup_memory()]

    def restore_memory(self, memory):
        self.memory.restore_memory(memory[0])

    def detach_memory(self):
        self.memory.detach_memory()


class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class BootCF(nn.Module):
    def __init__(self, device, n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64, lamb=None,
                 output=30, init_lamb=0.5):
        super(BootCF, self).__init__()
        self.CTDG_Encoder_o = CTDG_Encoder(device, n_nodes, node_features, message_dimension, memory_dimension, lamb,
                                           output, init_lamb)
        self.CTDG_Encoder_d = CTDG_Encoder(device, n_nodes, node_features, message_dimension, memory_dimension, lamb,
                                           output, init_lamb)
        self.linear = nn.Linear(memory_dimension * 2, memory_dimension)

    def forward(self, source_nodes, destination_nodes, timestamps, now_time, time_diffs):
        from_o_unique_sources, from_o_unique_messages, from_o_unique_timestamps = self.CTDG_Encoder_o.compute_message(
            source_nodes, destination_nodes, timestamps)
        from_d_unique_sources, from_d_unique_messages, from_d_unique_timestamps = self.CTDG_Encoder_d.compute_message(
            destination_nodes, source_nodes, timestamps)

        o_embedding = self.CTDG_Encoder_o.compute_embeddings(source_nodes, now_time, from_d_unique_sources,
                                                             from_d_unique_messages, from_d_unique_timestamps)
        d_embedding = self.CTDG_Encoder_d.compute_embeddings(destination_nodes, now_time, from_o_unique_sources,
                                                             from_o_unique_messages, from_o_unique_timestamps)
        return torch.cat([o_embedding, d_embedding], dim=1)

    def init_memory(self):
        self.CTDG_Encoder_o.init_memory()
        self.CTDG_Encoder_d.init_memory()

    def backup_memory(self):
        return [self.CTDG_Encoder_o.backup_memory(), self.CTDG_Encoder_d.backup_memory()]

    def restore_memory(self, memory):
        self.CTDG_Encoder_o.restore_memory(memory[0])
        self.CTDG_Encoder_d.restore_memory(memory[1])

    def detach_memory(self):
        self.CTDG_Encoder_o.detach_memory()
        self.CTDG_Encoder_d.detach_memory()
