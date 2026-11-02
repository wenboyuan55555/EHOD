import torch
import numpy as np
from torch import nn


class ExpLambsEmbedding(nn.Module):
    def __init__(self):
        super(ExpLambsEmbedding, self).__init__()

    def forward(self, memory, nodes=None, memory_dim=64):
        embeddings = memory[nodes, :memory_dim] / memory[nodes, memory_dim:]
        return embeddings, memory[nodes, :memory_dim]


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super(IdentityEmbedding, self).__init__()

    def forward(self, memory, nodes):
        return memory[nodes, :]


class ExpMemory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, device="cpu"):
        super(ExpMemory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.b_dim = 1
        self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(
            torch.cat([torch.zeros((self.n_nodes, self.memory_dimension)), torch.ones((self.n_nodes, self.b_dim))],
                      dim=-1), requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        return self.memory.data.clone(), self.last_update.data.clone()

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()


class Memory_lambs(nn.Module):

    def __init__(self, n_nodes, memory_dimension, device="cpu"):
        super(Memory_lambs, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()


class ExpMemoryUpdater(nn.Module):
    def __init__(self, memory, lamb, device):
        super(ExpMemoryUpdater, self).__init__()
        self.memory = memory
        self.lamb = lamb
        self.device = device

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        updated_memory = self.memory.memory.data.clone()
        updated_last_update = self.memory.last_update
        time_delta = (updated_last_update[unique_node_ids] - timestamps)
        updated_memory[unique_node_ids] = unique_messages + torch.exp(time_delta.unsqueeze(-1) / self.lamb) * \
                                          updated_memory[unique_node_ids]

        if timestamps is not None:
            self.memory.last_update[unique_node_ids] = timestamps
        self.memory.set_memory(unique_node_ids, updated_memory.detach()[unique_node_ids])
        return updated_memory, self.memory.last_update


class ExpMessageAggregator(nn.Module):
    def __init__(self, device, embedding_dimension, lamb):
        super(ExpMessageAggregator, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.lamb = lamb

    def forward(self, node_ids, messages):
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_timestamps.append(messages[node_id][1][-1])
                unique_messages.append(torch.sum(messages[node_id][0] * torch.exp(
                    (messages[node_id][1] - messages[node_id][1][-1]).unsqueeze(-1) / self.lamb), dim=0))
        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


class MLPMessageFunction(nn.Module):
    def __init__(self, raw_message_dimension, message_dimension):
        super(MLPMessageFunction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, (raw_message_dimension + message_dimension) // 2),
            nn.LeakyReLU(),
            nn.Linear((raw_message_dimension + message_dimension) // 2, message_dimension),
            nn.LeakyReLU()
        )

    def forward(self, raw_messages):
        messages = self.mlp(raw_messages)
        return messages
