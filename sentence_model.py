# tips from https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceModel(nn.Module):

    def __init__(self, embedding_size, num_classes, hidden_size=32, drop_rate=0.1):
        super(SentenceModel, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.dropout(F.gelu(x)))
        return F.log_softmax(x, dim=-1)




class SentenceEmbeddedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, classes, device):
        super().__init__()        
        self.embeddings = embeddings
        self.classes = classes
        self.device = device


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.classes[idx]


