import numpy as np
import torch
import torch.nn as nn

class ConLoss(nn.Module):
    def __init__(self) -> None:
        super(ConLoss, self).__init__()

    def forward(self, Zi, Zj, labels=None, mask=None):
        loss = 0
        temperature = 0.1

        def Cosine_Similarity(x,y):

            v = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

            return v

        for i in range(len(Zi)):
            #ZiとZjのコサイン類似度
            pos = Cosine_Similarity(Zi[i],Zj[i]) / temperature
            #Ziと負のサンプルとのコサイン類似度
            negs = 0
            for j in range(len(Zi)):
                if Zi[i] != Zi[j]:
                    negs += Cosine_Similarity(Zi[i],Zi[j]) / temperature
                    negs += Cosine_Similarity(Zi[i],Zj[j]) / temperature

            loss = pos / negs
  
        return loss

    