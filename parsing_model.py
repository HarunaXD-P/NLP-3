import torch
import torch.nn as nn
import torch.nn.functional as F

class ParsingModel(nn.Module):

    def __init__(self, ):
        """ 
        Initialize the parser model. You can add arguments/settings as you want, depending on how you design your model.
        NOTE: You can load some pretrained embeddings here (If you are using any).
              Of course, if you are not planning to use pretrained embeddings, you don't need to do this.
        """
        super(ParsingModel, self).__init__()
        pass


    def forward(self, t):
        """
        Input: input tensor of tokens -> SHAPE (batch_size, n_features)
        Return: tensor of predictions (output after applying the layers of the network
                                 without applying softmax) -> SHAPE (batch_size, n_classes)
        """
        
        pass
        
        return logits
