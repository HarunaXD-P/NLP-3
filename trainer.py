import torch
import torch.nn as nn
from utils import evaluate

class ParserTrainer():

    def __init__(
        self,
        train_data,
        dev_data,
        optimizer,
        loss_func,
        output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005, 
    ): # You can add more arguments
        """
        Initialize the trainer.
        
        Inputs:
            - train_data: Packed train data
            - dev_data: Packed dev data
            - optimizer: The optimizer used to optimize the parsing model
            - loss_func: The cross entropy function to calculate loss, initialized beforehand
            - output_path (str): Path to which model weights and results are written
            - batch_size (int): Number of examples in a single batch
            - n_epochs (int): Number of training epochs
            - lr (float): Learning rate
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.output_path = output_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        ### TODO: You can add more initializations here


    def train(self, parser, ): # You can add more arguments as you need
        """
        Given packed train_data, train the neural dependency parser (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        """
        best_dev_UAS = 0

        ### TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize

        for epoch in range(self.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epochs))
            dev_UAS, dev_LAS = self._train_for_epoch(parser, )
            # TODO: you can change this part, to use either uas or las to select best model
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                print("New best dev UAS! Saving model.")
                torch.save(parser.model.state_dict(), self.output_path)
            print("")


    def _train_for_epoch(self, parser, ): # You can add more arguments as you need
        """ 
        Train the neural dependency parser for single epoch.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        Return:
            - dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
        """
        parser.model.train() # Places model in "train" mode, e.g., apply dropout layer, etc.
        ### TODO: Train all batches of train_data in an epoch.
        ### Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)

        print("Evaluating on dev set",)
        parser.model.eval() # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        dependencies = parser.parse(self.dev_data)
        uas,las = evaluate(dependencies, gold_dependencies)  # To check the format of the input, please refer to the utils.py
        print("- dev UAS: {:.2f}".format(uas * 100.0), "- dev LAS: {:.2f}".format(las * 100.0))
        return uas, las