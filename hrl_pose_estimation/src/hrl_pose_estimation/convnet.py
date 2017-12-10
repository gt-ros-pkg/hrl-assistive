import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size


        hidden_dim1= 16
        hidden_dim2 = 32
        hidden_dim3 = 48
        hidden_dim4 = 128
        #hidden_dim5 = 128

        self.CNN_pack1 = nn.Sequential(
            torch.nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 1, padding = 0),
            torch.nn.ReLU(inplace = True),
        )

        self.CNN_pack2 = nn.Sequential(
            torch.nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=4, stride=2, padding= 0),
            torch.nn.ReLU(inplace=True),

        )

        self.CNN_pack3 = nn.Sequential(
            torch.nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=4, stride=2, padding= 1),
            torch.nn.ReLU(inplace=True),
            #torch.nn.MaxPool2d(2, 2),  # this cuts the height and width down by 2
            #
        )

        self.CNN_pack4 = nn.Sequential(
            torch.nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=4, stride=2, padding= 1),
            torch.nn.ReLU(inplace=True),
        )


        print 'x'
        self.CNN_fc = nn.Sequential(
            torch.nn.Linear(5760, 3000),
            torch.nn.Linear(3000, 1000),
            torch.nn.Linear(1000, 300),
            torch.nn.Linear(300, out_size),
        )

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, out_size) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        scores = images
        #print images.size(), 'CNN input size'
        scores = self.CNN_pack1(images)
        #score_size = scores.size()
        #print score_size, 'score conv1'


        scores = self.CNN_pack2(scores)
        #score_size = scores.size()
        #print score_size, 'score conv2'

        scores = self.CNN_pack3(scores)
        score_size = scores.size()
        #print score_size, 'score conv3'

        scores = self.CNN_pack4(scores)
        score_size = scores.size()
        #print score_size, 'score conv4'

        #scores = self.CNN_pack5(scores)
        #score_size = scores.size()
        #rint score_size, 'score_conv5'

        # This combines the height, width, and filters into a single dimension
        scores = scores.view(images.size(0),score_size[1] *score_size[2]*score_size[3] )

        #print scores.size(), 'score fc1'
        scores = self.CNN_fc(scores)

        #print scores.size(), 'score fc2'


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

