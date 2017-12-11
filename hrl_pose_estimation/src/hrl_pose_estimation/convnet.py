import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

        self.CNN_pack1 = nn.Sequential(
            nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 1, padding = 0),
            nn.ReLU(inplace = True),
        )

        self.CNN_pack2 = nn.Sequential(
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=4, stride=2, padding= 0),
            nn.ReLU(inplace=True),

        )

        self.CNN_pack3 = nn.Sequential(
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=4, stride=2, padding= 1),
            nn.ReLU(inplace=True),
            #torch.nn.MaxPool2d(2, 2),  # this cuts the height and width down by 2
            #
        )

        self.CNN_pack4 = nn.Sequential(
            nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=4, stride=2, padding= 1),
            nn.ReLU(inplace=True),
        )


        print 'x'
        self.CNN_fc = nn.Sequential(
            nn.Linear(5760, 3000),
            nn.Linear(3000, 1000),
            nn.Linear(1000, 300),
            nn.Linear(300, out_size),
        )






        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward_direct(self, images, targets):

        '''
        Take a batch of images and run them through the CNN to
        produce a scores for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, out_size) specifying the scores
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print images.size(), 'CNN input size'
        scores = self.CNN_pack1(images)
        #scores_size = scores.size()
        #print scores_size, 'scores conv1'


        scores = self.CNN_pack2(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv2'

        scores = self.CNN_pack3(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv3'

        scores = self.CNN_pack4(scores)
        scores_size = scores.size()
        #print scores_size, 'scores conv4'

        #scores = self.CNN_pack5(scores)
        #scores_size = scores.size()
        #rint scores_size, 'scores_conv5'

        # This combines the height, width, and filters into a single dimension
        scores = scores.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )

        #print scores.size(), 'scores fc1'
        scores = self.CNN_fc(scores)

        targets_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #print scores.size(), 'scores fc2'

        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
        scores = targets - scores
        scores = scores.pow(2)
        scores[:, 0] = scores[:, 0] + scores[:, 1] + scores[:, 2]
        scores[:,1] = scores[:,3]+scores[:,4]+scores[:,5]
        scores[:,2] = scores[:,6]+scores[:,7]+scores[:,8]
        scores[:,3] = scores[:,9]+scores[:,10]+scores[:,11]
        scores[:,4] = scores[:,12]+scores[:,13]+scores[:,14]
        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0,-10,0,0), "constant", 0)
        scores = scores.squeeze()
        scores = scores.sqrt()


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, targets_est

    def forward_kinematic_jacobian(self, images, targets, constraints):
        '''
        Take a batch of images and run them through the CNN to
        produce a scores for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, out_size) specifying the scores
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print images.size(), 'CNN input size'
        scores = self.CNN_pack1(images)
        #scores_size = scores.size()
        #print scores_size, 'scores conv1'


        scores = self.CNN_pack2(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv2'

        scores = self.CNN_pack3(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv3'

        scores = self.CNN_pack4(scores)
        scores_size = scores.size()
        #print scores_size, 'scores conv4'

        #scores = self.CNN_pack5(scores)
        #scores_size = scores.size()
        #rint scores_size, 'scores_conv5'

        # This combines the height, width, and filters into a single dimension
        scores = scores.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )

        #print scores.size(), 'scores fc1'
        scores = self.CNN_fc(scores)

        targets_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #print scores.size(), 'scores fc2'

        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
        scores = targets - scores
        scores = scores.pow(2)
        scores[:, 0] = scores[:, 0] + scores[:, 1] + scores[:, 2]
        scores[:,1] = scores[:,3]+scores[:,4]+scores[:,5]
        scores[:,2] = scores[:,6]+scores[:,7]+scores[:,8]
        scores[:,3] = scores[:,9]+scores[:,10]+scores[:,11]
        scores[:,4] = scores[:,12]+scores[:,13]+scores[:,14]
        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0,-10,0,0), "constant", 0)
        scores = scores.squeeze()
        scores = scores.sqrt()


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
