import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
import scipy.stats as ss

class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size, loss_vector_type):
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
        self.loss_vector_type = loss_vector_type

        print loss_vector_type, 'loss vector type'


        hidden_dim1 = 16
        hidden_dim2 = 24
        hidden_dim3 = 40
        hidden_dim4 = 64

        self.CNN_pack1 = nn.Sequential(
            nn.Conv2d(7, hidden_dim1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.CNN_pack2 = nn.Sequential(
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.CNN_pack3 = nn.Sequential(
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.CNN_pack4 = nn.Sequential(
            nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        print 'x'
        self.CNN_fc1 = nn.Sequential(
            nn.Linear(2304, 1500),
            nn.Linear(1500, 500),
            nn.Linear(500, 100),
            nn.Linear(100, out_size),
        )


    def forward_kinematic_jacobian(self, images, targets, kincons=None, prior_cascade = None, forward_only = False, body_side = None):
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
        targets_est = None
        lengths_est = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print images.size(), 'CNN input size'
        scores_cnn = self.CNN_pack1(images)
        #scores_size = scores_cnn.size()
        #print scores_size, 'scores conv1'

        scores_cnn = self.CNN_pack2(scores_cnn)
        #scores_size = scores_cnn.size()
        #print scores_size, 'scores conv2'

        scores_cnn = self.CNN_pack3(scores_cnn)
        #scores_size = scores_cnn.size()
        #print scores_size, 'scores conv3'

        scores_cnn = self.CNN_pack4(scores_cnn)
        scores_size = scores_cnn.size()
        #print scores_size, 'scores conv4'


        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )

        fc_noise = False #add noise to the output of the convolutions.  Only add it to the non-zero outputs, because most are zero.
        if fc_noise == True:
            bin_nonz = -scores_cnn
            bin_nonz[bin_nonz < 0] = 1
            x = np.arange(-900, 900)
            xU, xL = x + 0.5, x - 0.5
            prob = ss.norm.cdf(xU, scale=300) - ss.norm.cdf(xL, scale=300)  # scale is the standard deviation using a cumulative density function
            prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
            image_noise = np.random.choice(x, size=(1, 4096), p=prob) / 1000.
            image_noise = Variable(torch.Tensor(image_noise), volatile = True)
            image_noise = torch.mul(bin_nonz, image_noise)
            scores_cnn = torch.add(scores_cnn, image_noise)

        scores = self.CNN_fc1(scores_cnn)
        #scores_lengths = self.CNN_fc2(scores_cnn)
        #scores_torso = self.CNN_fc3(scores_cnn)


        #kincons_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #torso_scores = scores[:, 0:3]


        #angles_scores = scores[:, 11:19]

        if kincons is not None:
            kincons = kincons / 100


        scores, angles_est, pseudotargets_est = KinematicsLib().forward_kinematics_pytorch(images, scores, targets, self.loss_vector_type, kincons, prior_cascade = prior_cascade, forward_only = forward_only, body_side=body_side)

        targets_est = np.copy(scores[:, 0:6].data.numpy())*1000.

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (2, 6, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)

        #print targets[0, 3:6], 'torso'
        #print targets[0, 6:9], 'elbow'
        #print targets[0, 12:15], 'hand'

        #print scores[0, 2:8], 'scores'

        scores[:, 2:8] = torch.cat((targets[:, 6:9], targets[:, 12:15]), dim = 1)/1000. - scores[:, 2:8]
        scores[:, 8:14] = ((scores[:, 2:8])*1.).pow(2)

        scores[:, 0] = (scores[:, 8] + scores[:, 9] + scores[:, 10]).sqrt()
        scores[:, 1] = (scores[:, 11] + scores[:, 12] + scores[:, 13]).sqrt()



        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, -12, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return  scores, targets_est, angles_est, lengths_est, pseudotargets_est, #, lengths_scores