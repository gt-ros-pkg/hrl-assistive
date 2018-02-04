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

        hidden_dim1= 32
        hidden_dim2 = 48
        hidden_dim3 = 96
        hidden_dim4 = 96

        self.count = 0

        self.CNN_pack1 = nn.Sequential(
            # Vanilla
            nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            nn.Dropout(p=0.1, inplace=False),

            # 2
            # nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1, inplace=False),

            # 3
            # nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
        )

        self.CNN_pack2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 7, stride = 1, padding = 0),
            nn.ReLU(inplace = True),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=5, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=4, stride=1, padding= 0),
            nn.ReLU(inplace=True),
        )



        self.CNN_pack3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(inplace = True),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 10, kernel_size=1, stride=1, padding= 0),
            nn.ReLU(inplace=True),

        )


        #self.CNN_pack4 = nn.Sequential(
        # torch.nn.MaxPool2d(2, 2),  # this cuts the height and width down by 2
        #
        #)


        print 'x'
        self.CNN_fc1 = nn.Sequential(
            # Vanilla
            nn.Linear(8832, 2048), #4096 for when we only pad the sides by 5 each instead of 10
            #nn.ReLU(inplace = True),
            #nn.Linear(5760, 3000),
            nn.Linear(2048, 2048),
            #nn.ReLU(inplace = True),
            nn.Linear(2048, 256),
            nn.Linear(256, out_size),

            # nn.Linear(8832, out_size),
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
        scores_size = scores.size()
        #print scores_size, 'scores conv1'


        #scores = self.CNN_pack2(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv2'

        #scores = self.CNN_pack3(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv3'

        #scores = self.CNN_pack4(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv4'


        # This combines the height, width, and filters into a single dimension
        scores = scores.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3])

        #print scores.size(), 'scores fc1'
        scores = self.CNN_fc1(scores)

        targets_est = np.copy(scores.data.numpy())

        #print scores.size(), 'scores fc2'

        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
        scores = targets - scores
        scores = scores.pow(2)
        scores[:, 0] = scores[:, 0] + scores[:, 1] + scores[:, 2]
        scores[:,1] = scores[:,3]+scores[:,4]+scores[:,5]
        scores[:,2] = scores[:,6]+scores[:,7]+scores[:,8]
        scores[:,3] = scores[:,9]+scores[:,10]+scores[:,11]
        scores[:,4] = scores[:,12]+scores[:,13]+scores[:,14]
        scores[:,5] = scores[:,15]+scores[:,16]+scores[:,17]
        scores[:,6] = scores[:,18]+scores[:,19]+scores[:,20]
        scores[:,7] = scores[:,21]+scores[:,22]+scores[:,23]
        scores[:,8] = scores[:,24]+scores[:,25]+scores[:,26]
        scores[:,9] = scores[:,27]+scores[:,28]+scores[:,29]
        scores = scores[:, 0:10]
        scores = scores.sqrt()


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, targets_est

    def forward_confidence(self, images, targets_proj):


        scores_cnn = self.CNN_pack2(images)
        scores_size = scores_cnn.size()
        print scores_size, 'scores conv1'

        scores_cnn = self.CNN_pack3(scores_cnn)
        scores_size = scores_cnn.size()
        print scores_size, 'scores conv1'


        print images.size()
        print targets_proj.size()



    def forward_kinematic_jacobian(self, images, targets, kincons=None, prior_cascade = None, forward_only = False):

        scores = None
        targets_est = None
        lengths_est = None

        scores_cnn = self.CNN_pack1(images)
        scores_size = scores_cnn.size()
        #print scores_size, 'scores conv1'



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

        #kincons_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #torso_scores = scores[:, 0:3]


        #angles_scores = scores[:, 11:19]

        if kincons is not None:
            kincons = kincons / 100


        scores, angles_est, pseudotargets_est = KinematicsLib().forward_kinematics_pytorch(images, scores, targets, self.loss_vector_type, kincons, prior_cascade = prior_cascade, forward_only = forward_only)


        #print scores.size(), ''

        # targets_est = np.copy(scores[:, 17:47].data.numpy())*1000. #after it comes out of the forward kinematics
        # targets_est[:, 0:3] = np.copy(scores[:, 20:23].data.numpy())*1000. #after it comes out of the forward kinematics
        # targets_est[:, 3:6] = np.copy(scores[:, 17:20].data.numpy())*1000. #after it comes out of the forward kinematics
        # lengths_est = np.copy(scores[:, 0:17].data.numpy())
        targets_est = scores[:, 17:47].data*1000. #after it comes out of the forward kinematics
        targets_est[:, 0:3] = scores[:, 20:23].data*1000. #after it comes out of the forward kinematics
        targets_est[:, 3:6] = scores[:, 17:20].data*1000. #after it comes out of the forward kinematics
        lengths_est = scores[:, 0:17].data

        if forward_only == False:
            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (10, 30, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

            #print scores.size()
            #print targets.size()

            scores[:, 27:30] = targets[:, 3:6]/1000 - scores[:, 27:30]
            scores[:, 30:33] = targets[:, 0:3]/1000 - scores[:, 30:33]
            scores[:, 33:57] = targets[:, 6:30]/1000 - scores[:, 33:57]
            scores[:, 57:87] = ((scores[:, 27:57])*1.).pow(2)
            self.count += 1
            if self.count < 300:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*4# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
            elif self.count < 1000:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
            else:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2
            scores[:, 1] = (scores[:, 60] + scores[:, 61] + scores[:, 62]).sqrt()
            scores[:, 2] = (scores[:, 63] + scores[:, 64] + scores[:, 65]).sqrt()
            scores[:, 3] = (scores[:, 66] + scores[:, 67] + scores[:, 68]).sqrt()
            scores[:, 6] = (scores[:, 75] + scores[:, 76] + scores[:, 77]).sqrt()
            scores[:, 7] = (scores[:, 78] + scores[:, 79] + scores[:, 80]).sqrt()
            #if self.count < 1500:
            #    scores[:, 4] = (scores[:, 69] + scores[:, 70] + scores[:, 71]).sqrt()*0.5
            #    scores[:, 5] = (scores[:, 72] + scores[:, 73] + scores[:, 74]).sqrt()*0.5
            #    scores[:, 8] = (scores[:, 81] + scores[:, 82] + scores[:, 83]).sqrt()*0.5
            #    scores[:, 9] = (scores[:, 84] + scores[:, 85] + scores[:, 86]).sqrt()*0.5
            #else:
            scores[:, 4] = (scores[:, 69] + scores[:, 70] + scores[:, 71]).sqrt()
            scores[:, 5] = (scores[:, 72] + scores[:, 73] + scores[:, 74]).sqrt()
            scores[:, 8] = (scores[:, 81] + scores[:, 82] + scores[:, 83]).sqrt()
            scores[:, 9] = (scores[:, 84] + scores[:, 85] + scores[:, 86]).sqrt()

            print self.count


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -60, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return  scores, targets_est, angles_est, lengths_est, pseudotargets_est, #, lengths_scores
