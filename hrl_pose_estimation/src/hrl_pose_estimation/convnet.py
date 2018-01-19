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

        hidden_dim1= 16
        hidden_dim2 = 32
        hidden_dim3 = 48
        hidden_dim4 = 128

        self.count = 0

        self.CNN_pack1 = nn.Sequential(
            nn.Conv2d(3, hidden_dim1, kernel_size = 7, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=4, stride=2, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=4, stride=2, padding= 1),
            nn.ReLU(inplace=True),
        )

        #self.CNN_pack2 = nn.Sequential(

        #)

        #self.CNN_pack3 = nn.Sequential(
        #    #torch.nn.MaxPool2d(2, 2),  # this cuts the height and width down by 2
        #    #
        #)

        #self.CNN_pack4 = nn.Sequential(
        #)


        print 'x'
        #self.CNN_fc1 = nn.Sequential(
        #    nn.Linear(4096, 2500),
        #    #nn.ReLU(inplace = True),
        #    #nn.Linear(5760, 3000),
        #    nn.Linear(2500, 1000),
        #    #nn.ReLU(inplace = True),
        #    nn.Linear(1000, 300),
        #    nn.Linear(300, out_size),
        #)
        self.CNN_fc1 = nn.Sequential(
            nn.Linear(4096, 2500),
            #nn.ReLU(inplace = True),
            #nn.Linear(5760, 3000),
            nn.Linear(2500, 1000),
            #nn.ReLU(inplace = True),
            nn.Linear(1000, 300),
            nn.Linear(300, out_size),
        )
        #self.CNN_fc2 = nn.Sequential(
        #    nn.Linear(4096, 500),
        #    nn.ReLU(inplace = True),
        #    nn.Linear(500, 100),
        #    nn.ReLU(inplace = True),
        #    nn.Linear(100, 50),
        #    nn.Linear(50, 17),
        #)
        #self.CNN_fc3 = nn.Sequential(
        #    nn.Linear(4096, 200),
        #    nn.ReLU(inplace = True),
        #    nn.Linear(200, 50),
        #    nn.Linear(50, 30),
        #    nn.Linear(30, 3),
        #)





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
        scores = self.CNN_fc(scores)

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

    def forward_kinematic_jacobian(self, images, targets, kincons=None, prior_cascade = None, forward_only = False):
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
        #scores_lengths = self.CNN_fc2(scores_cnn)
        #scores_torso = self.CNN_fc3(scores_cnn)


        #kincons_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #torso_scores = scores[:, 0:3]


        #angles_scores = scores[:, 11:19]

        if kincons is not None:
            kincons = kincons / 100


        scores, angles_est, pseudotargets_est = KinematicsLib().forward_kinematics_pytorch(images, scores, targets, self.loss_vector_type, kincons, prior_cascade = prior_cascade, forward_only = forward_only)
        #scores_torso, scores_angles, pseudotargets_est = KinematicsLib().forward_kinematics_3fc_pytorch(images, scores_torso, scores_lengths, scores_angles, targets, self.loss_vector_type, kincons, prior_cascade = prior_cascade, forward_only = forward_only)

        if self.loss_vector_type == 'upper_angles':
            targets_est = np.copy(scores[:, 9:27].data.numpy())*1000. #this is after the forward kinematics
            targets_est[:, 0:3] = np.copy(scores[:, 12:15].data.numpy())*1000. #this is after the forward kinematics
            targets_est[:, 3:6] = np.copy(scores[:, 9:12].data.numpy())*1000. #this is after the forward kinematics
            lengths_est = np.copy(scores[:, 0:9].data.numpy())

            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (6, 18, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)


            scores[:, 21:33] = targets[:, 6:18]/1000 - scores[:, 21:33]
            scores[:, 15:18] = targets[:, 3:6]/1000 - scores[:, 15:18]
            scores[:, 18:21] = targets[:, 0:3]/1000 - scores[:, 18:21]

            scores[:, 33:51] = ((scores[:, 15:33])*1.).pow(2)

            scores[:, 0] = (scores[:, 33] + scores[:, 34] + scores[:, 35]).sqrt()
            scores[:, 1] = (scores[:, 36] + scores[:, 37] + scores[:, 38]).sqrt()
            scores[:, 2] = (scores[:, 39] + scores[:, 40] + scores[:, 41]).sqrt()
            scores[:, 3] = (scores[:, 42] + scores[:, 43] + scores[:, 44]).sqrt()
            scores[:, 4] = (scores[:, 45] + scores[:, 46] + scores[:, 47]).sqrt()
            scores[:, 5] = (scores[:, 48] + scores[:, 49] + scores[:, 50]).sqrt()


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -36, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

        elif self.loss_vector_type == 'angles':
            #print scores.size(), ''


            targets_est = np.copy(scores[:, 17:47].data.numpy())*1000. #after it comes out of the forward kinematics
            targets_est[:, 0:3] = np.copy(scores[:, 20:23].data.numpy())*1000. #after it comes out of the forward kinematics
            targets_est[:, 3:6] = np.copy(scores[:, 17:20].data.numpy())*1000. #after it comes out of the forward kinematics
            lengths_est = np.copy(scores[:, 0:17].data.numpy())

            #print scores_angles.size(), 'scores angles'
            #targets_est = np.zeros((targets.data.numpy().shape[0], 30)) #after it comes out of the forward kinematics
            #targets_est[:, 0:3] = np.copy(scores_angles[:, 0:3].data.numpy())*1000. #after it comes out of the forward kinematics
            #targets_est[:, 3:6] = np.copy(scores_torso[:, 0:3].data.numpy())*1000. #after it comes out of the forward kinematics
            #targets_est[:, 6:30] = np.copy(scores_angles[:, 3:27].data.numpy())*1000. #after it comes out of the forward kinematics
            #engths_est = np.copy(scores_lengths[:, 0:17].data.numpy())

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
                else:
                    scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
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
                #scores_angles = scores_angles.unsqueeze(0)
                #scores_angles = scores_angles.unsqueeze(0)
                #scores_angles = F.pad(scores_angles, (10, 27, 0, 0))
                #scores_angles = scores_angles.squeeze(0)
                #scores_angles = scores_angles.squeeze(0)
                #scores_torso = scores_torso.unsqueeze(0)
                #scores_torso = scores_torso.unsqueeze(0)
                #scores_torso = F.pad(scores_torso, (1, 3, 0, 0))
                #scores_torso = scores_torso.squeeze(0)
                #scores_torso = scores_torso.squeeze(0)

                #print scores_angles.size()
                #print targets.size(), 'target size'
                #
                # scores_torso[:, 1:4] = targets[:, 3:6]/1000 - scores_torso[:, 1:4]
                # scores_angles[:, 10:13] = targets[:, 0:3]/1000 - scores_angles[:, 10:13]
                # scores_angles[:, 13:37] = targets[:, 6:30]/1000 - scores_angles[:, 13:37]
                #
                # scores_angles[:, 37:64] = ((scores_angles[:, 10:37])*1.).pow(2)
                # scores_torso[:, 4:7] = ((scores_angles[:, 1:4])*1.).pow(2)
                #
                #
                # scores_torso[:, 0] = (scores_torso[:, 4] + scores_torso[:, 5] + scores_torso[:, 6]).sqrt()*2# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
                # scores_angles[:, 1] = (scores_angles[:, 37] + scores_angles[:, 38] + scores_angles[:, 39]).sqrt()
                # scores_angles[:, 2] = (scores_angles[:, 40] + scores_angles[:, 41] + scores_angles[:, 42]).sqrt()
                # scores_angles[:, 3] = (scores_angles[:, 43] + scores_angles[:, 44] + scores_angles[:, 45]).sqrt()
                # scores_angles[:, 4] = (scores_angles[:, 46] + scores_angles[:, 47] + scores_angles[:, 48]).sqrt()
                # scores_angles[:, 5] = (scores_angles[:, 49] + scores_angles[:, 50] + scores_angles[:, 51]).sqrt()
                # scores_angles[:, 6] = (scores_angles[:, 52] + scores_angles[:, 53] + scores_angles[:, 54]).sqrt()
                # scores_angles[:, 7] = (scores_angles[:, 55] + scores_angles[:, 56] + scores_angles[:, 57]).sqrt()
                # scores_angles[:, 8] = (scores_angles[:, 58] + scores_angles[:, 59] + scores_angles[:, 60]).sqrt()
                # scores_angles[:, 9] = (scores_angles[:, 61] + scores_angles[:, 62] + scores_angles[:, 63]).sqrt()
                #
                # #print scores_angles.size()
                # scores_angles = scores_angles.unsqueeze(0)
                # scores_angles = scores_angles.unsqueeze(0)
                # scores_angles = F.pad(scores_angles, (-1, -54, 0, 0))
                # scores_angles = scores_angles.squeeze(0)
                # scores_angles = scores_angles.squeeze(0)
                # scores_torso = scores_torso.unsqueeze(0)
                # scores_torso = scores_torso.unsqueeze(0)
                # scores_torso = F.pad(scores_torso, (0, -6, 0, 0))
                # scores_torso = scores_torso.squeeze(0)
                # scores_torso = scores_torso.squeeze(0)

                #print torch.cat((scores_lengths, scores_torso, scores_angles), dim=1).size()



        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return  scores, targets_est, angles_est, lengths_est, pseudotargets_est, #, lengths_scores