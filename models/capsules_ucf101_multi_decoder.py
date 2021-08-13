import sys
sys.path.insert(0, "../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.pytorch_i3d import InceptionI3d
from torchsummary import summary


class primarySentenceCaps(nn.Module):
    def __init__(self):
        super(primarySentenceCaps, self).__init__()
        # self.a = nn.Conv1d(in_channels=1, out_channels=1,
        #                    kernel_size=16, stride=16, bias=True)
        # self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x ', x.sum())
        p = x[:,0:128]
        p = p.view(-1,16,8)
        # print('p shape',p.size())
        # x = x.view(-1,1,128)
        # print('x shape ',x.shape)
        a = x[:,128:136]
        a = a.view(-1,1,8)
        a = self.sigmoid(a)
        # print('a shape',a.size())
        # print('sentence activations', torch.sum(a[0]).item(), a[0].size())
        # print('p shape ',p.shape)
        # print('a shape ',a.shape)
        out = torch.cat([p, a], dim=1)
        # print('sentence out1',out.size())
        out = out.permute(0, 2, 1)
        # print('sentence out2',out.size())

        # print('sentence out size ',out.shape)
        return out

#PrimaryCaps(832, 8, 9, P, stride=1)
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    # def __init__(self, A=32, B=64, K=9, P=4, stride=1):
    def __init__(self,A, B, K, P, stride):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.pose.weight.data.normal_(0.0, 0.1)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out

#ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B, C, K, P, stride, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        self._lambda = 1e-6
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # self.ln_2pi = torch.cuda.HalfTensor(1).fill_(math.log(2*math.pi))

        # params
        # Note that \beta_u and \beta_a are per capsul/home/bruce/projects/capsulese type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.randn(C,self.psize))
        self.beta_a = nn.Parameter(torch.randn(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K[0]*K[1]*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        cost_h = cost_h.sum(dim=2)


        cost_h_mean = torch.mean(cost_h,dim=1,keepdim=True)

        cost_h_stdv = torch.sqrt(torch.sum(cost_h - cost_h_mean,dim=1,keepdim=True)**2 / C + eps)
        # self._lambda = 1e-03
        # a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))


        # cost_h_mean = cost_h_mean.sum(dim=2)
        # cost_h_stdv = cost_h_stdv.sum(dim=2)

        a_out = self.sigmoid(self._lambda*(self.beta_a - (cost_h_mean -cost_h)/(cost_h_stdv + eps)))

        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(eps + a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        # r = torch.cuda.HalfTensor(b, B, C).fill_(1./C)
        # print(r.dtype)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for h_idx in range(0, h - K + 1, stride)] \
                for k_idx in range(0, K)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def add_pathes2(self,x, B, K=(3, 3), psize=4, stride=(1, 1)):
        b, h, w, c = x.shape
        assert c == B * (psize + 1)

        oh = int((h - K[0] + 1) / stride[0])
        ow = int((w - K[1] + 1) / stride[1])

        idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - K[0] + 1, stride[0])] for k_idx in range(0, K[0])]
        idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - K[1] + 1, stride[1])] for k_idx in range(0, K[1])]

        x = x[:, idxs_h, :, :]
        x = x[:, :, :, idxs_w, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)

        # coor_h = torch.cuda.HalfTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        # coor_w = torch.cuda.HalfTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            # x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)
            x, oh, ow = self.add_pathes2(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()

            p_in=p_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, 1)
            # p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            # a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            # print('conv cap activations',a_out[0].sum().item(),a_out[0].size())
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K[0] and 1 == self.K[1]
            assert 1 == self.stride[0] and 1 == self.stride[1]
            # assert 1 == self.K
            # assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class CapsNet(nn.Module):


    def __init__(self, P=4, pretrained_load=False):
        super(CapsNet, self).__init__()
        self.P = P

        

        self.conv1 = InceptionI3d(157, in_channels=3, final_endpoint='Mixed_4f')
        if pretrained_load:
            pt_path = '../weights/rgb_charades.pt'
            pretrained_weights = torch.load(pt_path)
            weights = self.conv1.state_dict()
            loaded_layers = 0
            for a in weights.keys():
                if a in pretrained_weights.keys():
                    weights[a] = pretrained_weights[a]
                    loaded_layers += 1
            self.conv1.load_state_dict(weights)
            print("Loaded I3D pretrained weights from ", pt_path, " for layers: ", loaded_layers)

        self.primary_caps = PrimaryCaps(832, 32, 9, P, stride=1)
        #self.conv_caps = ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
        self.conv_caps = ConvCaps(32, 24, (1, 1), P, stride=(1, 1), iters=3)

        #self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1 = nn.ConvTranspose2d(384, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1.weight.data.normal_(0.0, 0.02)


        self.upsample2 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample2.weight.data.normal_(0.0, 0.02)

        self.up_conv1 = nn.Conv3d(128, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.up_conv2 = nn.Conv3d(128, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.up_conv3 = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.up_conv4 = nn.Conv3d(128, 1, kernel_size=(3,3,3), padding=(1,1,1))
        #self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=1,output_padding=(0,1,1))
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample3.weight.data.normal_(0.0, 0.02)

        self.upsample4 = nn.ConvTranspose3d(128, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding = (1,1,1))
        self.upsample4.weight.data.normal_(0.0, 0.02)
        
        self.dropout3d = nn.Dropout3d(0.5)

        self.smooth = nn.ConvTranspose3d(128, 1, kernel_size=3,padding = 1)
        self.smooth.weight.data.normal_(0.0, 0.02)


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        # self.sentenceNet = sentenceNet()
        self.sentenceCaps = primarySentenceCaps()

        self.conv28 = nn.Conv2d(832, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv56 = nn.Conv3d(192, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv112 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))



    def load_pretrained_weights(self):
        saved_weights = torch.load('./savedweights/weights_referit')
        self.load_state_dict(saved_weights, strict=False)
        print('loaded referit pretrained weights for whole network')

    def load_previous_weights(self, weightfile):
        saved_weights = torch.load(weightfile)
        self.load_state_dict(saved_weights, strict=False)
        print('loaded weights from previous run: ', weightfile)

    def catcaps(self, wordcaps, imgcaps):
        num_wordcaps = wordcaps.size()[1]
        num_word_poses = int(wordcaps.size()[2] - 1)
        h = imgcaps.size()[1]
        w = imgcaps.size()[2]
        img_data = imgcaps.size()[3]
        num_imgcaps = int(img_data / (self.P * self.P))
        wordcaps = torch.unsqueeze(wordcaps, 1)
        wordcaps = torch.unsqueeze(wordcaps, 1)
        word_caps = wordcaps.repeat(1, h, w, 1, 1)

        word_poses = word_caps[:, :, :, :, :num_word_poses]
        word_poses = word_poses.contiguous().view(-1, h, w, num_wordcaps * num_word_poses)

        word_acts = word_caps[:, :, :, :, num_word_poses]
        word_acts = word_acts.view(-1, h, w, num_wordcaps)

        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, word_poses, img_acts, word_acts), dim=-1)
        return combined_caps
    
    def caps_reorder(self, imgcaps):
        h = imgcaps.size()[1]
        w = imgcaps.size()[2]
        img_data = imgcaps.size()[3]
        num_imgcaps = int(img_data / (self.P * self.P))
        
        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, img_acts), dim=-1)
        return combined_caps
        
        
    def forward(self, img, classification, decoder="trainable"):
        '''
        INPUTS:
        img is of shape (B, 3, T, H, W) - B is batch size, T is number of frames (4 in our experiments), H and W are the height and width of frames (224x224 in our experiments)
        sent is of shape (B, F, N) - B is batch size, F is feature length (300 for word2vec), N is the number of words in the sentence
        classification is of shape (B, ) - B is batch size - this contains the ground-truth class which will be used for masking at training time
        
        OUTPUTS:
        out is a list of segmentation masks (all copies of on another) of shape (B, T, H, W) - B is batch size, T is number of frames (4 in our experiments), H and W is the heights and widths (224x224 in our experiments)
        actor_prediction is the actor prediction (B, C) - B is batch size, C is the number of classes
        
        '''

        x, cross56, cross112 = self.conv1(img)
        # print("conv1 x: ", x.shape)
        
        # For 3d Dropout
        x = self.dropout3d(x)

        x = x.view(-1, 832, 28, 28)
        cross28 = x.clone()
        x = self.primary_caps(x)
        #print("primary_caps x: ", x.shape)

        #x = self.catcaps(sent_caps, x)
        x = self.caps_reorder(x)
        #print("caps_reorder x: ", x.shape)
        
        combined_caps = self.conv_caps(x)
        #print("combined_caps: ", combined_caps.shape)
        #exit()

        h = combined_caps.size()[1]
        w = combined_caps.size()[2]
        caps = int(combined_caps.size()[3] / ((self.P * self.P) + 1))
        range = int(caps * self.P * self.P)
        activations = combined_caps[:, :, :, range:range + caps]
        poses = combined_caps[:, :, :, :range]
        # print(activations.shape)
        actor_prediction = activations
        feat_shape = activations
        # print(feat_shape.shape)
        feat_shape = torch.reshape(feat_shape, (feat_shape.shape[0], feat_shape.shape[1]*feat_shape.shape[2], feat_shape.shape[3]) )
        #print(feat_shape.shape)
        #print(actor_prediction.shape)
        actor_prediction = torch.mean(actor_prediction, 1)
        # print(actor_prediction.shape)
        actor_prediction = torch.mean(actor_prediction, 1)

        #print("training: ", self.training)
        if self.training:
            activations = torch.eye(caps)[classification.long()]
            activations = activations
            activations = activations.cuda()
            # activations = activations.to(self.device)
            activations = activations.view(-1, caps, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations.cuda()
            # activations = activations.to(self.device)

        else:
            activations = torch.eye(caps)[torch.argmax(actor_prediction, dim=1)]
            activations = activations.cuda()
            # activations = activations.to(self.device)
            activations = activations.view(-1, caps, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations.cuda()
            # activations = activations.to(self.device)

        poses = poses.view(-1,h,w,caps,self.P*self.P)
        poses = poses * activations
        poses = poses.view(-1,h,w,range)
        poses = poses.permute(0, 3, 1, 2)

        x = poses
        x = self.relu(self.upsample1(x))

        x = x.view(-1, 64, 1, 28, 28)
        cross28 = cross28.view(-1, 832, 28, 28)
        cross28 = self.relu(self.conv28(cross28))
        cross28 = cross28.view(-1,64,1,28,28)
        x = torch.cat((x, cross28), dim=1)

        # print("shape before up2", x.shape)

        if decoder=='trainable':
            x = self.relu(self.upsample2(x))
            # print("up2: ", x.shape)

        else:
            x = self.relu(F.interpolate(self.up_conv1(x), scale_factor=(2)))
            # print("up2: ", x.shape)
        
        cross56 = self.relu(self.conv56(cross56))
        # print("cross56: ", cross56.shape)
        x = torch.cat((x, cross56), dim=1)

        if decoder=='trainable':
            x = self.relu(self.upsample3(x))
            # print("up3: ", x.shape)

        else:
            x = self.relu(F.interpolate(self.up_conv2(x), scale_factor=(2)))
            # print("up3: ", x.shape)

        cross112 = self.relu(self.conv112(cross112))
        x = torch.cat((x, cross112), dim=1)

        if decoder == 'trainable':
            x = self.upsample4(x)
            # out_upsample_4 = x
            # print("up4: ", x.shape)

        else:
            x = F.interpolate(self.up_conv3(x), scale_factor=(2))
            # print("up4: ", x.shape)
        
        # For 3d Dropout
        x = self.dropout3d(x)
        print(x.shape)

        if decoder=="trainable":
            x = self.smooth(x)
        else:
            x = self.up_conv4(x)

        print(x.shape)
        out_1 = x.view(-1,1,8,224,224)
        # out_1 = x.view(-1,1,8, 112, 112)

        return out_1, actor_prediction, feat_shape

# def get_activation(name):
#     def hook(model, input, output):
#         # print(activation[name])
#         activation[name] = output.detach()
#     return hook

outputs= []
def hook(module, input, output):
    outputs.append(output)

if __name__ == '__main__':
    activation = {}

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = CapsNet()
    # print(model)

    # model = model.cuda()
    # fstack = torch.rand(1, 3, 8, 224, 224).cuda()
    # actor = torch.Tensor([1]).view(1, 1).cuda()
    #sentence = torch.rand(1, 300, 16).cuda()
    model = model.to(device)
    summary(model, [(3, 8, 224, 224), [1, 1]])
    fstack = torch.rand(2, 3, 8, 224, 224).to(device)
    actor = torch.Tensor([1]).view(1, 1).to(device)
    # print(fstack.shape)
    out, ap, feat_shape = model(fstack, actor, 'trainble')
    print(out.shape)
    # print(out[0,0,0].shape)
    # np_out = out[0,0,0].cpu().detach().numpy()
    # print(len(np.unique(np_out)))
    # for name, param in model.named_parameters():
    #     # if param.requires_grad:
    #     print (name, param.data.shape)
    # model.conv_caps.register_forward_hook(hook)
    # # print(bs)
    # print("out_1: ", out[0].shape)
    # print("activations: ", ap.shape)
    # print(feat_shape.shape)
    # print(outputs)
