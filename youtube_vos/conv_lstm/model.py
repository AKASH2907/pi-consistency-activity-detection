import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models import resnet18
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=(28, 28, 1), n_classes=10):
        super(Model, self).__init__()
        h, w, ch = input_dim
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(ch, 32, (3, 3), (1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))
        self.mp1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=(1, 1))
        self.mp2 = nn.MaxPool2d((2, 2), (2, 2))

        self.fc = nn.Linear(64*7*7, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):

        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.mp1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.mp2(x)

        x = x.view(-1, 64*7*7)

        x = self.softmax(self.fc(x))

        return x


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet = resnet18(pretrained=True)
        self.resnet_children = list(self.resnet.children())[:7]

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m):
        f = (in_f - self.mean) / self.std
        m = in_m

        x = self.resnet_children[0](f) + self.conv1_m(m)

        children = []
        for i, child in enumerate(self.resnet_children[1:]):
            x = child(x)
            children.append(x)

        return children


class Decoder(nn.Module):
    def __init__(self, n_features_in):
        super(Decoder, self).__init__()
        self.skip_connection1 = nn.Sequential(nn.Conv3d(128, 128, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU())
        self.skip_connection2 = nn.Sequential(nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU())
        self.skip_connection3 = nn.Sequential(nn.Conv3d(64, 32, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU())

        self.conv_transpose1 = nn.Sequential(nn.ConvTranspose3d(n_features_in, 128, (3, 3, 3), (2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)), nn.ReLU())
        self.conv_transpose2 = nn.Sequential(nn.ConvTranspose3d(128, 64, (3, 3, 3), (2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)), nn.ReLU())
        self.conv_transpose3 = nn.Sequential(nn.ConvTranspose3d(64, 32, (3, 3, 3), (2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)), nn.ReLU())

        self.conv_smooth1 = nn.Sequential(nn.Conv3d(128 + 128, 128, (3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(128), nn.ReLU())
        self.conv_smooth2 = nn.Sequential(nn.Conv3d(64 + 64, 64, (3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(64), nn.ReLU())
        self.conv_smooth3 = nn.Sequential(nn.Conv3d(32 + 32, 32, (3, 3, 3), (1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(32), nn.ReLU())
        self.conv_smooth4 = nn.Sequential(nn.Conv3d(32, 32, (3, 3, 3), (1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(32), nn.ReLU())

        self.segment_layer = nn.Conv3d(32, 1, (1, 3, 3), padding=(0, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, cond_features1, cond_features2, cond_features3, vid_feats3):
        x = self.conv_transpose1(cond_features1)

        skip_connection1 = self.skip_connection1(cond_features2)
        x = torch.cat([x, skip_connection1], 1)
        x = self.conv_smooth1(x)

        x = self.conv_transpose2(x)

        skip_connection2 = self.skip_connection2(cond_features3)
        x = torch.cat([x, skip_connection2], 1)
        x = self.conv_smooth2(x)

        x = self.conv_transpose3(x)

        skip_connection3 = self.skip_connection3(vid_feats3)
        x = torch.cat([x, skip_connection3], 1)

        x = self.conv_smooth3(x)

        x = self.conv_smooth4(x)

        x = self.segment_layer(x)

        interp = F.interpolate(x.squeeze(1), scale_factor=2, mode='bilinear', align_corners=False).unsqueeze(1)
        interp_sigmoid = self.sigmoid(interp)
        return interp, interp_sigmoid


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class LSTMOp(nn.Module):
    def __init__(self, in_vid_feature_dim, in_frame_feature_dim, hidden_dim):
        super(LSTMOp, self).__init__()

        h_state_dim = hidden_dim

        self.init_h = nn.Conv2d(in_frame_feature_dim, h_state_dim, (3, 3), padding=(1, 1))
        self.init_c = nn.Conv2d(in_frame_feature_dim, h_state_dim, (3, 3), padding=(1, 1))

        self.conv_lstm = ConvLSTM(in_vid_feature_dim, h_state_dim, (3, 3), 1, batch_first=True)

    def forward(self, vid_features, frame_features, previous_state=None):
        # vid_features should have shape (B, F_vid, T, H, W)
        # frame_features should have shape (B, F_frames, H, W)

        vid_features_transposed = vid_features.permute(0, 2, 1, 3, 4)  # (B, T, F_frames, H, W)

        if previous_state is None:
            previous_state = [[self.init_h(frame_features), self.init_c(frame_features)]]

        layer_output_list, last_state_list = self.conv_lstm(vid_features_transposed, previous_state)
        layer_output = layer_output_list[-1].permute(0, 2, 1, 3, 4)

        return layer_output, last_state_list


class VOSModel(nn.Module):
    def __init__(self):
        super(VOSModel, self).__init__()

        self.r3d_model = r3d_18(pretrained=True, progress=True)
        self.r3d_model_children = list(self.r3d_model.children())[:5]

        self.frame_encoder = Encoder_M()

        self.lstm_op1 = LSTMOp(512, 256, 256)
        self.lstm_op2 = LSTMOp(256, 128, 128)
        self.lstm_op3 = LSTMOp(128, 64, 64)

        self.decoder = Decoder(256)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, video_input, first_frame, object_mask, previous_state=(None, None, None)):

        # passes the input video through the R3D network (First four layers)

        video_layers = []

        x = (video_input - self.mean) / self.std
        for i, child in enumerate(self.r3d_model_children):
            video_layers.append(child(x))
            x = video_layers[-1]
            #print(x.shape)

        # obtains the features from first frame + mask
        frame_layers = self.frame_encoder(first_frame, object_mask)  # (B, 256, H/16, H/16)

        layer_output, last_state_list = self.lstm_op1(video_layers[-1], frame_layers[-1], previous_state[-1])

        layer_output2, last_state_list2 = self.lstm_op2(video_layers[-2], frame_layers[-2], previous_state[-2])

        layer_output3, last_state_list3 = self.lstm_op3(video_layers[-3], frame_layers[-3], previous_state[-3])

        # decoder network
        logits, segmentation = self.decoder(layer_output, layer_output2, layer_output3, video_layers[1])

        return logits, segmentation, [last_state_list3, last_state_list2, last_state_list]


if __name__ == '__main__':
    video_inputs = torch.autograd.Variable(torch.rand(1, 3, 32, 224, 224))
    first_frame_inputs = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
    object_mask_inputs = torch.autograd.Variable(torch.rand(1, 1, 224, 224))
    if torch.cuda.is_available():
        video_inputs = video_inputs.cuda()
        first_frame_inputs = first_frame_inputs.cuda()
        object_mask_inputs = object_mask_inputs.cuda()

    criterion = nn.BCELoss(reduction='mean')

    model = VOSModel()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-7)
    if torch.cuda.is_available():
        #model = torch.nn.DataParallel(model)
        model.cuda()

    y_pred_logits, y_pred, hidden_state = model(video_inputs, first_frame_inputs, object_mask_inputs)
    print("done")

