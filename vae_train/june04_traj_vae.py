import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent

def one_hot(val: torch.LongTensor, num: int, num_first: bool = False) -> torch.FloatTensor:
    r"""
    Overview:
        Convert a ``torch.LongTensor`` to one hot encoding.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot``
    Arguments:
        - val (:obj:`torch.LongTensor`): each element contains the state to be encoded, the range should be [0, num-1]
        - num (:obj:`int`): number of states of the one hot encoding
        - num_first (:obj:`bool`): If ``num_first`` is False, the one hot encoding is added as the last; \
            Otherwise as the first dimension.
    Returns:
        - one_hot (:obj:`torch.FloatTensor`)
    Example:
        >>> one_hot(2*torch.ones([2,2]).long(),3)
        tensor([[[0., 0., 1.],
                 [0., 0., 1.]],
                [[0., 0., 1.],
                 [0., 0., 1.]]])
        >>> one_hot(2*torch.ones([2,2]).long(),3,num_first=True)
        tensor([[[0., 0.], [1., 0.]],
                [[0., 1.], [0., 0.]],
                [[1., 0.], [0., 1.]]])
    """
    assert (isinstance(val, torch.Tensor)), type(val)
    assert val.dtype == torch.long
    assert (len(val.shape) >= 1)
    old_shape = val.shape
    val_reshape = val.reshape(-1, 1)
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # To remember the location where the original value is -1 in val.
    # If the value is -1, then it should be converted to all zeros encodings and
    # the corresponding entry in index_neg_one is 1, which is used to transform
    # the ret after the operation of ret.scatter_(1, val_reshape, 1) to their correct encodings bellowing
    index_neg_one = torch.eq(val_reshape, -1).long()
    if index_neg_one.sum() != 0:  # if -1 exists in val
        # convert the original value -1 to 0
        val_reshape = torch.where(
            val_reshape != -1, val_reshape,
            torch.zeros(val_reshape.shape, device=val.device).long()
        )
    try:
        ret.scatter_(1, val_reshape, 1)
        if index_neg_one.sum() != 0:  # if -1 exists in val
            ret = ret * (1 - index_neg_one)  # change -1's encoding from [1,0,...,0] to [0,0,...,0]
    except RuntimeError:
        raise RuntimeError('value: {}\nnum: {}\t:val_shape: {}\n'.format(val_reshape, num, val_reshape.shape))
    if num_first:
        return ret.permute(1, 0).reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)

class VaeEncoder(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(VaeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.label_dim = 2
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.device = torch.device('cuda:0')

        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)

        enc_mid_dims = [self.h_dim, self.h_dim, self.h_dim, self.latent_dim]
        mu_modules = []
        sigma_modules = []
        in_channels = self.h_dim
        for m_dim in enc_mid_dims:
            mu_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            sigma_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.mean = nn.Sequential(*mu_modules) 
        self.log_var = nn.Sequential(*sigma_modules)
        self.encoder = nn.LSTM(self.embedding_dim + self.label_dim, self.h_dim, self.num_layers)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device)
        )
    def get_relative_position(self, abs_traj):
        # abs_traj shape: batch_size x seq_len x 4
        # rel traj shape: batch_size x seq_len -1 x 2
        rel_traj = abs_traj[:, 1:, :2] - abs_traj[:, :-1, :2]
        rel_traj = torch.cat([abs_traj[:, 0, :2].unsqueeze(1), rel_traj], dim = 1)
        rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:]],dim=2)
        #rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:].unsqueeze(2)],dim=2)
        # rel_traj shape: batch_size x seq_len x 4
        return rel_traj
    
    def encode(self, input, traj_label):
        # input meaning: a trajectory len 25 and contains x, y , theta, v
        # input shape: batch x seq_len x 4
        #data_traj shape: seq_len x batch x 4
        if self.use_relative_pos:
            input = self.get_relative_position(input)
            input = input[:,:,:2]
        traj_label_onehot = one_hot(traj_label.long(),num=2).unsqueeze(0)
        traj_label_onehot = traj_label_onehot.repeat(self.seq_len, 1, 1)
        data_traj = input.permute(1, 0, 2).contiguous()
        traj_embedding = self.spatial_embedding(data_traj.view(-1, 2))
        traj_embedding = traj_embedding.view(self.seq_len, -1, self.embedding_dim)
        # Here we do not specify batch_size to self.batch_size because when testing maybe batch will vary
        batch_size = traj_embedding.shape[1]
        hidden_tuple = self.init_hidden(batch_size)
        traj_embedding = torch.cat([traj_label_onehot, traj_embedding], 2)
        output, encoder_h = self.encoder(traj_embedding, hidden_tuple)
        mu = self.mean(encoder_h[0])
        log_var = self.log_var(encoder_h[0])
        #mu, log_var = torch.tanh(mu), torch.tanh(log_var)
        return mu, log_var

    def forward(self, input, traj_label):
        return self.encode(input, traj_label)


class VaeDecoder(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(VaeDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.label_dim = 2
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(4, self.embedding_dim)
        # input: h_dim, output: throttle, steer
        self.hidden2control = nn.Linear(self.h_dim, 2)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim, out_features = self.h_dim * self.num_layers)
        
        label_dims = [self.h_dim, self.h_dim, self.h_dim, self.label_dim]
        label_modules = []
        in_channels = self.latent_dim
        for m_dim in label_dims:
            label_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.label_classification = nn.Sequential(*label_modules) 

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        #pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 /2,3.14 /2)
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        output_label = self.label_classification(z)
        #output_label = F.softmax(output_label, dim=2)
        # decoder_input shape: batch_size x 4
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
            #decoder_h.unsqueeze(0)
        decoder_h = (decoder_h, decoder_h)
        for _ in range(self.seq_len):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder(decoder_input, decoder_h)
            control = self.hidden2control(output.view(-1, self.h_dim))
            curr_state = self.plant_model_batch(prev_state, control[:,0], control[:,1], self.dt)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj, output_label
    
    def forward(self, z, init_state):
        return self.decode(z, init_state)

class TrajVAE(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        kld_weight = 0.01,
        fde_weight = 0.1,
        ):
        super(TrajVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.kld_weight = kld_weight
        self.fde_weight = fde_weight
        self.dt = dt
        self.vae_encoder = VaeEncoder(
            embedding_dim = self.embedding_dim,
            h_dim = self.h_dim,
            latent_dim = self.latent_dim,
            seq_len = self.seq_len,
            use_relative_pos = self.use_relative_pos,
            dt = self.dt
        )
        self.vae_decoder = VaeDecoder(
            embedding_dim = self.embedding_dim,
            h_dim = self.h_dim,
            latent_dim = self.latent_dim,
            seq_len = self.seq_len,
            use_relative_pos = self.use_relative_pos,
            dt = self.dt
        )

    def reparameterize(self, mu, logvar):
        # mu shape: batch size x latent_dim
        # sigma shape: batch_size x latent_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        #return mu
    
    def forward(self, expert_traj, init_state, traj_label = None):
        mu, log_var = self.vae_encoder(expert_traj, traj_label)
        z = self.reparameterize(mu, log_var)
        #z = torch.tanh(z)
        recons_traj, output_label = self.vae_decoder(z, init_state)
        #recons_traj = recons_traj[:,:,[0,1,3]]
        return [recons_traj, expert_traj, mu.squeeze(0), log_var.squeeze(0), output_label.squeeze(0), traj_label]  

    def loss_function(self, *args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        output_label = args[4]
        ground_truth_label = args[5]
        ground_truth_label = ground_truth_label.long()
        traj_mask = args[6]
        # epoch = 0
        # if len(args) > 4:
        #     epoch = args[4]
        kld_weight = self.kld_weight
        recon_loss = 0
        classification_loss_function =torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_function(output_label, ground_truth_label)
        # reconstruction loss
        recons_loss = F.mse_loss(recons[:,:,:2] * traj_mask[:,:,:2], input[:,:,:2]*traj_mask[:,:,:2])
        #recons_loss += F.mse_loss(recons[:,:,3], input[:,:,3]) * 0.01
        vel_loss = F.mse_loss(recons[:,:,3]* traj_mask[:,:,3], input[:,:,3]* traj_mask[:,:,3]) * 0.01 
        #final displacement loss
        final_displacement_error = F.mse_loss(recons[:,-1, :2]* traj_mask[:,-1, :2], input[:, -1, :2]*traj_mask[:, -1, :2])
        theta_error = F.mse_loss(recons[:,:,2]*traj_mask[:,:,2], input[:,:,2] * np.pi / 180*traj_mask[:,:,2]) * 0.01
        final_theta_error = 0.0
        final_theta_error = F.mse_loss(recons[:,-1,2] *traj_mask[:,-1,2], input[:,-1,2]*traj_mask[:,-1,2] * np.pi / 180) * 10
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = 0.01
        loss = recons_loss  + kld_weight * kld_loss + self.fde_weight * final_displacement_error + theta_error  + vel_loss + final_theta_error + classification_loss
        # print('kld_weight: {}'.format(kld_weight))
        # print('epoch: {} '.format(epoch))
        return {'loss': loss, "reconstruction_loss": recons_loss, 'KLD': kld_loss, 'final_displacement_error' : final_displacement_error, 
        'final_theta_error': final_theta_error,'theta_error':theta_error, 'mu':mu[0][0], 'log_var': log_var[0][0], 'classification_loss': classification_loss}    

    def sample(self, batch_z, init_state):
        with torch.no_grad():
            samples, output_labels = self.vae_decoder(batch_z, init_state)
        return samples, output_labels


def create_model(params):
    
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    model = TrajVAE(
        embedding_dim = params.embedding_dim,
        h_dim = params.h_dim,
        latent_dim = params.latent_dim,
        seq_len = params.seq_len,
        dt = params.dt,
    )
    
    model = model.float()
    model.to(params.device)

    return model