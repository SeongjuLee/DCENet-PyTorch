import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENet(nn.Module):
    def __init__(self, config):
        super(DCENet, self).__init__()

        self.encoder_x = Encoder(config)
        self.encoder_y = Encoder(config)
        self.decoder = CVAE(config)

    def forward(self, x, x_occu, y=None, y_occu=None, train=True):
        x_encoded_dense = self.encoder_x(x, x_occu)
        y_encoded_dense = None

        if train:
            y_encoded_dense = self.encoder_y(y, y_occu)

        # out for train: pred_traj, mu, log_var | out for test: pred_traj
        out = self.decoder(x_encoded_dense, y_encoded_dense, train=train)

        return out


class Encoder(nn.Module):
    def __init__(self, config, x_or_y='x'):
        super(Encoder, self).__init__()

        self.config = config

        # For trajectory
        self.traj_conv1 = nn.Conv1d(in_channels=2, out_channels=config['n_hidden'] // 16, kernel_size=3, stride=1, padding=1)
        self.traj_fc = nn.Sequential(
            nn.Linear(in_features=config['n_hidden'] // 16, out_features=config['n_hidden'] // 8),
            nn.ReLU()
            )
        self.traj_pos_encode = PositionalEncoding(
            d_in=config['n_hidden'] // 8,
            d_model=config['{}_encoder_dim'.format(x_or_y)]
            )
        self.traj_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['{}_encoder_dim'.format(x_or_y)],
            nhead=config['{}_encoder_head'.format(x_or_y)],
            dim_feedforward=config['{}_encoder_dim'.format(x_or_y)]
            )
        self.traj_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.traj_transformer_encoder_layer, 
            num_layers=config['{}_encoder_layers'.format(x_or_y)]
            )
        self.traj_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
            )

        # For dynamic map
        self.occu_model = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),     # Tensorflow padding 'SAME' option
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),     # Tensorflow padding 'SAME' option
            nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0),
            nn.Dropout(p=config['o_drop']),
            nn.Flatten()
            )
        self.occu_time_distributed = TimeDistributed(self.occu_model, tdim=1)
        self.occu_pos_encode = PositionalEncoding(
            d_in=6144,
            d_model=config['occu_encoder_{}_dim'.format(x_or_y)]
            )
        self.occu_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['occu_encoder_{}_dim'.format(x_or_y)],
            nhead=config['occu_encoder_{}_head'.format(x_or_y)],
            dim_feedforward=config['occu_encoder_{}_dim'.format(x_or_y)]
            )
        self.occu_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.occu_transformer_encoder_layer,
            num_layers=config['occu_encoder_{}_layers'.format(x_or_y)]
            )
        self.occu_lstm = nn.LSTM(
            input_size=config['occu_encoder_{}_dim'.format(x_or_y)],
            hidden_size=config['hidden_size'],
            num_layers=1,
            batch_first=True,
            )
        self.occu_dropout = nn.Dropout(p=config['s_drop'])

        # For encoding
        self.encode_fc = nn.Sequential(
            nn.Linear(in_features=config['{}_encoder_dim'.format(x_or_y)] + config['hidden_size'], out_features=config['encoder_dim']),
            nn.ReLU()
        )

        # print(sum(p.numel() for p in self.occu_time_distributed.parameters() if p.requires_grad))

    def init_hidden(self, x):
        h0 = torch.zeros((1, x.size(0), self.config['hidden_size'])).cuda()
        c0 = torch.zeros((1, x.size(0), self.config['hidden_size'])).cuda()
        
        return h0, c0

    def forward(self, traj, dmap):
        traj = traj.transpose(1, 2)
        traj = self.traj_conv1(traj)
        traj = traj.transpose(1, 2)
        traj = self.traj_fc(traj)
        traj = traj.transpose(0, 1)  # (L, B, H)
        traj = self.traj_pos_encode(traj)
        traj = self.traj_transformer_encoder(traj)
        traj = traj.transpose(0, 1).transpose(1, 2)
        traj = self.traj_avg_pool(traj)

        dmap = self.occu_time_distributed(dmap)
        dmap = dmap.transpose(0, 1)
        dmap = self.occu_pos_encode(dmap)
        dmap = self.occu_transformer_encoder(dmap)
        dmap = dmap.transpose(0, 1)
        dmap_hidden = self.init_hidden(dmap)
        dmap_out, dmap_hidden = self.occu_lstm(dmap, dmap_hidden)
        dmap = dmap_out[:, -1, :]
        dmap = self.occu_dropout(dmap)

        out = torch.cat((traj, dmap), dim=1)
        out = self.encode_fc(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_in, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_model),
            nn.ReLU()
        )
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.fc(x)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            #only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.reshape(bs*seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            return out.view(bs, seq_len,*out_shape[1:])
    
    def low_mem_forward(self, *args, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)
    def __repr__(self):
        return f'TimeDistributed({self.module})'


class CVAE(nn.Module):
    def __init__(self, config):
        super(CVAE, self).__init__()
        self.z_dim = config['z_dim']
        self.pred_seq = config['pred_seq']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For concatenated input
        self.xy_encoded_fc1 = nn.Sequential(
            nn.Linear(in_features=config['encoder_dim'] * 2, out_features=config['n_hidden']),
            nn.ReLU()
            )
        self.xy_encoded_fc2 = nn.Sequential(
            nn.Linear(in_features=config['n_hidden'], out_features=config['n_hidden'] // 2),
            nn.ReLU()
            )
        self.mu = nn.Linear(config['n_hidden'] // 2, config['z_dim'])
        self.log_var = nn.Linear(config['n_hidden'] // 2, config['z_dim'])

        # decoder part
        self.z_fc = nn.Sequential(
            nn.Linear(config['z_dim'] + config['encoder_dim'], config['n_hidden'] // 2),
            nn.ReLU()
            )
        self.z_lstm = nn.LSTM(input_size=config['n_hidden'] // 2, hidden_size=config['z_decoder_dim'], num_layers=1, batch_first=True)
        self.z_dropout = nn.Dropout(p=config['z_drop'])

        self.y_decoder_model = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=config['z_decoder_dim'], stride=config['z_decoder_dim'])
        self.y_decoder = TimeDistributed(self.y_decoder_model, tdim=1)

    def encoder(self, x, y):
        concat_input = torch.cat([x, y], 1)
        h = self.xy_encoded_fc1(concat_input)
        h = self.xy_encoded_fc2(h)
        return self.mu(h), self.log_var(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, x):
        concat_input = torch.cat([z, x], 1)
        h = self.z_fc(concat_input)
        h = h.view(h.size(0), 1, -1).repeat(1, self.pred_seq, 1)   # 12 for predicted sequence length
        h, _ = self.z_lstm(h)
        h = self.z_dropout(torch.tanh(h))
        h = h.view(h.size(0), h.size(1), 1, -1)
        out = self.y_decoder(h).view(h.size(0), h.size(1), -1)
        return out
    
    def forward(self, x, y=None, train=True):
        if train:
            mu, log_var = self.encoder(x, y)
            z = self.sampling(mu, log_var)
            return self.decoder(z, x), mu, log_var
        else:
            z = torch.randn((x.size(0), self.z_dim)).to(self.device)
            return self.decoder(z, x)
