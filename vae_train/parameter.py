import torch

class hyper_parameter(object):
    def __init__(self, args):

        # dataset parameter
        self.batch_size = args.batch_size
        self.split_ratio = 0.8
        # 1 mean [0], 2 means [0.05], 3 means [0.1], 4 means [0, 0.05, 0.1]
        if args.noise_mode == 1:
            self.noise_lst = [0]
        elif args.noise_mode == 2:
            self.noise_lst = [0.05]
        elif args.noise_mode == 3:
            self.noise_lst = [0.1]
        elif args.noise_mode == 4:
            self.noise_lst = [0, 0.05, 0.1]
        # self.track_lst = args.track_lst

        # model parameter
        # self.latent_compact = args.latent_compact
        # self.latent_continuous = args.latent_continuous
        # self.LSTM_model = args.LSTM_model
        # self.LSTM_Attention_model = args.LSTM_Attention_model


        # self.N_LSTM_encoder_in_future_feat = 3
        # self.N_encoder_in_future_feat = 15
        # self.N_decoder_in_state_feat = 4
        # self.N_encoder_1 = args.hidden_dim*2
        # self.N_encoder_2 = args.hidden_dim*2
        # self.N_encoder_3 = args.hidden_dim*2
        # self.N_state_encoder = args.hidden_dim
        # self.latent_variable_dim = args.latent_variable_dim
        # self.latent_num_gaussians = 1
        # self.N_decoder_1 = args.hidden_dim*2
        # self.N_decoder_2 = args.hidden_dim*2
        # self.N_decoder_3 = args.hidden_dim*2
        # self.output_variable_dim = 2
        # self.output_num_gaussians = 1

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = 1

        self.restore_epoch = 0


        # self.model_name = 'naive_model'
        # self.load_model_name = 'naive_model'
        # self.model_name = 'naive_model_noise1'
        # self.load_model_name = 'naive_model_noise1'
        # self.model_name = 'naive_model_noise2'
        # self.load_model_name = 'naive_model_noise2'
        

        # training parameter
        self.train_the_model = args.train_the_model
        self.evaluate_model = args.evaluate_model
        self.roll_out_test = args.roll_out_test
        self.visualize_data_distribution = args.visualize_data_distribution
        self.restore_model = args.restore_model
        self.restore_epoch = args.restore_epoch

        self.n_epochs = args.n_epochs
        self.print_every_epoch = 1
        # self.learning_rate = 0.0005
        self.learning_rate = 0.0001
        self.adam_weight_decay = 5e-4
        self.lr_decay_step_size = 5 
        self.lr_decay_gamma = 0.6

        self.latent_compact_beta = 1
        self.latent_continuous_beta = 0.5

        # self.model_name = 'Naive_Model'
        # self.load_model_name = 'Naive_Model'

        self.exp_name = 'zt_jan13_018'
        self.test = True

        self.embedding_dim = 64
        self.h_dim = 64
        self.latent_dim = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.seq_len =  30 #20 # 20 # 30
        self.dt = 0.1
        self.use_relative_pos = True
        self.kld_weight = 20 #10  #20.0 # 0.01, 0.4
        self.fde_weight = 1.5
        self.cum_theta_weight = 1
        self.one_side_class_vae = False  



        self.val_freq = 1