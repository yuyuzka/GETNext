import torch.nn as nn
from model import *


class GETNext(nn.Module):
    def __int__(self, args, num_users, num_cats, num_pois, nodes_feature):
        super(GETNext)
        self.poi_embed_model = GCN(ninput=args.gcn_nfeat,
                                   nhid=args.gcn_nhid,
                                   noutput=args.poi_embed_dim,
                                   dropout=args.gcn_dropout)

        # Node Attn Model
        self.node_attn_model = NodeAttnMap(in_features=nodes_feature, nhid=args.node_attn_nhid, use_mask=False)

        # %% Model2: User embedding model, nn.embedding
        self.user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

        # %% Model3: Time Model
        self.time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

        # %% Model4: Category embedding model
        self.cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

        # %% Model5: Embedding fusion models
        self.embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
        self.embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

        # %% Model6: Sequence model
        args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
        self.seq_model = TransformerModel(num_pois,
                                          num_cats,
                                          args.seq_input_embed,
                                          args.transformer_nhead,
                                          args.transformer_nhid,
                                          args.transformer_nlayers,
                                          dropout=args.transformer_dropout)

        def forward(self, data):
            pass


