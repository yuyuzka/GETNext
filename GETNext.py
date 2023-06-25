import torch.nn as nn
from model import *


class GETNext(nn.Module):
    def __int__(self, args, num_users, num_cats, num_pois, nodes_feature):
        super(GETNext, self).__init__()

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

            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            # 通过GCN得到POI_embedding,图进图出
            poi_embeddings = self.poi_embed_model(X, A)

            for sample in data:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                input_seq_time = [each[1] for each in sample[1]]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)

        def input_traj_to_embeddings(sample, poi_embeddings):
            # Parse sample
            traj_id = sample[0]
            input_seq = [each[0] for each in sample[1]]
            input_seq_time = [each[1] for each in sample[1]]
            input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

            # User to embedding
            user_id = traj_id.split('_')[0]
            user_idx = user2id_dict[user_id]
            input = torch.LongTensor([user_idx]).to(device=args.device)
            user_embedding = user_embed_model(input)
            user_embedding = torch.squeeze(user_embedding)

            # POI to embedding and fuse embeddings
            input_seq_embed = []
            for idx in range(len(input_seq)):
                poi_embedding = poi_embeddings[input_seq[idx]]
                poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

                # Time to vector
                time_embedding = self.time_embed_model(
                    torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
                time_embedding = torch.squeeze(time_embedding).to(device=args.device)

                # Categroy to embedding
                cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
                cat_embedding = self.cat_embed_model(cat_idx)
                cat_embedding = torch.squeeze(cat_embedding)

                # Fuse user+poi embeds
                fused_embedding1 = self.embed_fuse_model1(user_embedding, poi_embedding)
                fused_embedding2 = self.embed_fuse_model2(time_embedding, cat_embedding)

                # Concat time, cat after user+poi
                concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

                # Save final embed
                input_seq_embed.append(concat_embedding)

            return input_seq_embed

        def adjust_pred_prob_by_graph(y_pred_poi):
            y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
            attn_map = self.node_attn_model(X, A)

            for i in range(len(batch_seq_lens)):
                traj_i_input = batch_input_seqs[i]  # list of input check-in pois
                for j in range(len(traj_i_input)):
                    y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

            return y_pred_poi_adjusted


