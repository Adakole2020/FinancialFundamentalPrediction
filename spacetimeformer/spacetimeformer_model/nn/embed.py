import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import spacetimeformer as stf

from .extra_layers import ConvBlock, Flatten
from einops import repeat

class SpacetimeformerEmbedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x,
        d_model=256,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=1,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        position_emb="abs",
        data_dropout=None,
        max_seq_len=None,
    ):
        """
        SpacetimeformerEmbedding class for embedding inputs in the Spacetimeformer model.

        Args:
            d_y (int): Dimensionality of the input y.
            d_x (int): Dimensionality of the input x.
            d_model (int, optional): Dimensionality of the embedding. Defaults to 256.
            time_emb_dim (int, optional): Dimensionality of the time embedding. Defaults to 6.
            method (str, optional): Embedding method. Can be "spatio-temporal" or "temporal". Defaults to "spatio-temporal".
            downsample_convs (int, optional): Number of downsampling convolutions. Defaults to 1.
            start_token_len (int, optional): Length of the start token. Defaults to 0.
            null_value (None, optional): Null value for masking. Defaults to None.
        """
        super().__init__()

        assert method in ["spatio-temporal", "temporal"]
        if data_dropout is None:
            self.data_drop = lambda y: y
        else:
            self.data_drop = data_dropout
            
        self.method = method

        time_dim = time_emb_dim * d_x
        self.time_emb = stf.Time2Vec(d_x, embed_dim=time_dim)# [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]

        assert position_emb in ["t2v", "abs"]
        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = stf.Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )

        y_emb_inp_dim = d_y if self.method == "temporal" else 1
        self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim, d_model)

        if self.method == "spatio-temporal":
            self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
            split_length_into = d_y
        else:
            split_length_into = 1

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [ConvBlock(split_length_into, d_model)for _ in range(downsample_convs)]
        )

        self.d_model = d_model
        self.null_value = null_value
        self.pad_value = pad_value
        self.is_encoder = is_encoder

    def __call__(self, y, x):
        """
        Forward pass of the SpacetimeformerEmbedding module.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.

        Returns:
            torch.Tensor: Embedded values, space embeddings, and variable indices.
        """
        if self.method == "spatio-temporal":
            emb = self.spatio_temporal_embed
        else:
            emb = self.temporal_embed
        return emb(y=y, x=x)
    
    def make_mask(self, y):
        # we make padding-based masks here due to outdated
        # feature where the embedding randomly drops tokens by setting
        # them to the pad value as a form of regularization
        if self.pad_value is None:
            return None
        return (y == self.pad_value).any(-1, keepdim=True)

    def temporal_embed(self, y, x):
        """
        Temporal embedding method.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.
            is_encoder (bool, optional): Whether the method is used in the encoder or decoder. Defaults to True.

        Returns:
            torch.Tensor: Embedded values and space embeddings.
        """
        bs, length, d_y = y.shape
        
        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)
        
        # position embedding ("local_emb")
        local_pos = torch.arange(length).to(x.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        if not self.TIME:
            x = torch.zeros_like(x)
        time_emb = self.time_emb(x) # [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]

        # val embedding
        if not self.VAL:
            y = torch.zeros_like(y)
        val_time_inp = torch.cat((y, time_emb), dim=-1) # [bs, length, d_y + time_emb_dim * d_x]
        val_time_emb = self.val_time_emb(val_time_inp) # [bs, length, d_model]

        # "given" embedding
        given = torch.ones((bs, length)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given) # [bs, length, d_model]
        emb += given_emb # [bs, length, d_model]

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)
                
        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None

        return emb, space_emb, var_idxs, mask

    SPACE = True
    TIME = True
    VAL = True
    GIVEN = True

    def spatio_temporal_embed(self, y, x, is_encoder=True):
        """
        Spatio-temporal embedding method.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.
            is_encoder (bool, optional): Whether the method is used in the encoder or decoder. Defaults to True.

        Returns:
            torch.Tensor: Embedded values, variable embeddings, and variable indices.
        """
        bs, length, d_y = y.shape
        
        # position emb ("local_emb")
        local_pos = repeat(
            torch.arange(length).to(x.device), f"length -> {bs} ({d_y} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())
            
        # time emb
        if not self.TIME:
            x = torch.zeros_like(x)
        x = torch.nan_to_num(x)
        x = repeat(x, f"batch len x_dim -> batch ({d_y} len) x_dim")
        time_emb = self.time_emb(x).repeat(1, d_y, 1) # [bs, length, d_x] -> [bs, length * d_y, time_emb_dim * d_x]
        
        # protect against NaNs in y, but keep track for Given emb
        true_null = torch.isnan(y)
        y = torch.nan_to_num(y)
        if not self.VAL:
            y = torch.zeros_like(y)

        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1) # [bs, length, d_y] -> [bs, length * d_y]
        val_time_inp = torch.cat((y, time_emb), dim=-1) # [bs, length * d_y, 1 + time_emb_dim * d_x]
        val_time_emb = self.val_time_emb(val_time_inp) # [bs, length * d_y, d_model]

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T # [bs, length, d_y]
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0
            
            # if y was NaN, set Given = False
            given *= ~true_null
                
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1) # [bs, length, d_y] -> [bs, length * d_y]
            if self.null_value is not None:
                # mask null values
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask
                
            given_emb = self.given_emb(given) # [bs, length * d_y, d_model]
        else:
            given_emb=0.0
            
        val_time_emb = local_emb + val_time_emb + given_emb # [bs, length * d_y, d_model]
            
        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
        var_idx = repeat(
            torch.arange(dy).long().to(x.device), f"dy -> {bs} (dy {length})"
        )
        var_idx_true = var_idx.clone()
        if not self.use_space:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask

class SpacetimeformerEmbeddingWithCategoricals(SpacetimeformerEmbedding):
    """
    Spacetimeformer embedding with categorical variables.

    This embedding assumes that the categoricals have been discretized and are included in order as the final values of y.

    Args:
        d_y (int): Dimensionality of the y input.
        d_x (int): Dimensionality of the x input.
        d_model (int, optional): Dimensionality of the model. Defaults to 256.
        time_emb_dim (int, optional): Dimensionality of the time embedding. Defaults to 6.
        method (str, optional): Embedding method. Can be "spatio-temporal" or "temporal". Defaults to "spatio-temporal".
        downsample_convs (int, optional): Number of downsampling convolutions. Defaults to 1.
        start_token_len (int, optional): Length of the start token. Defaults to 0.
        null_value (int, optional): Null value for masking. Defaults to None.
        categorical_dict_sizes (list[int], optional): List of sizes of the categorical dictionaries. Defaults to None.
        categorical_embedding_dim (int, optional): Dimensionality of the categorical embeddings. Defaults to 32.
    """
    
    def __init__(
        self,
        d_y,
        d_x,
        d_model=256,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=0,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        position_emb="abs",
        data_dropout=None,
        max_seq_len=None,
        categorical_dict_sizes=None,
        categorical_embedding_dim=32,
    ):
        super().__init__(
            d_y,
            d_x,
            d_model,
            time_emb_dim,
            method,
            downsample_convs,
            start_token_len,
            null_value,
            pad_value,
            is_encoder,
            position_emb,
            data_dropout,
            max_seq_len,
        )
        if self.is_encoder:
            self.categorical_dict_sizes = categorical_dict_sizes
            self.categorical_embedding_dim = categorical_embedding_dim
        
            if categorical_dict_sizes:
                self.cat_emb = nn.ModuleList([
                    nn.Embedding(num_embeddings=size, embedding_dim=categorical_embedding_dim)
                    for size in categorical_dict_sizes
                ])
        
        if self.method == "temporal":
            if self.is_encoder:
                y_emb_inp_dim = d_y - (categorical_embedding_dim - 1) * len(categorical_dict_sizes)
            else:
                y_emb_inp_dim = d_y
        else:
            if self.is_encoder:
                y_emb_inp_dim = categorical_embedding_dim
            else:
                y_emb_inp_dim = 1
        
        time_dim = time_emb_dim * d_x
        self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim, d_model)
        
    
    def temporal_embed(self, y, x, is_encoder=True):
        """
        Temporal embedding method.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.
            is_encoder (bool, optional): Whether the method is used in the encoder or decoder. Defaults to True.

        Returns:
            torch.Tensor: Embedded values and space embeddings.
        """
        bs, length, d_y = y.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)
        
        # position embedding ("local_emb")
        local_pos = torch.arange(length).to(x.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        if not self.TIME:
            x = torch.zeros_like(x)
        time_emb = self.time_emb(x) # [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]

        # val embedding
        if not self.VAL:
            y = torch.zeros_like(y)

        # "given" embedding
        given = torch.ones((bs, length)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given) # [bs, length, d_model]
            
        if self.is_encoder and self.categorical_dict_sizes:
            k = len(self.categorical_dict_sizes)
            for i, emb in enumerate(self.cat_emb):
                cat = y[:, :, d_y-k+i].long()
                cat_emb = emb(cat)
                y = torch.cat((y, cat_emb), dim=-1)
            
            y = np.concatenate((y[:, :, :d_y-k], y[:, :, d_y:]), axis=2) # [bs, length, d_y] -> [bs, length, d_y - k + (categorical_embedding_dim - 1) * k]
        val_time_inp = torch.cat((y, time_emb), dim=-1) # [bs, length, d_y + time_emb_dim * d_x]
        val_time_emb = self.val_time_emb(val_time_inp) # [bs, length, d_model]

        
        emb = local_emb + val_time_emb + given_emb # [bs, length, d_model]

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)
                
        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None

        return emb, space_emb, var_idxs, mask
    
    def spatio_temporal_embed(self, y, x, is_encoder=True):
        """
        Spatio-temporal embedding method.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.
            is_encoder (bool, optional): Whether the method is used in the encoder or decoder. Defaults to True.

        Returns:
            torch.Tensor: Embedded values, variable embeddings, and variable indices.
        """
        bs, length, d_y = y.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)
        
        # position emb ("local_emb")
        local_pos = repeat(
            torch.arange(length).to(x.device), f"length -> {bs} ({d_y} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())
            
        # time emb
        if not self.TIME:
            x = torch.zeros_like(x)
        time_emb = self.time_emb(x).repeat(1, d_y, 1) # [bs, length, d_x] -> [bs, length * d_y, time_emb_dim * d_x]
        
        # protect against NaNs in y, but keep track for Given emb
        true_null = torch.isnan(y)
        if not self.VAL:
            y = torch.zeros_like(y)
            
        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1) # [bs, length, d_y] -> [bs, length * d_y, 1]

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T # [bs, length, d_y]
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0
            
            # if y was NaN, set Given = False
            given *= ~true_null
                
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1) # [bs, length, d_y] -> [bs, length * d_y]

            if self.null_value is not None:
                # mask null values
                null_mask = (y.squeeze(-1) != self.null_value)
                given *= null_mask
                
            given_emb = self.given_emb(given) # [bs, length * d_y, d_model]
        else:
            given_emb=0.0
        
        if self.is_encoder and self.categorical_dict_sizes:
            # Reshape the tensor to (bs, length_dy, 1)
            # y = torch.unsqueeze(y,1)
            # Concatenate zeros along the third dimension to make it (bs, length*d_y, categorical_embedding_dim)
            zeros = torch.zeros((bs, length * d_y, self.categorical_embedding_dim-1)).long().to(x.device) 
            y = torch.cat((y, zeros), dim=-1)
            
            categorical_dim = len(self.categorical_dict_sizes)
            # For each row in the batch, replace the last categorical_dim elements of the second dimension with the embeddings
            for i in range(bs):
                for j in range(length*d_y):
                    if (j % d_y) >= (d_y-categorical_dim):
                        # Apply PyTorch embedding for this categorical variable
                        embedding = self.cat_emb[(j % d_y) - d_y + categorical_dim](y[i, j, 0].clone().detach().long())
                        # Replace the entire third dimension for that row with the embedding
                        y[i, j] = embedding.unsqueeze(0)
        
        val_time_inp = torch.cat((y, time_emb), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp) # [bs, length * d_y, d_model]
     
        val_time_emb = local_emb + val_time_emb + given_emb # [bs, length * d_y, d_model]
            
        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
        var_idx = repeat(
            torch.arange(d_y).long().to(x.device), f"dy -> {bs} (dy {length})"
        )
        var_idx_true = var_idx.clone()
        if not self.SPACE:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask