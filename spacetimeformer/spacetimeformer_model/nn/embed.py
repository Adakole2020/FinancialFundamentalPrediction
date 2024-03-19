import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import spacetimeformer as stf

from .encoder import VariableDownsample

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
        downsample_convs=1,
        start_token_len=0,
        null_value=None,
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
        )
        
        self.categorical_dict_sizes = categorical_dict_sizes
        self.categorical_embedding_dim = categorical_embedding_dim
        
        self.cat_emb = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=categorical_embedding_dim)
            for size in categorical_dict_sizes
        ])
        
        if self.method == "temporal":
            y_emb_inp_dim = d_y + (categorical_embedding_dim - 1) * len(categorical_dict_sizes) + (time_emb_dim * d_x)
        else:
            y_emb_inp_dim = categorical_embedding_dim + (time_emb_dim * d_x)

        self.y_emb = nn.Linear(y_emb_inp_dim, d_model)
        
    
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

        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        if not self.TIME:
            x = torch.zeros_like(x)
        x = torch.cat((x, local_pos), dim=-1) # [bs, length, x.size] -> [bs, length, x.size + 1]
        t2v_emb = self.x_emb(x) # [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]
        
        k = len(self.categorical_dict_sizes)
        for i, emb in enumerate(self.cat_emb):
            cat = y[:, :, d_y-k+i].long()
            cat_emb = emb(cat)
            y = torch.cat((y, cat_emb), dim=-1)
        
        y = np.concatenate((y[:, :, :d_y-k], y[:, :, d_y:]), axis=2) # [bs, length, d_y] -> [bs, length, d_y - k + (categorical_embedding_dim - 1) * k]
        
        # val embedding
        emb_inp = torch.cat((y, t2v_emb), dim=-1) # [bs, length, d_y - k + (categorical_embedding_dim - 1) * k] -> [bs, length, d_y - k + (categorical_embedding_dim - 1) * k + time_emb_dim * d_x]
        emb = self.y_emb(emb_inp) # [bs, length, d_model]

        # "given" embedding
        given = torch.ones((bs, length)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given) # [bs, length, d_model]
        emb += given_emb # [bs, length, d_model]

        if is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        return emb, torch.zeros_like(emb)
    
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

        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1) # [bs, length, d_y] -> [bs, length * d_y]
        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        ) # [bs, length, 1]
        x = torch.cat((x, local_pos), dim=-1) # [bs, length, x.size] -> [bs, length, x.size + 1]
        if not self.TIME:
            x = torch.zeros_like(x)
        if not self.VAL:
            y = torch.zeros_like(y)
        t2v_emb = self.x_emb(x).repeat(1, d_y, 1) # [bs, length, d_x] -> [bs, length * d_y, time_emb_dim * d_x]
        # Reshape the tensor to (bs, length_dy, 1)
        y = torch.unsqueeze(y,2)
        # Concatenate zeros along the third dimension to make it (bs, length*d_y, categorical_embedding_dim)
        zeros = torch.zeros((bs, length * d_y, self.categorical_embedding_dim-1)).long().to(x.device) 
        y = torch.cat((y, zeros), dim=-1)
        
        categorical_dim = len(self.categorical_dict_sizes)
        
        # For each row in the batch, replace the last categorical_dim elements of the second dimension with the embeddings
        for i in range(bs):
            for j in range(length*d_y):
                if j % d_y >= (d_y-categorical_dim):
                    # Apply PyTorch embedding for this categorical variable
                    embedding = self.cat_emb[(j % d_y) - categorical_dim](y[i, j, 0].clone().detach().long())
                    # Replace the entire third dimension for that row with the embedding
                    y[i, j] = embedding.unsqueeze(0)
        
        
        val_time_inp = torch.cat((y, t2v_emb), dim=-1) # [bs, length * d_y, categorical_embedding_dim + time_emb_dim * d_x]
        val_time_emb = self.y_emb(val_time_inp) # [bs, length * d_y, d_model]

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T # [bs, length, d_y]
            if not is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1) # [bs, length, d_y] -> [bs, length * d_y]
            if self.null_value is not None:
                # mask null values
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask
            given_emb = self.given_emb(given) # [bs, length * d_y, d_model]
            val_time_emb += given_emb # [bs, length * d_y, d_model]

        if is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # var embedding
        var_idx = torch.Tensor([[i for j in range(length)] for i in range(d_y)])
        var_idx = var_idx.long().to(x.device).view(-1).unsqueeze(0).repeat(bs, 1)
        var_idx_true = var_idx.clone()
        if not self.SPACE:
            var_idx = torch.zeros_like(var_idx)
        var_emb = self.var_emb(var_idx)

        return val_time_emb, var_emb, var_idx_true



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
        self.method = method

        # account for added local position indicator "relative time"
        d_x += 1

        self.x_emb = stf.Time2Vec(d_x, embed_dim=time_emb_dim * d_x) # [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]

        if self.method == "temporal":
            y_emb_inp_dim = d_y + (time_emb_dim * d_x)
        else:
            y_emb_inp_dim = 1 + (time_emb_dim * d_x)

        self.y_emb = nn.Linear(y_emb_inp_dim, d_model)

        if self.method == "spatio-temporal":
            self.var_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [VariableDownsample(d_y, d_model) for _ in range(downsample_convs)]
        )

        self._benchmark_embed_enc = None
        self._benchmark_embed_dec = None
        self.d_model = d_model
        self.null_value = null_value

    def __call__(self, y, x, is_encoder=True):
        """
        Forward pass of the SpacetimeformerEmbedding module.

        Args:
            y (torch.Tensor): Input y.
            x (torch.Tensor): Input x.
            is_encoder (bool, optional): Whether the module is used in the encoder or decoder. Defaults to True.

        Returns:
            torch.Tensor: Embedded values, space embeddings, and variable indices.
        """
        if self.method == "spatio-temporal":
            val_time_emb, space_emb, var_idxs = self.spatio_temporal_embed(
                y, x, is_encoder
            )
        else:
            val_time_emb, space_emb = self.temporal_embed(y, x, is_encoder)
            var_idxs = None

        return val_time_emb, space_emb, var_idxs

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

        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        if not self.TIME:
            x = torch.zeros_like(x)
        x = torch.cat((x, local_pos), dim=-1) # [bs, length, x.size] -> [bs, length, x.size + 1]
        t2v_emb = self.x_emb(x) # [bs, length, d_x] -> [bs, length, time_emb_dim * d_x]

        # val embedding
        emb_inp = torch.cat((y, t2v_emb), dim=-1) # [bs, length, d_y + time_emb_dim * d_x]
        emb = self.y_emb(emb_inp) # [bs, length, d_model]

        # "given" embedding
        given = torch.ones((bs, length)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given) # [bs, length, d_model]
        emb += given_emb # [bs, length, d_model]

        if is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        return emb, torch.zeros_like(emb)

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

        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1) # [bs, length, d_y] -> [bs, length * d_y]
        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        ) # [bs, length, 1]
        x = torch.cat((x, local_pos), dim=-1) # [bs, length, x.size] -> [bs, length, x.size + 1]
        if not self.TIME:
            x = torch.zeros_like(x)
        if not self.VAL:
            y = torch.zeros_like(y)
        t2v_emb = self.x_emb(x).repeat(1, d_y, 1) # [bs, length, d_x] -> [bs, length * d_y, time_emb_dim * d_x]
        val_time_inp = torch.cat((y, t2v_emb), dim=-1) # [bs, length * d_y, 1 + time_emb_dim * d_x]
        val_time_emb = self.y_emb(val_time_inp) # [bs, length * d_y, d_model]

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T # [bs, length, d_y]
            if not is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1) # [bs, length, d_y] -> [bs, length * d_y]
            if self.null_value is not None:
                # mask null values
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask
            given_emb = self.given_emb(given) # [bs, length * d_y, d_model]
            val_time_emb += given_emb # [bs, length * d_y, d_model]

        if is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # var embedding
        var_idx = torch.Tensor([[i for j in range(length)] for i in range(d_y)])
        var_idx = var_idx.long().to(x.device).view(-1).unsqueeze(0).repeat(bs, 1)
        var_idx_true = var_idx.clone()
        if not self.SPACE:
            var_idx = torch.zeros_like(var_idx)
        var_emb = self.var_emb(var_idx)

        return val_time_emb, var_emb, var_idx_true
