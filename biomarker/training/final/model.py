import torch 
import torch.nn as nn
import math 
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial

class MageEncodingViT(nn.Module):
    def __init__(self, args):
        super(MageEncodingViT, self).__init__()
        self.args = args 
        ## needs args: num_heads, feature_dim, fc1_size, dropout, num_classes, fc2_size 
        # have to make sure the input shape is : ##n, c, h, w 
        print(args.fc2_size, type(args.fc2_size))
        
        if self.args.age_input:
            self.age_embed = nn.Embedding(4, args.feature_dim)
            # self.age_embedding = nn.Sequential(nn.Linear(1,args.feature_dim), nn.ReLU()), nn.Linear(args.feature_dim, args.feature_dim)
        if self.args.sex_input:
            self.sex_embed = nn.Embedding(2, args.feature_dim)
            
        if self.args.modality_input: 
            self.modality_embed = nn.Embedding(3, args.feature_dim)
        
        self.model = VisionTransformer(args, image_size=[8,self.args.MAGE_INPUT_SIZE],patch_size=1,num_layers=args.num_layers_vit,num_heads=args.num_heads,hidden_dim=args.feature_dim,mlp_dim=args.fc1_size,dropout=args.dropout,attention_dropout=args.dropout / 2, num_classes=args.num_classes, representation_size=args.fc2_size if args.fc2_size != -1 else None)
    def forward(self, x, age=None, gender=None, rf=None, return_embeddings=False, return_embeddings_pred=False, t5_emb=None, modality=None):
        if age is not None and self.args.age_input:
            # age[age >= 80] = 79
            age = torch.clamp(age, 0, 79) // 20
            # if torch.isnan(age).any():
            #     bp() 
            age_processed = torch.where(torch.isnan(age), torch.tensor(0, device=age.device, dtype=torch.int64), age.int())
            age_embedded = self.age_embed(age_processed)
            age_embedded = torch.where(torch.isnan(age).unsqueeze(-1), torch.zeros_like(age_embedded), age_embedded)
            age = age_embedded
            
            # age = torch.where(torch.isnan(age), torch.zeros(self.args.feature_dim, device=age.device), self.age_embed(age.int()))
        else:
            age = None
        if gender is not None and self.args.sex_input:
            sex_processed = torch.where(torch.isnan(gender), torch.tensor(0, device=gender.device, dtype=torch.int64), gender.int() - 1)
            sex_embedded = self.sex_embed(sex_processed)
            sex_embedded = torch.where(torch.isnan(gender).unsqueeze(-1), torch.zeros_like(sex_embedded), sex_embedded)
            sex = sex_embedded
            # sex = torch.where(torch.isnan(gender), torch.zeros(self.args.feature_dim, device=gender.device), self.sex_embed(gender.int() - 1))
        else:
            sex = None 
        if modality is not None and self.args.modality_input:
            modality_embedded = self.modality_embed(modality.int())
            modality = modality_embedded
        else:
            modality = None
        x = x.swapaxes(1,2).swapaxes(2,3)
        return self.model(x, rf=rf, return_embeddings=return_embeddings, return_embeddings_pred=return_embeddings_pred, age=age, sex=sex, t5_emb=t5_emb, modality=modality)

class MagePredViT(nn.Module):
    def __init__(self, args):
        super(MagePredViT, self).__init__()
        self.args = args 
        ## needs args: num_heads, feature_dim, fc1_size, dropout, num_classes, fc2_size 
        # have to make sure the input shape is : ##n, c, h, w 
        #V1: 
        if self.args.mage_img_slice_as_feature:
            self.model = VisionTransformer(args, image_size=[1,1024],patch_size=1,num_layers=args.num_layers_vit,num_heads=args.num_heads,hidden_dim=args.feature_dim,mlp_dim=args.fc1_size,dropout=args.dropout,attention_dropout=args.dropout / 2, num_classes=args.num_classes, representation_size=args.fc2_size if args.fc2_size != -1 else None, in_channels=256)
        else:
            self.model = VisionTransformer(args, image_size=[256,1024],patch_size=32,num_layers=args.num_layers_vit,num_heads=args.num_heads,hidden_dim=args.feature_dim,mlp_dim=args.fc1_size,dropout=args.dropout,attention_dropout=args.dropout / 2, num_classes=args.num_classes, representation_size=args.fc2_size if args.fc2_size != -1 else None, in_channels=1)
        #V2: 
        # 
    def forward(self, x, age=None, gender=None, rf=None, return_embeddings=False, return_embeddings_pred=False, t5_emb=None):
        x = x.unsqueeze(2)
        if not self.args.mage_img_slice_as_feature:
            x = x.swapaxes(1,2)
        
        return self.model(x, rf=rf, return_embeddings=return_embeddings, return_embeddings_pred=return_embeddings_pred, age=age, sex=gender, t5_emb=t5_emb)

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class EncoderTwoTail(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        which_layer_combine=0
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_rf = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.which_layer_combine = which_layer_combine
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(self.which_layer_combine, num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        
        
        if self.which_layer_combine > 0:
            layers_rf = OrderedDict()
            layers_belt = OrderedDict()
            for i in range(self.which_layer_combine):
                layers_rf[f"encoder_layer_{i}"] = EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
                layers_belt[f"encoder_layer_{i}"] = EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
            self.layers_rf = nn.Sequential(layers_rf)
            self.layers_belt = nn.Sequential(layers_belt)
        
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, rf=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        if rf is None:
            input = input + self.pos_embedding
            if self.which_layer_combine > 0:
                input = self.layers_belt(self.dropout(input))
            else:
                input = self.dropout(input)
            print('no rf detecting, probably a bug')
            return self.ln(self.layers(input))
        
        indices_tensor = rf #torch.tensor(rf, device=input.device)
        mask = torch.zeros(input.shape[0], dtype=torch.bool, device=input.device)
        mask[indices_tensor] = True
        
        # Split the batch into two subsets
        subset_A = input[mask]
        subset_B = input[~mask]
        
        # Process each subset with the respective module
        
        # Prepare a tensor to hold the combined outputs, with the same size and device as the inputs
        combined_output = torch.empty_like(input).to(input.device)

        subset_A = subset_A + self.pos_embedding_rf
        if self.which_layer_combine > 0:
            processed_A = self.layers_rf(self.dropout(subset_A))
        else:
            processed_A = self.dropout(subset_A)
        
        subset_B = subset_B + self.pos_embedding
        if self.which_layer_combine > 0:
            processed_B = self.layers_belt(self.dropout(subset_B))
        else:
            processed_B = self.dropout(subset_B)
        
        # Re-insert the processed outputs back into the correct positions
        combined_output[mask] = processed_A
        combined_output[~mask] = processed_B

        
        return self.ln(self.layers(combined_output))


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        args,
        image_size,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        in_channels=768
    ):
        super().__init__()
        # torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        print(representation_size)
        self.norm_layer = norm_layer
        self.args = args 
        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        
        
        if hasattr(self.args, 'num_tokens'):
            self.num_tokens = self.args.num_tokens
        else:
            self.num_tokens = 1
        
        if self.args.separate_head:
            self.num_tokens += 1
        
        
        if self.args.mage_pred_vit_model:
            seq_length = (image_size[0] * image_size[1]) // (patch_size **2)
        else:
            seq_length = (image_size[0] * image_size[1]) // patch_size

        # Add a class token
        for token in range(self.num_tokens):
            setattr(self, f'class_token_{str(token)}', nn.Parameter(torch.zeros(1, 1, hidden_dim)))
            seq_length += 1
        
        if self.args.age_input:
            seq_length += 1
        if self.args.sex_input:
            seq_length += 1
        if self.args.modality_input:
            seq_length += 1
        if self.args.t5_demographics:
            seq_length += 1
        if self.args.t5_demographics_nomean:
            seq_length += 13
        
        ## self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))        
        # if self.separate_head:
        #     self.subtype_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        #     seq_length += 1
        # seq_length += 1

        if args.tail_length_vit == -1:
            self.encoder = Encoder(
                seq_length,
                num_layers,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        else:
            self.encoder = EncoderTwoTail(
                seq_length,
                num_layers,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                which_layer_combine=args.tail_length_vit
            )
        self.seq_length = seq_length


        for token in range(self.num_tokens):
            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            if representation_size is None:
                heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
            else:
                heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
                heads_layers["act"] = nn.Tanh()
                heads_layers["head"] = nn.Linear(representation_size, num_classes)

            if token == 0:
                self.heads = nn.Sequential(heads_layers)
                self.heads1 = self.heads
            else:
                setattr(self, f'heads{str(token + 1)}', nn.Sequential(heads_layers))
            
        if self.args.label == 'simul':
            assert False ## no longer supported , use the num_tokens instead 
        
        if self.args.t5_demographics or self.args.t5_demographics_nomean:
            self.t5_mapping = nn.Linear(768, hidden_dim)
        # heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        # if representation_size is None:
        #     heads_layers["head"] = nn.Identity() #nn.Linear(hidden_dim, num_classes)
        # else:
        #     heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        #     heads_layers["act"] = nn.Tanh()
        #     if self.separate_head:
        #         heads_layers["head"] = nn.Linear(representation_size, num_classes)

        # self.heads = nn.Sequential(heads_layers)
        
        # if self.separate_head:
        #     heads2_layers: OrderedDict[str, nn.Module] = OrderedDict()
        #     if representation_size is None:
        #         heads2_layers["head"] = nn.Linear(hidden_dim, num_classes)
        #     else:
        #         heads2_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        #         heads2_layers["act"] = nn.Tanh()
        #         heads2_layers["head"] = nn.Linear(representation_size, 4)

        #     self.heads2 = nn.Sequential(heads2_layers)
        # else:
        #     if representation_size is None:
        #         representation_size = hidden_dim
        #     self.head1 = nn.Linear(representation_size, num_classes)
        #     if self.args.label == 'simul':
        #         self.head2 = nn.Linear(representation_size, 4)
        #         self.head3 = nn.Linear(representation_size, 5)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        for token in range(1,self.num_tokens):
            if hasattr(self,f"heads{token + 1}") and hasattr(getattr(self,f"heads{token + 1}"), "pre_logits") and isinstance(getattr(self,f"heads{token + 1}").pre_logits, nn.Linear):
                fan_in = getattr(self,f"heads{token + 1}").pre_logits.in_features
                nn.init.trunc_normal_(getattr(self,f"heads{token + 1}").pre_logits.weight, std=math.sqrt(1 / fan_in))
                nn.init.zeros_(getattr(self,f"heads{token + 1}").pre_logits.bias)

        for token in range(self.num_tokens):
                nn.init.zeros_(getattr(self,f"heads{token + 1}").head.weight)
                nn.init.zeros_(getattr(self,f"heads{token + 1}").head.bias)

        # if hasattr(self,"heads1") and isinstance(self.head1, nn.Linear):
        #     nn.init.zeros_(self.head1.weight)
        #     nn.init.zeros_(self.head1.bias)
        # if not self.separate_head and self.args.label == 'simul':
        #     nn.init.zeros_(self.head2.weight)
        #     nn.init.zeros_(self.head2.bias)
        #     nn.init.zeros_(self.head3.weight)
        #     nn.init.zeros_(self.head3.bias)
        

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size[0], f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size[1], f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        if not self.args.no_conv_proj:
            x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, rf=None, return_embeddings=False, return_embeddings_pred=False, sex=None, age=None, t5_emb=None, modality=None):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        if self.args.t5_demographics or self.args.t5_demographics_nomean:
            if t5_emb is not None:
                t5_x = self.t5_mapping(t5_emb.float())
                if len(t5_x.shape) == 2:
                    t5_x = t5_x.unsqueeze(1)
                x = torch.cat([x,t5_x],dim=1)
            else:
                if self.args.t5_demographics:
                    x = torch.cat([x,torch.zeros(x[:,0:1,:].shape).to(x.device)],dim=1)
                elif self.args.t5_demographics_nomean:
                    x = torch.cat([x,torch.zeros(x[:,0:13,:].shape).to(x.device)],dim=1)
        
        if self.args.sex_input:
            if sex is not None:
                x = torch.cat([x,sex.unsqueeze(1)],dim=1)
            else:
                x = torch.cat([x,torch.zeros(x[:,0:1,:].shape).to(x.device)],dim=1)
        if self.args.age_input:
            if age is not None:
                x = torch.cat([x,age.unsqueeze(1)],dim=1)
            else:
                x = torch.cat([x,torch.zeros(x[:,0:1,:].shape).to(x.device)],dim=1)
        if self.args.modality_input:
            if modality is not None:
                x = torch.cat([x,modality.unsqueeze(1)],dim=1)
            else:
                x = torch.cat([x,torch.zeros(x[:,0:1,:].shape).to(x.device)],dim=1)
        
        
        n = x.shape[0]

        # Expand the class token to the full batch
        if self.args.separate_head:
            assert False 
            batch_subtype_token = self.subtype_token.expand(n, -1, -1)
            x = torch.cat([batch_subtype_token, x], dim=1)
        
        for token in range(self.num_tokens):
            x = torch.cat([getattr(self,f'class_token_{str(token)}').expand(n, -1, -1),x],dim=1)

        # batch_class_token = self.class_token.expand(n, -1, -1)
        # x = torch.cat([batch_class_token, x], dim=1)
        if rf is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, rf=rf)

        # Classifier "token" as used by standard language architectures
        
        other_results = [x[:, i] for i in range(self.num_tokens)] 

        if return_embeddings:
            return other_results

        # if self.separate_head:
        #     x2 = x[:, 1]
        #     x2 = self.heads2(x)

        result = [getattr(self,f'heads{str(token + 1)}')(other_results[token]) for token in range(self.num_tokens)]
        
        if return_embeddings_pred:
            return [result[0] if len(result) == 1 else result , other_results]
        return result[0] if len(result) == 1 else result 

        
        # if self.separate_head:
        #     return x, x2, torch.zeros(x2.shape[0],5).to(self.args.device)
        # else:
        #     if self.args.label == 'simul':
        #         return self.head1(x), self.head2(x), self.head3(x)
        #     else:
        #         x = self.head1(x)
        # return x
        
