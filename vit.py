import torch
from torch import nn
# create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # init the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        self.patch_size = patch_size
        # create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        # create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,end_dim=3) # only flatten the feature map dimensions into a single vector
    # define the forward method
    def forward(self, x):
        # create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

class ViT(nn.Module):
    def __init__(self,
                 img_size=224,#from table3, training resolution
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768,#table1
                 dropout=0.1,
                 mlp_size=3072,#table1
                 num_transformer_layers=12,#table1
                 num_heads=12,#table1
                 num_classes=1000):
        super().__init__()

        #Assert image size is divisible by patch size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        # 1 create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        # 2 create class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)
        # 3 create positional embedding
        num_patches = (img_size*img_size) // patch_size**2 # N=HW/P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
        # 4 create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)
        # 5 create transformes encoder layer (single)
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
        #                                                             nhead=num_heads,
        #                                                             dim_feedforward=mlp_size,
        #                                                             activation="gelu",
        #                                                             batch_first=True,
        #                                                             norm_first=True)
        # 5 create transformes encoder layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                    nhead=num_heads,
                                                                                                    dim_feedforward=mlp_size,
                                                                                                    activation="gelu",
                                                                                                    batch_first=True,
                                                                                                    norm_first=True),
                                                                                        num_layers=num_transformer_layers)

        # 6 create mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
    def forward(self, x):
        #get some dimensions
        batch_size=x.shape[0]
        #patch embed
        x=self.patch_embedding(x)

        #we need a class token per image in our batch so expand the class token across the batch size
        class_token = self.class_token.expand(batch_size,-1,-1)
        #prepend the class token to the patch embedding
        x = torch.cat((class_token,x),dim=1)
        # add the pos embed to patch embed with class token
        x=self.positional_embedding+x
        #dropout on patch+pos embed
        x = self.embedding_dropout(x)
        #pass embed thwough transformer encoder stack
        x = self.transformer_encoder(x)
        # pass just 0zt index(class token) of x through MLP head
        x = self.mlp_head(x[:,0])
        return x

