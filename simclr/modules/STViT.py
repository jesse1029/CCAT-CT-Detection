import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import timm
from vit.vit_pytorch.vit_pytorch import ViT
import numpy as np
from einops.layers.torch import Rearrange
from simclr.modules.gmlp import gMLPVision as mlp
from efficientnet_pytorch import EfficientNet


class SingleViT(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, device, args, mode='train'):
        super(SingleViT, self).__init__()
        
        
        self.mode=mode
        self.FREQ=args.FREQ
        self.backbone=args.backbone
        self.device = device
        self.n_features = args.n_features
        self.crop_size = args.crop_size
        self.image_size = args.image_size
        self.FRR=args.FRR
        self.FREQ=args.FREQ
        self.L1_depth=args.L1_depth
        self.L2_depth=args.L2_depth
        self.twoStream = args.twoStream
        
#         model = timm.create_model('cspdarknet53', pretrained=True)
#         model = EfficientNet.from_pretrained('efficientnet-b4')

        in_feat_dim = 2048
        if 'resnet50' in self.backbone:
            model = torchvision.models.resnet50(pretrained=True) #featMap: -2
        elif 'resnet101' in self.backbone:
            model = torchvision.models.resnet101(pretrained=True) 
        elif 'nfnf0' in self.backbone:
            in_feat_dim = 2304 # nfnet_l0
            model = timm.create_model('nfnet_l0', pretrained=True) # featmap: -1
        elif 'nfnf1' in self.backbone:
            in_feat_dim = 2304
            model = timm.create_model('eca_nfnet_l0', pretrained=True) # featmap: -1
        elif 'resnest50' in self.backbone:
            in_feat_dim = 2048
            model = timm.create_model('resnest50d', pretrained=True) # featmap: -1
        elif 'cspdarknet53' in self.backbone:
            in_feat_dim = 1024
            model = timm.create_model('cspdarknet53', pretrained=True) # featmap: -1
        elif 'efficientnet' in self.backbone:
            in_feat_dim = 1280 #1792 for b4
            model = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            print('Backbone fail!!')
            raise
        
        if 'efficientnet' not in self.backbone:
            print('extract the last-%d-th layer' % -args.useFeatMap)

            model = torch.nn.Sequential(*list(model.children())[:args.useFeatMap])
        
        self.encoder = model  #output shape [B, 1024, 8, 8]
        self.reshape =  Rearrange('b c h w -> b (h w) (c)')
        num_patches = (self.crop_size//28 -1) ** 2
        if args.heads==0:
            self.spatial_transformer = mlp(dim = in_feat_dim,
                                           num_patches=num_patches, # Feat map size
                                           depth = args.L1_depth
                                          )
        
            self.transformer = mlp(dim=in_feat_dim, # Feat map size
                                    num_patches = args.FRR,
                                    depth = self.L2_depth
                                  )
        else:
            self.spatial_transformer  = ViT(dim = in_feat_dim,
                        depth = args.L1_depth,
                        heads = args.heads,
                        num_patches=num_patches,
                        mlp_dim = args.n_features,
                        dropout = 0.1,
                        emb_dropout = 0.1)

            self.transformer  = ViT(dim = in_feat_dim,
                        depth = args.L2_depth,
                        heads = args.heads,
                        num_patches=args.FRR,
                        mlp_dim = args.n_features,
                        dropout = 0.1,
                        emb_dropout = 0.1)
      
        
        
       
        self.inter_feat = nn.Sequential(
            nn.Linear(in_feat_dim*args.FRR+in_feat_dim, args.n_features, bias=False),
            nn.LeakyReLU(inplace=True),
            )
        
        self.cls = nn.Sequential(
            nn.Linear(args.n_features, args.n_features//2),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(args.n_features//2, args.n_features//4),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(args.n_features//4, 2)
            )
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        
    def stforward(self, X):
        lstm_i, module_outputs = [], []
        
        if self.mode=='train' and len(X['img'])==self.FRR:
            
            if self.twoStream:
                for k in range(len(X['img'])): # for spatial
                    if 'efficientnet' in self.backbone:
                        x = self.encoder.extract_features(X['img'][k].to(self.device, non_blocking=True))
                        x2 = self.encoder.extract_features(X['img2'][k].to(self.device, non_blocking=True))
                    else:
#                         import pdb
#                         pdb.set_trace()
                        x = self.encoder(X['img'][k].to(self.device, non_blocking=True))
                        x2 = self.encoder(X['img2'][k].to(self.device, non_blocking=True))

                    x = self.spatial_transformer(self.reshape(x))
                    x2 = self.spatial_transformer(self.reshape(x2))
                    module_outputs.append(x)
                    lstm_i.append(x2)
                
                
                feat = torch.stack(lstm_i)## time-seq x batch x feat-dim
                feat = feat.permute(1,0,2)
                feat = self.transformer(feat)

                module_outputs.append(feat)
                # Concatenate current image CNN output 
                X = torch.cat(module_outputs, dim=-1)
                X = self.inter_feat(X)
                
            else:
                for k in range(len(X['img'])):
                    if 'efficientnet' in self.backbone:
                        x = self.encoder.extract_features(X['img'][k].to(self.device, non_blocking=True))
                    else:
                        x = self.encoder(X['img'][k].to(self.device, non_blocking=True))
#                     print(x.shape)
                    x = self.spatial_transformer(self.reshape(x))
                    lstm_i.append(x)
                    module_outputs.append(x)

                feat = torch.stack(lstm_i)## time-seq x batch x feat-dim
                feat = feat.permute(1,0,2)
                feat = self.transformer(feat)

                module_outputs.append(feat)
                # Concatenate current image CNN output 
                X = torch.cat(module_outputs, dim=-1)
                X = self.inter_feat(X)
        else:
            tlen = len(X['img'])
            assert tlen>=self.FRR
            
            copies = tlen//self.FREQ
            copies = 1 if copies==0 else copies
            Xs = []
            FREQ=self.FREQ
            additive = 0
            if tlen-self.FRR*FREQ<=0:
                additive = 1
            if tlen-self.FRR*FREQ + additive<=0:
                FREQ=1
#             while self.FRR*self.FREQ > tlen:
#                 if self.FREQ<=1:
#                     break
#                 self.FREQ-=1
                
                    
            for c in range(0, tlen-self.FRR*FREQ + additive, FREQ):
                module_outputs, lstm_i=[], []
                
                for k in range(c, c+self.FRR*FREQ, FREQ):
                    if 'efficientnet' in self.backbone:
                        x = self.encoder.extract_features(X['img'][k].to(self.device, non_blocking=True))
                    else:
                        x = self.encoder(X['img'][k].to(self.device, non_blocking=True))
                    x = self.spatial_transformer(self.reshape(x))

                    lstm_i.append(x)
                    module_outputs.append(x)

                feat=torch.stack(lstm_i)## time-seq x batch x feat-dim
                feat = feat.permute(1,0,2)
                feat = self.transformer(feat)

                module_outputs.append(feat)
                # Concatenate current image CNN output 
                feat = torch.cat(module_outputs, dim=-1)
                feat = self.inter_feat(feat)
                Xs.append(feat)
            
            return Xs
            
        return X
        
        
    def forward(self, X, forward_mode='train'):
        X=self.stforward(X)
        if forward_mode=='test' and isinstance(X, list):
            for i in range(len(X)):
                X[i] = self.cls(X[i])

        else:
            X=self.cls(X)
        return X
    
