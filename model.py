
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from functools import partial
import clip
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from functools import reduce
class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                embed_dim= 512,
                context_length= 77,
                vocab_size= 50257,
                transformer_width= 512,
                transformer_heads= 32,
                transformer_layers= 18,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        print("learning_rate",learning_rate)
        from transformers import EncoderDecoderModel
        from transformers import MBartConfig, MBartForConditionalGeneration, EncoderDecoderModel
        config=MBartConfig(vocab_size=vocab_size,
                           max_position_embeddings = 1024,
                           encoder_layers = 16,
                           encoder_ffn_dim = 1024,
                           encoder_attention_heads = 8,
                           decoder_layers = 16,
                           decoder_ffn_dim = 1024,
                           decoder_attention_heads = 8,
                           encoder_layerdrop = 0.1,
                           decoder_layerdrop = 0.1,
                           use_cache = True,
                           is_encoder_decoder = True,
                           activation_function = 'gelu',
                           d_model = 512,
                           dropout = 0.1,
                           attention_dropout = 0.1,
                           activation_dropout = 0.1,
                           init_std = 0.04,
                           classifier_dropout = 0.01,
                           scale_embedding = False,
                           pad_token_id = 1,
                           bos_token_id = 49406,
                           eos_token_id = 49407)

        self.model = MBartForConditionalGeneration(config=config)
        self.model.train()
        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        
        self.encode_image= VisionTransformer(
                input_resolution=224,
                patch_size=16,
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                output_dim=embed_dim
            )
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.loss=torch.nn.CrossEntropyLoss(reduction='mean')

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # create linear layer with the same parameter as token embedding for one-hot input
        self.token_embedding_linear=nn.Linear(vocab_size,transformer_width,bias=False)
        #make sure the linear layer has the same parameter as token embedding
        self.token_embedding_linear.weight=torch.nn.Parameter(self.token_embedding.weight.T)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.handles=[]

        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0

        self.initialize_parameters()
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
 
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
        for _,layer in self.encode_image.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:

                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1)
                nn.init.zeros_(layer.bias)
        for _,layer in self.encoder.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=fc_std)
                nn.init.zeros_(layer.bias)
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text_en(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x
    def encode_text_pseudo_en(self, text):

        #psuedo embedding of one hot...
        # b, 77, V -> b, 77, F  so we @ v,f -> b, 77, F
        x=text@self.token_embedding.weight 
        #sum over the one hot dimension 
        #x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), torch.max(text.argmax(dim=-1),dim=-1).indices] 
        return x
    def translate(self,es,en):
        #take ES and convert to EN and return logits and encoder output
        #es is shape [batchsize,5,77]
        # print(es.shape)
        outputs=self.model(input_ids=es, return_dict=True)
        logits=outputs.logits
        encoderoutput=outputs.encoder_last_hidden_state
        #select the encoder output for the last token of each caption (the token that is the end token) 
        # print("EOT Indexes are: ",es.argmax(dim=-1).tolist())
        # print("EOT out are: ",es[torch.arange(encoderoutput.shape[0]),es.argmax(dim=-1)])
        encoderoutput=encoderoutput[torch.arange(encoderoutput.shape[0]),es.argmax(dim=-1)]        
        #print(encoderoutput.shape)# [batchsize*5,77,512]

        return logits, encoderoutput
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        n=16
        labels=torch.arange(batch[0].shape[0],device=self.device)
        im,captions= batch[0],batch[1].squeeze()
    
        split=captions.shape[1]//2
        Es=captions[:,split:]
        En=captions[:,:split]
        #im is shape [batchsize,3,224,224]
        #captions is shape [batchsize,10,1,77] and is made of en and es captions (5 each)
        TranlatedEnTokenlogits,TrencoderLogits=self.translate(es=Es.flatten(0,1), en=En.flatten(0,1))
        TranlatedEnTokenlogits=torch.nn.functional.gumbel_softmax(TranlatedEnTokenlogits.reshape(Es.shape[0],Es.shape[1],77,-1),dim=-1)
        esLogits=TrencoderLogits.reshape(Es.shape[0],Es.shape[1],self.transformer_width)
        
        enLogits=self.encode_text_en(En.flatten(0,1)).reshape(En.shape[0],En.shape[1],self.transformer_width)
        TrEnLogits=self.encode_text_pseudo_en(TranlatedEnTokenlogits.flatten(0,1)).reshape(TranlatedEnTokenlogits.shape[0],TranlatedEnTokenlogits.shape[1],self.transformer_width)
        ImLogits=self.encode_image(im).unsqueeze(1) @ self.text_projection
        #esLogits=self.encode_text_es(Es.flatten(0,1)).reshape(Es.shape[0],Es.shape[1],self.transformer_width)
       
        TrEnLogits=TrEnLogits/TrEnLogits.norm(dim=-1,keepdim=True)
        enLogits=enLogits/enLogits.norm(dim=-1,keepdim=True)
        esLogits=esLogits/esLogits.norm(dim=-1,keepdim=True)
        ImLogits=ImLogits/ImLogits.norm(dim=-1,keepdim=True)

        TrEnLogits=TrEnLogits* 400 /TrEnLogits.shape[1]
        enLogits=enLogits*400 /enLogits.shape[1]
        esLogits=esLogits *400 /esLogits.shape[1]
        ImLogits=ImLogits *400 /ImLogits.shape[1]

        logits=torch.cat((TrEnLogits,esLogits,enLogits,ImLogits),dim=1).permute(1,0,2) # convert shape B,16,H to 16,B,H

        # LossLogits=reduce(torch.add,[reduce(torch.add,[item@x.T  for x in logits]) for item in logits]) *self.logit_scale.exp() *256
        # #LossLogits=LossLogits- torch.diag(LossLogits).diag() #remove the diagonal
        # #LossLogits=LossLogits- (reduce(torch.add,[item@item.T for item in logits]) *self.logit_scale.exp() *256)

        # #When done try the above line removed? See ErnieVil paper


        # loss =self.loss(LossLogits, labels)
        


        loss= reduce(torch.add,[self.loss(y@x.T *self.logit_scale.exp(),labels) for x in logits for y in logits])
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        

        labels=torch.arange(batch[0].shape[0],device=self.device)
        im,captions= batch[0],batch[1].squeeze()
    
        split=captions.shape[1]//2
        Es=captions[:,split:]
        En=captions[:,:split]
        #im is shape [batchsize,3,224,224]
        #captions is shape [batchsize,10,1,77] and is made of en and es captions (5 each)
        outputs=self.model(input_ids=Es.flatten(0,1),labels=En.flatten(0,1), return_dict=True)
        logits=outputs.logits
        modelLoss=outputs.loss

        ENTokens=torch.cat((torch.argmax(logits.reshape(Es.shape[0],Es.shape[1],77,-1),dim=-1),En),dim=1)
        # print(ENTokens.shape)#shape [batchsize,10,77]
        enLogits=self.encode_text_en(ENTokens.flatten(0,1)).reshape(ENTokens.shape[0],ENTokens.shape[1],self.transformer_width)
        ImLogits=self.encode_image(im).unsqueeze(1)#@ self.text_projection

        logits=torch.cat((enLogits,ImLogits),dim=1).permute(1,0,2) # convert shape B,21,H to 21,B,H
        LossLogits=reduce(torch.add,[reduce(torch.add,[item@ x.T for x in logits]) for item in logits])*self.logit_scale.exp()
        loss = self.loss(LossLogits, labels)
        loss = loss.mean()

        self.log('val_loss_contr', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        self.log('val_loss_model', modelLoss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        loss=loss+modelLoss
        self.log('val_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        return {"loss": loss}
    
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        model=self.model
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            #             {
            #     "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            #     "weight_decay": self.hparams.weight_decay,
            # },
            #             {
            #     "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            #     "weight_decay": self.hparams.weight_decay,
            # },
                        {
                "params": [p for n, p in self.named_parameters()  if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
 
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

       

        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

#     def on_validation_epoch_start(self):
#         self.log("Mean Projection Value",self.text_projection.mean(),enable_graph=False)
#         self.log("logit scale",self.logit_scale.exp())

#         self.naninfcount=0
#         self.model2,_ = clip.load("ViT-B/32", device=self.device)
#         self.model2.eval()
#         self._insert_hooks()
#         self.IMhsic_matrix0=torch.zeros([],device=self.device)
#         self.IMhsic_matrix1=torch.zeros([],device=self.device)
#         self.IMhsic_matrix2=torch.zeros([],device=self.device)
#         self.CAPhsic_matrix0=torch.zeros([],device=self.device)
#         self.CAPhsic_matrix1=torch.zeros([],device=self.device)
#         self.CAPhsic_matrix2=torch.zeros([],device=self.device)
        
#         self.eval()

#     def validation_step(self,batch,*args):
#         #do stock loss here
#         labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
#         self.model1_features = {}  #reset list of forward hooks
#         self.model2_features = {}  #reset list of forward hooks
#         image_features=self.encode_image(batch[0])
#         self.model2.encode_image(batch[0])# to compare supervision model
#         a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
#         self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200)) 
#         a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
#         self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200))
#         joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
#         self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
#         ##Now Do Text
#         self.model1_features = {}  #reset list of forward hooks
#         self.model2_features = {}  #reset list of forward hooks
#         choice=torch.randint(0,5,(1,)).item()
#         #print("choice", choice)
#         c=batch[1][:,choice]
#         c=c.squeeze()

#         captions=self.encode_text(c) #run through main mode

#         # c=captions.detach().clone().cpu()
#         #run through main mode
        
#         self.model2.encode_text(c)# to compare supervision model
#         a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
#         self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,batch_HSIC2(a)) 
#         a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
#         self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,batch_HSIC2(a))
#         joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
#         self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
#         if self.projection=="inv":
#             image_features=image_features@ self.text_projection
#         elif self.projection=="iinv":
#             image_features=image_features@torch.inverse(self.text_projection)
#         elif self.projection=="None":
#             captions=captions@self.text_projection


#         # print("self.logit scale is 14 right? ",self.logit_scale.exp())
#         logitsI,logitsT=self.calculate_lossStock(image_features, captions) 
#         self.log("mean validation stock logits ", logitsI.mean())
        
#         lossim = self.loss(logitsI*(self.logit_scale.exp()), labels)
#         loss1 = self.loss(logitsT*(self.logit_scale.exp()), labels)
#         loss = lossim+loss1
#         loss=loss/2
#         loss = loss.mean()
#         return {"loss": loss, "imfeatures":image_features, "tfeatures":captions,"classes":batch[2]}

#     def validation_epoch_end(self,acc_val):
#         imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in acc_val],dim=0)).cpu().numpy()
#         tfeatures=torch.nan_to_num(torch.cat([val["tfeatures"] for val in acc_val],dim=0)).cpu().numpy()
#         labels=torch.cat([val["classes"] for val in acc_val],dim=0).cpu().numpy()
#         if not hasattr(self,"Iclassifier"):
#             self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
#         if not hasattr(self,"Tclassifier"):
#             self.Tclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1,  n_jobs=-1)
        
#         self.Iclassifier.fit(imfeatures, labels)
#         self.log( "ImProbe",self.Iclassifier.score(imfeatures, labels))
#         self.Tclassifier.fit(tfeatures, labels)
#         self.log( "TProbe",self.Tclassifier.score(tfeatures, labels))

#         self.log('val_loss-stock', torch.stack([val["loss"] for val in acc_val],dim=0).mean(), prog_bar=True,enable_graph=False, rank_zero_only=True)


#         self.unfreeze()
#         self.train()
#         self.plot_results("IM","IMHSIC{}.jpg".format(self.current_epoch))
#         self.plot_results("CAP","CAPHSIC{}.jpg".format(self.current_epoch))
#         if self.logger is not None:
#             self.logger.log_image(key="IMHSIC{}".format(self.current_epoch), images=["IMHSIC{}.jpg".format(self.current_epoch)])        
#             self.logger.log_image(key="CAPHSIC{}".format(self.current_epoch), images=["CAPHSIC{}.jpg".format(self.current_epoch)])
#         for handle in self.handles:
#             handle.remove()
#         print(self.naninfcount)
#         del self.model2
#         if self.prune:
#             for hook in self.pruneHooks:
#                     global_entropy = hook.retrieve()
#                     hook.remove()        

#                     # im_scores =map(lambda name, block: prune_Residual_Attention_block(block, global_entropy[name], self.args["prune_eta"]), filter(lambda name,block: isinstance(block, ResidualAttentionBlock) and name in global_entropy.keys(), self.encode_image.named_modules()[:-1]))
#                     # for imscoredict in im_scores:
#                     #     for (param_to_prune, im_score) in imscoredict.items():
#                     #         prune_module(param_to_prune, im_score, self.args)
#                     #then purun accordingly 
        
#     def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
#         if isinstance(out, tuple):
#             out=out[0]       
#         if out.shape[0] == self.hparams.train_batch_size:
#             self.__store(out,name,model,layer)
#         elif out.shape[1] == self.hparams.train_batch_size:
#             self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

#     def __store(self,out,name, model,layer):
#         X = out.flatten(1)
#         X= torch.nan_to_num((X @ X.t()).fill_diagonal_(0))
#         if (torch.isnan(X).any() or torch.isinf(X).any()):
#             self.naninfcount+=1
#         if model == "model1":
#             while name in self.model1_features:
#                 name=name+"1"
#             self.model1_features[name] = X

#         elif model == "model2":
#             while name in self.model1_features:
#                 name=name+"1"
#             self.model2_features[name] = X

#         else:
#             raise RuntimeError("Unknown model name for _log_layer.")

#     def _insert_hooks(self):
#         self.handles=[]
#         # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
#         self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()]) 
#         self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules() ]) 
#         self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
#         self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        
  
#     def export(self):
      
#         return {
#             "model1_name": "Trained",
#             "model2_name": "PretrainedModel",
#             "IMCKA":self.IMhsic_matrix1 / (torch.sqrt(self.IMhsic_matrix0.unsqueeze(1))*torch.sqrt(self.IMhsic_matrix2.unsqueeze(0))),
#             "CAPCKA":self.CAPhsic_matrix1 / (torch.sqrt(self.CAPhsic_matrix0.unsqueeze(1))*torch.sqrt(self.CAPhsic_matrix2.unsqueeze(0))),
#             "model1_layers": self.named_modules(),
#             "model2_layers": self.model2.named_modules(),
#         }

#     def plot_results(self,
#                      model_name: str,
#                      save_path: str = None,
#                      title: str = None):
#         title =model_name+" HSIC" if title is None else model_name+title
#         fig, ax = plt.subplots()
#         if model_name=="IM":
#             print(self.IMhsic_matrix0) #46 #Comes out inf on val step
#             print(self.IMhsic_matrix2) # 110
#             t=self.IMhsic_matrix0.unsqueeze(1)*self.IMhsic_matrix2.unsqueeze(0) #46 x 110
#         #print(torch.sum(torch.abs(t)==t))
#             r=torch.sqrt(torch.abs(t))
#             r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
#             print("im1",self.IMhsic_matrix1)
#             print("r", r)
#             hsic_matrix = torch.div(self.IMhsic_matrix1.squeeze().t(), r)
#             print("hsic",hsic_matrix)
#         else:
#             print(self.CAPhsic_matrix0.shape,self.CAPhsic_matrix2.shape)
#             t=self.CAPhsic_matrix0.unsqueeze(1)*self.CAPhsic_matrix2.unsqueeze(0)
#             r=torch.sqrt(torch.abs(t))
#             r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
#             print("cap1", self.CAPhsic_matrix1.shape)
#             print("r",r.shape)
#             hsic_matrix = torch.div(self.CAPhsic_matrix1.squeeze().t() , r)
#         hsic_matrix=torch.nan_to_num(hsic_matrix,nan=0)
#         im = ax.imshow(hsic_matrix.cpu(), origin='lower', cmap='magma')
#         ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
#         ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)
#         if title is not None:
#             ax.set_title(f"{title}", fontsize=18)
#         else:
#             ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)
#         add_colorbar(im)
#         plt.tight_layout()
#         if save_path is not None:
#             plt.savefig(save_path, dpi=300)

#     def test_step(self,batch,*args):
#         #do stock loss here
#         image_features=self.encode_image(batch[0])

#         return {"imfeatures":image_features, "classes":batch[1]}

#     def test_epoch_end(self,acc_val):
#         imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in acc_val],dim=0)).cpu().numpy()
#         labels=torch.cat([val["classes"] for val in acc_val],dim=0).cpu().numpy()
#         if not hasattr(self,"Iclassifier"):
#             self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
   
#         self.Iclassifier.fit(imfeatures, labels)
#         self.log( "TopK Imagenet",self.Iclassifier.score(imfeatures, labels))
        
# def batch_HSIC2(K):
#     #K is Layers x B x B
#     a=torch.sum(K,dim=-1)
#     #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
#     b=torch.sum(K,dim=-2)
#     c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
#     #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
#     output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
#     return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
#     #check for why pos infs... 
# def batch_HSIC3(K,L):
    # K=K.unsqueeze(1) # 46,1,B,B
    # L=L.unsqueeze(0) # 1,46, B,B
    # a=torch.sum(L,dim=-1) #1,46,10
    # b=torch.sum(K,dim=-2) #46,1,10
    # #print(a.shape,b.shape)
    # c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
    # #print(c.shape) # expect LayerK, LayerL, 
    # return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
    # #returns many pos infs 