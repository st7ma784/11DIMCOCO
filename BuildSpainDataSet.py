import pytorch_lightning as pl
import os
import wandb
wandb.login()
import pandas as pd
import torch,time
from transformers import AutoTokenizer
from pySmartDL import SmartDL

from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
Rs=Resize((224,224),interpolation=Image.NEAREST)
T= Compose([Rs,ToTensor()])
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
preprocess=Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
class TriModal(torch.utils.data.Dataset):
    def __init__(self,dir,transform=preprocess,tokenizer=None):
        self.dir=dir
        self.tokenizer=tokenizer
        if self.tokenizer== None:
            self.tokenizer= AutoTokenizer.from_pretrained("gpt2",cache_dir=dir)
            self.tokenizer.vocab["</s>"] = self.tokenizer.vocab_size -1
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.func= lambda x: self.tokenizer(
                    x,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids']
            
        self.transform=transform
        self.imdir=os.path.join(dir,"images")
        self.datadir=os.path.join(dir,"data")
        filea="train_machine_spanish.xlsx"
        fileb="train_machine_english.xlsx"
        Spdataframe=pd.read_excel(os.path.join(self.datadir,filea))
        Endataframe=pd.read_excel(os.path.join(self.datadir,fileb))
        self.spanSentences=torch.cat(Spdataframe['caption'].apply(self.func).values.tolist(),dim=0)
        self.enSentences=torch.cat(Endataframe['caption'].apply(self.func).values.tolist(),dim=0)
        self.filenames=Endataframe['image_id']
    def __len__(self):
        return len(self.enSentences)
    def __getitem__(self,idx):
    
        imid=self.filenames[idx]
        imid="".join([(12-len(str(imid)))*"0"]+[str(imid)]+[".jpg"])
        img=Image.open(os.path.join(self.imdir,imid)).convert("RGB")
        if self.transform:
            img=self.transform(img)
        else:
            img=Rs(img)

            img=T(img)
        return self.enSentences[idx],self.spanSentences[idx], img

class DataSet(torch.utils.data.Dataset):
    def __init__(self,dir,filenames,captions=[],transform=None):
        self.dir=dir
        self.entries= list(zip(filenames,*[c for c in captions]))
        self.transform=transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self,idx):
        imid,captions=self.entries[idx][0],self.entries[idx][0:]

        imid="".join([(12-len(str(imid)))*"0"]+[str(imid)]+[".jpg"])
        img=Image.open(os.path.join(self.dir,imid))
        if self.transform:
            return self.transform(Rs(img))
        return T(img),captions
class DataModule(pl.LightningDataModule):

    def __init__(self,dir="MS-COCO-ES",batch_size=3,tokenizer=None):
        super().__init__(batch_size)
        self.tokenizer=tokenizer
        if self.tokenizer== None:
            self.tokenizer= AutoTokenizer.from_pretrained("gpt2",cache_dir=dir)
            self.tokenizer.vocab["</s>"] = self.tokenizer.vocab_size -1
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.func= lambda x: self.tokenizer(
                    x,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids']
            
        self.batch_size=batch_size
        self.datadir=os.path.join(dir,"data")
        #self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.imdir=os.path.join(dir,"images")
        #self.clip,self.preprocess = clip.load("ViT-B/32", jit=False, device=self.device)

    @torch.no_grad()
    def download_data(self):
        # wget https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_human_spanish.xlsx
        # wget https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_machine_english.xlsx
        # wget https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_human_spanish.xlsx
        urls= ["https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_human_spanish.xlsx",
        "https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_machine_english.xlsx",
        "https://github.com/carlosGarciaHe/MS-COCO-ES/raw/master/data/train_human_spanish.xlsx",
        ]
        objs=[]
        for url in urls:
            #print("url:",url)
            name=str(url).split('/')[-1]
            obj=SmartDL(url,os.path.join(self.datadir,name),progress_bar=False, verify=False)
            obj.FileName=name
            if not os.path.exists(os.path.join(self.datadir,name)):
                print(os.path.join(self.datadir,name))
                objs.append(obj)
                obj.start(blocking=False,  )#There are security problems with Hostename 'images.cocodataset.org' and Certificate 'images.cocodataset.org' so we need to disable the SSL verification
        for obj in objs:
            while not obj.isFinished():
                time.sleep(5)
        filea="train_machine_spanish.xlsx"
        fileb="train_machine_english.xlsx"
        dataframeA=pd.read_excel(os.path.join(self.datadir,filea),engine="openpyxl")
        dataframeB=pd.read_excel(os.path.join(self.datadir,fileb),engine="openpyxl")
        
        sentencesa= torch.stack(dataframeA['caption'].apply(self.func).values.tolist(),dim=0)
        sentencesa=sentencesa.reshape((int(sentencesa.shape[0]/5),5))

        sentencesb= torch.stack(dataframeB['caption'].apply(self.func).values.tolist(),dim=0)
        sentencesb=sentencesb.reshape((int(sentencesb.shape[0]/5),5))
        imagesids=dataframeA["image_id"]
        dataset=DataSet(dir=self.imdir,filenames=imagesids,captions=(sentencesa,sentencesb),transform=self.preprocess)
        

        path="pretrain.pt"
        torch.save(dataset, path)
        #print("datasets {} : ".format(data[0]))
        train_size = int(0.8 * len(data))
        val_size = int(0.1 *  len(data))
        test_size= len(data) - (train_size+val_size)

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
        torch.save(self.train_dataset, "train.pt")
        torch.save(self.val_dataset, "validation.pt")
        torch.save(self.test_dataset, "test.pt")
      
    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            #check if "espretrain.pt") exists in the directory
            if os.path.exists("train.pt"):
                self.train_dataset=torch.load("train.pt")
            else:
                self.download_data()
            
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            #check if esprevalidation.pt exists in the directory
            if os.path.exists("validation.pt"):
                self.val_dataset=torch.load("validation.pt")
            else:
                self.download_data()
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            #check for espretest.pt in the directory
            if os.path.exists("test.pt"):
                self.test_dataset=torch.load("test.pt")
            else:

                self.download_data()

        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
if __name__=="__main__":
    data=DataModule(batch_size=3)
    data.download_data()