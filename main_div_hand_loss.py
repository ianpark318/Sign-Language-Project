import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchmetrics
from model_dim import skeleTransLayer as skelTrans2
from dataset import KSLDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def custom_collate(batch):
    input_list = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]
    len_list = [item['frame_len'] for item in batch]
    input_list= pad_sequence(input_list, batch_first=True, padding_value=0.)
    return {'input': input_list.contiguous(),
            'label': labels,
            'len_list': len_list}

def gen_src_mask(total_len, len_list):
    batch_len = len(len_list)
    zero = torch.zeros(batch_len, total_len+1)
    for tens, t in zip(zero, len_list):
        mask = torch.ones(total_len-t) 
        tens[t+1:] = mask
    ret = zero.bool()
    return torch.transpose(ret, 0, 1)

with open('labels.json','r') as f:
    dict_json = json.load(f)
    classes = dict_json['labels']
    num_classes = len(classes)
    # print(classes)

class TrainerModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.model1 = skelTrans2(num_classes, 21, 1, 60, 1, mask = False)
        self.model2 = skelTrans2(num_classes, 21, 1, 60, 1, mask = False)
        self.model3 = skelTrans2(num_classes, 42, 1, 60, 1, mask = False)
        self.accuracy = torchmetrics.classification.Accuracy(multiclass=True, num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['label']
        l = batch['len_list']
        _, total_len, _, _ = x.shape
        mask = gen_src_mask(total_len,l).to('cuda')
        # mask = None
        temp_list = []
        global classes
        for label in y:
            k = classes.index(label)
            temp_list.append(k)
        labels = torch.tensor(temp_list).to("cuda")
        z = self.model1(x[:, :, :21, :], mask)
        z2 = self.model2(x[:, :, 21:, :], mask)
        z3 = self.model3(x, mask)
        loss = F.cross_entropy(z, labels)
        loss2 = F.cross_entropy(z2, labels)
        loss3 = F.cross_entropy(z3, labels)
        loss_sum = loss+ loss2 +loss3
        self.log('train_loss_step', loss_sum)
        return loss_sum
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['label']
        l = batch['len_list']
        _, total_len, _, _ = x.shape
        mask = gen_src_mask(total_len,l).to('cuda')
        mask = None
        temp_list = []
        global classes
        for label in y:
            k = classes.index(label)
            temp_list.append(k)
        labels = torch.tensor(temp_list).to("cuda")
        z = self.model1(x[:, :, :21, :], mask)
        z2 = self.model2(x[:, :, 21:, :], mask)
        z3 = self.model3(x, mask)
        z = (z+z2+z3)/3
        
        acc = self.accuracy(z, labels)
        print(z, labels)
        self.log('val_acc_step', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer    
    

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.morp_path = '/home/jaehyeong/sl/data/morpheme'
        self.keyp_path = '/home/jaehyeong/sl/data/keypoint'
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.nia_train = KSLDataset(is_train=True, morp_path=self.morp_path, keyp_path=self.keyp_path)
        self.nia_val = KSLDataset(is_train=False, morp_path=self.morp_path, keyp_path=self.keyp_path)
    
    def train_dataloader(self):
        return DataLoader(self.nia_train, batch_size = self.batch_size, num_workers=10, collate_fn=custom_collate, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.nia_val, batch_size = self.batch_size, num_workers = 10, collate_fn=custom_collate)
    
def get_arguments():
    return

if __name__=="__main__":

    trainer = pl.Trainer(accelerator="gpu", max_epochs=100, devices=1)
    custom_callback = []
    #custom_callback.append(TensorBoardLogger(args.save_dir, ))
    #trainer = pl.Trainer().from_argparse_args(args)
    model = TrainerModule()
    data = DataModule()
    trainer.fit(model, data)



