import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchsummary import summary

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import pickle
from tqdm import tqdm

class CAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
    self.unpool = nn.MaxUnpool2d(kernel_size=2)

    self.l1 = nn.Sequential(
      nn.Conv2d(1,32,kernel_size=3, padding=2), #out is of size (178,322)
      nn.ReLU(),
      )

    self.l2 = nn.Sequential(
      nn.Conv2d(32,16,kernel_size=3, padding=2), #
      nn.ReLU(),
      )

    self.l3 = nn.Sequential(
      nn.Conv2d(16,8,kernel_size=3, padding=2),
      nn.ReLU(),
      )

    self.l4 = nn.Sequential(
      nn.Conv2d(8,4,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.l5 = nn.Sequential(
      nn.Conv2d(4,1,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.drop_out = nn.Dropout(p=0.2)

    self.up1 = nn.ConvTranspose2d(1,4,kernel_size=3, padding=1)
    self.up2 = nn.ConvTranspose2d(4,8,kernel_size=3, padding=1)
    self.up3 = nn.ConvTranspose2d(8,16,kernel_size=3, padding=2)
    self.up4 = nn.ConvTranspose2d(16,32,kernel_size=3, padding=2)
    self.up5 = nn.ConvTranspose2d(32,1,kernel_size=3, padding=2)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.l1(x)
    x, i1 = self.pool(x)
    x = self.l2(x)
    x, i2 = self.pool(x)
    x = self.l3(x)
    x, i3  = self.pool(x)
    x = self.l4(x)
    x, i4 = self.pool(x)
    x = self.l5(x)
    x, i5 = self.pool(x)
    #x = self.drop_out(x)

    bottleneck = torch.flatten(x)

    x = self.unpool(x, i5, output_size=(11,20))
    x = self.up1(x, output_size=(11,20))
    x = self.relu(x)
    x = self.unpool(x, i4, output_size=(23,41))
    x = self.up2(x, output_size=(23,41))
    x = self.relu(x)
    x = self.unpool(x, i3, output_size=(47,83))
    x = self.up3(x, output_size=(45,81))
    x = self.relu(x)
    x = self.unpool(x, i2, output_size=(91,163))
    x = self.up4(x, output_size=(89,161))
    x = self.relu(x)
    x = self.unpool(x, i1, output_size=(178,322))
    x = self.up5(x, output_size=(176,320))
    x = self.sigmoid(x)

    return x, bottleneck

class VAE(nn.Module):
  def __init__(self, latent_dim=32):
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
    self.unpool = nn.MaxUnpool2d(kernel_size=2)

    self.l1 = nn.Sequential(
      nn.Conv2d(1,32,kernel_size=3, padding=2), #out is of size (178,322)
      nn.ReLU(),
      )

    self.l2 = nn.Sequential(
      nn.Conv2d(32,16,kernel_size=3, padding=2), #
      nn.ReLU(),
      )

    self.l3 = nn.Sequential(
      nn.Conv2d(16,8,kernel_size=3, padding=2),
      nn.ReLU(),
      )

    self.l4 = nn.Sequential(
      nn.Conv2d(8,4,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.l5 = nn.Sequential(
      nn.Conv2d(4,1,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.fc1 = nn.Linear(50, latent_dim)
    self.fc2 = nn.Linear(50, latent_dim)
    self.fc3 = nn.Linear(latent_dim, 50)

    self.up1 = nn.ConvTranspose2d(1,4,kernel_size=3, padding=1)
    self.up2 = nn.ConvTranspose2d(4,8,kernel_size=3, padding=1)
    self.up3 = nn.ConvTranspose2d(8,16,kernel_size=3, padding=2)
    self.up4 = nn.ConvTranspose2d(16,32,kernel_size=3, padding=2)
    self.up5 = nn.ConvTranspose2d(32,1,kernel_size=3, padding=2)

    self.flt = nn.Sequential(nn.Flatten())
    self.unflt = nn.Sequential(nn.Unflatten(1, torch.Size([1,5,10])))
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def reparameterize(self, mu, sigma):
    std = sigma.mul(0.5).exp_()
    # return torch.normal(mu, std)
    esp = torch.randn(*mu.size())
    z = mu + std * esp
    return z

  def forward(self, x):
    x = self.l1(x)
    x, i1 = self.pool(x)
    x = self.l2(x)
    x, i2 = self.pool(x)
    x = self.l3(x)
    x, i3  = self.pool(x)
    x = self.l4(x)
    x, i4 = self.pool(x)
    x = self.l5(x)
    x, i5 = self.pool(x)

    x = self.flt(x)
    mu = self.fc1(x)
    sigma = self.fc1(x)
    bottleneck = self.reparameterize(mu, sigma)
    x = self.fc3(bottleneck)
    x = self.unflt(x)

    x = self.unpool(x, i5, output_size=(11,20))
    x = self.up1(x, output_size=(11,20))
    x = self.relu(x)
    x = self.unpool(x, i4, output_size=(23,41))
    x = self.up2(x, output_size=(23,41))
    x = self.relu(x)
    x = self.unpool(x, i3, output_size=(47,83))
    x = self.up3(x, output_size=(45,81))
    x = self.relu(x)
    x = self.unpool(x, i2, output_size=(91,163))
    x = self.up4(x, output_size=(89,161))
    x = self.relu(x)
    x = self.unpool(x, i1, output_size=(178,322))
    x = self.up5(x, output_size=(176,320))
    x = self.sigmoid(x)

    return x, mu, sigma, bottleneck

def loss_fn(recon_x, x, mu, sigma):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())

    return BCE + KLD, BCE, KLD

##### DATALOADER #####
trf = T.Compose([T.ToTensor()])
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path

class BDD100K(Dataset):
  def __init__(self,img_dir,gt_dir,seg_model=None,gt=False,pytorch=True):
    super().__init__()
    # Loop through the files in red folder and combine, into a dictionary, the other bands
    self.files = [self.combine_files(f, gt_dir) for f in img_dir.iterdir() if not f.is_dir()]
    self.pytorch = pytorch
    self.gt = gt
      
  def combine_files(self, img_file: Path, gt_dir):
    files = {'img': img_file,
             'gt': Path(str(gt_dir/img_file.name).split('.')[0] + '_drivable_id.png')}
    return files
                                     
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    if self.gt:
      trf2 = T.Compose([T.Resize((176,320)),T.ToTensor()])
      return 0, trf2(Image.open(self.files[index]['gt']))
    else:
      datas = pickle.load(open('./dataset/inform/bdd100k_inform.pkl', "rb"))
      img = Image.open(self.files[index]['img'])
      image = np.asarray(img, np.float32)
      image = resize(image, (176,320), order=1, preserve_range=True)

      image -= datas['mean']
      # image = image.astype(np.float32) / 255.0
      image = image[:, :, ::-1]  # revert to RGB
      image = image.transpose((2, 0, 1))  # HWC -> CHW

      image = torch.from_numpy(image.copy())

      segmentation.eval()
      y = segmentation(image.unsqueeze(0))
      y = y.cpu().data[0].numpy()
      y = y.transpose(1, 2, 0)

      y = np.asarray(np.argmax(y, axis=2), dtype=np.float32)
      y[y==2] = 0
      y = torch.from_numpy(y.copy()).unsqueeze(0)

      return trf(img), y
##############

def train_CAE(model,dl,criterion,optimizer,epochs):
  l = []
  for epoch in range(1,epochs+1):
    train_loss = 0.0
    t = tqdm(total=len(dl),desc='Episodes')
    i=0
    for _, images in dl:
      optimizer.zero_grad()
      outputs, _ = model(images)
      loss = criterion(outputs,images)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()*images.size(0)
      i+=1
      if i%50 == 0:
        torch.save(model, "../seg_weights/last_CAE.pt")
      t.set_description(f'Episodes (loss: {round(float(loss),6)})')
      t.update(1)
    t.close()
    train_loss = train_loss/len(dl)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    torch.save(model, "../seg_weights/CAE_"+str(epoch)+".pt")
    l.append(train_loss)
  return l

def train_VAE(model,dl,criterion,optimizer,epochs):
  l = []
  for epoch in range(1,epochs+1):
    train_loss = 0.0
    t = tqdm(total=len(dl),desc='Episodes')
    i=0
    for _, images in dl:
      optimizer.zero_grad()
      outputs, mu, sigma, _ = model(images)
      loss, bce, kld = criterion(outputs, images, mu, sigma)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()*images.size(0)
      i+=1
      if i%50 == 0:
        torch.save(model, "../seg_weights/last_VAE.pt")
      t.set_description("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch, 
                                epochs+1, loss.item()/images.size(0), bce.item()/images.size(0), kld.item()/images.size(0)))
      t.update(1)
    t.close()
    train_loss = train_loss/len(dl)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    torch.save(model, "../seg_weights/VAE_"+str(epoch)+".pt")
    l.append(train_loss)
  return l

##############
if __name__ == "__main__":
  from builders.model_builder import build_model

  autoencoder = VAE().cpu()
  #print(summary(CAE().cuda(),(1,176,320)))
  segmentation = build_model('FastSCNN',num_classes=3)
  checkpoint = torch.load('./checkpoint/bdd100k/FastSCNNbs200gpu1_train/model_8.pth', map_location=torch.device('cpu'))
  segmentation.load_state_dict(checkpoint['model'])
  
  train_ds = BDD100K( Path('./dataset/bdd100k/images/100k/train/'),
                     Path('./dataset/bdd100k/drivable_maps/labels/train/'),
                     segmentation,False
                     )
  valid_ds = BDD100K( Path('./dataset/bdd100k/images/100k/val/'),
                     Path('./dataset/bdd100k/drivable_maps/labels/val/'),
                     segmentation,False
                     )

  train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
  valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)

  ######TRAINING##########
  training_mode = False
  if training_mode:
    autoencoder = torch.load("../seg_weights/last_VAE.pt")
    opt = torch.optim.Adam(autoencoder.parameters(),lr=0.001)
    train_loss = train_VAE(autoencoder, train_dl, loss_fn, opt, epochs=15)

    pickle.dump(train_loss, open("../seg_weights/loss_stats_autoencoder.pkl","wb"))
    torch.save(autoencoder,"../seg_weights/last_autoencoder.pt")

  #######TESTING###########
  trained_autoencoder = torch.load("../seg_weights/last_VAE.pt", map_location="cpu")
  trans = T.ToPILImage(mode='RGB')
  trans2 =T.ToPILImage(mode='L')
  x,y = valid_ds[2]
  #plt.imshow(trans(x.squeeze())); plt.show()
  plt.imshow(trans2(y.squeeze())); plt.show()
  print(y.shape,y)

  start = time.time()
  pred, _,_,bottleneck = trained_autoencoder(y.unsqueeze(0))
  print(time.time()-start,'seconds')
  print(bottleneck)
  plt.imshow(trans2(pred.squeeze())); plt.show()

  loss = pickle.load(open("../seg_weights/loss_stats_autoencoder.pkl","rb"))
  plt.plot(loss)
  plt.show()
  for j in range(10):
    x,y = valid_ds[j]
    trans(x).save("./result/autoencoder/gt_"+str(j)+".png","PNG")
    for i in range(1,2):
      mod = torch.load("../seg_weights/last_VAE3.pt", map_location="cpu").cpu()
      pred, _,_,bottleneck = mod(y.unsqueeze(0).cpu())
      pred = trans2(pred.squeeze())
      datas = pred.getdata()
      new_img2 = []
      for pixel in datas:
        #print(pixel)
        if pixel==0: new_img2.append((255))
        else: new_img2.append((0))
      pred.putdata(new_img2)
      pred.save("./result/autoencoder/"+str(j)+"pred_"+str(i)+"last3.png","PNG")