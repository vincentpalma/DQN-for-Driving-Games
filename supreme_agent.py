import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
from skimage.transform import resize
import pickle

import pyautogui  #for keyboard control & screenshots
import win32gui
import cv2
from directkeys import PressKey, ReleaseKey, W, A, S, D

from model.FastSCNN import FastSCNN
from autoencoder import VAE

AVAILABLE_ACTIONS = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]

class DQN(nn.Module):
   def __init__(self):
      nn.Module.__init__(self)
      self.l1 = nn.Linear(36, 256)    # input is img vector of 32 dimensions
      self.l2 = nn.Linear(256, 256)
      self.l3 = nn.Linear(256, len(AVAILABLE_ACTIONS))       

   def forward(self, x):   # forward propagation
      x = F.relu(self.l1(x))
      x = F.relu(self.l2(x))
      x = self.l3(x)
      return x

def get_window_coords(window_title=''):
   hwnd = win32gui.FindWindow(None, window_title)
   if hwnd:
      win32gui.SetForegroundWindow(hwnd)
      x, y, x1, y1 = win32gui.GetClientRect(hwnd)
      x, y = win32gui.ClientToScreen(hwnd, (x, y))
      x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
      return x, y, x1, y1
   else:
      print('Window not found!')

def step(action):
   print(action, end=' ')        #control keyboard with PyAutoGui

   # ReleaseKey(S) #reset all keys
   # ReleaseKey(W)
   # ReleaseKey(A)
   # ReleaseKey(D)

   if action[0] == -1:  #brake 
      PressKey(S) 
      time.sleep(50)
      ReleaseKey(S)
   elif action[0] == 1: #accelerate
      PressKey(W) 
      time.sleep(50)
      ReleaseKey(W)
   if action[1] == -1:  #turn left
      PressKey(A) 
      time.sleep(50)
      ReleaseKey(A)
   elif action[1] == 1: #turn right
      PressKey(D) 
      time.sleep(50)
      ReleaseKey(D)

if __name__ == '__main__':
   device = torch.device('cpu')

   datas = pickle.load(open('./SavedWeights/bdd100k_inform.pkl', "rb"))

   segmentation = FastSCNN(classes=3)
   checkpoint = torch.load('./SavedWeights/model_8.pth', map_location=device)
   segmentation.load_state_dict(checkpoint['model'])

   autoencoder = torch.load("./SavedWeights/last_VAE.pt", map_location=device)
   agent = torch.load('./SavedWeights/dqn.pt', map_location=device)
   trans = T.ToPILImage(mode='L')

   last_actions = np.zeros(4)
   last_action = np.zeros(2)

   x_coord, y_coord, x1_coord, y1_coord = get_window_coords('VDrift')

   while True:
      start = time.time()
      img = pyautogui.screenshot(region=(x_coord, y_coord, x1_coord, y1_coord))
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
      y = torch.from_numpy(y.copy()).unsqueeze(0) # our image is now a tensor ready to be fed

      # Encoding part
      pred, _,_,bottleneck = autoencoder(y.unsqueeze(0))
      encoded = bottleneck.data.cpu().numpy()
      decoded = pred.squeeze().data.cpu().numpy()

      # Deep Q-learning part
      action = agent(torch.FloatTensor([np.append(encoded,last_actions)])).data.max(1)[1].view(1, 1)
      action = AVAILABLE_ACTIONS[action.numpy()[0, 0]]
      last_actions = np.append(last_action,action)
      last_action = action
      step(last_action)
      print('Time :',time.time()-start)

   
      #cv2.imshow('window', decoded)
      #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
      if cv2.waitKey(25) & 0xFF == ord('q'):
         cv2.destroyAllWindows()
         break
      