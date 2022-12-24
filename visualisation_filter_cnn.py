import warnings 
warnings.filtewarning("ignore")
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets 
import torchvision.models as models 
import torchvision.transforms as transforms


def plot_filters_multi_channel(weight_tensor):
  n_kernels=weight_tensor.shape[0]
  n_cols=12
  n_rows=n_kernels
  fig=plt.figure(figsize=(n_cols,n_rows))
  for i in range(n_kernels):
    ax1=fig.add_subplot(n_rows,n_cols,i+1)
    #converting tensor to numy img
    npimg=np.array(weight_tensor[i].numpy(),np.float32)
    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
    npimg = npimg.transpose((1, 2, 0))
    ax1.imshow(npimg)
    ax1.axis('off')
    ax1.set_title(str(i))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
  plt.tight_layout()
  plt.show()
def plot_filters_single_channel(t):
    
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    nrows = 1 + nplots//ncols
    
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    
    fig = plt.figure(figsize=(ncols, nrows))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()

def plot_filters_single_channel_big(t):
    
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
          
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='Greys', ax=ax, cbar=False)

#visulaizing alexnet 
alexnet=models.alexnet(pretrained=True)
def plot_weights(model,layer_num,single_channel=True,collated=False):
  layer=model.features[layer_num]
  if  isinstance(layer,nn.Conv2d):
    weight_tensor=model.features[layer_num].weight.data
    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)
    else:
      if  weight_tensor.shape[1]==3:
         plot_filters_multi_channel(weight_tensor)
      else:
        print("cannot display the  images")
  else:
    print("only cnn layers")
'''plot_weights(alexnet, 0, single_channel = False)
plot_weights(alexnet, 0, single_channel = True)

plot_weights(alexnet, 0, single_channel = True, collated = True)
plot_weights(alexnet, 3, single_channel = True, collated = True)'''

