from model.BASNet import BASNet
import torch
import torch.nn as nn
class chain_BASNet(nn.Module):
    def __init__(self,n_channels=3,n_classes=1):
        super(chain_BASNet,self).__init__()
        self.lung_basnet=BASNet(3,1)
        self.infection_basnet=BASNet(3,1)
    def forward(self,x):
        lung_mask,lung1,lung2,lung3,lung4,lung5,lung6,lung7=self.lung_basnet(x)
        lung_x=lung_mask*x
        infection_mask,infection1,infection2,infection3,infection4,infection5,infection6,infection7=self.infection_basnet(lung_x)

        lung_multiscale=[lung1,lung2,lung3,lung4,lung5,lung6,lung7]
        infection_multiscale=[infection1,infection2,infection3,infection4,infection5,infection6,infection7]

        return lung_mask,lung_multiscale,infection_mask,infection_multiscale



import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
model = chain_BASNet()
macs, params = get_model_complexity_info(model, (3, 224,224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import time
time1=time.time()
inputs=torch.ones((1,3,224,224))
with torch.no_grad():
    r=model(inputs)
time2=time.time()
print("Inf time: ",time2-time1)