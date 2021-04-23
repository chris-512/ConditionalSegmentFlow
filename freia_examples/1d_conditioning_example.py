import pdb
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from models.subnet_coupling import subnet_coupling_layer
from models.coupling_layers import glow_coupling_layer


def subnet_fc(c_in, c_out):
    print(c_in, c_out)
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))


fc_cond_length = 80
batch_size = 16

cinn = Ff.SequenceINN(2000 * 2)
for k in range(12):
    cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=(
        fc_cond_length,), subnet_constructor=subnet_fc, permute_soft=False)
cinn = cinn.cuda()

optimizer = torch.optim.Adam(cinn.parameters(), lr=0.001)

for i in range(100):
    print(i)
    x = torch.randn((batch_size, 2000 * 2)).cuda()
    cond = torch.randn((batch_size, fc_cond_length)).cuda()
    z, log_jac_det = cinn(x, c=[cond])
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / batch_size
    loss.backward()
    optimizer.step()

z = torch.randn(batch_size, 2000 * 2).cuda()
cond = torch.randn((batch_size, fc_cond_length)).cuda()
samples, _ = cinn(z, c=[cond], rev=True)
print(samples.shape)
