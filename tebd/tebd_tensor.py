import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("-J",type = float,default=1,help="")
parser.add_argument("-Hx",type = float,default=0.5,help="")
parser.add_argument("-delta",type = float,default=0.001,help="")
parser.add_argument("-iters",type = int,default=10000,help="")
parser.add_argument("-cut", type = int, default=10,help="")


args = parser.parse_args()

lamB = torch.randn(args.cut).to(dtype=torch.float64)
lamA = torch.randn(args.cut).to(dtype=torch.float64)

GA = torch.randn(args.cut,2,args.cut).to(dtype=torch.float64)
GB = torch.randn(args.cut,2,args.cut).to(dtype=torch.float64)

sz = torch.tensor([[1,0],[0,-1]]).to(dtype=torch.float64)
sx = torch.tensor([[0,1],(1,0)]).to(dtype=torch.float64)
I2 = torch.eye(2).to(dtype=torch.float64)

H = -args.J*torch.einsum("ij,kl->ikjl",sz,sz)+0.5*-args.Hx*(torch.einsum("ij,kl->ikjl",sx,I2)+torch.einsum("ij,kl->ikjl",I2,sx))

Hp = H*-args.delta

e,v = torch.eig(Hp.reshape(4,4),True)
E = torch.exp(e[:,0])
U = torch.matmul(torch.matmul(v,torch.diag(E)),v.t()).reshape(2,2,2,2)

for i in range(args.iters):
    theta = torch.einsum("ab,bcd,de,efg,gh->acfh",torch.diag(lamB),GA,torch.diag(lamA),GB,torch.diag(lamB))
    thetap = torch.einsum("abcd,bcef->aefd",theta,U).reshape(args.cut*2,args.cut*2)
    u,s,vt = torch.svd(thetap)
    s = s[:args.cut]
    u = u[:,:args.cut].reshape(args.cut,2,args.cut)
    v = vt.t()[:args.cut,:].reshape(args.cut,2,args.cut)
    lamA = s/(torch.sqrt((s**2).sum()))
    lamBinv = lamB**-1
    GA = torch.einsum("ab,bcd->acd",torch.diag(lamBinv),u)
    GB = torch.einsum("abc,cd->abd",v,torch.diag(lamBinv))

    (lamA,lamB) = (lamB,lamA)
    (GA,GB) = (GB,GA)

theta = torch.einsum("ab,bcd,de,efg,gh->acfh",torch.diag(lamB),GA,torch.diag(lamA),GB,torch.diag(lamB))
thetaU = torch.einsum("abcd,bcef->aefd",theta,H)

E = torch.einsum("abcd,abcd->",theta,thetaU)

(lamA,lamB) = (lamB,lamA)
(GA,GB) = (GB,GA)

theta = torch.einsum("ab,bcd,de,efg,gh->acfh",torch.diag(lamB),GA,torch.diag(lamA),GB,torch.diag(lamB))
thetaU = torch.einsum("abcd,bcef->aefd",theta,H)

E += torch.einsum("abcd,abcd->",theta,thetaU)

E /= 2

print("E:",E.item())


