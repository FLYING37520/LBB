import os,sys
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from torchvision.models import *

from torch.utils.data import DataLoader
import tqdm
from Net.resnet import ResNet18
from mydataloader import getBatchSample
import ray#import行为会导致sys.path[0]指向解释器目录，
curpath=os.path.dirname(os.path.realpath(__file__))
os.chdir(curpath)

bs=128
lr=0.1
epoch=120
nsoft=1
num_workers = 4
iterations= ( round(50000/bs/nsoft) )   #每个epoch的迭代数量
netname='resnet18' # resnet18 vit EfficientNetB0 shufflenetv2



def get_model(netname='resnet18',num_classes=10):
    net=None
    import Net
    if netname=='resnet18':
        net=ResNet18(num_classes=num_classes)
    if netname=='vit':
        from Net.vit import ViT
        lr=0.0001
        net=  ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = num_classes,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    if netname=='EfficientNetB0':
        import Net.efficientnet
        net=Net.efficientnet.EfficientNetB0(num_classes=num_classes)

    if netname=='':
        import Net.shufflenetv2
        net=Net.shufflenetv2.ShuffleNetV2(1.0,num_classes=num_classes)
    return net
def get_data_loader():
    os.chdir(os.getcwd())
    """Safely downloads data. Returns training/validation set dataloader."""
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(f"{curpath}/data/mylock.lock"):
        train_loader = DataLoader(
            datasets.CIFAR10(
                f"{os.getcwd()}/data/", train=True, download=True, transform=transform_train
            ),
            batch_size=bs,
            shuffle=True,
            num_workers=2,prefetch_factor=2
        )
        test_loader = DataLoader(
            datasets.CIFAR10(f"{os.getcwd()}/data/", train=False, download=False,transform=transform_test),
            batch_size=bs,
            shuffle=True,
        )
    return train_loader, test_loader


def evaluate(model, test_loader,testDevice=2):
    """Evaluates the accuracy of the model on a validation dataset."""
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_loader = DataLoader(
        datasets.CIFAR10(
            f"{os.getcwd()}/data/", train=False, download=True, transform=transform_test
        ),
        batch_size=bs,
        shuffle=True,
        num_workers=2,prefetch_factor=2
    )

    correct_local=0
    lossTest_local=0
    accTestPercent=0
    pbarTest=tqdm.tqdm(enumerate(test_loader),dynamic_ncols=True)
    for batch_idx, (data, target) in pbarTest:
        data, target=data.cuda(2),target.cuda(2)
        torch.cuda.synchronize(2)
        model.zero_grad()
        model.cuda(2)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        torch.cuda.synchronize()
        lossTest_local+=loss.item()
        _, predicted = output.max(1)
        correct_local += predicted.eq(target).sum().item()
        accTestPercent=correct_local/bs/(batch_idx+1)
        lossTest_local__=lossTest_local/(batch_idx+1)
        pbarTest.set_postfix_str(f'TEST loss :{lossTest_local__:.3f} TEST acc{accTestPercent:.3f}')
    return round(lossTest_local__,4) ,round(accTestPercent,4) 


def get_weights (model:torch.nn.Module):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def set_weights(model:torch.nn.Module, weights):
    model.load_state_dict(weights)

def get_gradients(model:torch.nn.Module):
    grads = []
    for p in model.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

def set_gradients(model:torch.nn.Module, gradients):
    for g, p in zip(gradients, model.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)

@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ResNet18(num_classes=10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch)

    def getLR(self):
        curLr=self.optimizer.state_dict()['param_groups'][0]['lr']
        return curLr
    def schStep(self):
        self.scheduler.step()
        
    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)  #####除以nsoft是自己家的
        ]
        self.optimizer.zero_grad()
        set_gradients(self.model,summed_gradients)
        self.optimizer.step()
        return get_weights(self.model)

    # def apply_gradients(self, *gradients):
    #     for gradient in gradients:
    #         self.optimizer.zero_grad()
    #         set_gradients(self.model,gradient)
    #         self.optimizer.step()
    #         return get_weights(self.model)

    def get_weights(self):
        return get_weights(self.model)

@ray.remote(num_gpus=1)
class DataWorker(object):
    def __init__(self,rank):
        self.rank=rank
        self.device=ray.get_gpu_ids()[0]
        print(f'rank{rank} get device{self.device} {torch.cuda.is_available()} count:{torch.cuda.device_count()}')
        sleep(1)
        self.model = ResNet18(num_classes=10)
        self.data_iterator = iter(get_data_loader()[0])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.dataset=datasets.CIFAR10(f"{os.getcwd()}/data/", train=True, download=True, transform=transform_train)
        # print(f'dataset长度')
        # print(self.dataset.__len__())

    def compute_gradients(self, weights):
        self.model.cuda()
        set_weights(self.model,weights)
        # try:
        #     data, target = next(self.data_iterator)
        # except StopIteration:  # When the epoch ends, start a new epoch.
        #     self.data_iterator = iter(get_data_loader()[0])
        #     data, target = next(self.data_iterator)
        # data=torch.rand(bs,3,32,32).cuda()
        # target=torch.randint(0,10,(bs,)).cuda()

        
        data, target=getBatchSample(self.dataset,bs=bs)
        data, target=data.cuda(),target.cuda()
        torch.cuda.synchronize()
        self.model.zero_grad()
        output = self.model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        torch.cuda.synchronize()

        _, predicted = output.max(1)
        correct_local = predicted.eq(target).sum().item()
        # print(f'train loss :{loss.item():.3f} train acc{correct_local/bs:.3f}')
        
        return get_gradients(self.model)




print("Running Asynchronous Parameter Server Training.")

ray.init(ignore_reinit_error=True)
ps:ParameterServer = ParameterServer.remote(lr)
workers = [DataWorker.remote(i) for i in range(num_workers)]


model = ResNet18(num_classes=10).cuda(2)
import time
recordFilePath=f'recordAsync/nsoft_{nsoft}_{netname}_'+time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
recordDICT={'loss':[], 'acc':[], 'dutation':[]}
test_loader = get_data_loader()[1]
current_weights = ps.get_weights.remote()
gradients = {}
for worker in workers:
    gradients[worker.compute_gradients.remote(current_weights)] = worker
t=time.time()
for ep in range(epoch):
    pbar=tqdm.tqdm(range(iterations ),dynamic_ncols=True)
    epstart=time.time()
    for i in pbar:
        start=time.time()
        # print('########################')
        # print(gradients) 
        # exit()
        ready_gradient_list, _ = ray.wait(list(gradients),num_returns=nsoft)
        # print(ready_gradient_list.__len__())
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        ps:ParameterServer
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker
        duration=time.time()-start
        # ray.get()
        if i ==0:
            curLR = ray.get( ps.getLR.remote() )#######################
        pbar.set_postfix_str(f'throuput:{nsoft*bs/duration:.3f} t:{duration:.3f} LR:{curLR}')

    epdura= round (time.time()-epstart  ,4)
    ray.get(ps.schStep.remote())#######################
    
            
    set_weights(model,ray.get(current_weights))
    lossTest,accuracy = evaluate(model, test_loader)
    print(f"epoch {ep} Test:acc:{accuracy:.2f} loss:{lossTest:.2f}")
    recordDICT['loss'].append(lossTest)
    recordDICT['acc'].append(accuracy)
    recordDICT['dutation'].append(epdura)
    np.save(recordFilePath,recordDICT,allow_pickle=True)
    print(recordDICT)
