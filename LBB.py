'''
'''
from datetime import timedelta
from calendar import EPOCH
import functools
from multiprocessing import managers
import os
curpath=os.path.dirname(os.path.realpath(__file__))
os.chdir(curpath)
from pprint import pprint
import copy
import time
import pynvml
from multiprocessing import Manager
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import JoinableQueue, Manager, Queue
import multiprocessing
from multiprocessing.managers import BaseManager
from torch.utils.data import DataLoader
from math import ceil
from random import Random
import torchvision
from multiprocessing import Process,JoinableQueue
import multiprocessing
import pandas as pd
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from torchvision.models import *
import scipy.optimize as optimize
from tqdm import tqdm
import os,sys
import Net.resnet
from Net.vit import ViT
import Net.efficientnet
import Net.mobilenetv2
import Net.shufflenetv2
import Net.attention
import Net.VGG
from torchvision.datasets import ImageFolder
import math
import mydataloader

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
# os.environ['NCCL_BLOCKING_WAIT']='1'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'



def list_elem_eq(list):
    ret=True
    a=list[0]
    for ele in list:
        if ele!=a:
            return False
    return True
def list_round_up(l:list):
    lsorted=sorted(l,reverse=True)
    length=len(l)
    for i in range(length):
        if i<length/2:
            l[  l.index( lsorted[i] ) ]=math.ceil(lsorted[i])
            ...
        else:
            l[  l.index( lsorted[i] ) ]=math.floor(lsorted[i])
            ...
        pass
    return l






'''---本地统计类变量'''
best_acc=0
batchidx=0
dataset_total_sNum=50000
num_classes=0



'''---训练的超参数'''
datasetName='Caltech101'   #cifar100  cifar10 tinyImageNet   Caltech101
netname='resnet18'    # resnet18  EfficientNetB0  vit shufflenetv2 #vgg16 vgg19 attention56 vgg11  #attention 模型需要0.01lr
Wavg=1   # 记号，用于对比加权平均与非加权平均   #localsgd 好像一定要是0 也就是普通的平均
minitor_val_worker=1  #用于 监视与test的worker
size = 4   #wordsize
# bslist=[30,30,25,2] 
bslist=[169, 171, 151, 18]  #初始值. 不适用warmup与动态调整的时候使用 ,会作为全局batchsize的参考.#resnet18:[169, 171, 151, 18]
# bslist=[128,128,128,128]
useDynamicTune=   False #localsgd False
needWarmup=    False   #用于跳过warmup 跳过warmup就是初始化的bs生效
localSGD_Step= 0#使用localsgd的信号，上面的两项应该全为False  不使用local SGD时 此项应该为0  ONESHOT为98 97  24 48
skipSSGD=False #lcoal step>0生效，变成SkipSSGD算法
isbbsp= useDynamicTune or needWarmup
gbs=sum( bslist ) 
lr=0.1   #默认  vit时会变为0.0001
num_classes= 10 
epoch=120
needCoolTime=False  #False


##########改模型 numclass 学习率 记录名称
# '''---训练的对象'''
if datasetName == 'cifar100':
    num_classes=100
    dataset_total_sNum=50000

if datasetName == 'tinyImageNet':
    num_classes=200
    dataset_total_sNum=100000

if datasetName == 'cifar10':
    num_classes=10
    dataset_total_sNum=50000
if datasetName == 'Caltech101':
    num_classes=101
    dataset_total_sNum=8677
    


    

if netname=='attention56':
    net=Net.attention.attention56(num_classes=num_classes)
if netname=='vgg16':
    net=Net.VGG.vgg16_bn(num_classes=num_classes)
if netname=='vgg11':
    net=Net.VGG.vgg11_bn(num_classes=num_classes)
if netname=='vgg19':
    net=Net.VGG.vgg19_bn(num_classes=num_classes)

if netname=='resnet18':
    # net=Net.resnet.ResNet18(num_classes=num_classes)
    net=model = torchvision.models.__dict__['resnet18']()
    input_num = model.fc.in_features
    model.fc = nn.Linear(input_num, 101)    
if netname=='vit':
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

if netname=='shufflenetv2':
    import Net.shufflenetv2
    net=Net.shufflenetv2.ShuffleNetV2(1.0,num_classes=num_classes)


# net=torchvision.models.swin_t(num_classes=10)      #后面再初始化
criterion = nn.CrossEntropyLoss()
optimizer = None#后面再初始化
scheduler = None#后面再初始化

if datasetName=='cifar10':
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

if datasetName=='cifar100':
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751, 0.48654, 0.440917843), (0.267334281, 0.25643846, 0.276104)),
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.508896466, 0.48739301, 0.441942), (0.2682515, 0.2573637, 0.277092)),
    ])

if datasetName=='tinyImageNet':
    __mean = [0.4802, 0.4481, 0.3975]
    __std = [0.2302, 0.2265, 0.2262]

    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(__mean, __std)
    ])
    transform_test=transform_train


if datasetName=='Caltech101':
    transform_train = transforms.Compose([
        transforms.Grayscale(3),  # 将图像转换为3通道灰度图像
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test=transform_train










if datasetName=='cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)
if datasetName=='cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

if datasetName=='tinyImageNet':
    trainset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
    testset = ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_train)
    testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

if datasetName=='Caltech101':
    trainset = torchvision.datasets.Caltech101(root='./data', download=False, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    testset = torchvision.datasets.Caltech101(root='./data', download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=1)


    


def run(rank,size ,gdict,trainCPTimeList:managers.ListProxy,trainCMTimeList:managers.ListProxy,BSassignedment:managers.ListProxy): #ns以备不时之需
    global localSGD_Step
    if rank==minitor_val_worker:
        NOfastDynamicCount=0   #非快速调整计数器，每次进入微调自增，进入快调时判断是否大于20才能快调，快调时清零
        recordFileName=f'./record/LBB{datasetName}_{netname}_{len(BSassignedment)}GPU_lstep{localSGD_Step}_skipSSGD{skipSSGD}_bbsp{isbbsp}_DT{useDynamicTune}_wavg{Wavg}_bs{gbs}_lr{lr}_'+time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
        print(f'记录保存在{recordFileName}')

        # log_file = open(recordFileName+'.log', "w")
        # sys.stdout = log_file
        # console = open("/dev/tty", "w")
        # sys.stderr = console




        data_recorder = {"epoch": [],
                    "train_loss": [],
                    "cp_time_list": [],
                    "cm_time_list": [],
                    "val_loss": [],
                    "val_accuracy": [],
                    "batchsizes": [],
                    "allTrainTimeStamp": [],
                    "straggler_lv_list":[],
                    "tuneBS_list":[]
                    }





    global net
    net=net.cuda(rank)
    # net=nn.SyncBatchNorm.convert_sync_batchnorm(net,)  #=====================
    optimizer = optim.SGD(net.parameters(), lr=lr,)
    if netname=='vit':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,100],gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[60,120,160,200],0.2)

    
    '''
    #########################################################
    warmup 开始
    #########################################################
    '''
    if needWarmup:
        profXList=[2**i for i in range(1,9)]
        # profXList=[i for i in range(2,130)]
        profYList=[]
        pbarwarmup=tqdm(profXList)
        netWarmup=copy.deepcopy(net)
        netWarmup.cuda(rank)
        optWarmup= optim.SGD(netWarmup.parameters(), lr=lr,)
        if netname=='vit':
            optWarmup = optim.Adam(netWarmup.parameters(), lr=lr)
        # optWarmup =optim.Adam(netWarmup.parameters(), lr=lr)
        pointYret=[] #记录所有bs的情况
        for bs in pbarwarmup:
            pointY=[] #记录每个bs的情况
            for i in range (9):
                input=torch.rand ( bs,3,32,32).cuda(rank)
                tar=torch.randint(1,10,(bs,)).cuda(rank)
                torch.cuda.synchronize(rank)    
                start=time.time()
                output=netWarmup(input)
                lossWarmup = criterion(output, tar )
                lossWarmup.backward()
                optWarmup.step()
                torch.cuda.synchronize(rank)    
                warmupIterT=round ( (time.time()-start)*1000 ,12 )
                pointY.append(warmupIterT)

            
            pointY.remove(max(pointY))
            pointY.remove(max(pointY))
            pointY.remove(max(pointY))
            pointY.remove(min(pointY))
            pointY.remove(min(pointY))
            pointY.remove(min(pointY))
            pavgtime=round( np.mean(pointY),12)
            profYList.append(pavgtime)  #所有bs对应的时间 
            pointYret.append( (bs,pavgtime) ) #返回所有bs对应的时间tuple

        del netWarmup
        del optWarmup
        del lossWarmup
        pbarwarmup.close()
        del pbarwarmup
        torch.cuda.empty_cache()
        '''用于函数化封装返回'''   
        gdict['profileSample'][rank]=pointYret
        dist.barrier()
        if rank==0:
            # print(gdict)
            for i in range(size):
                print( gdict['profileSample'][i])

        '''开始拟合'''
        print(f'rank{rank} 开始拟合')
        def Fun(p,x):                        # 废弃定义拟合函数形式 
            a1,a2,a3,a4 = p
            return a1*x**3+a2*x**2+a3*x+a4
        def error (p,x,y):                    # 废弃拟合残差
            return func(p,x)-y 
        def func(x, a, b, c,d):
            return a+b*x+c*x*x+d*x*x*x
        def fitfuncmy(x,a,b):
            return a+b*x

        x=np.array(profXList)
        y=np.array(profYList)
        para,prop =optimize.curve_fit(func, x,y) # 进行拟合
        y_fitted = func (x,*para) # 画出拟合后的曲线
        dist.barrier()
        # if rank==3:
        #     print(f'原始时间：{y}')
        #     print(f'预测时间：{y_fitted}')
        #     print(f'预测时间0:{func (0,*para)}')
        #     print(f'我的简单模型预测时间128:{fitfuncmy (128,21,0.00078125*1000)}')
        #     print(f'fitPara:{para}')
        #     print(f'rank:{rank}: {y_fitted-y}')

        '''把para作为参数放到gdict['fit_para'] 由监视进程获取并得到结果放到bsassigndedment 数组'''
        gdict['fit_para'][rank]=para
        if rank==3:
            print(f'收集到拟合参数如下：')
            print(gdict['fit_para'])
        
        # 开始进行分配
        if rank==minitor_val_worker: #monitor
            def tarFunc(bs:np.array,fitpara): #目标函数
                timetuple=()
                for rank in range (len(fitpara)):
                    timetuple+= (func(bs[rank],*fitpara[rank]),)
                # timetuple+=(func(bs[3],*g750_popt),)
                return max(timetuple)-min(timetuple)
            def consfunc(bslist,gbs):    #约束条件
                return sum(bslist)-gbs
            cons = ({'type': 'eq', 'fun': functools.partial(consfunc, gbs=gbs)})   #约束条件  
            startpoint=tuple( [2]*size )  #初始值
            bounds= [[0,None]]*size
            fitpara=gdict['fit_para']   #获取拟合参数
            # print(f'\n\n\n最终收集到fitpara')
            res = optimize.minimize(tarFunc, startpoint, args=(fitpara,),method='SLSQP',constraints=cons,bounds=bounds) #优化  8ms
            print(res)
            predictTime=[  round (func(res.x[rank],*fitpara[rank]) ,4) for rank in range(size) ]
            print(f'预测的时间:{predictTime}')
            resBS=list_round_up( list (res.x) )
            predictTime=[  round (func(resBS[rank],*fitpara[rank]) ,4) for rank in range(size) ]
            print(f'取舍后的预测的时间:{predictTime}')
            
            for i in range(len(resBS)):
                BSassignedment[i]=resBS[i]

            '''记录拟合结果'''
            
            warmupRecord={'r0':[],'r1':[],'r2':[],'r3':[]}
            for rank__ in range(size):
                for x_ in profXList:
                    y_=round (func(x_,*fitpara[rank__]) ,4)
                    warmupRecord[f'r{rank__}'].append(y_)
            print(f'预测的执行时间情况')
            print(warmupRecord)
            ls__=[]  #存放保存在文件里的最终结果


            ls__.append([f'我们首先放进去ground truth'])
            for i in range(size):
                ls__.append( gdict['profileSample'][i])  #放入所有客观样本
            for rank__ in range(size):
                ls__.append(warmupRecord[f'r{rank__}']) #放入所有预测点
            ls__.append( [f'放入所有的预测点,这下面的是warmup阶段的精细预测'])
            for rank__ in range(size):       # 放入所有预测点
                rankpredict=[] 
                for i in range(2,200):
                    rankpredict.append( round( func(i,*fitpara[rank__]) , 4 )   )
                ls__.append(rankpredict)


            for rank__ in range(size):
                ls__.append(fitpara[rank__])

            df=pd.DataFrame(ls__)
            df.to_csv(f'./warmupRecord/batchVStime.csv',header=False)


                
            
            print(f'###################\n\nafterwarmup: {BSassignedment}')
        

    '''
    #########################################################
    warmup 结束
    #########################################################
    '''

    '''
    #########################################################
    Train start ********************************************
    #########################################################
    
    '''


    ''' gpu temp limit'''
    if rank==minitor_val_worker:
        hotest=0
        pynvml.nvmlInit()
        gpuHandlerList=[]
        gpuTempList=[0]*size
        for i in range(size):
            gpuHandlerList.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        cooltime=0.01
    ''' gpu temp limit end , rest is in ep begin'''

    allTrainTimeStamp=0  #记录训练时间  由所有的iter的cp cm相加组成，每个ep结束会进行一次记录
    
    for ep in range(epoch):
        ''' --training init--'''
        # print(f'===============ep"{ep} init==============')
        batchidx=0
        global_totlesample=0
        train_loss_local = 0
        correct_local = 0
        total_local = 0
        trainCPTimeList_ep=[]#统计一整个epoch中每个iter的所有worker的计算时间
        trainCMTimeList_ep=[]
        bs_list_ep=[]
        straggler_lv_list=[]
        tuneBS_list=[]    #记录每个迭代的bs变化
        dist.barrier()
        bs=BSassignedment[rank]
        # 模型重归一点
        for name, param in net.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data/=size
            # param.data *=avg_weight
        ''' --training init--end'''
        net.train()
        
        if rank==minitor_val_worker:
            pbar=tqdm(total=dataset_total_sNum,dynamic_ncols=True,)   
            pbar.dynamic_miniters



        '''gpu temp limit'''
        
       


        lsgd_IterTIme=0
        lstep=1
        '''-----------训练循环----一个epoch'''
        while True:
            # 一个iter
                bs=BSassignedment[rank]
                inputs,targets=mydataloader.getBatchSample(trainset=trainset,bs=bs)

                batchidx+=1 #动态微调时 无意义无意义
                
                
                inputs, targets = inputs.to(rank), targets.to(rank)

                optimizer.zero_grad()
                torch.cuda.synchronize(rank)

                dist.barrier()   #这个打开统计比较好
                cp_start=time.time()
                outputs = net(inputs)#
                loss:torch.Tensor = criterion(outputs, targets)#
                loss.backward()#
                torch.cuda.synchronize(rank)
                cp_time=time.time()-cp_start

                dist.barrier()   #这个一起打开用于统计数据传输时间
                if localSGD_Step==0:  #没有localsgd 直接同步参数并更新
                    torch.cuda.synchronize(rank)

                    if Wavg==1:
                        avg_weight=float( BSassignedment[rank]/gbs)
                    elif Wavg==0:
                        avg_weight=float(1/size)
                    cm_start=time.time()
                    for _name,param in net.named_parameters():
                        if param.grad!=None:
                            param.grad.data*=avg_weight
                            req=dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM,async_op=False)
                            
                        else:
                            pass
                        # param.grad.data  /= size
                    torch.cuda.synchronize(rank)
                    optimizer.step()
                    cm_time=time.time()-cm_start #cmtime 包含了等待用的时间
                    
                else: #有localsgd
                    # if rank==1: print(f'进入localsgd lstep{lstep} ')
                    if lstep<localSGD_Step:
                        # if rank==1: print(f'本地更新 lstep{lstep}')
                        lstep+=1
                        _upStart=time.time()
                        if skipSSGD == False:
                            optimizer.step()   #本地步骤，更新模型
                        _upTime=time.time()-_upStart
                        lsgd_IterTIme+=cp_time+_upTime
                        continue
                    elif lstep>=localSGD_Step: #全局同步
                        # if rank==1: print(f'全局同步 lstep{lstep} ')
                        optimizer.step()
                        lsgd_IterTIme_ok=lsgd_IterTIme+cp_time #lsgd_IterTIme_ok每个h步的local update做完了才存在这个变量
                        lsgd_IterTIme=0
                        lstep=1
                        torch.cuda.synchronize(rank)



                        # 数据传输
                        if Wavg==1:
                            avg_weight= float( BSassignedment[rank]/gbs)
                        elif Wavg==0:
                            avg_weight=float(1/size)

                        torch.cuda.synchronize(rank)
                        dist.barrier()
                        cm_start=time.time()
                        for _name,param in net.named_parameters():
                            if param.data!=None:
                                param.data*=avg_weight
                                req=dist.all_reduce(param.data, op=dist.ReduceOp.SUM,async_op=False)
                                
                                # req=dist.all_reduce(param.grad*avg_weight, op=dist.ReduceOp.SUM,async_op=False)
                            else:
                                pass
                        torch.cuda.synchronize(rank)
                        cm_time=time.time()-cm_start 


                    ...
                
                

                '''----local统计----'''
                loss_global=loss.clone().detach()
                dist.all_reduce(loss_global, op=dist.ReduceOp.SUM,async_op=False)
                loss_global_data=loss_global.item()/size
                train_loss_local += loss.item()
                train_loss_local_avg=train_loss_local/batchidx
                _, predicted = outputs.max(1)
                total_local += targets.size(0)
                correct_local += predicted.eq(targets).sum().item()
                train_acc_local=correct_local/total_local

                correct_globalIter=correct_local
                dist.all_reduce(torch.tensor(correct_globalIter),)
                total_globalIter=total_local
                dist.all_reduce(torch.tensor(total_globalIter),)
                ACC_global_Iter= round(correct_globalIter/total_globalIter,4)

                
                

                '''---内进度统计数据计算,后面才展示'''
                if localSGD_Step==0:
                    iterSampleNum=sum(BSassignedment)
                    global_totlesample+=iterSampleNum
                    trainCPTimeList[rank]=round (cp_time,3)  #当前iter,rank的执行时间
                    trainCMTimeList[rank]=round (cm_time,3) #包含了等待用的时间
                    dist.barrier()
                    if rank==minitor_val_worker:
                        trainCPTimeList_ep.append (trainCPTimeList._getvalue()) #提交给ep list , 当前iter,rank的执行时间  
                        trainCMTimeList_ep.append (trainCMTimeList._getvalue()) #本地变量包含等待时间
                        bs_list_ep.append(BSassignedment._getvalue()) #本地变量 batch size

                    CPTimeAvg= round (np.max(trainCPTimeList),3 )
                    CMTimeAvg=  round (np.min(trainCMTimeList),3 )  #这里是说一个iter的全局计算时间
                    allTrainTimeStamp+=CPTimeAvg+CMTimeAvg  #记录一个epoch的所有运行时间   ## iter统计
                else:
                    iterSampleNum=sum(BSassignedment)*localSGD_Step
                    global_totlesample+=iterSampleNum
                    trainCPTimeList[rank]=round (lsgd_IterTIme_ok,3)  #当前iter,rank的执行时间
                    trainCMTimeList[rank]=round (cm_time,3)
                    dist.barrier()
                    if rank==minitor_val_worker:
                        trainCPTimeList_ep.append (trainCPTimeList._getvalue()) #提交给ep list , 当前iter,rank的执行时间  
                        trainCMTimeList_ep.append (trainCMTimeList._getvalue())
                        bs_list_ep.append(BSassignedment._getvalue())

                    CPTimeAvg= round (np.max(trainCPTimeList),3 )
                    CMTimeAvg=  round (np.min(trainCMTimeList),3 )#这里是说一个iiter的全局计算时间
                    allTrainTimeStamp+=CPTimeAvg+CMTimeAvg  #记录一个epoch的所有运行时间   #TImestamp的时间是正确的时间，
                    



                
                # 这里代码搬家了,搬到最下面了
                # if rank==minitor_val_worker:
                #     pbar.update(iterSampleNum)
                #     curLr=optimizer.state_dict()['param_groups'][0]['lr']
                #     pbar.set_postfix_str(f'ep:{ep}lr:{curLr:.6f} loss:{train_loss_local_avg:.2f} gloss:{loss_global_data:.2f} gacc={ACC_global_Iter:.3f}  bs:{BSassignedment} cp:{trainCPTimeList} cm:{CMTimeAvg} ')  


                
                '''
                ####################################
                ####################################
                动态微调↓
                ####################################
                ####################################
                '''
                if rank==minitor_val_worker:
                    TuneLV=None
                    NOfastDynamicWindow=30
                    
                    curBSassignedment=BSassignedment._getvalue() #未调整的bs
                    if useDynamicTune:
                        ## ##func应该被更新为simplefunc
                        view_windows=[True]*5
                        max_t=max(trainCPTimeList)
                        min_t=min(trainCPTimeList)
                        maxt_idx=trainCPTimeList.index(max_t)
                        mint_idx=trainCPTimeList.index(min_t)
                        straggle_lv=round( (max_t-min_t)/ np.mean(trainCPTimeList),4)
                        if 0<straggle_lv<=0.06:
                            '''不调'''
                            TuneLV='0'
                            NOfastDynamicCount+=1
                            pass
                        if 0.05<straggle_lv<=0.15 or NOfastDynamicCount<NOfastDynamicWindow:
                            '''细调'''
                            TuneLV='1'
                            NOfastDynamicCount+=0
                            BSassignedment[maxt_idx]-=1
                            BSassignedment[mint_idx]+=1

                        
                        if 0.15<straggle_lv<=999999:   
                            if NOfastDynamicCount>NOfastDynamicWindow:
                                NOfastDynamicCount=0
                                '''粗调'''
                                TuneLV='2'
                                '''动态快调 重新评估'''
                                simpleFitPara:list=[[]]*size
                                def simpleFit(x,a,b):
                                    return x/a+b
                                for r in range (size):
                                    startup=func(0.01,*fitpara[r])/1000 #毫秒转为秒
                                    cp_timelist= trainCPTimeList._getvalue()[r] #秒
                                    iterThrp=BSassignedment[r]/(cp_timelist-startup) #秒
                                    print(f'{torch.cuda.get_device_name(r)}startup:{startup:.2f} iterThrp:{iterThrp:.2f}\
                                        tarBS{BSassignedment[r]} simpleFitFunction:{simpleFit(BSassignedment[r],iterThrp,startup):.4f}')
                                    simpleFitPara[r]=[iterThrp,startup]

                                '''======使用simpleFit替代tarFunc中部分func'''
                                def FastTune_TarFunc(bs:np.array,fitpara): #目标函数
                                    timetuple=()
                                    # print(f'##动态微调目标函数的参数列表长度:{len(fitpara)} 内容{fitpara}')
                                    for rank in range (len(fitpara)):
                                        timetuple+= (simpleFit(bs[rank],*fitpara[rank]),)
                                    return max(timetuple)-min(timetuple)
                                def consfunc(bslist,gbs):    #约束条件
                                    return sum(bslist)-gbs
                                cons = ({'type': 'eq', 'fun': functools.partial(consfunc, gbs=gbs)})   #约束条件  
                                startpoint=tuple( [2]*size )  #初始值
                                bounds= [[0,None]]*size

                                simpleFitPara  #获取拟合参数
                                # print(f'\n\n\n最终收集到fitpara')
                                res = optimize.minimize(FastTune_TarFunc, startpoint, args=(simpleFitPara,),method='SLSQP',constraints=cons,bounds=bounds) #优化  8ms
                                predictTime=[  round (simpleFit(res.x[rank],*simpleFitPara[rank]) ,4) for rank in range(size) ]
                                WarmupBasedPredictTime= [  round ( func(res.x[rank],*fitpara[rank])/1000 ,4) for rank in range(size) ] #warmup阶段模型的预期
                                print(f'动态快调预测的时间:{ list ( zip(   [round(i) for i in list(res.x)]    ,predictTime))} 原始模型预测的时间:{WarmupBasedPredictTime}')
                                resBS=list (res.x)
                                for i in range(len(resBS)):
                                    BSassignedment[i]=round(resBS[i])
                                pass


                                ####这里只是想记一下线性函数图像：
                                fastTune_predictTime_list=[[]]
                                for rank___ in range(size):
                                    fastTuneYlist=[]
                                    for i in range(150):
                                        fastTuneYlist.append( round(   simpleFit( i,*simpleFitPara[rank___])*1000 ,4))
                                    fastTune_predictTime_list.append(fastTuneYlist)
                                fastTune_predictTime_list_DF=pd.DataFrame(fastTune_predictTime_list)
                                fastTune_predictTime_list_DF.to_csv (f'./warmupRecord/fastTunePredict.csv',header=False)



                        '''if max trainCPTimeList - min trainCPTimeList / min trainCPTimeList >0.03'''
                        '''
                        ####################################
                        ####################################
                        动态微调结束
                        ####################################
                        ####################################
                        '''
                '''更新tqdm,iter内'''
                if rank==minitor_val_worker:
                    pbar.update(iterSampleNum)
                    curLr=optimizer.state_dict()['param_groups'][0]['lr']
                    nbs=[BSassignedment[i]- curBSassignedment[i]   for i in range(len(curBSassignedment))] #反应bs的变化
                    if useDynamicTune==False:
                        straggle_lv=0
                    straggler_lv_list.append(straggle_lv)
                    tuneBS_list.append(nbs)    #记录每个迭代的bs变化
                    pbar.set_postfix_str(f'ep:{ep}lr:{curLr:.6f}  loss:{train_loss_local_avg:.2f} gloss:{loss_global_data:.2f} gacc={ACC_global_Iter:.3f} cbs:{curBSassignedment}  nbs:{nbs} cp:{trainCPTimeList} cm:{trainCMTimeList} lv{straggle_lv:.3f} DT{TuneLV} DC{NOfastDynamicCount}') 
                #落实动态batch
                bs=BSassignedment[rank]
            
                '''一个epoch的结束 end while'''
                if  global_totlesample>dataset_total_sNum-iterSampleNum-2:
                    if rank==minitor_val_worker:
                        pbar.close()
                    break
        
            
        scheduler.step() #一个训练正式结束 凉快凉快
        if rank==minitor_val_worker:
            if needCoolTime:
                for i in range(size):
                    gpuTempList[i]=pynvml.nvmlDeviceGetTemperature(gpuHandlerList[i], 0)
                    hotest=max(gpuTempList)


                # time.sleep( max(  0.01, cooltime) )

                if hotest>68:
                    cooltime+=1
                elif hotest>73:
                    cooltime+=3
                elif hotest<=66:
                    cooltime-=1
                elif hotest<=64:
                    cooltime=0
                cooltime=max(0.01,cooltime)
                time.sleep( max(  0.01, cooltime) )
                print(f'gpu temp:{gpuTempList} cooltime:{cooltime}')
        dist.barrier()






        '''-----验证'''
        # 参数聚合
        for name, param in net.named_parameters():
            dist.all_reduce(param.data  , op=dist.ReduceOp.SUM)
            param.data/=size
            # param.data /= float(size)


        torch.cuda.synchronize( )
        dist.barrier()
        if rank==minitor_val_worker:


            test_acc,test_loss=test(epoch=ep,device= minitor_val_worker)   
            train_loss_local_avg
            train_acc_local
            '''---记录epoch  都是一个epoch内发生的所有事情'''
            ''' 
                    "train_loss": [],"cp_time_list": [],"cm_time_list": [], "val_loss": [], "val_accuracy": [], "batchsizes": [], "allTrainTimeStamp": [],
                    }'''

            data_recorder['batchsizes'].append(bs_list_ep)
            data_recorder['train_loss'].append(train_loss_local_avg)
            data_recorder['cp_time_list'].append(trainCPTimeList_ep)
            data_recorder['cm_time_list'].append(trainCMTimeList_ep)
            data_recorder['val_loss'].append(test_loss)
            data_recorder['val_accuracy'].append(test_acc)
            data_recorder['allTrainTimeStamp'].append(allTrainTimeStamp)
            data_recorder['straggler_lv_list'].append(straggler_lv_list)
            data_recorder['tuneBS_list'].append(tuneBS_list)

            np.save(recordFileName,data_recorder,allow_pickle=True)
            # dic=np.load('./record/cifar10_resnet18_1070_1070_m40.npy',allow_pickle=True).item()#不用了
            print(f'=========epoch {ep}记录完成=====')
            # print(dic)

        ...

        req__=dist.barrier()


    ...
def test(epoch,device):

    teststart=time.time()
    global best_acc,net
    net.eval()###
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_loss_avg=test_loss/ (batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc=acc
    torch.cuda.synchronize(device)

    print(f'===============')
    print(f"===验证完成 time:{time.time()-teststart} epoch:{epoch} curacc:{acc:.3f} bestacc :{best_acc:.3f} testLossAvg:{test_loss_avg:.3f}")
    return acc,test_loss_avg




def getNet(netName:str): 
    if netName=='resnet18':
        return Net.resnet.ResNet18(num_classes=num_classes)
def roundList(list,pos):
    return  [ round(elem,pos)  for elem in list ]
def list_elem_eq(list):
    ret=True
    a=list[0]
    for ele in list:
        if ele!=a:
            return False
    return True



def init_processes(rank, size ,ns,trainCPTimeList,trainCMTimeList,BSassignedment,fn, backend='gloo'):

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'  # ##
    os.environ['NCCL_ASYNC_ERROR_HANDLING ']='1'
    dist.init_process_group(backend, rank=rank, world_size=size,timeout=timedelta(seconds=2000))
    fn(rank, size,ns,trainCPTimeList,trainCMTimeList,BSassignedment)


manager =None

if __name__ == '__main__':    
    multiprocessing.set_start_method("spawn")
    manager = Manager()    
    gdict = manager.dict()       #用来承担各种杂活参数
    trainCPTimeList=manager.list([0]*(size) ) #计算时间
    trainCMTimeList=manager.list([0]*(size) ) #通信时间
    BSassignedment=manager.list(bslist)
    

    print(f'main strat')
    gdict['profileSample']= manager.list([0]*size)
    #gdict.update({'fit_para':manager.list([0]*4) }) #嵌套对象的写法
    gdict['fit_para']=manager.list([0]*size) #和上一句的功能一样
    
    for i in range(0,size):
        print(f'cuda{i} :{torch.cuda.get_device_name(i)}')
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size ,gdict,trainCPTimeList,trainCMTimeList,BSassignedment, run))
        p.start()
        processes.append(p)

    time.sleep(0.05)

    processes.append(p)

    for p in processes:
        p.join()




