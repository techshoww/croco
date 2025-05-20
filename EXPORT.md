本文档描述croco模型的导出过程。

## 1. 从github clone源码  
```
git clone git@github.com:techshoww/croco.git
```

## 2. 新建export分支   
```
git checkout -b export  
```

## 3. 安装python环境  

```
conda create -n croco  python=3.10.15 -y
conda install pytorch torchvision -c pytorch
pip install onnx==1.17.0 onnxruntime==1.20.1 onnx-simplifier==0.4.36
```

## 4. 下载pth模型  
这里选用了最基础的模型
```
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth
```

## 5. 修改mask的实现方式，规避NonZero算子  
```
x = x[~masks].view(B, -1, C)
posvis = pos[~masks].view(B, -1, 2)
```
原代码中的这种写法会在导出onnx时产生NonZero算子。NonZero是onnx动态shape算子，工具链不支持，需要将其规避掉。  

原代码中生成mask的代码  
```
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask
```
noise已经是随机的，只需要将argsort变量取出前self.num_mask个元素即是mask索引，取出剩余的元素即是非mask索引，所以将代码改成这样：  
```
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        print("noise",noise.shape)
        argsort = torch.argsort(noise, dim=1)
        # return argsort < self.num_mask
        return argsort[:, self.num_mask:].reshape(-1), argsort < self.num_mask       # nonmask ids, mask map
```
相应地，将 corco.py 中类似
```
x = x[~masks].view(B, -1, C)
```
的代码都改成：
```
x = x[:,nonmasks,:].view(B,-1,C)
```
变量名mask，给为nonmask即可。

## 6. 替换 einsum 算子  
这个函数中出现了 einsum 算子,需要用 permute 替换掉    
```
x = torch.einsum('nchpwq->nhwpqc', x)
```
改为
```
x = x.permute(0,2,4,3,5,1)
```

## 7. 导出 onnx 

(1). 导出onnx 
```
python export_onnx.py 
```

(2). 测试onnx
```
python demo_onnx.py
```

(3). 对比pth结果  
```
python demo.py
```

## 8. 编译onnx  
进入工具链环境，执行  
```
bash build.sh
```

## 9. 开发板运行axmodel  
```
ax_run_model -m croco.axmodel -w 10 -r 100
```
```
   Run AxModel:
         model: croco.axmodel
          type: 3 Core
          vnpu: Disable
      affinity: 0b001
        warmup: 10
        repeat: 100
         batch: { auto: 0 }
      parallel: false
   pulsar2 ver: 4.0 ce2fb6a4
    engine ver: 2.12.0s
      tool ver: 2.5.1a
      cmm size: 218760166 Bytes
  ---------------------------------------------------------------------------
  min =  13.016 ms   max =  13.350 ms   avg =  13.104 ms  median =  13.091 ms
   5% =  13.043 ms   90% =  13.170 ms   95% =  13.201 ms     99% =  13.350 ms
  ---------------------------------------------------------------------------
```