# nas_stegan

## Datasets
CIFAR10

ImageNet can find [here](http://www.image-net.org/).

## Experiments
You need to add your file path in the code first. Then run the following commands.
```
### for searching architecture
python main_nas.py
```
You can use our search structure to train directly or add your own search structure to `genotypes.py` to train.
```
### for training 
python main.py --arch 'your architecture'
```
You can modify the `test_ckpt` to test specified model.
```
### for testing
python main.py --mode test --test_ckpt 'your file path'
```

## Pre-trained weights
Pre-trained weights can be accessed here(). 

## **Acknowledgments**
Codes are heavily borrowed from [UDH](https://github.com/ChaoningZhang/Universal-Deep-Hiding.git) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS.git).
