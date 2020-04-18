# 非bp网络训练探索

## baseline实验

实验配置在experiments/ddtp.yaml。

## 解绑目标反传算法实验

实验配置在experiments/directdtp.yaml，验证了网络可以通过直接回传目标进行学习。具体思路是：

![image/ddtp](https://github.com/IanyePKU/tp/tree/master/materials/ddtp.PNG)

在18年，hinton的论文指出dtp训练针对超参数很敏感，所以调整了几组学习率，训练了50个epoch看结果。具体的可以利用tensorboard看Log中的内容。

![image/ddtp_result](https://github.com/IanyePKU/tp/tree/master/materials/ddtp_result.png)

可以看到结果是能够训练的，但是效果比不上dtp的准确率，处在0.9的水平（baseline dtp处于0.96左右的水平 bp处于0.97的水平）。具体的实验配置可以看experiment的yaml。
