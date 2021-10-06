# NLP tf.estimator 项目模版
tensorflow的Estimator高级API，将模型的train, eval, predict, save规范化，
免去了tensorflow的Session.run的操作，并且很好地结合了tf.data.Dataset作为数据处理的包装，
使得整个模型从数据到模型产出的整体思路抽象封装成接口，方便用户自定义设计model

主要依赖
- tensorflow-gpu 115
- numpy 1.16.0 解决warning
- cudatookit 10.0, cudnn 7.4

## 主体架构
![img.png](img.png)


## 参考链接
- https://zhuanlan.zhihu.com/p/112062303