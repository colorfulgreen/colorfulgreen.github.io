---
layout: post
title:  "ML 笔记：降维与度量学习"
date:   2020-08-14 09:00:00 +0800
categories: machine-learning 
---

降维的方法可分为线性降维和非线形降维两类。

线性降维先将样本做中心化处理，即平移样本空间，然后以原点为中心旋转样本空间。旋转到的位置根据对低维子空间性质的不同要求而定。例如 PCV 要求低维子空间对样本具有最大可分性。

非线形降维有两种思路。第一种思路是核化，其使用一个很大的核（宽度与样本数量相同），将样本空间映射到线性可分的高维空间，然后再线性降维。第二种思路是将样本空间在局部近似看做线性，包括利用流形在局部上与欧⽒空间同胚性质的 Isomap，以及固定邻域内样本间线性表出关系的 LLE.

<object data="/assets/slides/20200814降维与度量学习.pdf#navpanes=1" type="application/pdf" style="min-height:100vh;width:100%;height=100%">
    <embed src="/assets/slides/20200814降维与度量学习.pdf" />
</object>
