---
layout: post
title: Normalisation is dead, long live normalisation!
tags: [normalisation, initialisation, propagation]
authors: Hoedt, Pieter-Jan, JKU; 

---

Since the advent of Batch Normalisation (BN) almost every state-of-the-art (SOTA) method uses some form of normalisation.
After all, normalisation generally speeds up learning and leads to models that generalise better than their unnormalised counterparts.
This turns out to be especially useful when using some form of skip connections, which are prominent in residual networks (ResNets), for example.
However, [Brock et al. (2021)](#brock21characterizing) suggest that SOTA performance can also be achieved using **ResNets without normalisation**!

The fact that Brock et al. went out of their way to get rid of something as simple as BN in ResNets for which BN happens to be especially helpful, does raise a few questions:

 1. Why get rid of BN in the first place?
 2. How (easy is it) to get rid of BN?
 3. Can this also work for other architectures?
 4. Does this allow to gain insights in why normalisation works so well?
 5. Wait a second... Are they getting rid of BN or normalisation as a whole?

The goal of this blog post is to provide some insights w.r.t. these questions using the results from [Brock et al. (2021)](#brock21characterizing).


## Normalisation

To set the scene for a world without normalisation, we start with an overview of normalisation layers in neural networks.
Batch Normalisation is probably the most well known method, but there are plenty of alternatives.
However, all of these methods build on the same principle ideas.

### Origins

The design of modern normalisation layers in neural networks was inspired by data normalisation ([Lecun et al., 1998](#lecun98efficient); [Schraudolph, 1998](#schraudolph98centering); [Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
In the setting of a simple linear regression, it can be shown (see e.g., [Lecun et al., 1998](#lecun98efficient)) that the the second order derivative, i.e., the Hessian, of the objective is exactly the covariance of the input data, $\mathcal{D}$:

$$\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, y) \in \mathcal{D}} \nabla_{\boldsymbol{w}}^2 \frac{1}{2}(\boldsymbol{w} \boldsymbol{x} - y)^2 = \frac{1}{|\mathcal{D}|}  \sum_{(\boldsymbol{x}, y) \in \mathcal{D}}\boldsymbol{x} \boldsymbol{x}^\mathsf{T}.$$

By enforcing that the Hessian (= covariance of the data) is (close to) the identity matrix, the optimisation problem becomes a lot easier.
However, whitening the data can be costly and might even hurt generalisation ([wadia et al., 2021](#wadia21whitening)).
Therefore, typical data normalisation consists of centring (to get zero mean) and scaling (to get unit variance) the data to at least improve the condition of the optimisation problem.

When considering multi-layer networks, things get more complicated.
However, in the end, it turns out that normalising the inputs to a layer should provide the same kind of benefits for the optimisation of the weights in that layer ([Lecun et al., 1998](#lecun98efficient)).
Using these insights [Schraudolph (1998)](#schraudolph98centering) showed empirically that centring the activations (and gradients) can effectively be used to speed up learning.
Also common initialisation methods generally build on these principles (e.g., [Lecun et al., 1998](#lecun98efficient); [Glorot & Bengio, 2010](#glorot10understanding); [He et al., 2015](#he15delving)).

### Batch Normalisation



### Alternative Flavours


## Skip Connections


## Normaliser-Free ResNets


## Insights


---

## References

<span id="brock21characterizing">Brock, A., De, S., & Smith, S. L. (2021). Characterizing signal propagation to close the performance gap in unnormalized ResNets. 
International Conference on Learning Representations 9.</span>
([link](https://openreview.net/forum?id=IX3Nnir2omJ),
 [pdf](https://openreview.net/pdf?id=IX3Nnir2omJ))

 <span id="wadia21whitening">Wadia, N., Duckworth, D., Schoenholz, S. S., Dyer, E., & Sohl-Dickstein, J. (2021). Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization.
 Proceedings of the 38th International Conference on Machine Learning, 139, 10617–10629.</span> 
 ([link](http://proceedings.mlr.press/v139/wadia21a.html),
  [pdf](http://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf))
