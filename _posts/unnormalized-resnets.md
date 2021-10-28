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
However, all of these methods build on the same principle idea.

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

Also initialisation strategies commonly build on these principles (e.g., [Lecun et al., 1998](#lecun98efficient); [Glorot & Bengio, 2010](#glorot10understanding); [He et al., 2015](#he15delving)).
Since the initial parameters are independent of the inputs, the weights can be set so that (pre-)activations are effectively normalised before the first update.
However, as soon as the network is being updated, the distributions change and the normalising properties of the initialisation get lost ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).

### Batch Normalisation

In contrast to classical initialisation methods, Batch Normalisation (BN) is able to maintain fixed mean and variance of the activations as the network is being updated ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
Concretely, this is achived by applying a typical data normalisation to every mini-batch of data, $\mathcal{B}$:

$$\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}_\mathcal{B}}{\boldsymbol{\sigma}_\mathcal{B}}.$$

Here $\boldsymbol{\mu}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\boldsymbol{x} \in \mathcal{B}} \boldsymbol{x}$ is the mean over the inputs in the mini-batch and $\boldsymbol{\sigma}_\mathcal{B}$ is the corresponding standard deviation.
Also note that the division is element-wise and generally is numerically stabilised by some $\varepsilon$ when implemented.
In case a zero mean and unit variance is not desired, it is also possible to apply an affine transformation $\boldsymbol{y} = \boldsymbol{\gamma} \odot \boldsymbol{x} + \boldsymbol{\beta}$ with learnable scale $(\boldsymbol{\gamma})$ and mean ($\boldsymbol{\beta}$) parameters ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).

The original reason for introducing BN is to alleviate the so-called _internal covariate shift_, i.e. the change of distributions as the network updates.
More recent research has pointed out, however, that internal covariate shift does not necessarily deteriorate learning dynamics ([Santurkar et al., 2018](#santurkar18how); [Bjorck et al., 2018](#bjorck18understanding)).
Therefore it is generally believed that the improvements lie elsewhere.


### Alternatives


## Skip Connections


## Normaliser-Free ResNets


## Insights


---

## References

<span id="arpit16normprop">Arpit, D., Zhou, Y., Kota, B., & Govindaraju, V. (2016). Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks. 
Proceedings of The 33rd International Conference on Machine Learning, 48, 1168–1176.</span> 
([link](https://proceedings.mlr.press/v48/arpitb16.html),
 [pdf](http://proceedings.mlr.press/v48/arpitb16.pdf))

<span id="ba16layernorm">Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization [Preprint]. </span> 
([link](http://arxiv.org/abs/1607.06450),
 [pdf](http://arxiv.org/pdf/1607.06450.pdf))

<span id="balduzzi17shattered">Balduzzi, D., Frean, M., Leary, L., Lewis, J. P., Ma, K. W.-D., & McWilliams, B. (2017). The Shattered Gradients Problem: If resnets are the answer, then what is the question? 
Proceedings of the 34th International Conference on Machine Learning, 70, 342–350.</span> 
([link](https://proceedings.mlr.press/v70/balduzzi17b.html),
 [pdf](http://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf))

<span id="bjorck18understanding">Bjorck, N., Gomes, C. P., Selman, B., & Weinberger, K. Q. (2018). Understanding Batch Normalization. 
Advances in Neural Information Processing Systems, 31, 7694–7705. </span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/36072923bfc3cf47745d704feb489480-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/36072923bfc3cf47745d704feb489480-Paper.pdf))

<span id="brock21characterizing">Brock, A., De, S., & Smith, S. L. (2021). Characterizing signal propagation to close the performance gap in unnormalized ResNets. 
International Conference on Learning Representations 9.</span>
([link](https://openreview.net/forum?id=IX3Nnir2omJ),
 [pdf](https://openreview.net/pdf?id=IX3Nnir2omJ))

<span id="he15delving">He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. 
Proceedings of the IEEE International Conference on Computer Vision, 1026–1034.</span> 
([link](https://doi.org/10.1109/ICCV.2015.123),
 [pdf](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))

<span id="he16resnet">He, K., Zhang, X., Ren, S., & Sun, J. (2016a). Deep Residual Learning for Image Recognition. 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.</span> 
([link](https://doi.org/10.1109/CVPR.2016.90),
 [pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf))
 
<span id="he16preresnet">He, K., Zhang, X., Ren, S., & Sun, J. (2016b). Identity Mappings in Deep Residual Networks. 
In B. Leibe, J. Matas, N. Sebe, & M. Welling (Eds.), Computer Vision – ECCV 2016 (pp. 630–645). Springer International Publishing. </span> 
([link](https://doi.org/10.1007/978-3-319-46493-0_38),
 [pdf](https://arxiv.org/pdf/1603.05027.pdf))

<span id="huang17centred">Huang, L., Liu, X., Liu, Y., Lang, B., & Tao, D. (2017). Centered Weight Normalization in Accelerating Training of Deep Neural Networks. 
Proceedings of the IEEE International Conference on Computer Vision, 2822–2830.</span> 
([link](https://doi.org/10.1109/ICCV.2017.305),
 [pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Centered_Weight_Normalization_ICCV_2017_paper.pdf))

<span id="ioffe15batchnorm">Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 
Proceedings of the 32nd International Conference on Machine Learning, 37, 448–456.</span> 
([link](http://proceedings.mlr.press/v37/ioffe15.html),
 [pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))

<span id="klambauer17selfnorm">Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. 
Advances in Neural Information Processing Systems, 30, 971–980.</span> 
([link](https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html),
 [pdf](https://papers.nips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf))

 <span id="lecun98efficient">LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. 
 In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 9–50). Springer. </span> 
 ([link](https://doi.org/10.1007/3-540-49430-8_2),
  [pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))


<span id="li19understanding">Li, X., Chen, S., Hu, X., & Yang, J. (2019). Understanding the Disharmony Between Dropout and Batch Normalization by Variance Shift. 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2682–2690. </span> 
([link](https://doi.org/10.1109/CVPR.2019.00279),
 [pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf))

<span id="mishkin16lsuv">Mishkin, D., & Matas, J. (2016). All you need is a good init. 
International Conference on Learning Representations 4.</span> 
([link](http://arxiv.org/abs/1511.06422),
 [pdf](http://arxiv.org/pdf/1511.06422.pdf))

<span id="salimans16weightnorm">Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. 
Advances in Neural Information Processing Systems, 29, 901–909.</span> 
([link](https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf))

<span id="santurkar18how">Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? 
Advances in Neural Information Processing Systems, 31, 2483–2493.</span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf))

<span id="schraudolph98centering">Schraudolph, N. N. (1998). Centering Neural Network Gradient Factors. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 207–226). Springer.</span> 
([link](https://doi.org/10.1007/3-540-49430-8_11),
 [pdf](https://n.schraudolph.org/pubs/Schraudolph98.pdf))

<span id="vandersmagt98solving">van der Smagt, P., & Hirzinger, G. (1998). Solving the Ill-Conditioning in Neural Network Learning. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 193–206). Springer.</span> 
([link](https://doi.org/10.1007/3-540-49430-8_10))

<span id="wadia21whitening">Wadia, N., Duckworth, D., Schoenholz, S. S., Dyer, E., & Sohl-Dickstein, J. (2021). Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization.
Proceedings of the 38th International Conference on Machine Learning, 139, 10617–10629.</span> 
([link](http://proceedings.mlr.press/v139/wadia21a.html),
 [pdf](http://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf))

<span id="zhang19fixup">Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. 
International Conference on Learning Representations 6. </span> 
([link](https://openreview.net/forum?id=H1gsz30cKX),
 [pdf](https://openreview.net/pdf?id=H1gsz30cKX))

