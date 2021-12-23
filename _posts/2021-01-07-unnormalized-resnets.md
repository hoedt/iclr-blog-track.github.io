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

 1. Why get rid of BN in the first place[?](#alternatives)
 2. How (easy is it) to get rid of BN in ResNets[?](#moment-control)
 3. Can this also work for other architectures?
 4. Does this allow to gain insights in why normalisation works so well?
 5. Wait a second... Are they getting rid of BN or normalisation as a whole?

The goal of this blog post is to provide some insights w.r.t. these questions using the results from [Brock et al. (2021)](#brock21characterizing).


## Normalisation

To set the scene for a world without normalisation, we start with an overview of normalisation layers in neural networks.
Batch Normalisation is probably the most well known method, but there are plenty of alternatives.
Despite the variety of normalisation methods, they all build on the same principle ideas.

### Origins

The design of modern normalisation layers in neural networks is mainly inspired by data normalisation ([Lecun et al., 1998](#lecun98efficient); [Schraudolph, 1998](#schraudolph98centering); [Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
In the setting of a simple linear regression, it can be shown (see e.g., [Lecun et al., 1998](#lecun98efficient)) that the the second order derivative, i.e., the Hessian, of the objective is exactly the covariance of the input data, $\mathcal{D}$:

$$\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, y) \in \mathcal{D}} \nabla_{\boldsymbol{w}}^2 \frac{1}{2}(\boldsymbol{w} \boldsymbol{x} - y)^2 = \frac{1}{|\mathcal{D}|}  \sum_{(\boldsymbol{x}, y) \in \mathcal{D}}\boldsymbol{x} \boldsymbol{x}^\mathsf{T}.$$

By enforcing that the Hessian (= covariance of the data) is (close to) the identity matrix, the optimisation problem becomes a lot easier ([Lecun et al., 1998](#lecun98efficient)).
However, whitening the data can be costly and might even hurt generalisation ([Wadia et al., 2021](#wadia21whitening)).
Therefore, typical data normalisation consists of centring (to get zero mean) and scaling (to get unit variance) the data to at least improve the condition of the optimisation problem.

When considering multi-layer networks, things get more complicated.
However, in the end, it turns out that normalising the inputs to a layer should provide the same kind of benefits for the optimisation of the weights in that layer ([Lecun et al., 1998](#lecun98efficient)).
Using these insights [Schraudolph (1998)](#schraudolph98centering) showed empirically that centring the activations can effectively be used to speed up learning.

Also initialisation strategies commonly build on these principles (e.g., [Lecun et al., 1998](#lecun98efficient); [Glorot & Bengio, 2010](#glorot10understanding); [He et al., 2015](#he15delving)).
Since the initial parameters are independent of the inputs, the weights can be set so that (pre-)activations are effectively normalised before the first update.
However, as soon as the network is being updated, the distributions change and the normalising properties of the initialisation get lost ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).

### Batch Normalisation

In contrast to classical initialisation methods, Batch Normalisation (BN) is able to maintain fixed mean and variance of the activations as the network is being updated ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).
Concretely, this is achieved by applying a typical data normalisation to every mini-batch of data, $\mathcal{B}$:

$$\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}_\mathcal{B}}{\boldsymbol{\sigma}_\mathcal{B}}.$$

Here $\boldsymbol{\mu}\_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum\_{\boldsymbol{x} \in \mathcal{B}} \boldsymbol{x}$ is the mean over the inputs in the mini-batch and $\boldsymbol{\sigma}_\mathcal{B}$ is the corresponding standard deviation.
Also note that the division is element-wise and generally is numerically stabilised by some $\varepsilon$ when implemented.
In case a zero mean and unit variance is not desired, it is also possible to apply an affine transformation $\boldsymbol{y} = \boldsymbol{\gamma} \odot \boldsymbol{x} + \boldsymbol{\beta}$ with learnable scale $(\boldsymbol{\gamma})$ and mean ($\boldsymbol{\beta}$) parameters ([Ioffe & Szegedy, 2015](#ioffe15batchnorm)).

The above description explains the core operation of BN during training.
However, during inference, it is not uncommon to desire predictions for single samples.
Obviously this would cause trouble because a mini-batch with a single sample has zero variance.
Therefore, it is common to accumulate the statistics ($\boldsymbol{\mu}\_\mathcal{B}$ and $\boldsymbol{\sigma}\_\mathcal{B}^2$) that are used for normalisation, during training.
These accumulated statistics can then be used as estimators for the mean and variance during inference.
This makes it possible for BN to be used on single samples during inference.

The original reason for introducing BN was to alleviate the so-called _internal covariate shift_, i.e. the change of distributions as the network updates.
More recent research has pointed out, however, that internal covariate shift does not necessarily deteriorate learning dynamics ([Santurkar et al., 2018](#santurkar18how)).
Also [Ioffe & Szegedy (2015)](#ioffe15batchnorm) seem to have realised that simply normalising the signal does not suffice: 

 > [...] the model blows up when the normalization parameters are computed outside the gradient descent step.

 All of this seems to indicate that part of the success of BN is due to the effects it has on the gradient signal.
 The affine transformation simply scales the gradient, such that $\nabla_{\hat{\boldsymbol{x}}} L = \boldsymbol{\gamma} \odot \nabla_{\boldsymbol{y}} L.$
 The normalisation operation, on the other hand, transforms the gradient, $\boldsymbol{g} = \nabla_{\hat{\boldsymbol{x}}} L$, as follows:

 $$\nabla_{\boldsymbol{x}} L = \frac{1}{\boldsymbol{\sigma}_\mathcal{B}} \big(\boldsymbol{g} - \mu_g \,\boldsymbol{1} - \operatorname{cov}(\boldsymbol{g}, \hat{\boldsymbol{x}}) \odot \hat{\boldsymbol{x}} \big),$$

where $\mu_g = \sum_{\boldsymbol{x} \in \mathcal{B}} \nabla_{\hat{\boldsymbol{x}}} L$ and $\operatorname{cov}(\boldsymbol{g}, \hat{\boldsymbol{x}}) = \frac{1}{|\mathcal{B} |} \sum_{\boldsymbol{x} \in \mathcal{B}} \boldsymbol{g} \odot \hat{\boldsymbol{x}}.$
Note that this directly corresponds to centering the gradients, which should also improve learning speed ([Schraudolph, 1998](#schraudolph98centering)).

In the end, everyone seems to agree that one of the main beneftis of BN is that it enables higher learning rates ([Ioffe & Szegedy, 2015](#ioffe15batchnorm); [Bjorck et al., 2018](#bjorck18understanding); [Santurkar et al., 2018](#santurkar18how); [Luo et al., 2019](#luo19towards)), which results in faster learning and better generalisation.
An additional benefit is that BN is scale-invariant and therefore much less sensitive to weight initialisation ([Ioffe & Szegedy, 2015](#ioffe15batchnorm); [Ioffe, 2017](#ioffe17batchrenorm)).

### Alternatives

<figure id="fig_dims">
    <img src="/public/images/data_dimensions.svg" alt="visualisation of different input data types">
    <figcaption>
        Figure&nbsp;1: Different input types in terms of their typical 
        batch size ($|\mathcal{B}|$), the number of channels/features ($C$) and the <em>size</em> of the signal ($S$) (e.g. width times height for images).
        Image inspired by (<a href="#wu18groupnorm">Wu & He, 2018</a>).
    </figcaption>
</figure>

Although BN provides important benefits, it also comes with a few downsides:

 - BN does not work well with **small batch sizes** ([Ba et al., 2016](#ba16layernorm); [Salimans & Kingma, 2016](#salimans16weightnorm); [Ioffe, 2017](#ioffe17batchrenorm)).
   For a batch-size of one, we have zero standard deviation, but also with a few samples, the estimated statistics are often not accurate enough.
 - BN is not directly applicable to certain input types ([Ba et al. 2016](#ba16layernorm)) and performs poorly when there are **dependencies between samples** in a mini-batch ([Ioffe, 2017](#ioffe17batchrenorm)).
 - BN uses **different statistics for inference** than those used during training ([Ba et al., 2016](#ba16layernorm); [Ioffe, 2017](#ioffe17batchrenorm)).
   This is especially problematic if the distribution during inference is different or drifts away from the training distribution.
 - BN does not play well with **other regularisation** methods ([Hoffer et al., 2018](#hoffer18norm)).
   This is especially known for $L_2$ regularisation ([Hoffer et al., 2018](#hoffer18norm)) and dropout ([Li et al., 2019](#li19understanding)).
 - BN introduces a significant **computational overhead** during training ([Ba et al., 2016](#ba16layernorm); [Salimans & Kingma, 2016](#salimans16weightnorm); [Gitman and Ginsburg, 2017](#gitman17comparison)).
   Because of the running averages, also memory requirements increase when introducing BN.

Therefore, alternative normalisation methods have been proposed to solve one or more of the problems listed above while trying to maintain the benefits of BN.

One family of alternatives simply computes the statistics along different dimensions (see figure&nbsp;[2](#fig_norm)).
**Layer Normalisation (LN)** is probably the most prominent example in this category ([Ba et al., 2016](#ba16layernorm)).
Instead of computing the statistics over samples in a mini-batch, LN uses the statistics of the feature vector itself.
This makes LN invariant to weight shifts and scaling individual samples.
BN, on the other hand, is invariant to data shifts and scaling individual neurons.
LN generally outperforms BN in fully connected and recurrent networks, but does not work well for convolutional architectures according to [Ba et al. (2016)](#ba16layernorm).
**Group Normalisation (GN)** is a slightly modified version of LN that also works well for convolutional networks ([Wu et al., 2018](#wu18groupnorm)).
The idea of GN is to compute statistics over groups of features in the feature vector instead of over all features.
For convolutional networks that should be invariant to changes in contrast, statistics can also be computed over single image channels for each sample.
This gives rise to a technique known as **Instance Normalisation (IN)**, which proved especially helpful in the context of style transfer ([Ulyanov et al., 2017](#ulyanov17improved)).

<figure id="fig_norm">
    <img src="/public/images/normalisation_dimensions.svg" alt="visualisation of normalisation methods">
    <figcaption>
        Figure&nbsp;2: Normalisation methods (Batch, Layer, Instance and Group Normalisation) and the parts of the input they compute their statistics over.
        $|\mathcal{B}|$ is the batch size, $C$ represents the number of channels/features and $S$ is the size of the signal (e.g. width times height for images).
        The lightly shaded region for LN indicates how it is typically used for image data.
        Image has been adapted from (<a href="#wu18groupnorm">Wu & He, 2018</a>).
    </figcaption>
</figure>

Instead of normalising the inputs, it is also possible to get a normalising effect by rescaling the weights of the network ([Arpit et al., 2016](#arpit16normprop)).
Especially in convolutional networks, this can significantly reduce the computational overhead.
With **Weight Normalisation (WN)** ([Salimans & Kingma, 2016](#salimans16weightnorm)), the weight vectors for each neuron are normalised to have unit norm.
This idea can also be found in a(n independently developed) technique called **Normalisation Propagation (NP)** ([Arpit et al., 2016](#arpit16normprop)).
However, in contrast to WN, NP accounts for the effect of (ReLU) activation functions.
In some sense, NP can be interpreted as a variant of BN where the statistics are computed theoretically (in expectation) rather than on-the-fly.
**Spectral Normalisation (SN)**, on the other hand, makes use of an induced matrix norm to normalise the entire weight matrix ([Miyato et al., 2018](#miyato18spectralnorm)).
Concretely, the weights are scaled by the reciprocal of an approximation of the largest singular value of the weight matrix.

Whereas WN, NP and SN still involve the computation of some weight norm, it is also possible to obtain normalisation without computational overhead.
By creating a forward pass that induces attracting fixed points in mean and variance, **Self-Normalising Networks (SNNs)** ([Klambauer et al., 2017](#klambauer17selfnorm)) are able to effectively normalise the signal.
To achieve these fixed points, it suffices to carefully scale the ELU activation function ([Clevert et al., 2016](#clevert16elu)) and the initial variance of the weights.
Additionally, [Klambauer et al. (2017)](#klambauer17selfnorm) provide a way to tweak dropout so that it does not interfere with the normalisation.
Maybe it is useful to point out that SNNs do not consist of explicit normalisation operations.
In this sense, an SNN could already be seen as some form of _normaliser-free_ network.


## Skip Connections

[Brock et al. (2021)](#brock21characterizing) mainly aim to rid residual networks (ResNets) of normalisation.
Therefore, it probably makes sense to revisit the key component of these ResNets: _skip connections_.
Apart from a bit of historical context, we aim to provide some intuition as to why normalisation methods can be so helpful in the context of skip connections, and what alternatives are available.

### History

_Shortcut_ or _skip connections_ are a way to allow information to bypass one or more layers in a neural network.
Mathematically, skip connections are typically written down something like

$$\boldsymbol{y} = \boldsymbol{x} + f(\boldsymbol{x}),$$

where $f$ represents some non-linear transformation ([He et al., 2016a](#he16resnet), [2016b](#he16preresnet)).
This non-linear transformation is typically a sub-network that is commonly referred to as the _residual branch_ or _residual connection_.
When the outputs of the residual branch have different dimensions, it is typical to use a linear transformation to match the output dimension of the skip connection with that of the residual connection.

Skip connection became very popular in computer vision due to the work of He et al. ([2016a](#he16resnet)).
However, they were already commonly used as a trick to improve learning in multi-layer networks before deep learning was a thing ([Ripley, 1996](#ripley96pattern)).
Similar to normalisation methods, skip connections can improve the condition of the optimisation problem by making it harder for the Hessian to become singular ([van der Smagt & Hirzinger, 1998](#vandersmagt98solving)).
Also in the forward pass, skip connections have benefits:
e.g., [Srivastava et al. (2015)](#srivastava15highway) argue that information can flow through the network without being altered.
[He et al., (2016a)](#he16resnet), on the other hand, claim that learning should be easier if the linear term of the transformation can be ignored.

<figure id="fig_skip">
    <img src="/public/images/skip_connections.svg" alt="visualisation of different types of skip connections">
    <figcaption>
        Figure&nbsp;3: Variations on skip connections in ResNets, Densenets and Highway networks.
        The white blocks correspond to the input / skip connection and the blue blocks correspond to the output of the non-linear transformation.
        The greyscale blocks are values between zero and one and correspond to masks.
    </figcaption>
</figure>

The general formulation of skip connection that we provided earlier, captures the idea of skip connections very well.
As you might have expected, however, there are plenty of variations on the exact formulation (a few of which are illustrated in figure&nbsp;[3](#fig_skip)).
Strictly speaking, even [He et al., (2016a)](#he16resnet) do not strictly adhere to their own formulation because they use an activation function on what we denoted as $\boldsymbol{y}$ ([He et al., 2016](#he16preresnet)).
E.g., in DenseNet ([G. Huang et al., 2017](#huang17densenet)), the outputs of the skip and residual connections is concatenated instead of aggregated by means of a sum.
This retains more of the information for subsequent layers.
Other variants of skip-connections make use of masks to select which information is passed on.
Highway networks ([Srivasta et al., 2015](#srivasta15highway)) make use of a gating mechanism similar to that in Long Short-Term Memory (LSTM) ([Hochreiter et al., 1997](#hochreiter97lstm)).
These gates enable the network to learn how information from the skip connection is to be combined with that of the residual branch.
Similarly, transformers ([Vaswani et al., 2017](#vaswani17attention)) could be interpreted as highway networks without a residual connection.
Additionally, the gate of the skip connection is replaced by a more complex attention mask.


### Moment Control

Traditional initialisation techniques manage to provide a stable starting point for the propagation of mean and variance in fully connected layers, but they do not work so well in ResNets.
The key problem is that the variance must increase when the two branches are added together.
After all, the variance is linear and unless the non-linear transformation branch would output a zero-variance signal, the output variance will be greater than the input variance.
Moreover, if the signal would have a strictly positive mean, also the mean would start drifting as the network becomes deeper.
By inserting normalisation layers after every residual block, these drifiting effects can effectively be countered.

Instead of relying on normalisation methods to resolve the drifting effects, it is also possible to implement other measures against these drift effects.
Similar to standard initialisation methods, the key idea is to stabilise the variance propagation.
To this end, a slightly modified formulation of residual networks is typically used (e.g., [Szegedy et al., 2016](#szegedy16inceptionv4); [Balduzzi et al., 2017](#balduzzi17shattered); [Hanin & Rolnick, 2018](#hanin18how)):

$$\boldsymbol{y} = \alpha x + \beta f(\alpha x),$$

which is equivalent to the traditional formulation when $\alpha = \beta = 1.$
The key advantage of this formulation is that the variance can be controlled (to some extent) by the newly introduced scaling factors $\alpha$ and $\beta.$

A very simple counter-measure to the variance explosion in ResNets is to set $\alpha = 1 / \sqrt{2}$ ([Balduzzi et al., 2017](#balduzzi17shattered)).
Assuming that the residual branch approximately preserves the variance, the variances of $\boldsymbol{y}$ and $\boldsymbol{x}$ should be roughly the same.
In practice, however, it seems to be more common to tune the $\beta$ factor instead of $\alpha$ ([Balduzzi et al., 2017](#balduzzi17shattered)).
For instance, simply setting $\beta$ to some small value (e.g., in the range $[0.1, 0.3]$) can already help ResNets (with BN) to stabilise training ([Szegedy et al., 2016](#szegedy16inceptionv4)).
It turns out that having small values for $\beta$ can help to preserve correlations between gradients, which should benefit learning ([Balduzzi et al., 2017](#balduzzi17shattered)).
Similar findings were established through the analysis of the variance propagation in ResNets by [Hanin & Rolnick (2018)](#hanin18how).
Eventually they propose to set $\beta = b^l$ in the $l$-th layer, with $0 < b < 1$ to make sure that the sum of scaling factors converges.
[Arpit et al. (2019)](#arpit19how) also take the backward pass into account and show that $\beta = L^{-1}$ provides stable variance propagation in a ResNet with $L$ skip connections.
Also learning the scaling factor $\beta$ in each layer can make it possible to keep the variance under control ([Zhang et al., 2019](#zhang19fixup); [De & Smith, 2020](#de20skipinit)).

Obviously, there are also workarounds that do not quite fit the general formulation with scaling factors $\alpha$ and $\beta.$
One possible workaround is to make use of an empirical approach to weight initialisation ([Mishkin et al., 2016](#mishkin16lsuv)).
By rescaling random orthogonal weight matrices by the empirical variance of the output activations at each layer, [Mishkin et al. (2016)](#mishkin16lsuv) show that it is possible to train ResNets without BN.
In some sense, this approach can be interpreted as choosing a scaling factor for each layer in the residual branch (and in some of the skip connections).
Instead of using the reciprocal of the empirical variance as scaling factor, [Zhang et al. (2019)](#zhang19fixup) scale the initial weights of the $k$-th layer in each of the $L$ residual branches by a factor $L^{-1/(2k-2)}.$


## Normaliser-Free ResNets

In some sense, it could be argued that the current popularity of skip connections is due to BN, rather than ResNets.
After all, without BN the skip connections in ResNets would have suffered from the drifting effects discussed earlier and ResNets would probably not have become so popular.
However, BN does have a few practical issues (see [earlier](#alternatives)) and it does seem to be the case that the drifting effects can be controlled using other techniques.
Therefore, it seems natural to find out whether it is possible to replace BN by one of these other techniques to get the best of both worlds.

### Prior Work

The idea of training ResNets without BN is practically as old as ResNets themselves.
With their Layer-Sequential Unit-Variance (LSUV) initialisation, [Mishkin et al. (2016)](#mishkin16lsuv) showed that it is possible to replace BN with good initialisation for small datasets (CIFAR-10).
Similarly, [Arpit et al. (2019)](#arpit19) are able to close the gap between Weight Normalisation (WN) and BN by reconsidering the initialisation of the weights.

Getting rid of BN in ResNets has been posed as an explicit goal by [Zhang et al. (2019)](#zhang19fixup), who proposed the so-called FixUp initialisation scheme.
The main idea of FixUp is to scale the weights in the $k$-th layer in each residual branch by $L^{-1/(2k-2)}$.
Moreover, they set the initial weights for the last layer in each residual branch to zero and introduce learnable $\beta$ scaling factors as well as scalar biases before every layer in the network.
With these tricks, Zhang et al. are able to show that FixUp can provide _almost_ the same benefits as BN for ResNets in terms of trainability and generalisation.
Using a different derivation, [De & Smith (2020)](#de20skipinit) end up with a very similar solution to train ResNets without BN, which they term SkipInit.
The key difference with FixUp is that the initial value for the learnable $\beta$ parameter must be less than $1 / \sqrt{L}.$
In return, SkipInit does not require the rescaling of initial weights in the residual branch or setting weights to zero, which are considered crucial parts in the FixUp strategy ([Zhang et al. (2019)](#zhang19fixup)).

### Current Work

Although the results of prior works look promising, there is still a performance gap compared to ResNets with BN.
To close this gap, [Brock et al. (2021)](#brock21characterizing) suggest to study the propagation of mean and variance through ResNets by means of so-called Signal Propagation Plots (SPPs).
These SPPs simply visualise the squared mean and variance of the activations after each skip connection, as well as the variance at the end of every residual branch (before the skip connection).
Figure&nbsp;[4](#fig_spp) provides an example of the SPPs for a pre-activation ResNets (or v2 ResNets, cf. [He et al., 2016b](#he16identity)) with and without BN.
First of all, the SPPs on the left side clearly illustrate that BN transforms the exponential growth to a linear propagation in ResNets, as described in theory (e.g., [Balduzzi et al., 2017](#balduzzi17shattered); [De & Smith, 2020](#de20skipinit)).
When focusing on ResNets with BN (on the right side), it is clear that mean and variance are reduced after every layer, each of which consists of a few skip connections.
This reduction is due to the _pre-activation_ block (BN + ReLU) that is inserted after every layer in these ResNets.

<figure id="fig_spp">
    <img src="/public/images/spp.svg" alt="Image with two plots. The left plot shows two signal propagation plots: one for ResNets with (increasing gray lines) and one for ResNets without (approximately flat blue lines) Batch Normalisation on a logarithmic scale. The right plot shows the zig-zag lines that represent the squared mean and variance after each residual branch." width="100%">
    <figcaption>
        Figure&nbsp;4: Example Signal Propagation Plots (SPPs) for a pre-activation (v2) ResNet-50 at initialisation.
        SPPs plot the squared mean ($\mu^2$) and variance ($\sigma^2$) of the pre-activations after each skip connection ($x$-axis), as well as the variance of the residuals before the skip connection ($\sigma_f^2$, $y$-axis on the right).
        The left plot illustrates the difference between ResNets with and without BN layers.
        The plot on the right shows the same SPP for a ResNet with BN without the logarithmic scaling.
        Note that ResNet-50 has four layers with 3, 4, 6 and 3 residual branches, respectively.
    </figcaption>
</figure>

The goal of Normaliser-Free ResNets (NF-ResNets) is to get rid of the BN layers in ResNets, while preserving the characteristics visualised in the SPPs ([Brock et al., 2021](#brock21characterizing)).
To get rid of the exponential increase in variance in unnormalised ResNets, it suffices to set $\alpha = 1 / \sqrt{\operatorname{Var}[\boldsymbol{x}]}$ in our modified formulation of ResNets.
This effectively implements the scaling that is normally a part of BN.
Unlike BN, however, the scaling in NF-ResNets is computed analytically for every skip connection.
This is possible if the inputs to the network are properly normalised (i.e., have unit variance) and the residual branch, $f$, is properly initialised (i.e., preserves variance).
Although this scheme should allow to reduce the variance after every skip-connection, it is only used after every layer, which typically consists of multiple skip-connections.
For all the other skip connections, the $\alpha$ rescaling is only applied on the residual branch and not on the skip connection, such that $\boldsymbol{y} = x + \beta f(\alpha x).$
This is necessary to imitate the variance drops in the reference SPP, which are due to the pre-activation blocks between layers in the ResNets (see figure&nbsp;[4](#fig_spp)).
The $\beta$ parameter, on the other hand, is simply used as a hyper-parameter to directly control the variance increase after every skip connection.

<figure id="fig_nfresnet">
    <img src="/public/images/spp_nfresnet.svg" alt="Image with two plots. The left plot shows two SPPs: one for a ResNet with Batch Normalisation (gray lines) and one for a Normaliser-Free ResNet (blue lines). The curves representting variance for both models are very close to each other, but the curve for the mean is quite different. The right plot is similar, but now the blue mean and residual variance curves are zero and one everywhere, respectively." width="100%">
    <figcaption>
        Figure&nbsp;5: SPPs comparing a NF-ResNet-50 to a Resnet with BN at initialisation.
        The NF-ResNet in the left plot only uses the $\alpha$ and $\beta$ scaling parameters.
        The right plot displays the behaviour of a NF-ResNet with Centred Weight Normalisation.
        Note that for the right plot, the scale for the SPPs is practically defined by the residual variance.
    </figcaption>
</figure>

As can be seen on the left plot in figure&nbsp;[5](#fig_nfresnet), a plain NF-ResNet effectively mimics the variance propagation of the baseline ResNet pretty accurately.
The propagation of the squared mean in NF-ResNets, on the other hand, looks nothing like that from the BN model.
After all, the considerations that lead to the scaling parameters only considers the variance paropagation.
On top of that, it turns out that the variance of the residual branches (right before it is merged with the skip connection) is not particularly steady.
This indicates that the residual branches might not properly preserve variance, which is necessary for the analytic computations of $\alpha$ to be correct.
It turns out that both of these discrepancies can be resolved by introducing a variant of Centred Weight Normalisation (CWN; [L. Huang et al., 2017](#huang17centred)) to NF-ResNets.
CWN simply applies WN after subtracting the weight mean from each weight vector, which ensures that every output has zero mean and variance can propagate steadily.
The effect of including CWN in NF-ResNets is illustrated in the right part of figure&nbsp;[5](#fig_nfresnet).

### Future Work


## Insights


---

## References

<span id="arpit16normprop">Arpit, D., Zhou, Y., Kota, B., & Govindaraju, V. (2016). Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks. 
Proceedings of The 33rd International Conference on Machine Learning, 48, 1168–1176.</span> 
([link](https://proceedings.mlr.press/v48/arpitb16.html),
 [pdf](http://proceedings.mlr.press/v48/arpitb16.pdf))

<span id="arpit19how">Arpit, D., Campos, V., & Bengio, Y. (2019). How to Initialize your Network? Robust Initialization for WeightNorm & ResNets. 
Advances in Neural Information Processing Systems, 32, 10902–10911.</span>
([link](https://papers.nips.cc/paper/2019/hash/e520f70ac3930490458892665cda6620-Abstract.html),
 [pdf](https://papers.nips.cc/paper/2019/file/e520f70ac3930490458892665cda6620-Paper.pdf))

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

<span id="clevert16elu">Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). 
International Conference on Learning Representations 4.</span> 
([link](http://arxiv.org/abs/1511.07289),
 [pdf](http://arxiv.org/pdf/1511.07289.pdf))

<span id="de20skipinit">De, S., & Smith, S. L. (2020). Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. 
Advances in Neural Information Processing Systems, 33, 19964–19975.</span>
([link](https://proceedings.neurips.cc//paper/2020/hash/e6b738eca0e6792ba8a9cbcba6c1881d-Abstract.html),
 [pdf](https://proceedings.neurips.cc//paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-Paper.pdf))

<span id="gitman17comparison">Gitman, I., & Ginsburg, B. (2017). Comparison of Batch Normalization and Weight Normalization Algorithms for the Large-scale Image Classification [Preprint]. </span> 
([link](http://arxiv.org/abs/1709.08145),
 [pdf](http://arxiv.org/pdf/1709.08145.pdf))

<span id="hanin18how">Hanin, B., & Rolnick, D. (2018). How to Start Training: The Effect of Initialization and Architecture. 
Advances in Neural Information Processing Systems, 31, 571–581.</span>
([link](https://proceedings.neurips.cc/paper/2018/hash/d81f9c1be2e08964bf9f24b15f0e4900-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/d81f9c1be2e08964bf9f24b15f0e4900-Paper.pdf))

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

<span id="hochreiter97lstm">Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. 
Neural Computation, 9(8), 1735–1780. </span> 
([link](https://doi.org/10.1162/neco.1997.9.8.1735),
 [pdf](https://ml.jku.at/publications/older/2604.pdf))

<span id="hoffer18norm">Hoffer, E., Banner, R., Golan, I., & Soudry, D. (2018). Norm matters: Efficient and accurate normalization schemes in deep networks. 
Advances in Neural Information Processing Systems, 31, 2160–2170. </span> 
([link](https://proceedings.neurips.cc/paper/2018/hash/a0160709701140704575d499c997b6ca-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2018/file/a0160709701140704575d499c997b6ca-Paper.pdf))

<span id="huang17centred">Huang, L., Liu, X., Liu, Y., Lang, B., & Tao, D. (2017). Centered Weight Normalization in Accelerating Training of Deep Neural Networks. 
Proceedings of the IEEE International Conference on Computer Vision, 2822–2830.</span> 
([link](https://doi.org/10.1109/ICCV.2017.305),
 [pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Centered_Weight_Normalization_ICCV_2017_paper.pdf))

<span id="huang17densenet">Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. 
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261–2269. </span> 
([link](https://doi.org/10.1109/CVPR.2017.243),
 [pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))

<span id="ioffe15batchnorm">Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 
Proceedings of the 32nd International Conference on Machine Learning, 37, 448–456.</span> 
([link](http://proceedings.mlr.press/v37/ioffe15.html),
 [pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))

<span id="ioffe17batchrenorm">Ioffe, S. (2017). Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. 
Advances in Neural Information Processing Systems, 30, 1945–1953. </span> 
([link](https://proceedings.neurips.cc/paper/2017/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf))

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

<span id="luo19towards">Luo, P., Wang, X., Shao, W., & Peng, Z. (2019). Towards Understanding Regularization in Batch Normalization. 6. </span>
([link](https://openreview.net/forum?id=HJlLKjR9FQ),
 [pdf](https://openreview.net/pdf?id=HJlLKjR9FQ))

<span id="miyato18spectralnorm">Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. 
International Conference on Learning Representations 6.</span> 
([link](https://openreview.net/forum?id=B1QRgziT-),
 [pdf](https://openreview.net/pdf?id=B1QRgziT-))


<span id="mishkin16lsuv">Mishkin, D., & Matas, J. (2016). All you need is a good init. 
International Conference on Learning Representations 4.</span> 
([link](http://arxiv.org/abs/1511.06422),
 [pdf](http://arxiv.org/pdf/1511.06422.pdf))

<span>Ripley, B. D. (1996). Pattern Recognition and Neural Networks. Cambridge University Press. </span> 
([link](https://doi.org/10.1017/CBO9780511812651))

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

<span id="srivasta15highway">Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. 
Advances in Neural Information Processing Systems, 28, 2377–2385. </span> 
([link](https://papers.nips.cc/paper/2015/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html), 
 [pdf](https://papers.nips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf))

<span id="szegedy16inceptionv4">Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2016). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning [Preprint].</span>
([link](http://arxiv.org/abs/1602.07261),
 [pdf](http://arxiv.org/pdf/1602.07261.pdf))

<span id="vandersmagt98solving">van der Smagt, P., & Hirzinger, G. (1998). Solving the Ill-Conditioning in Neural Network Learning. 
In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (1st ed., pp. 193–206). Springer.</span> 
([link](https://doi.org/10.1007/3-540-49430-8_10))

<span id="vaswani17attention">Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. 
Advances in Neural Information Processing Systems, 30, 5998–6008.</span> 
([link](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
 [pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))


<span id="wadia21whitening">Wadia, N., Duckworth, D., Schoenholz, S. S., Dyer, E., & Sohl-Dickstein, J. (2021). Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization.
Proceedings of the 38th International Conference on Machine Learning, 139, 10617–10629.</span> 
([link](http://proceedings.mlr.press/v139/wadia21a.html),
 [pdf](http://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf))

<span id="wu18groupnorm">Wu, Y., & He, K. (2018). Group Normalization. 
Computer Vision – ECCV 2018, 3–19. Springer International Publishing. </span> 
([link](https://doi.org/10.1007/978-3-030-01261-8_1),
 [pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf))


<span id="zhang19fixup">Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. 
International Conference on Learning Representations 6. </span> 
([link](https://openreview.net/forum?id=H1gsz30cKX),
 [pdf](https://openreview.net/pdf?id=H1gsz30cKX))
