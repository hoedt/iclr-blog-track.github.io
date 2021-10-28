---
layout: post
title: Normalisation is dead, long live normalisation!
tags: [normalisation, initialisation, propagation]
authors: Hoedt, Pieter-Jan, JKU; 

---

Since the advent of Batch Normalisation (BN) almost every state-of-the-art (SOTA) method uses some form of normalisation.
After all, normalisation generally speeds up learning and leads to models that generalise better than their unnormalised counterparts.
This turns out to be especially useful when using some form of skip connections, which are prominent in residual networks (ResNets), for example.
However, <a href="#brock21characterizing">Brock et al. (2021)</a> suggest that SOTA performance can also be achieved using **ResNets without normalisation**!

The fact that Brock et al. went out of their way to get rid of something as simple as BN in ResNets for which BN happens to be especially helpful, does raise a few questions:

 1. Why get rid of BN in the first place?
 2. How (easy is it) to get rid of BN?
 3. Can this also work for other architectures?
 4. Does this allow to gain insights in why normalisation works so well?
 5. Wait a second... Are they getting rid of BN or normalisation as a whole?

The goal of this blog post is to provide some insights w.r.t. these questions using the results from <a href="#brock21characterizing">Brock et al. (2021)</a>.


## Normalisation


## Skip Connections


## Normaliser-Free ResNets


## Insights


---

## References

<span id="brock21characterizing">Brock et al. (2021)</span>
(<a href="">link</a>,
 <a href="">pdf</a>)