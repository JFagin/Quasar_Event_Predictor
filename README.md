# Predicting High Magnification Events in Microlensed Quasars in the Era of LSST using RNNs

Code for "Predicting High Magnification Events in Microlensed Quasars in the Era of LSST using Recurrent Neural Networks
" [[arXiv]](https://arxiv.org/abs/2409.08999). Here we train a recurrent neural network to predict the onset of high magnification events in microlensed quasar light curves through real time classification. Our method is applicable to predicting any events or transients in irregularly smapled, multivariate time series.

## Background

[Quasar](https://en.wikipedia.org/wiki/Quasar) are bright and unobscured [AGN](https://en.wikipedia.org/wiki/Active_galactic_nucleus) powered by [super massive black holes (SMBH)](https://en.wikipedia.org/wiki/Supermassive_black_hole) at the center of galaxies. [Strong gravitational lensing](https://en.wikipedia.org/wiki/Strong_gravitational_lensing) of quasars occurs when there is a galaxy lens along the line of sight of a quasar source and the observer, causing multiple images of the quasar. Each image can independently undergo [gravitational microlensing](https://en.wikipedia.org/wiki/Gravitational_microlensing) when individual stars in the lensing galaxy affects each image. Microlensing can zoom in on different regions of the [accretion disk](https://en.wikipedia.org/wiki/Accretion_disk) at extreme cosmological distances, providing a unique opportunity to resolve spatial variations in temperature, structure, and emission properties that are otherwise impossible to observe directly. This microlensing causes a so-called high magnification event. Ideally these high magnifaction events will be observed in high cadence multi-band or spectroscop follow-up. We train a recurrent neural network to predict the onset of these high magnification events in simulated 10-year microlensing quasar [light curves](https://en.wikipedia.org/wiki/Light_curve) in preperation for the [Rubin Observatory Legacy Survey of Space and Time (LSST)](https://en.wikipedia.org/wiki/Vera_C._Rubin_Observatory). Our network can be continuously applied throughout the LSST survey, providing crucial alerts to optimize follow-up resources.

## Citation

If you found this codebase useful in your research, please consider citing:

```
@misc{Fagin_2024,
      title={Predicting High Magnification Events in Microlensed Quasars in the Era of LSST using Recurrent Neural Networks}, 
      author={Joshua Fagin and Eric Paic and Favio Neira and Henry Best and Timo Anguita and Martin Millon and Matthew O'Dowd and Dominique Sluse and Georgios Vernardos},
      year={2024},
      eprint={2409.08999},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA},
      url={https://arxiv.org/abs/2409.08999}, 
}
```

### Contact
For inquiries or to request the full training set, reach out to: jfagin@gradcenter.cuny.edu
