Yet Another NEAT Implementation
###############################

This is an implementation of `NeuroEvolution of Augmenting Topologies`_, a method for evolving neural network
topologies. It uses TensorFlow 2.0 for running the networks. I plan to add support for HyperNEAT and maybe
ES-HyperNEAT.

.. _`NeuroEvolution of Augmenting Topologies`: https://doi.org/10.1162/106365602320169811

Install
#######

I recommend using a conda environment; the tensorflow 2.0.0alpha0 requirement can fail otherwise::

    git clone https://github.com/tkclough/yaNEAT
    cd yaNEAT
    pip install .
