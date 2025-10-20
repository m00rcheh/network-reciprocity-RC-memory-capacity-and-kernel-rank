# network-reciprocity-RC-memory-capacity-and-kernel-rank

This repository contains the source code used to assess the computational implications of reciprocity gradients in binary and weighted neural networks. The findings are discussed in the accompanying manuscript.

The NRC algorithms, available in the utils directory and also accessible in the [NRC_binary_and_weighted_Network_Reciprocity_Control](https://github.com/m00rcheh/NRC_binary_and_weighted_Network_Reciprocity_Control) repository, were used to control the degree of asymmetry and reciprocity in both binary and weighted networks while preserving key network properties.


## Abstract
Cerebral cortical networks in the mammalian brain exhibit a non-random organization that systematically avoids strong reciprocal projections, particularly in sensory hierarchies. This “no-strong-loops” principle is thought to prevent runaway excitation and maintain stability, yet its computational impact remains unclear. Here, we use computational analysis and modeling to show that connectivity asymmetry supports high working-memory capacity, whereas increasing reciprocity reduces memory capacity and representational diversity in reservoir-computing models of recurrent neural networks. We systematically examine synthetic architectures inspired by mammalian cortical connectivity and find that sparse, modular, and hierarchical networks achieve superior performance, relative to random, small-world, or core-periphery graphs, but only when reciprocity is constrained. Validated on directed macaque and marmoset connectomes, these results indicate that restricting reciprocal motifs yields functional benefits in sparse networks, consistent with an evolutionary strategy for stable, efficient information processing in the brain. These findings suggest a biologically-inspired design principle for artificial neural systems.

![Graphical Abstract](Figs/Main/Graphical%20abstract.jpg)


## Features

- **Network Reciprocity Control**: Efficient algorithms to control reciprocity and asymmetry in binary and weighted networks.
- **Benchmark Networks**: Tested on synthetic networks (random, small-world, hierarchical modular, hierarchical core-periphery, and hierarchical modular core-periphery) and directed brain connectomes obtained from non-human primates (Macaque and Marmoset).
- **Key Computational Metrics**: Memory capacity (MC) to assess temporal information retention and kernel rank (KR) to quantify representational diversity.

## Usage
The `utils` script contains functions for generating networks, measuring binary and weighted reciprocity, and adjusting reciprocity in networks. 
The `metrics` script contains functions for computing memory capacity and kernel rank. 
  

## Citation
If you use these algorithms in your research, please cite the corresponding manuscripts:

Hadaeghi, Fatemeh, et al. "A computational perspective on the no-strong-loops principle in brain networks." bioRxiv (2025): 2025-09. 
link: [https://www.biorxiv.org/content/10.1101/2025.09.24.678310v1]

Hadaeghi, Fatemeh, Kayson Fakhar, and Claus Christian Hilgetag. "Controlling Reciprocity in Binary and Weighted Networks: A Novel Density-Conserving Approach." bioRxiv (2024).
link: [https://www.biorxiv.org/content/10.1101/2024.11.24.625064v1]
