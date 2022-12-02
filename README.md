## Long-Term Cognitive Network for Pattern Classification

This repository introduces the Long-Term Cognitive Network (LTCN) model for structured pattern classification problems. This recurrent neural network incorporates a quasi-nonlinear reasoning rule that allows controlling the ammout of nonlinearity in the reasoning mechanism. Furthermore, this neural classifier uses a recurrence-aware decision model that evades the issues posed by the unique fixed point while introducing a deterministic learning algorithm to compute the tunable parameters. The simulations show that our interpretable model obtains competitive results when compared to state-of-the-art white and black-box models.

Firstly, we propose a new reasoning rule that introduces a nonlinearity coefficient $\phi \in [0,1]$ controlling the extent to which the model will take into account the value produced by the transfer function over the neuron's initial activation value.

$A_k^{(t)} = \phi f \left( A_k^{(t-1)}W + B \right)+ (1-\phi) A_k^{(0)}$

where $B_{1 \times M}$ denotes the bias matrix, which can be understood as the amount of external information impacting the neuron's state \bluetext{(i.e., what cannot be explained through the neuronal concepts describing the problem domain)}. The matrices $B$ and $W$ will be computed from historical data during the \textit{unsupervised learning step}.

### Reference

Nápoles, G., Salgueiro, Y., Grau, I., & Espinosa, M. L. (2022). Recurrence-Aware Long-Term Cognitive Network for Explainable Pattern Classification. IEEE Transactions on Cybernetics. [paper](https://arxiv.org/pdf/2107.03423.pdf)
