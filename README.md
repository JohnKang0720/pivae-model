# Poisson-Informed VAE
The model was implemented from the paper below, as well as the repository linked with it.
The model is fit against synthetic Poisson spike data, but with some different parameters.
1. Learning Rate of 1e-5 is optimal
2. Epochs of 50~70 should be good.
3. 20 neurons for Prior
4. ~60 neurons for Encoder
5. Extra GIN Layer with ReLU activation.

Reference: https://arxiv.org/pdf/2011.04798
