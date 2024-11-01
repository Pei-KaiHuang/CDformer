# CDformer

### Channel difference transformer for face anti-spoofing

## Overview of the proposed Channel Difference Self-Attention (CDSA). In addition to the original queries \textbf{q}, keys \textbf{k}, and values \textbf{v} in Self-Attention (SA)  \cite{vaswani2017attention}, CDSA further includes the channel-wise differences $\hat{\textbf{z}}$ to obtain the projected $ \hat{\textbf{q}} $, $ \hat{\textbf{k} }$ and $ \hat{\textbf{v}} $, and then aggregates both $[ \textbf{q},  \textbf{k}, \textbf{v} ]$ and $[\hat{\textbf{q}},  \hat{\textbf{k} },  \hat{\textbf{v}} ] $ for modeling long-range data dependencies.
![plot](figures/CDSA.png)
