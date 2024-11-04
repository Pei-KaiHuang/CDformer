# CDformer

### Channel difference transformer for face anti-spoofing

This is the first work focusing on exploring  semantically complementary channel information  to enhance the discriminability of channel-wise features for face anti-spoofing. 

## Overview of the proposed Channel Difference Self-Attention (CDSA).
![Screenshot](figures/CDSA.png)


## Multi-Head Channel Difference Self-Attention (MCDSA) consists of k CDSA operations running in parallel.
![Screenshot](figures/MCDSA.png)

## Requirements
```
numpy==1.23.3
pytz==2022.4
requests==2.28.1
scikit_learn==1.2.0
timm==0.6.7
torch==1.10.1
torchvision==0.11.2
```

## Training & Testing
Run `train.py` to train LDCformer

Run `test.py` to test LDCformer

## Citation

If you use the LDCformer/Decoupled-LDC, please cite the paper:
```

```
