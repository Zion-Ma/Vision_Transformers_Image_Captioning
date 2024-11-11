# RNN
> In this section, I used LSTM as the decoder to generate captions for given image features. Here are some hyper parameter settings and metrics.
## Hyper Parameter Settings
    1. hidden layer size = 256
    2. window size = 20
    3. batch size = 100
    4. learning rate = 0.001
    5. epoch = 4
## Result
    1. accuracy = 0.325
    2. perplexity = 18.911

# Transformers
> In this section, I used Transformers' decoder as the decoder to generate captions for given image features. Here are some hyper parameter settings and metrics.
# Hyper Parameter Settings
    1. hidden layer size = 256
    2. window size = 20
    3. batch size = 128
    4. learning rate = 0.001
    5. epoch = 4
    6. nhead = 3
    7. sublayer = 6
## Result
    1. accuracy = 0.345
    2. perplexity = 16.427

# Bug report 
    1. I didn't specify the window size (sequence length) for Tensorflow's Embedding layer. It led to a dimension mismatch error.
    2. I misplaced image representation as "inputs" and word embedding for "context sequence", which led to bad training results.
    3. I didn't augment dimension for image representation and sent it to Transformers' decoder. It caused some dimensionality error. 