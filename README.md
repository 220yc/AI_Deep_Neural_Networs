# AI_Deep_Neural_Networs
Hardware Accelerators for AI Deep Neural Networs

### AI model : GoogLeNet
![image](https://github.com/user-attachments/assets/8ec1b74d-8aed-40cf-ae38-6ed9472b1709)
![image](https://github.com/user-attachments/assets/973f0fe5-333f-418e-8445-4bfd4ca9daf2)
![image](https://github.com/user-attachments/assets/5841cebc-1f97-4ca7-af62-91379d8da76f)

# GoogLeNet Performance Analysis

## Parameter Count and Computational Load

In GoogLeNet, the parameter count for the first convolutional layer is 9.2K, with a computational load of 118M. Overall, GoogLeNet has a weight parameter count of approximately 6.99M and a MAC operation count of about 1580M.

## How Does GoogLeNet Improve Performance?

Unlike AlexNet and VGGNet, which increase the number of layers and channel widths, GoogLeNet adopts a more efficient approach to reduce parameter counts and enhance the utilization of computational resources. As depth increases, networks are prone to overfitting and may experience gradient vanishing or explosion; thus, simply increasing depth is not sufficient.

## Comparison of Small and Large Convolutional Kernels

The advantages of using small convolutional kernels include:

1. **Accuracy Improvement**: Stacking multiplesmall convolutional kernels can improve accuracy more than a single large convolutional kernel, without decreasing the receptive field and actually reducing computational load.

2. **Increased Nonlinear Functions**: Stacking multiple small convolutional kernels allows for more nonlinear functions, enhancing the discriminative power of the decision function.

3. **Reduced Parameters**: The parameter count of two 3x3 convolutional kernels compared to one 5x5 convolutional kernel reduces parameters by 28%, effectively lowering the overall parameter count.

## Computation and Memory Bandwidth

Deep neural network computation requires the use of weight data and input data (maps, activations). Weights are the parameters of the neural network, while input data is the data being processed from one layer to the next.

### Data Reuse

Generally, if computations can reuse data, less memory bandwidth is required. Data reuse can be achieved by:

- Sending more inputs to be processed by the same weights.
- Sending more weights to process the same inputs.

If there is no input or weight data reuse, the bandwidth will be at a maximum for a given application.

### Bandwidth Requirements for Linear and Convolutional Layers

**Linear Layers**: A weight matrix of M by M is used to process a vector of M values, resulting in a total data transfer of `b(M+M²)` or approximately `bM²`.

If a linear layer is used for only one vector, the entire M² weight matrix must be sent during computation. If your system performs T operations/second, the time required for computation is `bM²/T`. Therefore, bandwidth BW = total data transferred / time. In the case of linear layers, BW = T.

**Convolutional Layers**: The bandwidth requirements for convolution operations are usually lower since input map data can be used in several convolution operations in parallel, and convolution weights are relatively small.

For example, a 3x3 convolution operation involving a 13 x 13 pixel map from 192 input maps to 192 output maps requires approximately 4MB of weight data and 0.1MB of input data. This may require about 3.2 GB/s of bandwidth on a 128 G-ops/s system with approximately 99% efficiency.

**RNNs**: Memory bandwidth for recurrent neural networks is among the highest. Systems like Deep Speech 2 use 4 RNN layers of size 400. Each layer employs the equivalent of 3 linear-layer-like matrix multiplications in a GRU model. During inference, the input batch is typically just 1 or a small number, requiring a high amount of memory bandwidth, often making it impossible to fully utilize even efficient hardware.

## Visualizing These Concepts

Refer to the following figure (add relevant visuals as needed).
![image](https://github.com/user-attachments/assets/fee8e3f5-44f9-4b33-8875-a4d316319374)

---------------

# 1.1 Dimensionality Reduction

Dimensionality reduction can decrease the scale of convolutional kernel parameters. For example, if the input feature map is **28×28×192**:

- **1×1 Convolution** with 64 channels
- **3×3 Convolution** with 128 channels
- **5×5 Convolution** with 32 channels

The parameters for the left diagram (a) will be:
192 × (1×1 × 64) + 192 × (3×3 × 128) + 192 × (5×5 × 32) = 387072

In the right diagram (b), if we add **1×1 convolutions** with **96** and **16** channels before the **3×3** and **5×5** convolution layers respectively, the convolutional kernel parameters change to:
192 × (1×1 × 64) + (192 × 1×1 × 96 + 96 × 3×3 × 128) + (192 × 1×1 × 16 + 16 × 5×5 × 32) = 157184

Adding a **1×1 convolution layer** after the **max pooling layer** can also reduce the output feature map's number of channels:

- **Left Diagram Feature Map Count**:
64 + 128 + 32 + 192 (pooling does not change feature maps) = 416

*(If each module does this, the network output will gradually increase.)*

- **Right Diagram Feature Map Count**:
64 + 128 + 32 + 32 (added a 1×1 convolution with 32 channels after pooling) = 256

# 1.2 Dimensionality Expansion

Suppose the input is a **(3, 3, 3)** feature map followed by a **1×1×3** convolution kernel. After convolution, we will get a **(3, 3, 3)** feature map, and if there are **64** of these **1×1×3 convolution kernels**, we will obtain **64 (3, 3, 3)** feature maps. 

By linearly combining the parameters at the same position, we can create a **3×3×64** feature map. Thus, through the **1×1 convolution kernel**, we expand the feature map from **3×3×3** to **3×3×64**, achieving dimensionality expansion while using the least number of parameters to widen the network's dimensions.

## Key Tips
- The number of **filters** (convolution kernels) determines the number of channels in the feature map after convolution.
- The input layer's number of channels must match the number of channels in the filters.
- Therefore, the next layer's convolution kernels will have the same number of channels as the output from the previous layer.

## Summary of Dimensionality Expansion and Reduction

Both dimensionality expansion and reduction using **1×1 convolution kernels** can significantly reduce the number of parameters compared to traditional methods.

## (2) Cross-Channel Information Interaction

Using a **1×1 convolution kernel** to achieve dimensionality reduction and expansion is essentially a linear combination of information across channels. For example, if we add a **1×1 kernel** with **28 channels** after a **3×3 kernel** with **64 channels**, it transforms into a **3×3 kernel** with **28 channels**. 

The original **64 channels** can be understood as having undergone a linear combination to become **28 channels**, thus facilitating information interaction across channels. This means that channels that were originally parallel can now interact and transform through linear combinations.

## (3) Increasing Non-linear Features

The **1×1 convolution kernel** can maintain the scale of the feature map while utilizing subsequent non-linear activation functions to significantly increase non-linear characteristics, allowing the network to become much deeper. 

For example, in the dimensionality expansion scenario, each of the **64 1×1×3 convolution kernels** can have an activation function added afterward, enhancing non-linear features.



