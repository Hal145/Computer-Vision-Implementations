# Computer-Vision
Various CV implementations

1- Color Quantization
Color quantization is utilized for various reasons in image processing and computer graphics. 
- Reduced Memory Usage: Useful when working with limited resources or optimizing file sizes for efficient storage or transmission.
- Compression and Bandwidth Optimization: Easier to transmit or store images, especially in situations with limited bandwidth.

2-Connected Component Analysis
A fundamental image processing technique used to identify and analyze distinct regions or objects within an image.

3- Bag-of-Features (BoF) Implementation
kornia, kornia_moons, and OpenCV libraries are used. The implementation includes the following steps: 
- Feature Extraction and encoding: This step extracts local features (detectors) from images using OpenCV. Once the local features are extracted, they are encoded into a fixed-length vector representation (descriptors).  SIFT and HyNet methods are used for feature encoding.
- Codebook Construction and Feature Quantization: The codebook is a collection of visual codewords obtained through clustering the local features. Each visual codeword represents a cluster center or a prototype of similar features. The codebook captures the statistical distribution of visual features in the dataset. The k-means technique is used for feature quantization.
- Histogram Creation: The quantized visual codewords are then used to construct a histogram representation, the Bag-of-Features representation. The histogram counts the occurrence of each visual codeword in the image.
- Classification: MLP and SVM are used as classifiers. 




