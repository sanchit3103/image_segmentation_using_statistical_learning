# Image segmentation using DCT and Bayesian Decision Rule

<p align="justify">
The grayscale 'cheetah' image shown below is segmented into foreground and background using Discrete Cosine Transform (DCT) and Bayesian Decision Rule (BDR). Here, we viewed the image as a collection of 8x8 blocks. For each block, DCT was computed which provided us with an array of 8x8 frequency coefficients. This was performed because the cheetah in foreground and the grass in background have different textures, with different frequency decompositions and the two classes should be better separated in the frequency domain. Each 8x8 array was converted into a 64 dimensional vector following the zig-zag pattern. For each vector, we computed the index of 2nd largest energy value, giving us the feature for each vector. Through these indexes, we obtained the class conditionals and the priors were calculated using the training data. These probability estimates were given to BDR to estimate the actual class for each block.
</p>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418033-8ce8e52c-351b-4501-a6cc-d456ee2d5a8d.png" width="250"/>  
</p>

The resulting segmented image is shown below. It looks somewhat like a cheetah but it is very noisy and blocky.
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418150-bd1b1b32-cfd8-4f43-bc77-bef487ef0d02.png" width="330"/>  
</p>

## Project Report
[Sanchit Gupta, 'Image segmentation using DCT and Bayesian Decision Rule', ECE 271A, Course Homework, UCSD](https://github.com/sanchit3103/image_segmentation_using_statistical_learning/blob/main/segmentation_using_DCT_and_BDR/Report.pdf)
