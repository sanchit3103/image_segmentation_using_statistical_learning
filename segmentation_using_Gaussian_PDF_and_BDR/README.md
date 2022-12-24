# Image segmentation using Maximum Likelihood Estimate (MLE), Gaussian PDF and Bayesian Decision Rule

<p align="justify">
The grayscale 'cheetah' image shown below is segmented into foreground and background using Maximum Likelihood (ML) estimate, Gaussian PDF and Bayesian Decision Rule (BDR). It was assumed that the class-conditional densities are multivariate Gaussians of 64 dimensions. Using the training data, the ML estimates for the prior probabilities of two classes and the parameters of class conditionals under the Gaussian assumption were computed. The plots with marginal densities of the two classes for each of 64 features were created and by visual inspection, the best 8 features and worst 8 features were found for classification purposes. 
</p>
<p align="justify">
We viewed the image as a collection of 8x8 blocks. For each block, DCT was computed which provided us with an array of 8x8 frequency coefficients. Each 8x8 array was converted into a 64 dimensional vector following the zig-zag pattern. The estimates calculated earlier for all 64 features and best 8 features were given to BDR following Gaussian PDF to estimate the actual class for each block.
</p>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418033-8ce8e52c-351b-4501-a6cc-d456ee2d5a8d.png" width="250"/>  
</p>

The resulting segmented image obtained using all 64 features is shown below.
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418809-c99ca025-6a60-4b70-b640-cc459030cb8d.png" width="330"/>  
</p>

The resulting segmented image obtained using the best 8 features is shown below. The mask obtained using best features has less probability of error in comparison to that of the mask obtained using all 64 features
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418830-07db499b-1f16-4810-8821-a7661ff0dc18.png" width="330"/>  
</p>

## Report
[Sanchit Gupta, 'Image segmentation using Maximum Likelihood Estimate (MLE), Gaussian PDF and Bayesian Decision Rule', ECE 271A, Course Homework, UCSD](https://github.com/sanchit3103/image_segmentation_using_statistical_learning/blob/main/segmentation_using_Gaussian_PDF_and_BDR/Report.pdf)

