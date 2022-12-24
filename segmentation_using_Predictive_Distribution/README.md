# Image segmentation using Predictive Distribution and Bayesian Parameter Estimation

<p align="justify">
The grayscale 'cheetah' image shown below is segmented into foreground and background using Maximum Likelihood (ML) estimate, Gaussian PDF and Bayesian Decision Rule (BDR). It was assumed that the class-conditional densities are multivariate Gaussians of 64 dimensions. Using the training data, the ML estimates for the prior probabilities of two classes and the parameters of class conditionals under the Gaussian assumption were computed. The Gaussian prior for mean and covariance were provided in the training data. Using the ML estimates and Gaussian priors, we computed the posterior mean and covariance and parameters of the predictive distribution.  
</p>

<p align="justify">
We viewed the image as a collection of 8x8 blocks. For each block, DCT was computed which provided us with an array of 8x8 frequency coefficients. Each 8x8 array was converted into a 64 dimensional vector following the zig-zag pattern. The posterior mean and covariance and parameters of the predictive distribution calculated earlier were given to BDR following Gaussian PDF to estimate the actual class for each block.
</p>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209418033-8ce8e52c-351b-4501-a6cc-d456ee2d5a8d.png" width="250"/>  
</p>

The resulting segmented image obtained through this procedure is shown below.
<p align="center">
  <img src = "https://user-images.githubusercontent.com/4907348/209419722-6dd0bb2e-b0dc-4cf5-9c38-3f85732f6681.png" width="330"/>  
</p>


## Report
[Sanchit Gupta, 'Image segmentation using Maximum Likelihood Estimate (MLE), Gaussian PDF and Bayesian Decision Rule', ECE 271A, Course Homework, UCSD](https://github.com/sanchit3103/image_segmentation_using_statistical_learning/blob/main/segmentation_using_Predictive_Distribution/Report.pdf)
