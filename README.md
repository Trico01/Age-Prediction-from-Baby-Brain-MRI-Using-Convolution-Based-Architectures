## Age Prediction from Baby Brain MRI Using Convolution-Based Architectures

## Dataset

The UNC / UMN Baby Connectome Project, which includes 343 healthy term-born subjects. Images from subjects aged 0-2 years are selected, which contain 1355 T1 and T2 MRI images.

<img src="Results/dist.png" alt="image" width="60%" height="60%">

## Result

The test performance of all four models under different input image and input mode based on MAE metric.

| Input Mode | Middle Slices - T1 | Middle Slices - T2 | All Brain - T1 | All Brain - T2 |
| ---------- | ------------------ | ------------------ | -------------- | -------------- |
| CNN-5      | 1.175              | 1.187              | 1.973          | 1.926          |
| ASPP-CNN   | 1.4464             | 1.333              | 2.213          | 2.062          |
| CBAM-CNN   | 1.396              | 1.404              | 2.132          | 2.205          |
| ResNet     | 1.169              | 1.107              | 1.912          | 1.895          |

<img src="Results/GradCAM.png" alt="image" width="100%" height="=100%">
