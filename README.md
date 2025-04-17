# Distillation-Water-Body-Detection

This research introduces the Efficient Water Body Detection Based on Knowledge Distillation for SAR Imagery, a novel knowledge distillation-based framework that significantly enhances water body detection accuracy in SAR satellite imagery while maintaining computational efficiency. 以Unet 和 SwinUnet 为例子展示。

# Sentinel-2 data Download link:
https://github.com/cloudtostreet/Sen1Floods11


# Input
This project utilizes [Sentinel-2 data](https://github.com/cloudtostreet/Sen1Floods11)  data as its dataset. Please download and extract the Sentinel-2 data dataset, and configure files under the `/dataset` directory.

# Output
All results will be logged to `XXX_training.log`, including the IoU and OA metrics for both the training and validation sets.

# Result
![](./res.png)

Detection Result Samples from the Spain Site. Visualization includes (a) SAR Image, (b) EO Image, (c) Label, and (d)-(f) comparative results of AttentiveUnet, DeeplabV3, PSPNet, SwinUnet, TransUNet, and UNet models before(left) and after(right) integrating proposed distillation learning methodology.

# Email:
B23160008@s.upc.edu.cn