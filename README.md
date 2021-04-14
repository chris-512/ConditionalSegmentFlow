# Learning Representative Points using Normalizing Flows

## TODO List
- COCO dataloader for points sampling from object segmentation mask.
- COCO dataloader for points sampling from object boundary.
- Implement semantic segmentation using NFs
 - Person segmentation (Simple binary segmentation of person vs background, N = 2)
 - Apply random cropping, jittering
 - Multi-class segmentation (N > 2)
- Implment boundary detection using NFs
 - Learn `circles` distribution by adding extra NF and use it as a prior flow.
 - Sample from learned prior flow and feed it to subsequent NF.
- Improve backbones
 - Use dilated convolutions
 - Use leaky ReLU
- Use Flow++, SoftFlow, Gradient Boosted NF instead of CNFs.

 ## Maintainers
 - Sae Young Kim