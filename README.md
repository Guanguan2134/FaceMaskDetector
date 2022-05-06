# FaceMaskDetector
The main purpose is to use a deep learning model to detect face mask, and We referenced the mask detector from this [repo](https://github.com/chandrikadeb7/Face-Mask-Detection) to make our own version.

Unlike the original version, we use personal items such as palms, mobile phones, and papers to add to the training dataset to avoid someone trying to deceive the machine with similar items.

At the same time, we also tried to use several classical image classification models for evaluation, and finally we chose ResNet50v2 as the final model, and got 96.6% accuracy in our results.

[TOC]

Dependencies
---
* python = 3.8
* tensorflow >= 1.15.2
* keras = 2.3.1
* imutils==0.5.3
* numpy==1.18.2
* opencv-python==4.2.0.*
* matplotlib==3.2.1
* argparse==1.1
* scipy==1.4.1

Results
---

Evaluation
---
![](https://i.imgur.com/DvNv3mg.png)


Reference
---
* [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)

TODO
---
- [x] Open the repo
- [x] Upload main code
- [ ] Complete MarkDown file
- [ ] Prove and refine the code
