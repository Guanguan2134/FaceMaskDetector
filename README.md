# FaceMaskDetector
The main purpose is to use a deep learning model to detect face mask, and We referenced the mask detector from this [repo](https://github.com/chandrikadeb7/Face-Mask-Detection) to make our own version.

Unlike the original mask recognition alone, we use personal items such as palms, mobile phones, and papers to add to the training data set to avoid someone trying to deceive the machine with similar items.

At the same time, we also tried to use several common image classification models for evaluation, and finally chose ResNet50v2 as the final model, and got 96.6% accuracy in our results.

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

Quick start
---
### Stream mask detection
```
python detect_mask_video.py
```
> You can put the arguments behind, the default setting is write in the `config.yml`. Please notice that any arguments will rewrite the setting in `config.yml`

### Training
1. Put your data into the folder `datasets/raw`, and seperate your datasets into `datasets/raw/with_mask` and `datasets/raw/without_mask`
2. Setting the ratio of the number of training data in `config.yml`
3. Run the data distribution:
    ```
    python dataset_generator.py
    ```
    Two folder will be generate, and the data structure is like:
    ```
    datasets
    ├─raw
    │  ├─without_mask
    │  └─with_mask
    ├─test
    │  ├─without_mask
    │  └─with_mask
    └─train
        ├─without_mask
        └─with_mask
    ```
4. Run `train_mask_detection.py`:
    ```
    python train_mask_detection.py
    ```
    
5. To do the evaluation, run:
    ```
    python evaluation.py
    ```

Evaluation
---
![](https://i.imgur.com/DvNv3mg.png)


Reference
---
* [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)

TODO
---
- [x] Open the repo
- [ ] Upload main code
- [ ] Complete MarkDown file
- [ ] Prove and refine the code

###### tags: `GitHub` `Python` `ML`