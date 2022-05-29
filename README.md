# FaceMaskDetector
The main purpose is to use a deep learning model to detect face mask, and We referenced the mask detector from this [repo](https://github.com/chandrikadeb7/Face-Mask-Detection) to make our own version.

Unlike the original mask recognition alone, we use personal items such as palms, mobile phones, and papers to add to the training data set to avoid someone trying to deceive the machine with similar items.

At the same time, we also tried to use several common image classification models for evaluation, and finally chose ResNet50v2 as the final model, and got the accuracy of 99.3% in our results.

# Table of Content

* [FaceMaskDetector](#FaceMaskDetector)
	* [Dependencies](#Dependencies)
	* [Demo](#Demo)
		* [Before / After data argumentation](#Before-//-After-data-argumentation)
	* [Quick start](#Quick-start)
		* [Stream mask detection](#Stream-mask-detection)
		* [Training](#Training)
	* [Evaluation](#Evaluation)
	* [Reference](#Reference)

Dependencies
---
* python = 3.8
* tensorflow = 2.5.0  

Others packages write in `requirements.txt`, try to run:
```
conda create -n facemask python=3.8 -y
conda activate facemask
pip install -r requirements.txt
```

Demo
---
### Before / After data argumentation
<img src="https://github.com/Guanguan2134/FaceMaskDetector/blob/main/fig/Before%20aug.gif" width="200"/>	<img src="https://github.com/Guanguan2134/FaceMaskDetector/blob/main/fig/After%20aug.gif" width="200" />

Quick start
---
### Stream mask detection
Use the default model (ResNet50v2):
```
python detect_mask_video.py
```

Use the model you trained:
```
python detect_mask_video.py -m model/trained_model.h5
```
> You can put the arguments behind, the default setting is write in the `config.yml`. Please notice that any arguments will rewrite the setting in `config.yml`

### Training
1. Put your data into the folder `datasets/raw`, and seperate your datasets into `datasets/raw/with_mask` and `datasets/raw/without_mask`
2. Setting the ratio of the training data in `config.yml`/`Data.train_split_ratio`
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
4. Run `train_mask_detection.py` and use the default model from Keras (ResNet50V2):
    ```
    python train_mask_detection.py
    ```
    
    or use arguments to train:
    ```
    python train_mask_detection.py -cv -m VGG16 -e 100
    ```
    
    The usage of each argument is down below:
    
    > -cv: Activate the function of cross-validation. The number of folds can be set in `config.yml`/`Train.cv.k-fold`. After training each fold, the result of each fold will be shown in command line (mean acc, best acc, std)
    
    > -m: Choose the base model you want to train from [Keras.application](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    
    > -e: Training epochs
    
5. To do the evaluation by the default model, run:
    ```
    python evaluation.py
    ```
    or choose the model you train:
    ```
    python evaluation.py -mp model/trained_model.h5
    ```

Evaluation
---
* Evaluate on 5 classic model in [Keras.application](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
![](https://i.imgur.com/DvNv3mg.png)

* The Training history and Testing confusion metrix of ResNet50V2

<figure class="half">
    <img src="https://i.imgur.com/b1zNBzP.png" width=500><img src="https://i.imgur.com/k6v9O4d.png" width=500>
</figure>




Reference
---
* [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)

###### tags: `GitHub` `Python` `ML`
