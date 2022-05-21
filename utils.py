import yaml
import itertools
import matplotlib.pyplot as plt
import numpy as np


def search(dicts, item, *args):
    if type(dicts) == type(dict()):
        if item in dicts and len(args) != 0 :
            return item
        else:
            for key in dicts:
                res = search(dicts[key], item, key)
                if res.split("/")[-1] == item:
                    return key+"/"+res
                else:
                    key += "/"+res
    else:
        return "END"
    
    return ""

def model_process_function(model_name):
    model_dict = {'VGG16':'vgg16',
                  'VGG19':'vgg19',
                  'MobileNetV2':'mobilenet_v2',
                  'MobileNetV3Large':'mobilenet_v3',
                  'MobileNetV3Small':'mobilenet_v3',
                  'InceptionV3':'inception_v3',
                  'DenseNet121':'densenet',
                  'DenseNet169':'densenet',
                  'DenseNet201':'densenet',
                  'ResNet50V2':'resnet50'}
    return model_dict[model_name]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_norm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j])+"\n("+format(cm_norm[i, j], fmt)+"%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
