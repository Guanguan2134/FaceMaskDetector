import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import yaml, argparse


with open("config.yml", "r") as f:
	config = yaml.safe_load(f)

ap = argparse.ArgumentParser()
ap.add_argument("-mp", "--model_path", type=str, default="model/mask_detector_best.h5", help="path to output face mask detector model")
args = vars(ap.parse_args())

test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = "datasets/test"
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args['model_path'])
print("[INFO] Success.")
def num2class(num):
    if num == 0:
        classes = 'with'
    else:
        classes = 'without'
    return classes

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=config['Train']['hyperparameter']['BS'],
                                                  class_mode='categorical', shuffle=False,)
prediction = model.predict(test_generator, verbose=0)
predict_label = np.argmax(prediction, axis=1)
true_label = test_generator.classes

# Which picture have wrong predict
imagenames = test_generator.filenames
errors = np.where(predict_label != test_generator.classes)[0]
print('\nWrong predict:')
for i, j in enumerate(errors):
    print(i+1, ':', imagenames[j], ', predict :', num2class(predict_label[j]))

test_loss, test_acc = model.evaluate(test_generator)
print('test acc:', round(test_acc, 3), ', test loss:', round(test_loss,3))

# Compute confusion matrix
cnf_matrix = confusion_matrix(true_label, predict_label)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['with_mask', 'without_mask'], normalize=True,
                      title='Confusion matrix, with normalization')
plt.savefig("fig/confusion_metrix.png")
plt.show()
