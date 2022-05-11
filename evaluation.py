import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd


test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = "datasets/test"
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("mask_detector_vgg16.model")
print("[INFO] Success.")
def num2class(num):
    if num == 0:
        classes = 'with'
    else:
        classes = 'without'
    return classes

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=30,
                                                  class_mode='categorical', shuffle=False,)
prediction = model.predict_generator(test_generator, verbose=1)
predict_label = np.argmax(prediction, axis=1)
true_label = test_generator.classes

test_loss, test_acc = model.evaluate_generator(test_generator, steps=34)
print('test acc:', round(test_acc, 3), ', test loss:', round(test_loss,3))

# Which picture have wrong predict
imagenames = test_generator.filenames
errors = np.where(predict_label != test_generator.classes)[0]
print('\nWrong predict:')
for i, j in enumerate(errors):
    print(i+1, ':', imagenames[j], ', predict :', num2class(predict_label[j]))

# Confusion matrix
true_label_list = list(true_label)
predict_label_list = list(predict_label)
for i in range(len(true_label_list)):
    true_label_list[i] = num2class(true_label_list[i])
    predict_label_list[i] = num2class(predict_label_list[i])
output = pd.crosstab(np.array(true_label_list), np.array(predict_label_list), rownames=['label'], colnames=['predict'])
print(output)