from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os, shutil
import yaml
from utils import search, model_process_function

with open("config.yml", "r") as f:
	config = yaml.safe_load(f)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_dir", type=str, help="path to dataset")
ap.add_argument("-p", "--plot", type=str, help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, help="which model to train: VGG16, MobileNetV2, InceptionV3, DenseNet121, ResNet50V2 from from tensorflow.keras.applications")
ap.add_argument("-md", "--model_dir", type=str, help="path to output face mask detector model directory")
ap.add_argument("-cv", "--cross_val", action='store_true', help="whether to use cross-validation or not")
ap.add_argument("-e", "--epoch", type=int, help="how many epochs to train")
args = vars(ap.parse_args())

for i in args:
    if args[i] not in [None, False, True]:
        ori_config = eval("config['"+search(config, i).replace("/", "']['")+"']")
        print(f"[INFO] change default setting {i} from {ori_config} to {args[i]}")
        exec("config['"+search(config, i).replace("/", "']['")+"'] = args[i]")
		
with open("config.yml", "w") as f:
	yaml.safe_dump(config, f)

exec(f"from tensorflow.keras.applications import {config['Train']['model']}")
exec(f"from tensorflow.keras.applications.{model_process_function(config['Train']['model'])} import preprocess_input")

def model_build(model, lr:float, epochs:int):
	# Loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in model.layers:
		layer.trainable = False

	# Construct the head of the model that will be placed on top of the
	# the base model
	headModel = model.output
	if "Inception" in config['Train']['model']:
		headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
	else:
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# Place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=model.input, outputs=headModel)

	# Compile our model
	opt = Adam(learning_rate=lr, decay=lr / epochs)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	return model
		
def make_val(dataset_dir, val_path):
	val_dir = os.path.join(dataset_dir, "val")
	if os.path.exists(val_dir):
		shutil.rmtree(val_dir)
	os.mkdir(val_dir)
	os.mkdir(os.path.join(val_dir, "with_mask"))
	os.mkdir(os.path.join(val_dir, "without_mask"))
	for path in val_path:
		shutil.move(path, os.path.join(val_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)))


def plot_metric(history, epoch, save_path):
	# Plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, epoch), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, epoch), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, epoch), history.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, epoch), history.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(save_path)

if __name__=="__main__":
	# Grab the list of images in our dataset directory, then initialize
	# the list of data (i.e., images) and class images
	print("[INFO] loading images...")
	imagePaths = list(paths.list_images(os.path.join(config['Data']['dataset_dir'], "train")))
	data = []
	labels = []
	INIT_LR = float(config['Train']['hyperparameter']['INIT_LR'])
	EPOCHS = config['Train']['hyperparameter']['epoch']
	BS = config['Train']['hyperparameter']['BS']
	seed = config['Train']['cv']['seed']
	k = config['Train']['cv']['k-fold']

	# construct the training image generator for data augmentation
	aug = ImageDataGenerator(
		rescale=1./255,
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

	if args['cross_val']:
		np.random.seed(config['Train']['cv']['seed'])
		kfold = StratifiedKFold(n_splits=config['Train']['cv']['k-fold'], shuffle=True, random_state=config['Train']['cv']['seed'])

	# Train the head of the network
	print("[INFO] Training model...")
	model_name = os.path.join(config['Train']['model_dir'], f"mask_detector_{config['Train']['model'].lower()}.h5")
	callbacks = [ModelCheckpoint(model_name.replace(".h5", "_tmp.h5"), monitor="val_loss", save_best_only=True), 
				 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=0.0001, cooldown=0, min_lr=0)]

	if args['cross_val']:
		count = 0
		cvscores = []
		for _, val in kfold.split(imagePaths, np.zeros(len(imagePaths))):
			print("[INFO] Executing {}-fold validation...".format(count+1))
			val_path = np.array(imagePaths)[val]
			model = eval(f"{config['Train']['model']}(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))")
			model = model_build(model, lr=INIT_LR, epochs=EPOCHS)
			make_val(dataset_dir=config['Data']['dataset_dir'], val_path=val_path)
			train_dir = os.path.join(config['Data']['dataset_dir'], "train")
			val_dir = os.path.join(config['Data']['dataset_dir'], "val")
			train_gen = aug.flow_from_directory(train_dir, target_size=(224, 224), batch_size=BS, class_mode='categorical')
			val_gen = aug.flow_from_directory(val_dir, target_size=(224, 224), batch_size=BS, class_mode='categorical')
			
			H = model.fit(train_gen,
				steps_per_epoch=(train_gen.samples // BS)+10, # The 10 more batches of images is for argumentaion images
				validation_data=val_gen,
				validation_steps=val_gen.samples // BS,
				epochs=EPOCHS,
				callbacks=callbacks)

			model = load_model(model_name.replace(".h5", "_tmp.h5"))
			scores = model.evaluate(val_gen, verbose=0)
			print("[Fold-%d] %s: %.2f%%" % (count+1, model.metrics_names[1], scores[1] * 100))
			if cvscores:
				if scores[1] * 100 > max(cvscores):
					print(f"[INFO] Accuracy improved from {max(cvscores)} to {scores[1] * 100}, save mask detector model to {model_name}...")
					model.save(model_name, save_format="h5")
			else:
				print(f"[INFO] First fold with the accuracy of {scores[1] * 100}, save mask detector model to {model_name}...")
				model.save(model_name, save_format="h5")
				plot_metric(H, EPOCHS, config['Train']['plot'])
			cvscores.append(scores[1] * 100)
			count += 1
			for path in val_path:
				shutil.move(os.path.join(val_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)), path)
			print("-"*50)
		print("Summary: acc: %.2f%%, best_acc: %.2f%%, std:(+/- %.2f%%)" % (np.mean(cvscores), np.max(cvscores), np.std(cvscores)))
		print("-"*50)
		print()
	else:
		model = eval(f"{config['Train']['model']}(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))")
		model = model_build(model, lr=INIT_LR, epochs=EPOCHS)
		test_size = int(round(len(imagePaths)*config['Train']['val_split']))
		_, val_path = train_test_split(imagePaths, test_size=test_size)
		make_val(dataset_dir=config['Data']['dataset_dir'], val_path=val_path)
		train_dir = os.path.join(config['Data']['dataset_dir'], "train")
		val_dir = os.path.join(config['Data']['dataset_dir'], "val")
		train_gen = aug.flow_from_directory(train_dir, target_size=(224, 224), batch_size=BS, class_mode='categorical')
		val_gen = aug.flow_from_directory(val_dir, target_size=(224, 224), batch_size=BS, class_mode='categorical')
		H = model.fit(train_gen,
			steps_per_epoch=(train_gen.samples // BS)+10,
			validation_data=val_gen,
			validation_steps=val_gen.samples // BS,
			epochs=EPOCHS,
			callbacks=callbacks)
		plot_metric(H, EPOCHS, config['Train']['plot'])
		for path in val_path:
			shutil.move(os.path.join(val_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)), path)
		print("[INFO] saving mask detector model...")
		model.save(model_name, save_format="h5")
	os.remove(model_name.replace(".h5", "_tmp.h5"))
	shutil.rmtree(val_dir)

	# Evaluation of train and test datasets
	print("[INFO] Evaluation testing data")
	test_dir = os.path.join(config['Data']['dataset_dir'], "test")
	test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=(224, 224), batch_size=BS, class_mode='categorical', shuffle=False)
	
	train_loss, train_acc = model.evaluate(train_gen)
	test_loss, test_acc = model.evaluate(test_gen)
	print(f'train acc: {train_acc*100:.2f}%, train loss: {train_loss:.4f}')
	print(f'test acc: {test_acc*100:.2f}%, test loss: {test_loss:.4f}')
