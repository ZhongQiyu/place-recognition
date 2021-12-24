# 0. CIFAR10(0) Demo ~ from Google Colab

# credit the Colab tutorial of CIFAR10:
# https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb

# 1. pre-processing

# download the CIFAR100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# display the size of each image in the dataset
# using the first image as an example

height, width, channels = x_train[0].shape
print(f"The height of the image is {height} pixels.")
print(f"The width of the image is {width} pixels.")
print(f"There are {channels} channels in each of the image at the dataset.")

# print the number of data points in the training set and the test set
train_count = len(x_train)  # or len(y_train)
test_count = len(x_test)  # or len(y_test)
print(f"There are {train_count} data points in the training set.")
print(f"There are {test_count} data points in the test set.")

# define the labels
CLASSES = set(y_train.flatten())  # or set(y_test.flatten())
CLASS_COUNT = len(CLASSES)
LABELS = list(CLASSES)
print(f"The labels in the dataset are ranging from {min(LABELS)} to {max(LABELS)}.")

# plot the first 25 instances in the dataset
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x_train[i][:])
  plt.xlabel(LABELS[y_train.flatten()[i]])
plt.show()

# normalize the data
# ImageDataGenerator()

# 2. set up the model: CIFAR100 Model - to apply onto the local image dset

# use a Sequential model 
model = Sequential()

# params
filter_counts = 32
conv_filter_size = 3
conv_1_filters = filter_counts * 3
conv_2_filters = filter_counts * 4
conv_3_filters = filter_counts * 5
pool_filter_size = 2
dropout_rate = 0.1

# use an input layer
model.add(Input(shape=(height, width, channels)))

# add convolution and maxpooling layers for compression
model.add(Conv2D(conv_1_filters, conv_filter_size, activation="relu"))
model.add(MaxPooling2D(pool_filter_size))

model.add(Conv2D(conv_2_filters, conv_filter_size, activation="relu"))
model.add(MaxPooling2D(pool_filter_size))

model.add(Conv2D(conv_3_filters, conv_filter_size, activation="relu"))
model.add(MaxPooling2D(pool_filter_size))

# flatten the output from the network
model.add(Flatten())

# add a fully-connected layer for learning non-linearity
# follow the flatten layer, cut down half of the input units to the FC layer
dense_output = model.output_shape[1] // 2
model.add(Dense(dense_output, activation="elu"))

# add a dropout layer for avoiding overfitting
model.add(Dropout(dropout_rate, input_shape=(height, width, channels)))

# add the output layer with softmax activation
model.add(Dense(CLASS_COUNT, activation="softmax"))

# summarize the model
model.summary()

# 3. compile the model

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. train the model

# *How to press the UserWarning? Is the warning concerning?
# - related to loss's from_logits=True; did I misunderstand
#   something about logits and output?

# (1) fit the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# (2) interpret the training process

# show basic information of the training process
params = history.params

for param, value in params.items():
  print(f"The parameter {param} has a value of {value} in this model.")

metrics = history.history.keys()
metric_count = len(metrics)
print(f"There are {metric_count} metrics that the history evaluates: " + 
      f"{list(metrics)}.")
epochs = params["epochs"]

print(f"Among the parameters, we have {epochs} epochs to train the model.")

# plot a training plot

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Training Process Overview")

ax1.plot({"data": [history.history["accuracy"], "label": "Accuracy (training)"})
plt.plot(history.history["val_accuracy"], label="Accuracy (testing)")
plt.title(f"Training Accuracy over {epochs} Epochs")
plt.xlabel("# of epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.show()

# 5. test the model

# make predictions
y_preds = model.predict(x_test)

# format the predictions to compute accuracy
y_max_preds = np.array([np.where(y_pred == max(y_pred))[0][0] for y_pred in y_preds])

# format the ground truth and compare with the predictions
accuracy = 0

for count in range(test_count):
  if y_max_preds[count] == y_test.flatten()[count]:
    accuracy += 1/test_count

print(f"The accuracy of the predictions is {accuracy * 100:.2f}%.")

# 6. visualize the results

# xlabel()
# ylabel()
print(y_max_preds)

# print the max number of each raw predictions,
# as the real prediction for that instance

# change to plt.plot()
