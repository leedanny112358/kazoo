import numpy as np
import os
import librosa, librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.preprocessing import image

training_dataset = {
  "spectrogram": [],
  "note": []
}
testing_dataset = {
  "spectrogram": [],
  "note": []
}

# loop through all data and preprocess
def process_data(path,dataset):
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
    if dirpath != path:
      print("Processing {} notes".format(dirpath))
      for file in filenames:
        if file != ".DS_Store":
          file_path = os.path.join(dirpath,file)
          signal,sr = librosa.load(file_path,sr=22050)
          stft = librosa.core.stft(signal[0:45050], n_fft=2047, hop_length=512)
          spectrogram = np.abs(stft)
          row, col = spectrogram.shape
          if col == 88:
            if dataset == "training":
              training_dataset["spectrogram"].append(spectrogram)   
              training_dataset["note"].append(file)
            if dataset == "testing":
              testing_dataset["spectrogram"].append(spectrogram)
              testing_dataset["note"].append(file)

def pre_process(X):
  norm_data = []
  for data in X:
    max_value = np.amax(data)
    norm_data.append(data/max_value)
  X = X.reshape((len(X), 1024, 88, 1))
  return X

def add_noise(specs):
  signal,sr = librosa.load("./noise.wav",sr=22050)
  noise = librosa.core.stft(signal[0:45050], n_fft=2047, hop_length=512)
  specs += noise
  return specs

# NOTE: Please refer to the following blocks of code to test out the different experimentations
print("0 - vanilla note generation\n")
print("1 - train noise / test clean\n")
print("2 - train clean / test noise\n")
print("3 - train notes / test chords\n")
print("4 - train chords / test notes\n")
experiment = int(input("Enter experiment number: "))

# vanilla notes
if experiment == 0:
  process_data("./dataset/experiment012","training")
  X_train  =  pre_process(np.array(training_dataset["spectrogram"][100:1000]))
  X_test  =  pre_process(np.array(training_dataset["spectrogram"][0:100]))
# train noise / test clean
if experiment ==  1:
  process_data("./dataset/experiment012","training")
  X_train = pre_process(np.array(add_noise(training_dataset["spectrogram"][100:1000])))
  X_test = pre_process(np.array(training_dataset["spectrogram"][0:100]))
# trained clean / test noise
if experiment == 2:
  process_data("./dataset/experiment012","training")
  X_train = pre_process(np.array(training_dataset["spectrogram"][100:1000]))
  X_test = pre_process(np.array(add_noise(training_dataset["spectrogram"][0:100])))
# trained notes / test chords
if experiment == 3:
  process_data("./dataset/experiment3_notes","training")
  X_train = pre_process(np.array(training_dataset["spectrogram"]))
  process_data("./dataset/experiment3_chords","testing")
  X_test  = pre_process(np.array(testing_dataset["spectrogram"]))
# trained chords /test notes
if experiment == 4:
  process_data("./dataset/experiment4_chords","training")
  X_train = pre_process(np.array(training_dataset["spectrogram"]))
  process_data("./dataset/experiment4_notes","testing")
  X_test  = pre_process(np.array(testing_dataset["spectrogram"]))

def show_data(X, n=10, height=88, width=1024, title=""):
    plt.figure(figsize=(5, 5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)

input_layer = Input(shape=(1024, 88, 1), name="INPUT")
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

code_layer = MaxPooling2D((2, 2), name="CODE")(x)

x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(code_layer)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(1, (3, 3), padding='same', name="OUTPUT")(x)

AE = Model(input_layer, output_layer)
AE.compile(optimizer='adam', loss='mse')
AE.summary()

AE.fit(X_train, X_train,
                epochs=15,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test))

AE.save("kazoo.h5")
decoded_data = AE.predict(X_test)

for i in range(10): #get 10 input/output pairs to show as examples for project
  synth_sound = decoded_data[i].reshape(1024,88)
  original_sound = X_test[i].reshape(1024,88)
  log_synth_sound = librosa.amplitude_to_db(synth_sound)
  log_original_sound = librosa.amplitude_to_db(original_sound)
  librosa.display.specshow(log_synth_sound,sr=22050,hop_length=512)
  if experiment <= 2:
    title = "Generated Spec for {}".format(training_dataset["note"][i])
  else:
    title = "Generated Spec for {}".format(testing_dataset["note"][i])
  plt.suptitle(title, fontsize = 20)
  plt.xlabel("time")
  plt.ylabel("frequency")
  plt.show()
  librosa.display.specshow(log_original_sound,sr=22050,hop_length=512)
  if experiment <= 2:
    title = "Original Spec for {}".format(training_dataset["note"][i])
  else:
    title = "Original Spec for {}".format(testing_dataset["note"][i])
  plt.suptitle(title, fontsize = 20)
  plt.xlabel("time")
  plt.ylabel("frequency")
  plt.show()
  audio = librosa.istft(synth_sound, hop_length=512)
  if experiment <=2:
    name = "synth_{}".format(training_dataset["note"][i])
  else:
    name = "synth_{}".format(testing_dataset["note"][i])
  sf.write(name,audio,22050)
