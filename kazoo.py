import numpy as np
import os
import json
import librosa, librosa.display
import soundfile as sf
from soundfile import SoundFile
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing import image
import keras.losses

dataset = {
    "piano": [],
    "spectrogram": []
  }


# loop through all data and preprocess
def process_data(path):
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
    if dirpath != path:
      print("Processing {} notes".format(dirpath))
      dataset["piano"].append(dirpath.split("/")[-1])
      for file in filenames:
        print("Processing file {} ".format(file))
        file_path = os.path.join(dirpath,file)
        signal,sr = librosa.load(file_path,sr=22050)
        stft = librosa.core.stft(signal[0:sr*2], n_fft=2048, hop_length=512)
        spectrogram = np.abs(stft)
        dataset["spectrogram"].append(spectrogram)   
  
  #with open("dataset.json", "w") as fp:
    #json.dump(dataset, fp, indent=2)

process_data("./dataset")
#with open("./dataset.json", 'r') as j:
  #process_data = json.loads(j.read())

def pre_process(X):
  X = X/200.0
  print(len(X))
  X = X.reshape((len(X), 89175))
  return X
print(np.array(dataset["spectrogram"]).shape)
X_train  =  pre_process(np.array(dataset["spectrogram"][100:1058]))
X_test  =  pre_process(np.array(dataset["spectrogram"][0:100]))
print(X_train.shape)

def show_data(X, n=10, height=87, width=1025, title=""):
    plt.figure(figsize=(5, 5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)

#show_data(X_train, title="train data")
#show_data(X_test, title="test data")

input_dim, output_dim = 89175, 89175
encode_dim = 100
hidden_dim = 256

# encoder
input_layer = Input(shape=(input_dim,), name="INPUT")
hidden_layer_1 = Dense(hidden_dim, activation='relu', name="HIDDEN_1")(input_layer)

# code
code_layer = Dense(encode_dim, activation='relu', name="CODE")(hidden_layer_1)

# decoder
hidden_layer_2 = Dense(hidden_dim, activation='relu', name="HIDDEN_2")(code_layer)
output_layer = Dense(output_dim, activation='sigmoid', name="OUTPUT")(hidden_layer_2)

AE = Model(input_layer, output_layer)
AE.compile(optimizer='adam', loss='binary_crossentropy')
AE.summary()

AE.fit(X_train, X_train, epochs=10)

decoded_data = AE.predict(X_test)

#show_data(X_test, title="original data")

#show_data(decoded_data, title="decoded data")
#plt.show()
for i in range(88):
  sound = decoded_data[i].reshape(1025,87)
  sound = sound * 255.0
  log_sound = librosa.amplitude_to_db(sound)
  librosa.display.specshow(log_sound,sr=22050,hop_length=512)
  plt.show()
  audio = librosa.istft(sound, hop_length=512)
  name = "audio_{}.wav".format(i)
  sf.write(name,audio,22050)
