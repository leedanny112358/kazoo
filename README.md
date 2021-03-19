# Kazoo

Kazoo is a simple piano synthesizer that utilizes a convolutional autoencoder to synthesize piano notes and chords on a sample-by-sample basis

## Installation

Clone this repo to your local machine

```bash
git clone https://github.com/leedanny112358/kazoo.git
```

Install all the dependencies needed for Kazoo (using a virtual env is highly recommended)

```bash
pip3 install -r requirements.txt
```

## Usage

Currently, there are 4 experiments that can be run with Kazoo. When you run the kazoo.py file, you will get the numbers that correspond to each experimentation. The program will prompt you to input an experiment number.

```bash
python3 ./kazoo.py
0 - vanilla note generation

1 - train noise / test clean

2 - train clean / test noise

3 - train notes / test chords

4 - train chords / test notes

Enter experiment number:
```

More details on what each experiment does can be found in the writeup pdf.

## Contact

For any questions, please contact me at lee.danny112358@gmail.com
