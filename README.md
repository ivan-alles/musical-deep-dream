# Musical Deep Dream

[![Video Intro](/assets/youtube_thumbnail.jpg)](https://youtu.be/45EKXZ8oSUE "Video")

This is music visualization using the deep dream approach. 
An audio file is processed by a musically motivated neural network (musicnn),
which extracts musical features from the song. 
These features are connected to layers of another neural network,
an image classifier turned upside down to create deep dream pictures.

## Setup
These instructions use Windows syntax. 

1. To run neural networks on a GPU (highly recommended), 
   install the required **[prerequisites](https://www.tensorflow.org/install/gpu)** for TensorFlow 2.
2. Get the source code into your working folder.
3. Install the dependencies: `pipenv sync`.
4. Activate the pipenv environment: `pipenv shell`.

## Run
Tweak the parameters in `main.py` and run:

`python main.py SONG.wav`

## Credits

[Musicnn](https://github.com/jordipons/musicnn) is used to exctact features from audio.

Inception v1 by Google is the deep dream model.
 


  