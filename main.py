import cv2
from musicnn.extractor import extractor
from musicnn import configuration
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf

AUDIO_FILE = r'songs\pop.00000.wav'
# AUDIO_FILE = r'songs\metal.00000.wav'
# AUDIO_FILE = r'songs\classical.00012.wav'

MUSICNN_MODEL = 'MSD_musicnn'
# MUSICNN_MODEL = 'MSD_vgg'

MUSICNN_INPUT_LENGTH = 1.834 * 2

# Possible values: mean_pool, max_pool, penultimate, taggram
FEATURE_NAME = 'mean_pool'
FEATURE_THRESHOLD = 0.1

DEEP_DREAM_MODEL = 'inception5h/tensorflow_inception_graph.pb'

IMAGENET_MEAN = 117.0
LAYER_NAMES = ['mixed3a', 'mixed4a', 'mixed4e', 'mixed5b']
LAYER_WEIGHTS = [3, 4, 2, 1]
LEARNING_RATE = 2
START_IMAGE_SIZE = 8
MAX_IMAGE_SIZE = 256
OCTAVE_SCALE = 2
ITERATION_COUNT = 5
MIX_RNG_SEED = 1
OUTPUT_IMAGE_SIZE = 1024

OUTPUT_DIR = 'output'

GAMMA = 0.9

configuration.SR = 64000
configuration.N_MELS = 64

fps = None

def make_frames():
    taggram, tags, feature_map = extractor(AUDIO_FILE, model=MUSICNN_MODEL, input_length=MUSICNN_INPUT_LENGTH)
    print(f'Musicnn features: {feature_map.keys()}')
    feature_map['taggram'] = taggram

    song_features = feature_map[FEATURE_NAME]

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with tf.gfile.FastGFile(DEEP_DREAM_MODEL, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # define input
    X = tf.placeholder(tf.float32, name="input")
    X2 = tf.expand_dims(X - IMAGENET_MEAN, 0)
    tf.import_graph_def(graph_def, {"input": X2})

    losses = []
    targets = []
    layers = []
    num_features = 0
    for layer_name in LAYER_NAMES:
        layer = graph.get_tensor_by_name("import/%s:0" % layer_name)
        layers.append(layer)
        num_features += int(layer.shape[-1])
        print(f'Layer {layer_name}, shape {layer.shape}')
        target = tf.placeholder(tf.float32, name="target")
        targets.append(target)
        # loss = tf.reduce_mean(tf.sqrt(tf.square(layer - target)))
        loss = tf.reduce_mean(layer * target)
        losses.append(loss)

    loss = losses[0] * LAYER_WEIGHTS[0]
    for i in range(1, len(losses)):
        loss = loss + losses[i] * LAYER_WEIGHTS[i]

    gradient = tf.gradients(loss, X)[0]

    def make_frame(image):
        frame = cv2.resize(image, (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC) / 255
        frame = np.clip(frame, 0, 1)
        frame = np.power(frame, GAMMA) * 255
        return frame

    image_sizes = [START_IMAGE_SIZE]
    while True:
        s = int(image_sizes[-1] * OCTAVE_SCALE)
        if s > MAX_IMAGE_SIZE:
            break
        image_sizes.append(s)

    image_size = START_IMAGE_SIZE
    image = np.full((image_size, image_size, 3), IMAGENET_MEAN, dtype=np.float32)
    frame_num = 0
    for fi in range(len(song_features)):
        target_values = []
        features = song_features[fi]
        scale = int(num_features / len(features) * 4)
        scale = scale if scale % 2 else scale + 1
        features = cv2.resize(np.tile(features, (scale, 1)), (num_features, scale),
                              interpolation=cv2.INTER_LINEAR)[scale // 2]
        features = (features > FEATURE_THRESHOLD).astype(np.float32)
        print(f'Non-zero features {features.sum() / len(features) * 100:0.1f}%')
        np.random.RandomState(MIX_RNG_SEED).shuffle(features)
        start = 0
        for l in range(len(layers)):
            layer = layers[l]
            target_size = int(layer.shape[3])
            t = features[start:start+target_size]
            start += target_size
            target_values.append(t)

        for si in range(len(image_sizes)):
            # l = sess.run(layer, {X: image})
            # print(f'size {image.shape} l shape {l.shape} l range {l.min()} {l.max()}')

            for batch in range(ITERATION_COUNT):
                args = {X: image}
                for t in range(len(targets)):
                    args[targets[t]] = target_values[t]
                g = sess.run(gradient, args)
                lr = LEARNING_RATE * (len(image_sizes) - si)
                image += lr * g / (np.abs(g).mean() + 1e-7)
                image = (image - image.min()) / (image.max() - image.min()) * 255
                frame = make_frame(image)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f'f-{frame_num:05d}.png'), frame)
                frame_num += 1
                cv2.imshow(f'image', frame / 255)
                cv2.waitKey(1)
            if si < len(image_sizes) - 1:
                image = cv2.resize(image, (image_sizes[si + 1], image_sizes[si + 1]), interpolation=cv2.INTER_CUBIC)

        # Keep this frame for a while
        for k in range(5):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'f-{frame_num:05d}.png'), frame)
            frame_num += 1

        downscaled = image
        for s in image_sizes[-2::-1]:
            downscaled = cv2.resize(downscaled, (s, s), interpolation=cv2.INTER_CUBIC)
            frame = make_frame(downscaled)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'f-{frame_num:05d}.png'), frame)
            frame_num += 1
            cv2.imshow(f'image', frame / 255)
            cv2.waitKey(100)

        global fps
        if fps is None:
            fps = int(np.round(frame_num / MUSICNN_INPUT_LENGTH))

        image = cv2.resize(image, (image_sizes[0], image_sizes[0]), interpolation=cv2.INTER_CUBIC)


def make_movie():
    subprocess.run(
        [
            'ffmpeg', '-y',
            '-pix_fmt', 'yuv420p',
            '-framerate', f'{fps}',
            '-start_number', '0',
            '-i', r'output\f-%05d.png',
            '-i', AUDIO_FILE,
            '-c:v', 'libx264',
            '-r', f'{fps}',
            os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(AUDIO_FILE))[0] + '.mp4')
        ]
    )

def run():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_frames()
    make_movie()


if __name__ == "__main__":
    run()