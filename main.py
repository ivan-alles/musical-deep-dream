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

MUSICNN_INPUT_LENGTH = 1.8

# Possible values: mean_pool, max_pool, penultimate, taggram
FEATURE_NAME = 'mean_pool'
FEATURE_THRESHOLD = 0.1

DEEP_DREAM_MODEL = 'inception5h/tensorflow_inception_graph.pb'

IMAGENET_MEAN = 117.0
LAYER_NAMES = ['mixed3a', 'mixed4a', 'mixed4e', 'mixed5b']
LAYER_WEIGHTS = [3, 4, 2, 1]
LEARNING_RATE = 2
MAX_IMAGE_SIZE = 256
# (size, iterations, learning_rate)
OCTAVE_PARAMS = [
    (8, 2, 10),
    (11, 2, 9),
    (16, 2, 8),
    (23, 2, 7),
    (32, 3, 5),
    (45, 3, 4),
    (64, 3, 3),
    (91, 4, 2),
    (128, 6, 2),
    (181, 8, 2),
    (256, 10, 2),
    (362, 16, 2)
]

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

    image = np.full((OCTAVE_PARAMS[0][0], OCTAVE_PARAMS[0][0], 3), IMAGENET_MEAN, dtype=np.float32)
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

        for oi in range(len(OCTAVE_PARAMS)):
            # l = sess.run(layer, {X: image})
            # print(f'size {image.shape} l shape {l.shape} l range {l.min()} {l.max()}')

            for batch in range(OCTAVE_PARAMS[oi][1]):
                args = {X: image}
                for t in range(len(targets)):
                    args[targets[t]] = target_values[t]
                g = sess.run(gradient, args)
                lr = OCTAVE_PARAMS[oi][2]
                image += lr * g / (np.abs(g).mean() + 1e-7)
                image = (image - image.min()) / (image.max() - image.min()) * 255
                frame = make_frame(image)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f'f-{frame_num:05d}.png'), frame)
                frame_num += 1
                cv2.imshow(f'image', frame / 255)
                cv2.waitKey(1)
            if oi < len(OCTAVE_PARAMS) - 1:
                image = cv2.resize(image, (OCTAVE_PARAMS[oi + 1][0], OCTAVE_PARAMS[oi + 1][0]), interpolation=cv2.INTER_CUBIC)

        downscaled = image
        for oi in range(len(OCTAVE_PARAMS) - 2, -1, -1):
            s = OCTAVE_PARAMS[oi][0]
            downscaled = cv2.resize(downscaled, (s, s), interpolation=cv2.INTER_CUBIC)
            frame = make_frame(downscaled)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'f-{frame_num:05d}.png'), frame)
            frame_num += 1
            cv2.imshow(f'image', frame / 255)
            cv2.waitKey(100)

        global fps
        if fps is None:
            fps = int(np.round(frame_num / MUSICNN_INPUT_LENGTH))

        image = cv2.resize(image, (OCTAVE_PARAMS[0][0], OCTAVE_PARAMS[0][0]), interpolation=cv2.INTER_CUBIC)


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