import cv2
import numpy as np
import os
import shutil
import subprocess
import sys
import tensorflow as tf

sys.path.append('musicnn')
from musicnn.extractor import extractor
from musicnn import configuration


MUSICNN_MODEL = 'MSD_musicnn'
# MUSICNN_MODEL = 'MSD_vgg'

# This shall be synchronized with the beats.
MUSICNN_INPUT_LENGTH = 2.416

# Possible values: mean_pool, max_pool, penultimate, taggram
FEATURE_NAME = 'max_pool'
FEATURE_THRESHOLD = 1.5

DEEP_DREAM_MODEL = 'inception5h/tensorflow_inception_graph.pb'

LAYER_NAMES = ['mixed3a', 'mixed4a', 'mixed4e', 'mixed5b']
LAYER_WEIGHTS = [2, 0.7, 0.5, 0.4]
MIX_RNG_SEED = 1

# (size, iterations, learning_rate)
OCTAVE_PARAMS = [
    (8, 3, 10),
    (11, 3, 9),
    (16, 2, 8),
    (23, 2, 7),
    (32, 3, 5),
    (45, 3, 4),
    (64, 3, 3),
    (91, 4, 2),
    (128, 6, 2),
    (181, 8, 2),
    (256, 8, 2),
    (362, 12, 2),
    (512, 16, 2),
]

OUTPUT_IMAGE_SIZE = 1024

OUTPUT_DIR = 'output'
TEMP_DIR = '.temp'

GAMMA = 0.9

configuration.SR = 64000
configuration.N_MELS = 64

# This is a constant, not a configurable parameter.
IMAGENET_MEAN = 117.0

mix_rng = np.random.RandomState(MIX_RNG_SEED)


fps = None

def make_frames(audio_file):
    taggram, tags, feature_map = extractor(audio_file, model=MUSICNN_MODEL, input_length=MUSICNN_INPUT_LENGTH)
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
        mix_rng.shuffle(features)
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
                frame = make_frame(image)
                cv2.imwrite(os.path.join(TEMP_DIR, f'f-{frame_num:05d}.png'), frame)
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
            cv2.imwrite(os.path.join(TEMP_DIR, f'f-{frame_num:05d}.png'), frame)
            frame_num += 1
            cv2.imshow(f'image', frame / 255)
            cv2.waitKey(100)

        global fps
        if fps is None:
            fps = int(np.round(frame_num / MUSICNN_INPUT_LENGTH))

        image = cv2.resize(image, (OCTAVE_PARAMS[0][0], OCTAVE_PARAMS[0][0]), interpolation=cv2.INTER_CUBIC)
        image = (image - image.min()) / (image.max() - image.min()) * 255


def make_movie(audio_file):
    for i in range(1000):
        filename = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(audio_file))[0] + f'-{i:003d}.mp4')
        if not os.path.exists(filename):
            break
    subprocess.run(
        [
            'ffmpeg', '-y',
            '-pix_fmt', 'yuv420p',
            '-framerate', f'{fps}',
            '-start_number', '0',
            '-i', fr'{TEMP_DIR}\f-%05d.png',
            '-i', audio_file,
            '-c:v', 'libx264',
            '-r', f'{fps}',
            filename
        ]
    )

def run(audio_file):
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_frames(audio_file)
    make_movie(audio_file)


if __name__ == "__main__":
    run(sys.argv[1])