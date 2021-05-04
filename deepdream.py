import cv2
import numpy as np
import tensorflow as tf

imagenet_mean = 117.0
layer_names = ['mixed3a', 'mixed4a', 'mixed4e', 'mixed5b']
layer_weights = [1, 0.3, 0.2, 0.05]

rng = np.random.RandomState(1)

def deep_dream(model_path):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # define input
    X = tf.placeholder(tf.float32, name="input")
    X2 = tf.expand_dims(X - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {"input": X2})

    losses = []
    targets = []
    layers = []
    for layer_name in layer_names:
        layer = graph.get_tensor_by_name("import/%s:0" % layer_name)
        layers.append(layer)
        print(f'Layer {layer_name}, shape {layer.shape}')
        target = tf.placeholder(tf.float32, name="target")
        targets.append(target)
        loss = tf.reduce_mean(tf.sqrt(tf.square(layer - target)))
        losses.append(loss)

    loss = losses[0] * layer_weights[0]
    for i in range(1, len(losses)):
        loss = loss + losses[i] * layer_weights[i]

    gradient = tf.gradients(loss, X)[0]

    learning_rate = 2
    for epoch in range(50):
        size = 7
        # image = rng.uniform(imagenet_mean - 5, imagenet_mean + 5, (size, size, 3))
        image = np.full((size, size, 3), imagenet_mean, dtype=np.float32)

        target_values = []
        for l in range(len(layers)):
            layer = layers[l]
            target_size = int(layer.shape[3])
            if True:
                t = rng.uniform(10, 100, target_size)
            else:
                t = rng.uniform(10, 100, int(target_size * 0.1))
                t = np.hstack((t, np.full(target_size - t.shape[0], 0.00001)))
                rng.shuffle(t)
            target_values.append(t)

        # for tv in target_values:
        #     tv *= 0.999

        while size < 300:
            l = sess.run(layer, {
                X: image})
            print(f'size {image.shape} l shape {l.shape} l range {l.min()} {l.max()}')

            for batch in range(5):
                args = {X: image}
                for t in range(len(targets)):
                    args[targets[t]] = target_values[t]
                g = sess.run(gradient, args)
                image += learning_rate * g / (np.abs(g).mean() + 1e-7)
                view_image = (image - image.min()) / (image.max() - image.min())
                view_image = np.power(view_image, 1.5)
                cv2.imshow(f'result-{size}', view_image)
                cv2.waitKey(1)
            size *= 2
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    deep_dream(r'D:\ivan\projects\music-visualization-project\inception5h\tensorflow_inception_graph.pb')
