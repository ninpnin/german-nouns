import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load preprocessed data
train_data = pickle.load(open("data/train.pkl", "rb"))
chars = set()

for wd, gender in train_data:
    print(wd, gender)

    for char in wd:
        chars.add(char)

chars = list(chars)
chars_map = {}
for ix, char in enumerate(list(chars)):
    chars_map[char] = ix
print(chars_map)

# Hyperparameters
gender_map = dict(masculine=0, feminine=1, neuter=2)
MAXLEN = 8
C = len(chars)

K = MAXLEN # How many characters we count
hidden_1 = 150
hidden_23 = 25
G = len(gender_map)

def preprocess_x(x):
    x = x[-MAXLEN:]
    x = [chars_map[c] for c in x]
    return x

def preprocess(x, y):
    x = preprocess_x(x)
    y = gender_map[y]

    print(x)
    print(y)
    return x, y

def create_dataset(train_data):
    xys = [preprocess(x, y) for x,y in train_data]
    #print(xys)
    xs = [x for x, _ in xys]
    ys = [y for _, y in xys]
    print("Moi")

    #xs = np.array(xs)
    #ys = np.array(ys)
    print("xs:", xs)
    print("ys:", ys)

    xs = tf.one_hot(xs, depth=C)
    ys = tf.one_hot(ys, depth=G)

    print("xs:", xs.shape)
    print("ys:", ys.shape)

    return tf.data.Dataset.from_tensor_slices((xs, ys)).batch(128)
# Naive Neural Model

train_xys = create_dataset(train_data)

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(C * K,), input_shape=(C, K)),
    keras.layers.Dense(units=hidden_1, activation='relu'),
    keras.layers.Dense(units=hidden_23, activation='relu'),
    keras.layers.Dense(units=hidden_23, activation='relu'),
    keras.layers.Dense(units=G, activation='softmax')
])

model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

valid_data = pickle.load(open("data/valid.pkl", "rb"))
valid_xys = create_dataset(valid_data)

history = model.fit(
    train_xys.repeat(), 
    epochs=10, 
    steps_per_epoch=500,
    validation_data=valid_xys.repeat(), 
    validation_steps=2
)

dummy_word = "alkierung"
dummy_x = preprocess_x(dummy_word)
predictions = model.predict(valid_xys)

print(valid_data)
print(predictions)


test_data = pickle.load(open("data/test.pkl", "rb"))
test_xys = create_dataset(test_data)
predictions = model.predict(test_xys)

np.set_printoptions(formatter={'float':           "{0:0.8f}".format})

for ix, (wd, gd) in enumerate(test_data):
    print(predictions.shape)
    print(wd, gd, predictions[ix])

print(predictions.shape)

results = model.evaluate(test_xys, batch_size=128)

print(results)


