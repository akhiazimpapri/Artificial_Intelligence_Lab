from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input((1,))
h1 = Dense(4, activation = 'relu')(inputs)
h2 = Dense(3, activation = 'relu')(h1)
outputs = Dense(1, activation = 'softmax')(h2)
model = Model(inputs, outputs)
model.summary(show_trainable=True)
