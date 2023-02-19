# import dependencies
import keras.layers
import tensorflow as tf
import numpy as np

# create callbacks concept
class Callbacks(tf.keras.callbacks.Callback): # inherits from base class
    def stop_training_by_iteration_loss(self,log=dict()):
        if log.get("loss") <= 0.3:
            print("obtained loss below threshold, cancelling training!!!")
            self.model.stop_training = True

# instantiate the callbacks
callbacks = Callbacks()

# define data
xs_data = np.array([1, 2, 4, 6, 8, -1, 3, 4,], dtype=float)
ys_data = np.array([1, 3, 7, 11, 15, -3, 5, 7], dtype=float)

# create model
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

# fit data and train model
model.fit(xs_data, ys_data, epochs=100, callbacks=[callbacks])

# predict using unseen data
print(type(model.predict([11.0])))





