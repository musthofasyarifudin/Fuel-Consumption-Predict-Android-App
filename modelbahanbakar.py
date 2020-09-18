#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_excel('measurements2.xlsx')
ad = df[['distance', 'consume']]
td=df[['distance']]
td = td.values
ts = df[['consume']]


# In[3]:


train = ad[:300]
test = ad[300:]


# In[4]:


import seaborn as sns
sns.pairplot(df[["distance", "consume"]], diag_kind="kde")


# In[5]:


train_stats = train.describe()
train_stats.pop("consume")
train_stats = train_stats.transpose()
train_stats


# In[6]:


train_labels = train.pop('consume')
test_labels = test.pop('consume')


# In[7]:


y = train.mean()
y


# In[8]:


def norm(x):
  return (x - train.mean()) / train.std()
normed_train_data = norm(train)
normed_test_data = norm(test)


# In[9]:


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[10]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# In[11]:


model = build_model()


# In[12]:


model.summary()


# In[13]:


example_batch = normed_train_data[:10]
example_batch = example_batch.values
example_result = model.predict(example_batch)
example_result


# In[14]:


print (train_labels)


# In[15]:


train


# In[18]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 10000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# In[19]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[20]:


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)


# In[56]:


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# In[21]:


normed_test_data = normed_test_data.values


# In[22]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# In[23]:


test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[24]:


error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[104]:


test_predictions = model.predict([4.5]).flatten()
test_predictions


# In[76]:


ad[:300]


# In[86]:


def modelnew():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[len(train.keys())]),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
    
    return model
    


# In[87]:


model2 = modelnew()


# In[88]:


model2.summary()


# In[97]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
model2 = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[len(train.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(1, activation='relu')])
model2.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
EPOCHS = 10000

history = model2.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[PrintDot()])


# In[96]:


plot_history(history)


# In[ ]:





# In[99]:


plot_history(history)


# In[110]:


test_predictions2 = model2.predict([12])
test_predictions2


# In[111]:


tr = ad[['distance']]
tr = tr.values
t = []
for x in tr:
    y = model2.predict([x]).flatten()
    t.append(y)


# In[115]:


plt.plot(t)
plt.plot(tr)


# In[116]:


tr = ad[['distance']]
tr = tr[:100]
tr = tr.values
t = []
for x in tr:
    y = model2.predict([x]).flatten()
    t.append(y)
plt.plot(t)
plt.plot(tr)


# In[117]:


loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} consume".format(mae))


# In[118]:


plt.scatter(t, tr)


# In[120]:


from keras.models import load_model

model2.save('consumtionmodel.h5')  # creates a HDF5 file 'my_model.h5'


# In[124]:


converter = tf.lite.TFLiteConverter.from_keras_model_file('consumtionmodel.h5')
tflite_model = converter.convert()
open("cumsumetionmodel.tflite", "wb").write(tflite_model)


# In[127]:





# In[ ]:





# In[ ]:




