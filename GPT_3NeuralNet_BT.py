import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


matches = pd.read_csv("EPL2022_16.csv") # index is already in the data
#new_games=pd.read_csv("new_games.csv")

#new_matches=new_games

matches=matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})

#new_matches=new_matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})


"""
#Unique code for each opponent
matches["HomeTeam"] = matches["HomeTeam"].astype("category").cat.codes
matches["AwayTeam"] = matches["AwayTeam"].astype("category").cat.codes

#new_matches["HomeTeam"] = new_matches["HomeTeam"].astype("category").cat.codes
#new_matches["AwayTeam"] = new_matches["AwayTeam"].astype("category").cat.codes

"""

data = matches


#all column names

cols5 = ['FTR','B365H','B365D','B365A']


data = data[cols5].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#data = data[cols5]

#data = data[cols5]


data = data.dropna(axis=0)


matches = data


X_train,X_test,y_train,y_test = train_test_split(matches,matches["FTR"],test_size=0.15,shuffle=True)

X_train = X_train.drop("FTR",axis=1)
X_test = X_test.drop("FTR",axis=1)


# Building the neural network

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='elu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='elu'))
model.add(tf.keras.layers.Dense(64, activation='elu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# Compiling the model
model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=30, batch_size=250)

# Evaluating the model
score = model.evaluate(X_test, y_test, batch_size=250)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

#Making predictions

y_preds = model.predict(X_test)

#Unsquishing results

true_test_result = y_test*2

#Unsquishing preds

true_preds = np.round(y_preds*2)

#make combinable

true_preds = true_preds.flatten()

#check accuracy

combined = pd.DataFrame(dict(actual=true_test_result, GPT_3=true_preds))

print(pd.crosstab(index=combined["actual"], columns=combined["GPT_3"]))

#merge to make readable

merged = pd.concat([combined, X_test], axis=1)

merged = merged.drop('actual', axis=1)

print(merged)


#64 & 65% repsectively for 0 and 2, 1's has accuracy of 27.5%