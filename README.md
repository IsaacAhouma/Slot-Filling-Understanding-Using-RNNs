# Slot-Filling-Understanding-Using-RNNs
Slot Filling Understanding Project

Keras implementation of A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding by Xiaodong Zhang and Houfeng Wang.

In this project, I build several Keras models that can be used to predict user intent and to perform semantic slot filling.

# Data

For this project, I have created a custom data set composed of flight related queries as well as weather related queries. For the flight related queries,
I used the widely available Air Travel Information System (ATIS) dataset which consists of about 5000 travel related queries. 
Each query in the dataset is associated with its intent as well as the slot (the role) of each word.

Finding a weather query dataset came to be very challenging, so I ended up building my own. I wrote about 100 different weather related queries (these can be found in data/weather_questions.txt) and wrote a script to generate random but distinct queries that followed the same patterns.
I generated this weather query dataset such that for each query I was able to keep track of its associated slots.

Once this was done, I combined both datasets and shuffled the result randomly to form the dataset used for training the models.

# Models
From the resource papers that I read and from my own understanding of NLP, I decided that using some type of Recurrent Neural Network would be the best way to obtain a powerful model able to predict intents and assign slots to each word in user queries.
So I decided to try both a bidirectional LSTM and a bidirectional GRU to see which one of the two would work best. In addition to that, I also tested models that make small changes in the parameters to find the model giving the best results. In some models, I also added max pooling layers as the last layer in the network but unlike in the resource paper, doing this actually made the performance of the model worse.
I found that overall, the GRU architecture performs better than the LSTM for thes combined tasks. I also noticed that increasing the number of hidden units of the GRU improved its performance. Not surprisingly, the more data I used for my training set,
the better the model performed.

The architecture of both type of networks is the same except for the recurrent layer which is either an LSTM layer or a GRU layer. This architecture consists of:

single input layer - embedding layer - convolutional layer - dropout layer (optional) - bidirectional rnn - maxpooling layer (optional) - two output layers (one for the intent classification task and the other for the slot filling task).
This is way more interesting to understand by looking at the images available for each model in the models folder.

In this project, performance was measured by the following metrics: accuracy (for the intent classification task) as well as precision, recall and f1-score (for the slot filling task).

The best performing model I built has an accuracy of 98.4%, a precision of 95.47, a recall value of 95.41, and an f1 score of 95.44. It uses embedding vectors of size 300, has 1000 hidden units and is trained for n_epochs using a train data of length 6000 and test data of length 1500. It doesn't use max pooling.

I built a Model class on top of the Keras Model class that allows me to easily retain information such as its parameters and performance on each metric. This helps to compare models, save and load them and to use them to make predictions on new queries without the need to retrain them.
For example, let's say we have a model named best_model. You can get a summary of the model by calling best_model.summary.
To get its accuracy, you would call best_model.accuracy, and to get its Keras model object you would call best_model.model.

Additionally, for each model that I built, I also generate an image of its networks. Everything related to all the models that I have tried can be found in the models folder.

Please see the example.py file for more examples on how to use the model class.

# Predicting on new data
Making predictions using a model is done through a Model object. When you load a model or build and train a new model, you can use the evaluate function defined in the Model class to make predictions on queries.

For example,

-best_model.evaluate("it is so hot today") would return the following:

{'intent': 'weather_intent', 'slots': [{'name': 'city', 'value': 'Missing City Value'}]}

-best_model.evaluate("it is so hot today in singapore") would return the following:

{'intent': 'weather_intent', 'slots': [{'name': 'city', 'value': 'Singapore'}]}

Note: If the model is unable to fill one of the slots, it assigns 'Missing SlotName Value' as its value, where "SlotName" is the name of the missing slot. 


