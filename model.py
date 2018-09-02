# Base Class for Our Recurrent Neural Network Models
from keras.models import Model as kerasModel
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Convolution1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from data_processing import n_vocab, n_slots, n_classes, convert_sentence_to_vector, convert_vector_to_sentence, \
    flight_booking_intents
from data_processing import ids2slots, ids2words, ids2intents, training_data_slots, training_data_intents, \
    training_data_queries, remove_punctuation
from data_processing import test_data_slots, test_data_intents, test_data_queries
import progressbar
import numpy as np
import json
from keras.utils.vis_utils import plot_model
import joblib


class Model:
    def __init__(self, embedding_dimension=300, dropout_parameter=0.2, bidirectional=True, rnn_type='GRU',
                 maxPooling=True, averagePooling=False, rnn_units=100,
                 name='model'):
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.dropout_parameter = dropout_parameter
        self.bidirectional = bidirectional
        self.maxPooling = maxPooling
        self.averagePooling = averagePooling
        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.model = None
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.summary = {'name': name, 'embedding_dimension': embedding_dimension,
                        'dropout_parameter': dropout_parameter,
                        'bidirectional': bidirectional, 'maxPooling': maxPooling, 'averagePooling': averagePooling,
                        'rnn_type': rnn_type, 'rnn_units': rnn_units}

    def build_model(self):
        main_input = Input(shape=(None,), dtype='int32', name='main_input')
        x = Embedding(output_dim=self.embedding_dimension, input_dim=n_vocab)(main_input)
        x = Convolution1D(64, 5, padding='same', activation='relu')(x)

        if self.dropout_parameter > 0.0:
            x = Dropout(self.dropout_parameter)(x)

        if self.rnn_type is 'GRU':
            rnn = GRU(self.rnn_units, return_sequences=True)
        elif self.rnn_type is 'LSTM':
            rnn = LSTM(self.rnn_units, return_sequences=True)
        else:
            rnn = SimpleRNN(self.rnn_units)

        if self.bidirectional:
            x = Bidirectional(rnn)(x)
        else:
            x = rnn(x)

        if self.maxPooling:
            x = MaxPooling1D(strides=1, padding='same')(x)
            print("Using MaxPooling")
        elif self.averagePooling:
            x = AveragePooling1D(strides=1, padding='same')(x)
            print("Using AveragePooling")
        slot_output = TimeDistributed(Dense(n_slots, activation='softmax'), name='slot_output')(x)
        intent_output = TimeDistributed(Dense(n_classes, activation='softmax'), name='intent_output')(x)
        model = kerasModel(inputs=[main_input], outputs=[intent_output, slot_output])

        # rmsprop is recommended for RNNs https://stats.stackexchange.com/questions/315743/rmsprop-and-adam-vs-sgd
        model.compile(optimizer='rmsprop',
                      loss={'intent_output': 'categorical_crossentropy', 'slot_output': 'categorical_crossentropy'})
        plot_model(model, 'models/' + self.name + '.png')

        self.model = model

        return

    def predict(self, sentence, is_words=True):
        assert (self.model != None)
        if is_words == False:
            sentence = convert_vector_to_sentence(sentence)
        sent = sentence.split(' ')
        word2vec = convert_sentence_to_vector(sentence)
        predicted_label, predicted_slots = self.model.predict_on_batch(word2vec)
        predicted_label = np.argmax(predicted_label, -1)[0]
        predicted_label = ids2intents[predicted_label[0]]

        predicted_slots = [(ids2slots[np.argmax(predicted_slots[0][i], -1)], sent[i]) for i in
                           range(len(predicted_slots[0]))]

        return predicted_label, predicted_slots

    def process_prediction(self, predicted_label, predicted_slots):
        if predicted_label in flight_booking_intents:
            predicted_label = 'flight_booking_intent'
        if predicted_label == 'flight_booking_intent':
            origin_b = list(filter(lambda pair: pair[0] == 'B-fromloc.city_name', predicted_slots))
            if origin_b:
                _, origin_b = origin_b[0]
            else:
                origin_b = 'Missing origin value'
            origin_i = list(filter(lambda pair: pair[0] == 'I-fromloc.city_name', predicted_slots))
            temp = ''
            if origin_i:
                for pair in origin_i:
                    temp += pair[1] + ' '
            origin_i = temp.strip()
            destination_b = list(filter(lambda pair: pair[0] == 'B-toloc.city_name', predicted_slots))
            if destination_b:
                _, destination_b = destination_b[0]
            else:
                destination_b = 'Missing destination value'
            destination_i = list(filter(lambda pair: pair[0] == 'I-toloc.city_name', predicted_slots))
            temp = ''
            if destination_i:
                for pair in destination_i:
                    temp += pair[1] + ' '
            destination_i = temp.strip()
            response = {'intent': predicted_label, 'slots': []}
            response['slots'].append(
                {'name': "origin", "value": remove_punctuation(origin_b.title() + " " + origin_i.title()).strip()})
            response['slots'].append(
                {'name': "destination",
                 "value": remove_punctuation(destination_b.title() + " " + destination_i.title()).strip()})
        elif predicted_label == 'weather_intent':
            city_b = list(filter(lambda pair: pair[0] == 'B-city_name', predicted_slots))
            temp = ''
            if city_b:
                for pair in city_b:
                    temp += pair[1] + ' '
                city_b = temp.strip()
            else:
                city_b = 'Missing city value'
            city_i = list(filter(lambda pair: pair[0] == 'I-city_name', predicted_slots))
            temp = ''

            if city_i:
                for pair in city_i:
                    temp += pair[1] + ' '
            city_i = temp.strip()
            response = {'intent': predicted_label, 'slots': []}
            response['slots'].append(
                {'name': "city", "value": remove_punctuation(city_b.title() + " " + city_i.title()).strip()})
        else:
            response = {'intent': predicted_label, 'slots': []}

        return response

    # predict slots and label given a string
    def evaluate(self, sentence):
        predicted_label, predicted_slots = self.predict(sentence)
        return self.process_prediction(predicted_label, predicted_slots)

    # function used for training phase of the model
    def train_model(self, n_train=2000):
        bar = progressbar.ProgressBar(len(training_data_queries[:n_train]))
        for n_batch, sent in bar(enumerate(training_data_queries[:n_train])):
            label = training_data_intents[n_batch]
            slot = training_data_slots[n_batch]
            label = np.eye(n_classes)[label][np.newaxis, :]
            slot_one_hot = np.eye(n_slots)[slot][np.newaxis, :]
            sentence = sent[np.newaxis, :]

            if sentence.shape[1] > 1:  # ignore 1 word sentences
                self.model.train_on_batch(sentence, [label, slot_one_hot])

    # function used for validation phase
    def test_model(self, n_test=500):
        predictions = []
        results = []
        misclassified_examples = []
        true_labels = []
        predicted_labels = []
        predicted_slots = []
        true_slots = []
        words = []
        avgLoss = 0
        bar = progressbar.ProgressBar(len(test_data_queries[:n_test]))

        for n_batch, sent in bar(enumerate(test_data_queries[:n_test])):
            label = test_data_intents[n_batch]
            true_label = ids2intents[label[0]]

            if true_label in flight_booking_intents:
                true_label = 'flight_booking_intent'

            true_labels.append(true_label)

            slot = test_data_slots[n_batch]
            true_slot = [ids2slots[entry] for entry in slot]
            true_slots.append(true_slot)

            label = np.eye(n_classes)[label][np.newaxis, :]
            slot_one_hot = np.eye(n_slots)[slot][np.newaxis, :]
            sent = sent[np.newaxis, :]
            predicted_label, predicted_slot = self.predict(sent, is_words=False)
            if predicted_label in flight_booking_intents:
                predicted_label = 'flight_booking_intent'
            predicted_slot = [pair[0] for pair in predicted_slot]

            if sent.shape[1] > 1:
                loss = self.model.test_on_batch(sent, [label, slot_one_hot])
                avgLoss += loss[0]

            prediction = self.process_prediction(predicted_label, predicted_slot)
            predicted_labels.append(predicted_label)

            assert (len(true_labels) == len(predicted_labels))

            predicted_slots.append(predicted_slot)

            original_sentence = ' '.join(list(map(lambda word_id: ids2words[word_id], sent[0].tolist())))
            result = {'sentence': original_sentence, 'prediction': prediction}
            results.append(json.dumps(result))
            predictions.append(json.dumps(prediction))
            words.append(original_sentence)

            if not (true_label == predicted_label):
                misclassified_examples.append(json.dumps(result))

        print("Average Loss = " + str(avgLoss / n_test))

        return words, results, predictions, misclassified_examples, predicted_labels, true_labels, predicted_slots, true_slots

    # compute accuracy of the model on label predictions
    def get_accuracy(self, predicted, ground_truth):
        assert (len(predicted) == len(ground_truth))
        self.accuracy = sum([ground_truth[i] == predicted[i] for i in range(len(ground_truth))]) / len(ground_truth)

        return self.accuracy

    def save_results(self, predictions, results, misclassified_examples):
        file = open('results/predictions.txt', 'w')
        for item in predictions:
            file.write("%s\n" % item)

        file = open('results/results.txt', 'w')
        for item in results:
            file.write("%s\n" % item)

        file = open('results/misclassified_examples.txt', 'w')
        for item in misclassified_examples:
            file.write("%s\n" % item)

    def save_model(self):
        joblib.dump(self.summary, 'models/' + self.name + '.txt')
        self.model.save('models/' + self.name + '.h5')
        print("Saved model to disk")
