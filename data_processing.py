# processing the data and utility functions

import _pickle as cPickle
from generate_weather_intent_data import generate_weather_queries
import numpy as np
from sklearn.utils import shuffle
import string

with open('atis/atis.dict.vocab.csv') as f:
    atis_vocab = f.readlines()
atis_vocab = [s.strip() for s in atis_vocab]


def atisfull():
    with open('atis/atis.train.pkl', 'rb') as f:
        train_set, dic = cPickle.load(f)
    with open('atis/atis.test.pkl', 'rb') as f:
        test_set, _ = cPickle.load(f)
    return train_set, test_set, dic


train, test, dic = atisfull()

word_ids, slot_ids, intent_ids = dic['token_ids'], dic['slot_ids'], dic['intent_ids']

weather_data = generate_weather_queries(5500)

with open('data/weather.dict.vocab.csv') as f:
    weather_vocab = f.readlines()

weather_vocab = [s.strip() for s in weather_vocab]
weather_vocab = [word for word in weather_vocab if word not in atis_vocab]

weather_query_slots = ['B-weather_forecast_temperature',
                       'B-weather_condition', 'B-time_horizon', 'I-time_horizon']

start = len(slot_ids)
for i in range(len(weather_query_slots)):
    slot = weather_query_slots[i]
    slot_ids[slot] = start + i

intent_ids['weather_intent'] = len(intent_ids)

start = len(atis_vocab)
for i in range(len(weather_vocab)):
    word_ids[weather_vocab[i]] = start + i

word_ids['unknown'] = start + i

ids2words = dict((v, k) for k, v in word_ids.items())
ids2slots = dict((v, k) for k, v in slot_ids.items())
ids2intents = dict((v, k) for k, v in intent_ids.items())

n_classes = len(ids2intents)
n_vocab = len(ids2words)
n_slots = len(ids2slots)

example_query = list(map(lambda x: ids2words[x], train['query'][0]))  # map from token ids to words
example_slots = list(map(lambda x: ids2slots[x], train['slot_labels'][0]))  # map from words to slot labels
example_intent = list(map(lambda x: ids2intents[x], train['intent_labels'][0]))  # map from query to intent

vectorized_weather_queries = list(
    map(lambda sentence: np.array(list(map(lambda word: word_ids[word], sentence))), weather_data[0]))
vectorized_weather_slots = list(
    map(lambda slots: np.array(list(map(lambda slot: slot_ids[slot], slots))), weather_data[1]))
vectorized_weather_intents = list(map(lambda l: np.array(l), [[26]] * len(weather_data[0])))
training_data_queries = train['query'] + vectorized_weather_queries[:-1000]
training_data_slots = train['slot_labels'] + vectorized_weather_slots[:-1000]
training_data_intents = train['intent_labels'] + vectorized_weather_intents[:-1000]

training_data_queries, training_data_slots, training_data_intents = shuffle(training_data_queries, training_data_slots,
                                                                            training_data_intents)

test_data_queries = test['query'] + vectorized_weather_queries[-1000:]
test_data_slots = test['slot_labels'] + vectorized_weather_slots[-1000:]
test_data_intents = test['intent_labels'] + vectorized_weather_intents[-1000:]

np.random.seed(170)
test_data_queries, test_data_slots, test_data_intents = shuffle(test_data_queries, test_data_slots, test_data_intents)

words_train = list(
    map(lambda w: list(map(lambda x: ids2words[x] if not ids2words[x].isdigit() else len(ids2words[x]) * "DIGIT", w)),
        train['query']))
words_train = words_train + weather_data[0][:-1000]
intents_train = list(map(lambda w: list(map(lambda x: ids2intents[x], w)), train['intent_labels']))
intents_train = intents_train + [['weather_intent']] * len(training_data_queries)
slots_train = list(map(lambda w: list(map(lambda x: ids2slots[x], w)), train['slot_labels']))
slots_train = slots_train + weather_data[1][:-1000]
words_test = list(
    map(lambda w: list(map(lambda x: ids2words[x] if not ids2words[x].isdigit() else len(ids2words[x]) * "DIGIT", w)),
        test['query']))
words_test = words_test + weather_data[0][-1000:]
intents_test = list(map(lambda w: list(map(lambda x: ids2intents[x], w)), test['intent_labels']))
intents_test = intents_test + [['weather_intent']] * len(test_data_queries)
slots_test = list(map(lambda w: list(map(lambda x: ids2slots[x], w)), test['slot_labels']))
slots_test = slots_test + weather_data[1][-1000:]


def get_sentences(dataset='train'):
    if dataset == 'test':
        return list(map(lambda sentence: ' '.join(sentence), words_test))
    return list(map(lambda sentence: ' '.join(sentence), words_train))


def get_sentence(dataset, index):
    return get_sentences(dataset)[index]


def get_intents(dataset='train'):
    if dataset == 'test':
        return [intent[0] for intent in intents_test]
    return [intent[0] for intent in intents_train]


def get_intent(dataset, index):
    return get_intents(dataset)[index]


def get_slots(dataset='train'):
    if dataset == 'test':
        return [slot for slot in slots_test]
    return [slot for slot in slots_train]


def get_sentence_slots(dataset, index):
    return get_slots(dataset)[index]


def get_vocab():
    return list(word_ids.keys())


def map_sentences_to_slot(dataset):
    if dataset == 'train':
        sentences = words_train
        slots_filled = slots_train
    elif dataset == 'test':
        sentences = words_test
        slots_filled = slots_test
    else:
        return []
    return list(map(lambda words, slots: list(zip(words, slots)), sentences, slots_filled))


def map_sentence_to_slot(dataset, index):
    return map_sentences_to_slot(dataset)[index]


def map_sentences_to_intent(dataset):
    if dataset == 'train':
        sentences = words_train
        intents = intents_train
    elif dataset == 'test':
        sentences = words_test
        intents = intents_test
    else:
        return []
    return list(map(lambda sentence, intent: (' '.join(sentence), intent[0]), sentences, intents))


def map_sentence_to_intent(dataset, index):
    return map_sentences_to_intent(dataset)[index]


def convert_sentence_to_vector(sentence):
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    word2vec = list(map(lambda word: word_ids[word] if word in word_ids.keys() else word_ids['unknown'], sentence))
    word2vec = np.array(word2vec)
    word2vec = word2vec[np.newaxis, :]
    return word2vec


def convert_vector_to_sentence(vector):
    vector = vector[0]
    sentence = list(map(lambda x: ids2words[x], vector))
    sentence = ' '.join(sentence)
    return sentence


def remove_punctuation(x):
    table = str.maketrans({key: None for key in string.punctuation})
    return x.translate(table)


flight_booking_intents = ['aircraft+flight+flight_no', 'airfare',
                          'airfare+flight', 'airfare+flight_time', 'airline',
                          'airline+flight_no', 'airport', 'cheapest', 'city',
                          'day_name', 'flight', 'flight+airfare',
                          'flight+airline', 'flight_no', 'flight_no+airline', 'flight_time',
                          'meal']

print(n_vocab)
print(word_ids['unknown'])
