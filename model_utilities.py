from model import Model
from keras.models import load_model as load
import joblib


def load_model(model_name):
    loaded_model_summary = joblib.load('models/' + model_name + '.txt')
    embedding_dimension = loaded_model_summary['embedding_dimension']
    dropout_parameter = loaded_model_summary['dropout_parameter']
    bidirectional = loaded_model_summary['bidirectional']
    maxPooling = loaded_model_summary['maxPooling']
    averagePooling = loaded_model_summary['averagePooling']
    rnn_type = loaded_model_summary['rnn_type']
    rnn_units = loaded_model_summary['rnn_units']
    model = Model(embedding_dimension=embedding_dimension, dropout_parameter=dropout_parameter,
                  bidirectional=bidirectional, rnn_type=rnn_type,
                  maxPooling=maxPooling, averagePooling=averagePooling, rnn_units=rnn_units,
                  name=model_name)
    model.model = load('models/' + model_name + '.h5')
    model.accuracy = loaded_model_summary['accuracy']
    model.precision = loaded_model_summary['precision']
    model.recall = loaded_model_summary['recall']
    model.f1 = loaded_model_summary['f1']
    return model
