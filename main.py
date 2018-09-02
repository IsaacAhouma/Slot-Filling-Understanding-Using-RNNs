# build a model, run it and save it to be used later
from model import Model
from accuracy import conlleval
import joblib

models = {}


def main(n_epochs=1, n_train=8000, n_test=1500, embedding_dimension=300, dropout_parameter=0.2, bidirectional=True,
         rnn_type='GRU', rnn_units=100, model_name='model', maxPooling=True, averagePooling=False, save=True):
    print("Working on Model: " + model_name)
    model = Model(embedding_dimension=embedding_dimension, dropout_parameter=dropout_parameter,
                  bidirectional=bidirectional,
                  rnn_type=rnn_type, rnn_units=rnn_units, name=model_name, maxPooling=maxPooling,
                  averagePooling=averagePooling)
    model.build_model()
    for i in range(n_epochs):
        print("Training epoch {}".format(i + 1))
        model.train_model(n_train=n_train)
        words, results, predictions, misclassified_examples, predicted_labels, true_labels, predicted_slots, true_slots = model.test_model(
            n_test=n_test)
        # format my words list such that they are in the correct format expected by the conlleval script
        words = [sentence.split() for sentence in words]
        _ = [sentence.pop(0) for sentence in words]
        _ = [sentence.pop(-1) for sentence in words]

        con_dict = conlleval(predicted_slots, true_slots,
                             words, 'measure.txt')

        print('Precision = {}, Recall = {}, F1 = {}'.format(
            con_dict['p'], con_dict['r'], con_dict['f1']))

        accuracy = model.get_accuracy(predicted_labels, true_labels)
        print('Accuracy = ' + str(accuracy))

    model.precision, model.recall, model.f1 = con_dict['p'], con_dict['r'], con_dict['f1']

    accuracy = model.get_accuracy(predicted_labels, true_labels)

    model.save_results(predictions, results, misclassified_examples)

    models[model_name] = {'accuracy': accuracy, 'precision': con_dict['p'], 'recall': con_dict['r'],
                          'f1': con_dict['f1']}
    model.summary['accuracy'] = accuracy
    print(model.summary['accuracy'])
    model.summary['precision'], model.summary['recall'], model.summary['f1'] = con_dict['p'], con_dict['r'], con_dict[
        'f1']
    if save:
        model.save_model()

    return model


# some of the models that were built
dummy_model = main(n_epochs=1, n_train=100, n_test=20, model_name='dummy_model')
simple_gru = main(n_epochs=20, n_train=1000, n_test=200, model_name='simple_gru')
simple_gru50 = main(n_epochs=20, n_train=1000, n_test=200, model_name='simple_gru50', rnn_units=500)
lstm_pooling = main(n_epochs=10, n_train=4000, n_test=1000, rnn_type='LSTM', model_name='lstm', rnn_units=500)
gru_pooling = main(n_epochs=10, n_train=4000, n_test=1000, model_name='gru', rnn_units=500)
lstm_nopooling = main(n_epochs=10, n_train=4000, n_test=1000, rnn_type='LSTM', model_name='lstm_nopooling',
                      maxPooling=False, rnn_units=500)
gru_nopooling = main(n_epochs=10, n_train=4000, n_test=1000, model_name='gru_nopooling', rnn_units=500,
                     maxPooling=False)
lstm_nopooling300 = main(n_epochs=10, n_train=4000, n_test=1000, rnn_type='LSTM', model_name='lstm_nopooling300',
                         maxPooling=False)
gru_nopooling300 = main(n_epochs=10, n_train=4000, n_test=1000, model_name='gru_nopooling300',
                        maxPooling=False)
gru_nopooling20epochs = main(n_epochs=20, n_train=4000, n_test=1000, model_name='gru_nopooling20epochs', rnn_units=500,
                             maxPooling=False)

gru_nopooling_moredata = main(n_epochs=10, n_train=6000, n_test=1500, model_name='gru_nopooling_moredata',
                              rnn_units=700,
                              maxPooling=False)

gru_nopooling_moredata2 = main(n_epochs=10, n_train=6000, n_test=1500, model_name='gru_nopooling_moredata2',
                               rnn_units=1000,
                               maxPooling=False)

gru_nopooling_moredata3 = main(n_epochs=10, n_train=8000, n_test=1500, model_name='gru_nopooling_moredata3',
                               rnn_units=1000,
                               maxPooling=False)

joblib.dump(models, 'results/models_summary.txt')


def load_models_summary():
    return joblib.load('results/models_summary.txt')


models = load_models_summary()
