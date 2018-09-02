# example predictions
from model_utilities import load_model


def examples(my_model):
    print(my_model.evaluate("current weather in djakarta"))
    print(my_model.evaluate("I will be here until 6"))
    print(my_model.evaluate("I want to travel from miami to lagos"))
    print(my_model.evaluate("I want to leave accra for lagos on tuesday"))
    print(my_model.evaluate("London weather"))
    print(my_model.evaluate("London expected weather"))
    print(my_model.evaluate("what time will the rain start in London"))
    print(my_model.evaluate("Will it rain today in london"))
    print(my_model.evaluate("I would like to know when is the next flight from Miami to Los Angeles"))
    print(my_model.evaluate("I am in Toronto but I want to fly to Bangkok"))
    print(my_model.evaluate("I wanna fly from New York to Toronto"))
    print(my_model.evaluate("Book a flight to Winnipeg from Halifax"))
    print(my_model.evaluate("Book a flight from Halifax to Winnipeg"))
    print(my_model.evaluate("what’s the weather in Paris"))
    print(my_model.evaluate("temperature in San Francisco"))
    print(my_model.evaluate("My friend wants to travel from Westeros to King's Landing"))
    print(my_model.evaluate("Seattle forecast for wednesday"))
    print(my_model.evaluate("What is Seattle forecast for wednesday"))
    print(my_model.evaluate("Current temperature in Westeros"))
    print(my_model.evaluate("Book a flight to Winnipeg from Gotham"))
    print(my_model.evaluate("Give me flights from New York City to Winnipeg"))


def examples2(my_model):
    print(my_model.evaluate("What's the weather like in London?"))
    print(my_model.evaluate("London weather"))
    print(my_model.evaluate("what’s the weather in Paris"))
    print(my_model.evaluate("temperature in San Francisco"))
    print(my_model.evaluate("I wanna fly from New York to Toronto"))
    print(my_model.evaluate("I wanna fly to Toronto from New York"))
    print(my_model.evaluate("Book a flight from Winnipeg to Halifax"))
    print(my_model.evaluate("Book a flight to Halifax from Winnipeg"))


# These are the three models that were saved on disk
simple_gru = load_model('simple_gru')
simple_gru50 = load_model('simple_gru50')
gru = load_model('gru')
lstm = load_model('lstm')
lstm_nopooling300 = load_model('lstm_nopooling300')
gru_nopooling300 = load_model('gru_nopooling300')
lstm_nopooling = load_model('lstm_nopooling')
gru_nopooling = load_model('gru_nopooling')
gru_nopooling_moredata = load_model('gru_nopooling_moredata')  # best so far
gru_nopooling_moredata2 = load_model('gru_nopooling_moredata2')  # best so far

examples(simple_gru)
examples(simple_gru50)
examples(gru)
examples(lstm)
examples(lstm_nopooling300)
examples(gru_nopooling300)
examples(lstm_nopooling)
examples(gru_nopooling)
examples2(gru_nopooling)

examples(gru_nopooling_moredata2)
examples2(gru_nopooling_moredata2)
# to make a single prediction you just call
# model_used_to_predict.evaluate(sentence_to_predict)
