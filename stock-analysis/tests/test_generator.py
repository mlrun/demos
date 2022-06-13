from functions.generator import *
from mlrun import code_to_function


def test_stocks_generator():
    gen = StocksGenerator()
    print(gen.generate(number_of_stocks=4, start_delta=7, end_delta=3, interval='5m', path='myfile.csv'))


def test_stocks_generator_job():
    fn = code_to_function(filename="../functions/generator.py",
                          handler="StocksGenerator::generate",
                          kind='job'
                          )

    fn.run(params={'start_delta': 3,
                   'end_delta': 0,
                   'interval': '5m',
                   'number_of_stocks': 4,
                   'path': 'mycsv.csv'},
           local=True)


def test_news_generator_job():
    fn = code_to_function(filename="../functions/generator.py",
                          handler="NewsGenerator::generate",
                          kind='job'
                          )

    fn.run(params={'number_of_stocks': 4,
                   'path': 'mycsv.csv'},
           local=True)
