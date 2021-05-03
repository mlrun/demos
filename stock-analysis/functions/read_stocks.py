import yfinance as yf
import os
import pandas as pd
import v3io_frames as v3f
import ast
import mlrun.feature_store as fs


def update_tickers(context, period):

    stocks_df = pd.DataFrame()
    for sym in context.stock_syms:
        hist = yf.Ticker(sym).history(period=period, interval='1m')
        time = hist.index[len(hist) - 1]
        record = hist.loc[time]
        last = context.last_trade_times.get(sym)
        context.logger.info(f'Received {sym} data from yfinance, including {len(hist)} candles ending at {last}')

        # update the stocks table and TSDB metrics in case of new data
        if not last or time > last:

            # update NoSQL table with stock data
            stock = {'symbol': sym, 'price': record['Close'], 'volume': record['Volume'], 'last_updated': time}
            expr = context.expr_template.format(**stock)
            context.logger.debug_with('update expression', symbol=sym, expr=expr)
            context.v3c.execute('kv', context.stocks_kv_table, 'update', args={'key': sym, 'expression': expr})
            context.logger.info(f'Updated records from {last} to {time}')
            # update time-series DB with price and volume metrics (use pandas dataframe with a single row, indexed by date)
            context.last_trade_times[sym] = time
            hist['symbol'] = sym
            hist = hist.reset_index()
            # hist = hist.set_index(['Datetime', 'symbol'])
            # hist = hist.loc[:, ['Close', 'Volume']]
            # hist = hist.rename(columns={'Close': 'price', 'Volume': 'volume'})
            hist = hist.set_index(['Datetime'])
            hist = hist.loc[:, ['symbol', 'Close', 'Volume']]
            hist = hist.rename(columns={'symbol': 'ticker', 'Close': 'price', 'Volume': 'volume'})
            stocks_df = stocks_df.append(hist)
            context.logger.info(f'Added records {hist.shape[0]} records for {sym} to history')
        else:
            context.logger.info(f'No update was made, current TS: {last} vs. new data {time}')

    if stocks_df.shape[0] > 0:
        # ingest to feature store
        context.logger.info("ingesting to feature store")
        quotes_df = stocks_df
        quotes_df.reset_index(inplace=True)
        quotes_df
        fs.infer_metadata(
            context.quotes_set,
            quotes_df,
            entity_columns=["ticker"],
            timestamp_key="Datetime",
            options=fs.InferOptions.default(),
        )
        df = fs.ingest(context.quotes_set, quotes_df)

        stocks_df = stocks_df.sort_index(level=0)
        context.logger.debug_with('writing data to TSDB', stocks=stocks_df)
        stocks_df.to_csv('history.csv')
        context.v3c.write(backend='tsdb', table=context.stocks_tsdb_table, dfs=stocks_df)


def init_context(context):
    # Setup V3IO Client
    v3io_framesd = os.getenv('V3IO_FRAMESD', 'framesd:8081')
    token = os.getenv('TOKEN', '')
    container = os.getenv('V3IO_CONTAINER', 'bigdata')
    client = v3f.Client(v3io_framesd, token=token, container=container)
    setattr(context, 'v3c', client)

    # Create V3IO Tables and add reference to context
    setattr(context, 'stocks_kv_table', os.getenv('STOCKS_KV_TABLE', 'stocks/stocks_kv'))
    setattr(context, 'stocks_tsdb_table', os.getenv('STOCKS_TSDB_TABLE', 'stocks/stocks_tsdb'))
    context.v3c.create(backend='tsdb', table=context.stocks_tsdb_table, rate='1/m', if_exists=1)

    stocks = os.getenv('STOCK_LIST', 'GOOGL,MSFT,AMZN,AAPL,INTC')
    if stocks.startswith('['):
        stock_syms = ast.literal_eval(stocks)
    else:
        stock_syms = stocks.split(',')
    setattr(context, 'stock_syms', stock_syms)

    # v3io update expression template
    expr_template = os.getenv('EXPRESSION_TEMPLATE',
                              "symbol='{symbol}';price={price};volume={volume};last_updated='{last_updated}'")
    setattr(context, 'expr_template', expr_template)

    last_trade_times = {}
    setattr(context, 'last_trade_times', last_trade_times)

    # create a new feature set
    quotes_set = fs.FeatureSet("stock-quotes", entities=[fs.Entity("ticker")])
    setattr(context, 'quotes_set', quotes_set)


    # Run first initial data preparation
    update_tickers(context, '7d')


def handler(context):
    update_tickers(context, '5m')
    return context.Response(body='Tickers updated :]',
                            headers={},
                            content_type='text/plain',
                            status_code=200)
