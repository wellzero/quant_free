
def alpha091(data, dependencies=['close', 'volume', 'low'], max_window=45):
    # ((RANK(CLOSE-MAX(CLOSE,5))*RANK(CORR(MEAN(VOLUME,40),LOW,5)))*-1)
    # 感觉是TSMAX
    part1 = (data['close'] - data['close'].rolling(window=5, min_periods=5).max()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=40, min_periods=40).mean()).rolling(window=5, min_periods=5).corr(data['low']).rank(pct=True)
    return (part1 * part2) * (-1)

def alpha092(data, dependencies=['close', 'amount', 'volume'], max_window=209):
    # (MAX(RANK(DECAYLINEAR(DELTA(CLOSE*0.35+VWAP*0.65,2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    w5 = np.array(range(1, 6))
    w5 = w5/w5.sum()
    part1 = ((data['close'] * 0.35 + vwap * 0.65).diff(2)).rolling(window=3, min_periods=3).apply(lambda x: np.dot(x, w3)).rank(axis=0, pct=True)
    part2 = abs((data['volume'].rolling(window=180, min_periods=180).mean()).rolling(window=13, min_periods=13).corr(data['close']))
    part2 = part2.rolling(window=5, min_periods=5).apply(lambda x: np.dot(x, w5))
    part2 = part2.rolling(window=15, min_periods=15).apply(lambda x: x.rank(pct=True)[-1])
    return np.maximum(part1, part2) * (-1)

def alpha093(data, dependencies=['Open', 'low'], max_window=21):
    # SUM(OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW,OPEN-DELAY(OPEN,1)),20)
    condition = data['Open'].diff(1) >= 0.0
    alpha= pd.Series(0,index=data.index)
    alpha[~condition] = np.maximum(data['Open'] - data['low'], data['Open'].diff(1))[~condition]
    return alpha.rolling(window=20, min_periods=20).sum()

def alpha094(data, dependencies=['close', 'volume'], max_window=31):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    part1 = np.sign(data['close'].diff(1)) * data['volume']
    return part1.rolling(window=30, min_periods=30).sum()

def alpha095(data, dependencies=['amount'], max_window=20):
    # STD(AMOUNT,20)
    return data['amount'].rolling(window=20, min_periods=20).std()

def alpha096(data, dependencies=['KDJ_D'], max_window=13):
    # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    # 就是KDJ_D
    part1 =data['close']- data['close'].rolling(window=9, min_periods=9).min()
    part2=data['high'].rolling(window=9, min_periods=9).max()-data['low'].rolling(window=9, min_periods=9).min()
    part4=SMA(part1/part2*100,3,1)
    part5=SMA(part4,3,1)
    return part5

def alpha097(data, dependencies=['VSTD10'], max_window=10):
    # STD(VOLUME,10)
    # 就是VSTD10
    return STD(data['volume'],10)

def alpha098(data, dependencies=['close'], max_window=201):
    # (DELTA(SUM(CLOSE,100)/100,100)/DELAY(CLOSE,100)<=0.05)?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))
    condition1 = (data['close'].rolling(window=100, min_periods=100).sum() / 100).diff(periods=100) / data['close'].shift(100) <= 0.05
    alpha = (data['close'] - data['close'].rolling(window=100, min_periods=100).min()) * (-1)
    alpha[~condition1] = data['close'].diff(3)[~condition1] * (-1)
    return alpha

def alpha099(data, dependencies=['close', 'volume'], max_window=5):
    # (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))
    # return COVIANCE(sorted(data['close']),sorted(data['volume']),5)*-1
    return COVIANCE(data['close'].rank(pct=True),data['volume'].rank(pct=True),5)*-1

def alpha100(data, dependencies=['VSTD20'], max_window=20):
    # STD(VOLUME,20), 就是VSTD10
    return STD(data['volume'],20)

def alpha101(data, dependencies=['amount', 'volume', 'high', 'close'], max_window=82):
    # (RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15)) < RANK(CORR(RANK(HIGH*0.1+VWAP*0.9),RANK(VOLUME),11)))*-1
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (data['volume'].rolling(window=30, min_periods=30).mean()).rolling(window=37, min_periods=37).sum()
    part1 = (part1.rolling(window=15, min_periods=15).corr(data['close'])).rank(axis=0, pct=True)
    part2 = (data['high'] * 0.1 + vwap * 0.9).rank(axis=0, pct=True)
    part2 = (part2.rolling(window=11, min_periods=11).corr(data['volume'].rank(axis=0, pct=True))).rank(axis=0, pct=True)
    return (part2 - part1) * (-1)

def alpha102(data, dependencies=['volume'], max_window=7):
    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    part1 = (np.maximum(data['volume'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    part2 = abs(data['volume'].diff(1)).ewm(adjust=False, alpha=float(1)/6, min_periods=0, ignore_na=False).mean()
    return (part1 / part2) * 100

def alpha103(data, dependencies=['low'], max_window=20):
    # ((20-LOWDAY(LOW,20))/20)*100
    return (20 - data['low'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmin(axis=0))) * 5.0

def alpha104(data, dependencies=['high', 'volume', 'close'], max_window=20):
    # -1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))
    part1 = (data['high'].rolling(window=5, min_periods=5).corr(data['volume'])).diff(5)
    part2 = (data['close'].rolling(window=20, min_periods=20).std()).rank(axis=0, pct=True)
    return (part1 * part2) * (-1)

def alpha105(data, dependencies=['Open', 'volume'], max_window=10):
    # -1*CORR(RANK(OPEN),RANK(VOLUME),10)
    alpha = (data['Open'].rank(axis=0, pct=True)).rolling(window=10, min_periods=10).corr(data['volume'].rank(pct=True))
    return alpha * (-1)

def alpha106(data, dependencies=['close'], max_window=21):
    # CLOSE-DELAY(CLOSE,20)
    return data['close'].diff(20)

def alpha107(data, dependencies=['Open', 'close', 'high', 'low'], max_window=2):
    # (-1*RANK(OPEN-DELAY(HIGH,1)))*RANK(OPEN-DELAY(CLOSE,1))*RANK(OPEN-DELAY(LOW,1))
    part1 = data['Open'] - data['high'].shift(1)
    part2 = data['Open'] - data['close'].shift(1)
    part3 = data['Open'] - data['low'].shift(1)
    return (part1 * part2 * part3) * (-1)

def alpha108(data, dependencies=['high', 'amount', 'volume'], max_window=126):
    # (RANK(HIGH-MIN(HIGH,2))^RANK(CORR(VWAP,MEAN(VOLUME,120),6)))*-1
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (data['high'] - data['high'].rolling(window=2,min_periods=2).min()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=120, min_periods=120).mean()).rolling(window=6, min_periods=6).corr(vwap).rank(axis=0, pct=True)
    return (part1 ** part2) * (-1)

def alpha109(data, dependencies=['high', 'low'], max_window=20):
    # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    part1 = (data['high']-data['low']).ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean()
    return (part1 / part1.ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean())

def alpha110(data, dependencies=['close', 'high', 'low'], max_window=21):
    # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    part1 = (np.maximum(data['high']-data['close'].shift(1), 0.0)).rolling(window=20,min_periods=20).sum()
    part2 = (np.maximum(data['close'].shift(1)-data['low'], 0.0)).rolling(window=20,min_periods=20).sum()
    return (part1 / part2) * 100.0

def alpha111(data, dependencies=['low', 'high', 'close', 'volume'], max_window=11):
    # SMA(VOL*(2*CLOSE-LOW-HIGH)/(HIGH-LOW),11,2)-SMA(VOL*(2*CLOSE-LOW-HIGH)/(HIGH-LOW),4,2)
    win_vol = data['volume'] * (data['close']*2-data['low']-data['high']) / (data['high']-data['low'])
    alpha = win_vol.ewm(adjust=False, alpha=float(2)/11, min_periods=0, ignore_na=False).mean() - win_vol.ewm(adjust=False, alpha=float(2)/4, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha112(data, dependencies=['close'], max_window=13):
    # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))
    # /(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    part1 = (np.maximum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()
    part2 = abs(np.minimum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()
    return ((part1-part2) / (part1+part2)) * 100

def alpha113(data, dependencies=['close', 'volume'], max_window=28):
    # -1*RANK(SUM(DELAY(CLOSE,5),20)/20)*CORR(CLOSE,VOLUME,2)*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))
    part1 = (data['close'].shift(5).rolling(window=20, min_periods=20).mean()).rank(axis=0, pct=True)
    part2 = data['close'].rolling(window=2, min_periods=2).corr(data['volume'])
    part3 = ((data['close'].rolling(window=5, min_periods=5).sum()).rolling(window=2, min_periods=2).corr(data['close'].rolling(window=20, min_periods=20).sum())).rank(axis=0, pct=True)
    return (part1 * part2 * part3) * (-1)

def alpha114(data, dependencies=['high', 'low', 'close', 'amount', 'volume'], max_window=8):
    # RANK(DELAY((HIGH-LOW)/(SUM(CLOSE,5)/5),2))*RANK(RANK(VOLUME))/((HIGH-LOW)/(SUM(CLOSE,5)/5)/(VWAP-CLOSE))
    # rank/rank貌似没必要
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = ((data['high']-data['low'])/(data['close'].rolling(window=5,min_periods=5).mean())).shift(2).rank(axis=0,pct=True)
    part2 = data['volume'].rank(axis=0, pct=True).rank(axis=0, pct=True)
    part3 = (data['high']-data['low'])/(data['close'].rolling(window=5,min_periods=5).mean())/(vwap-data['close'])
    return part1*part2*part3

def alpha115(data, dependencies=['high', 'low', 'volume', 'close'], max_window=40):
    # (RANK(CORR(HIGH*0.9+CLOSE*0.1,MEAN(VOLUME,30),10))^RANK(CORR(TSRANK((HIGH+LOW)/2,4),TSRANK(VOLUME,10),7)))
    part1 = ((data['high'] * 0.9 + data['close'] * 0.1).rolling(window=10, min_periods=10).corr(
        data['volume'].rolling(window=30, min_periods=30).mean())).rank(axis=0, pct=True)
    part2 = (((data['high'] * 0.5 + data['low'] * 0.5).rolling(window=4, min_periods=4).apply(lambda x: stats.rankdata(x)[-1]/4.0)).rolling(window=7, min_periods=7) .corr(data['volume'].rolling(window=10, min_periods=10).apply(lambda x: stats.rankdata(x)[-1]/10.0))).rank(axis=0,pct=True)
    return part1 ** part2
    
def alpha116(data, dependencies=['close'], max_window=20):
    # REGBETA(CLOSE,SEQUENCE,20)
    alpha = REGBETA(data['close'],pd.Series(range(1,21)),20)
    return alpha

def alpha117(data, dependencies=['volume', 'close', 'high', 'low'], max_window=32):
    # TSRANK(VOLUME,32)*(1-TSRANK(CLOSE+HIGH-LOW,16))*(1-TSRANK(RET,32))
    part1 = data['volume'].rolling(32).apply(lambda x:x.rank(pct=True)[-1])
    part2 = 1.0 - (data['close']+data['high']-data['low']).rolling(16).apply(lambda x:x.rank(pct=True)[-1])
    part3 = 1.0 - data['close'].pct_change(periods=1).rolling(32).apply(lambda x:x.rank(pct=True)[-1])
    return part1 * part2 * part3

def alpha118(data, dependencies=['high', 'Open', 'low'], max_window=20):
    # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    alpha = (data['high']-data['Open']).rolling(window=20,min_periods=20).sum() / (data['Open']-data['low']).rolling(window=20,min_periods=20).sum() * 100.0
    return alpha

def alpha119(data, dependencies=['amount', 'volume', 'Open'], max_window=62):
    # RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8))
    # 感觉有个TSMIN
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w7 = np.array(range(1, 8))
    w7 = w7/w7.sum()
    w8 = np.array(range(1, 9))
    w8 = w8/w8.sum()
    part1 = ((data['volume'].rolling(window=5,min_periods=5).mean()).rolling(window=26, min_periods=26).sum()).rolling(window=5, min_periods=5).corr(vwap)
    part1 = (part1.rolling(window=7,min_periods=7).apply(lambda x:np.dot(x,w7))).rank(axis=0,pct=True)
    part2 = ((data['volume'].rolling(window=15, min_periods=15).mean()).rank(axis=0,pct=True)).rolling(window=21,min_periods=21).corr(data['Open'].rank(axis=0,pct=True))
    part2 = (((part2.rolling(window=9, min_periods=9).min()).rolling(window=7,min_periods=7).apply(lambda x: stats.rankdata(x)[-1]/7.0)).rolling(window=8,min_periods=8).apply(lambda x:np.dot(x,w8))).rank(axis=0, pct=True)
    return part1-part2

def alpha120(data, dependencies=['amount', 'volume', 'close'], max_window=1):
    # RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    return ((vwap-data['close']) / (vwap+data['close']))

def alpha121(data, dependencies=['amount', 'volume'], max_window=83):
    # (RANK(VWAP-MIN(VWAP,12))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    part1 = (vwap - vwap.rolling(window=12, min_periods=12).min()).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=2, min_periods=2).apply(lambda x: stats.rankdata(x)[-1]/2.0)
    part2 = ((vwap.rolling(window=20, min_periods=20).apply(lambda x: stats.rankdata(x)[-1]/20.0)).rolling(window=18, min_periods=18).corr(part2)) .rolling(window=3, min_periods=3).apply(lambda x: stats.rankdata(x)[-1]/3.0)
    return (part1 ** part2) * (-1)

def alpha122(data, dependencies=['close'], max_window=40):
    # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    part1 = (np.log(data['close'])).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    part1 = (part1.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    return part1.pct_change(periods=1)

def alpha123(data, dependencies=['high', 'low', 'volume'], max_window=89):
    # (RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW,VOLUME,6)))*-1
    part1 = (data['high']*0.5+data['low']*0.5).rolling(window=20, min_periods=20).sum()
    part1 = ((data['volume'].rolling(window=60,min_periods=60).mean()).rolling(window=20,min_periods=20).sum()).rolling(window=9,min_periods=9).corr(part1).rank(axis=0, pct=True)
    part2 = (data['low'].rolling(window=6,min_periods=6).corr(data['volume'])).rank(axis=0, pct=True)
    return (part2 - part1) * (-1)

def alpha124(data, dependencies=['close', 'amount', 'volume'], max_window=32):
    # (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w2 = np.array(range(1, 3))
    w2 = w2/w2.sum()
    part1 = data['close'] - vwap
    part2 = ((data['close'].rolling(window=30,min_periods=30).max()).rank(axis=0,pct=True)).rolling(window=2,min_periods=2).apply(lambda x:np.dot(x,w2))
    return part1 / part2

def alpha125(data, dependencies=['close', 'amount', 'volume'], max_window=117):
    # RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(CLOSE*0.5+VWAP*0.5,3),16))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w20 = np.array(range(1, 21))
    w20 = w20/w20.sum()
    w16 = np.array(range(1, 17))
    w16 = w16/w16.sum()
    part1 = (data['volume'].rolling(window=80,min_periods=80).mean()).rolling(window=17,min_periods=17).corr(vwap)
    part1 = (part1.rolling(window=20,min_periods=20).apply(lambda x:np.dot(x,w20))).rank(axis=0, pct=True)
    part2 = ((data['close']*0.5+vwap*0.5).diff(periods=3)).rolling(window=16,min_periods=16).apply(lambda x:np.dot(x,w16)).rank(axis=0,pct=True)
    return part1 / part2

def alpha126(data, dependencies=['high', 'low', 'close'], max_window=1):
    # (CLOSE+HIGH+LOW)/3
    return (data['close'] + data['high'] + data['low']) / 3.0

def alpha127(data, dependencies=['close'], max_window=24):
    # MEAN((100*(CLOSE-MAX(CLOSE,12))/MAX(CLOSE,12))^2)^(1/2)
    # 这里貌似是TSMAX,MEAN少一个参数
    alpha = (data['close'] - data['close'].rolling(window=12,min_periods=12).max()) / data['close'].rolling(window=12,min_periods=12).max() * 100
    alpha = (alpha ** 2).rolling(window=12, min_periods=12).mean() ** 0.5
    return alpha

def alpha128(data, dependencies=['high', 'low', 'close', 'volume'], max_window=14):
    # 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/
    # SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    condition1 = ((data['high']+data['low']+data['close'])/3.0).diff(1) > 0.0
    condition2 = ((data['high']+data['low']+data['close'])/3.0).diff(1) < 0.0
    part1 = (data['high']+data['low']+data['close'])/3.0*data['volume']
    part2 = part1.copy(deep=True)
    part1[~condition1] = 0.0
    part1 = part1.rolling(window=14, min_periods=14).sum()
    part2[~condition2] = 0.0
    part2 = part2.rolling(window=14, min_periods=14).sum()
    return (100.0-(100.0/(1+part1/part2)))

def alpha129(data, dependencies=['close'], max_window=13):
    # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    return (abs(np.minimum(data['close'].diff(1), 0.0))).rolling(window=12, min_periods=12).sum()

def alpha130(data, dependencies=['low', 'high', 'volume', 'amount'], max_window=59):
    # (RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    w10 = np.array(range(1, 11))
    w10 = w10/w10.sum()
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    part1 = (data['volume'].rolling(window=40,min_periods=40).mean()).rolling(window=9,min_periods=9).corr(data['high']*0.5+data['low']*0.5)
    part1 = part1.rolling(window=10,min_periods=10).apply(lambda x: np.dot(x, w10)).rank(axis=0, pct=True)
    part2 = (data['volume'].rank(axis=0, pct=True)).rolling(window=7,min_periods=7).corr(vwap.rank(axis=0, pct=True))
    part2 = part2.rolling(window=3,min_periods=3).apply(lambda x: np.dot(x, w3)).rank(axis=0, pct=True)
    return part1 / part2

def alpha131(data, dependencies=['amount', 'volume', 'close'], max_window=86):
    # (RANK(DELAT(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
    # vwap = data['amount'] / (data['volume']*100) 
    vwap = data['vwap']
    part1 = vwap.diff(1).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=50, min_periods=50).mean()).rolling(window=18, min_periods=18).corr(data['close'])
    part2 = part2.rolling(window=18, min_periods=18).apply(lambda x:x.rank(pct=True)[-1])
    return (part1 ** part2)

def alpha132(data, dependencies=['amount'], max_window=20):
    # MEAN(AMOUNT,20)
    return data['amount'].rolling(window=20, min_periods=20).mean()

def alpha133(data, dependencies=['low', 'high'], max_window=20):
    # ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    part1 = (20 - data['high'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmax(axis=0))) * 5.0
    part2 = (20 - data['low'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmin(axis=0))) * 5.0
    return part1 -part2

def alpha134(data, dependencies=['close', 'volume'], max_window=13):
    # (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    return (data['close'].pct_change(periods=12) * data['volume'])

def alpha135(data, dependencies=['close'], max_window=42):
    # SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    alpha = (data['close']/data['close'].shift(20)).shift(1)
    return alpha.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha136(data, dependencies=['close', 'Open', 'volume'], max_window=10):
    # -1*RANK(DELTA(RET,3))*CORR(OPEN,VOLUME,10)
    part1 = data['close'].pct_change(periods=1).diff(3).rank(axis=0,pct=True)
    part2 = data['Open'].rolling(window=10, min_periods=10).corr(data['volume'])
    return (part1 * part2) * (-1)

def alpha137(data, dependencies=['Open', 'low', 'close', 'high'], max_window=2):
    # 16*(CLOSE+(CLOSE-OPEN)/2-DELAY(OPEN,1))/
    # ((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:
    # (ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
    # *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    part1 = data['close'] * 1.5 - data['Open'] * 0.5 - data['Open'].shift(1)
    part2 = abs(data['high']-data['close'].shift(1)) + abs(data['low']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    condition1 = np.logical_and(abs(data['high']-data['close'].shift(1)) > abs(data['low']-data['close'].shift(1)), 
                               abs(data['high']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)))
    condition2 = np.logical_and(abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['low'].shift(1)), 
                               abs(data['low']-data['close'].shift(1)) > abs(data['high']-data['close'].shift(1)))
    part2[~condition1 & condition2] = abs(data['low']-data['close'].shift(1)) + abs(data['high']-data['close'].shift(1)) / 2.0 + abs(data['close']-data['Open']).shift(1) / 4.0
    part2[~condition1 & ~condition2] = abs(data['high']-data['low'].shift(1)) + abs(data['close']-data['Open']).shift(1) / 4.0
    part3 = np.maximum(abs(data['high']-data['close'].shift(1)), abs(data['low']-data['close'].shift(1)))
    alpha = (part1 / part2 * part3 * 16.0)
    return alpha

def alpha138(data, dependencies=['low','amount','volume'], max_window=126):
    # ((RANK(DECAYLINEAR(DELTA(LOW*0.7+VWAP*0.3,3),20))
    # -TSRANK(DECAYLINEAR(TSRANK(
        # CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5)
        # ,19),16),7))* -1)
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    w20 = np.array(range(1, 21))
    w20 = w20/w20.sum()
    w16 = np.array(range(1, 17))
    w16 = w16/w16.sum()
    part1 = ((data['low']*0.7+vwap*0.3).diff(3)).rolling(window=20,min_periods=20).apply(lambda x: np.dot(x,w20)).rank(axis=0, pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=17,min_periods=17).apply(lambda x: stats.rankdata(x)[-1]/17.0)
    part2 = part2.rolling(window=5,min_periods=5).corr(data['low'].rolling(window=8,min_periods=8).apply(lambda x: stats.rankdata(x)[-1]/8.0))
    part2 = ((part2.rolling(window=19,min_periods=19).apply(lambda x: stats.rankdata(x)[-1]/19.0)).rolling(window=16,min_periods=16).apply(lambda x:np.dot(x,w16))).rolling(window=7,min_periods=7).apply(lambda x: stats.rankdata(x)[-1]/7.0)
    return (part1-part2) * (-1)

def alpha139(data, dependencies=['Open', 'volume'], max_window=10):
    # (-1*CORR(OPEN,VOLUME,10))
    return data['Open'].rolling(window=10,min_periods=10).corr(data['volume']) * (-1)

def alpha140(data, dependencies=['Open', 'low', 'high', 'close', 'volume'], max_window=99):
    # MIN(RANK(DECAYLINEAR(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
    w8 = np.array(range(1, 9))
    w8 = w8/w8.sum()
    w7 = np.array(range(1, 8))
    w7 = w7/w7.sum()
    part1 = data['Open'].rank(axis=0,pct=True)+data['low'].rank(axis=0,pct=True)-data['high'].rank(axis=0,pct=True)-data['close'].rank(axis=0,pct=True)
    part1 = part1.rolling(window=8,min_periods=8).apply(lambda x:np.dot(x,w8)).rank(axis=0,pct=True)
    part2 = (data['volume'].rolling(window=60, min_periods=60).mean()).rolling(window=20,min_periods=20).apply(lambda x: stats.rankdata(x)[-1]/20.0)
    part2 = part2.rolling(window=8,min_periods=8).corr(data['close'].rolling(window=8,min_periods=8).apply(lambda x: stats.rankdata(x)[-1]/8.0))
    part2 = (part2.rolling(window=7,min_periods=7).apply(lambda x:np.dot(x,w7))).rolling(window=3,min_periods=3).apply(lambda x: stats.rankdata(x)[-1]/3.0)  
    return np.minimum(part1,part2)

def alpha141(data, dependencies=['high', 'volume'], max_window=25):
    # (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)
    alpha = ((data['volume'].rolling(window=15,min_periods=15).mean().rank(axis=0,pct=True)).rolling(window=9,min_periods=9).corr(data['high'].rank(axis=0,pct=True))).rank(axis=0,pct=True)
    return alpha * (-1)

def alpha142(data, dependencies=['close', 'volume'], max_window=25):
    # -1*RANK(TSRANK(CLOSE,10))*RANK(DELTA(DELTA(CLOSE,1),1))*RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))
    part1 = (data['close'].rolling(window=10,min_periods=10).apply(lambda x: stats.rankdata(x)[-1]/10.0)).rank(axis=0,pct=True)
    part2 = (data['close'].diff(1)).diff(1).rank(axis=0,pct=True)
    part3 = (data['volume']/data['volume'].rolling(window=20,min_periods=20).mean()).rolling(window=5,min_periods=5).apply(lambda x: stats.rankdata(x)[-1]/5.0).rank(axis=0,pct=True)
    return (part1 * part2 * part3) * (-1)

def alpha143(data):
    # CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    condition = data['close']>data['close'].shift(1)
    alpha = data['close'].copy()
    alpha[condition] = ((1+data['close'].pct_change())*data['close'])[condition]
    return alpha

def alpha144(data, dependencies=['close','amount'], max_window=21):
    # SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    part1 = abs(data['close'].pct_change(periods=1)) / data['amount']
    part1[data['close'].diff(1)>=0] = 0.0
    part1 = part1.rolling(window=20, min_periods=20).sum()
    part2 = (data['close'].diff(1)<0.0).rolling(window=20,min_periods=20).sum()
    return part1 / part2

def alpha145(data, dependencies=['volume'], max_window=26):
    # (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    alpha = (data['volume'].rolling(window=9,min_periods=9).mean() - data['volume'].rolling(window=26,min_periods=26).mean()) / data['volume'].rolling(window=12,min_periods=12).mean() * 100.0
    return alpha

def alpha146(data, dependencies=['close'], max_window=121):
    # MEAN(RET-SMA(RET,61,2),20)*(RET-SMA(RET,61,2))/SMA(SMA(RET,61,2)^2,60)
    # 假设最后一个SMA(X,60,1)
    sma = (data['close'].pct_change(1)).ewm(adjust=False, alpha=float(2)/61, min_periods=0, ignore_na=False).mean()
    ret_excess = data['close'].pct_change(1) - sma
    part1 = ret_excess.rolling(window=20, min_periods=20).mean() * ret_excess
    part2 = (sma ** 2).ewm(adjust=False, alpha=float(1)/60, min_periods=0, ignore_na=False).mean()
    return part1 / part2


def alpha147(data, dependencies=['close'], max_window=24):
    # REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    ma_price = data['close'].rolling(window=12, min_periods=12).mean()
    alpha = REGBETA(ma_price,pd.Series(range(1,13)),12)
    return alpha

def alpha148(data, dependencies=['Open', 'volume'], max_window=75):
    # (RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),6))<RANK(OPEN-TSMIN(OPEN,14)))*-1
    part1 = (data['volume'].rolling(window=60,min_periods=60).mean()).rolling(window=9,min_periods=9).sum()
    part1 = part1.rolling(window=6,min_periods=6).corr(data['Open']).rank(axis=0,pct=True)
    part2 = (data['Open'] - data['Open'].rolling(window=14,min_periods=14).min()).rank(axis=0, pct=True)
    return (part2-part1) * (-1)

# def alpha149(data, dependencies=['close'], max_window=253):
#     # REGBETA(FILTER(RET,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),
#     # FILTER(BANCHMARK_INDEX_CLOSE/DELAY(BANCHMARK_INDEX_CLOSE,1)-1,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),252)
#     bm = (data['close'].mean(axis=0).diff(1) < 0.0)
#     part1 = data['close'].pct_change(periods=1).iloc[-252:][bm]
#     part2 = data['close'].mean(axis=0).pct_change(periods=1).iloc[-252:][bm]
#     alpha = pd.DataFrame([[stats.linregress(part1[col].values, part2.values)[0] for col in data['close'].columns]], 
#                  index=data['close'].index[-1:], columns=data['close'].columns)
#     return alpha
def alpha149(data, dependencies=['close', 'bm_index_close'], max_window=253):
    # REGBETA(FILTER(RET,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),
    # FILTER(BANCHMARK_INDEX_CLOSE/DELAY(BANCHMARK_INDEX_CLOSE,1)-1,BANCHMARK_INDEX_CLOSE<DELAY(BANCHMARK_INDEX_CLOSE,1)),252)
    bm = data['bm_index_close']
    bm = (bm.diff(1) < 0.0)
    part1 = data['close'].pct_change(periods=1)[bm]
    part2 = data['close'].rolling(252).mean().pct_change(periods=1)[bm]
    try:
        alpha = REGBETA(part1,part2,252)
    except Exception as e:
        print(e)
        return pd.Series(np.nan,index=data.index)
    return alpha


def alpha150(data, dependencies=['close', 'high', 'low', 'volume'], max_window=1):
    # (CLOSE+HIGH+LOW)/3*VOLUME
    return ((data['close'] + data['high'] + data['low']) / 3.0 * data['volume'])

def alpha151(data, dependencies=['close'], max_window=41):
    # SMA(CLOSE-DELAY(CLOSE,20),20,1)
    return (data['close'].diff(20)).ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha152(data, dependencies=['close'], max_window=59):
    # A=DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1)
    # SMA(MEAN(A,12)-MEAN(A,26),9,1)
    part1 = ((data['close'] / data['close'].shift(9)).shift(1)).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean().shift(1)
    alpha = (part1.rolling(window=12,min_periods=12).mean()-part1.rolling(window=26,min_periods=26).mean()).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean()
    return alpha

def alpha153(data, dependencies=['BBI'], max_window=24):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    # 就是BBI
    part1=[3,6,12,24]
    part2=[data['close'].rolling(window=x,min_periods=x).mean() for x in part1]
    alpha = sum(part2)/4
    return alpha

def alpha154(data, dependencies=['amount', 'volume'], max_window=198):
    # VWAP-MIN(VWAP,16)<CORR(VWAP,MEAN(VOLUME,180),18)
    # 感觉是TSMIN
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = vwap - vwap.rolling(window=16, min_periods=16).min()
    part2 = (data['volume'].rolling(window=180, min_periods=180).mean()).rolling(window=18, min_periods=18).corr(vwap)
    return part2-part1

def alpha155(data, dependencies=['volume'], max_window=37):
    # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    sma13 = data['volume'].ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    sma27 = data['volume'].ewm(adjust=False, alpha=float(2)/27, min_periods=0, ignore_na=False).mean()
    ssma = (sma13-sma27).ewm(adjust=False, alpha=float(2)/10, min_periods=0, ignore_na=False).mean()
    return (sma13 - sma27 - ssma)

def alpha156(data, dependencies=['amount', 'volume', 'Open', 'low'], max_window=9):
    # MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR((DELTA(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85)) * -1,3))) * -1
    w3 = np.array(range(1, 4))
    w3 = w3/w3.sum()
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    den = data['Open']*0.15+data['low']*0.85
    part1 = (vwap.diff(5)).rolling(window=3,min_periods=3).apply(lambda x:np.dot(x,w3))
    part2 = (den.diff(2)/den*(-1)).rolling(window=3,min_periods=3).apply(lambda x:np.dot(x,w3))
    return np.maximum(part1, part2) * (-1)

def alpha157(data, dependencies=['close'], max_window=12):
    # MIN(PROD(RANK(LOG(SUM(TSMIN(RANK(-1*RANK(DELTA(CLOSE-1,5))),2),1))),1),5) +TSRANK(DELAY(-1*RET,6),5)
    part1 = np.log((((data['close']-1.0).diff(5).rank(pct=True) * (-1)).rank(pct=True)).rolling(window=2, min_periods=2).min())
    part1 = (part1.rank(axis=0, pct=True)).rolling(window=5,min_periods=5).min()
    part2 = ((data['close'].pct_change(periods=1) * (-1)).shift(6)).rolling(5).apply(lambda x:x.rank(pct=True)[-1])
    return (part1 + part2)

def alpha158(data, dependencies=['low', 'high', 'close'], max_window=1):
    # (HIGH-LOW)/CLOSE
    return ((data['high'] - data['low']) / data['close'])

def alpha159(data, dependencies=['close', 'low', 'high'], max_window=25):
    # ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24
    # +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24
    # +(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    min_low_close = np.minimum(data['low'], data['close'].shift(1))
    max_high_close = np.maximum(data['high'], data['close'].shift(1))
    part1 = (data['close'] - min_low_close.rolling(window=6,min_periods=6).sum()) / (max_high_close-min_low_close).rolling(window=6,min_periods=6).sum() * 12 * 24
    part2 = (data['close'] - min_low_close.rolling(window=12,min_periods=12).sum()) / (max_high_close-min_low_close).rolling(window=12,min_periods=12).sum() * 6 * 24
    part3 = (data['close'] - min_low_close.rolling(window=24,min_periods=24).sum()) / (max_high_close-min_low_close).rolling(window=24,min_periods=24).sum() * 6 * 12
    return (part1+part2+part3)*100.0/(12*6+6*24+12*24)
    
def alpha160(data, dependencies=['close'], max_window=41):
    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    part1 = data['close'].rolling(window=20,min_periods=20).std()
    part1[data['close'].diff(1)>0] = 0.0
    part1[:19] = np.nan
    return part1.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha161(data, dependencies=['close', 'low', 'high'], max_window=13):
    # MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    part1 = np.maximum(data['high']-data['low'], abs(data['close'].shift(1)-data['high']))
    part1 = np.maximum(part1, abs(data['close'].shift(1)-data['low']))
    return part1.rolling(window=12,min_periods=12).mean()

def alpha162(data, dependencies=['close'], max_window=25):
    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    # -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    # /(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)
    # -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    den = (np.maximum(data['close'].diff(1), 0.0)).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean() /(abs(data['close'].diff(1))).ewm(adjust=False, alpha=float(1)/12, min_periods=0, ignore_na=False).mean() * 100.0
    alpha = (den - den.rolling(window=12,min_periods=12).min()) / (den.rolling(window=12,min_periods=12).max() - den.rolling(window=12,min_periods=12).min())
    return alpha

def alpha163(data, dependencies=['amount', 'volume', 'close', 'high'], max_window=20):
    # RANK((-1*RET)*MEAN(VOLUME,20)*VWAP*(HIGH-CLOSE))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    alpha = data['close'].pct_change(periods=1) * (data['volume'].rolling(window=20, min_periods=20).mean()) * vwap * (data['high'] - data['close']) * (-1)
    return alpha

def alpha164(data, dependencies=['close', 'high', 'low'], max_window=26):
    # SMA(((CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1,12))/(HIGH-LOW)*100,13,2)
    part1 = 1.0 / data['close'].diff(1)
    part1[data['close'].diff(1)<=0] = 1.0
    part2 = part1.rolling(window=12, min_periods=12).min()
    alpha = (part1-part2)/(data['high']-data['low'])*100.0
    return alpha.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()

def alpha165(data, dependencies=['close'], max_window=144):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    # SUMAC少了前N项和,TSMAX/TSMIN
    part1 = ((data['close']-data['close'].rolling(window=48,min_periods=48).mean()).rolling(window=48,min_periods=48).sum()).rolling(window=48,min_periods=48).max()
    part2 = ((data['close']-data['close'].rolling(window=48,min_periods=48).mean()).rolling(window=48,min_periods=48).sum()).rolling(window=48,min_periods=48).min()
    part3 = data['close'].rolling(window=48,min_periods=48).std()
    return (part1-part2/part3)

def alpha166(data, dependencies=['close'], max_window=41):
    # -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1))^2,20))^1.5)
    part1 = data['close'].pct_change(periods=1)-(data['close'].pct_change(periods=1).rolling(window=20,min_periods=20).mean())
    part1 = part1.rolling(window=20,min_periods=20).sum() * ((-20) * 19 ** 1.5)
    part2 = (((data['close']/data['close'].shift(1)) ** 2).rolling(window=20,min_periods=20).sum() ** 1.5) * 19 * 18
    return (part1 / part2)

def alpha167(data, dependencies=['close'], max_window=13):
    # SUM(CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0,12)
    return (np.maximum(data['close'].diff(1), 0.0)).rolling(window=12, min_periods=12).sum()

def alpha168(data, dependencies=['volume'], max_window=20):
    # -1*VOLUME/MEAN(VOLUME,20)
    return (data['volume']/(data['volume'].rolling(window=20,min_periods=20).mean())) * (-1)

def alpha169(data, dependencies=['close'], max_window=48):
    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    part1 = (data['close'].diff(1).ewm(adjust=False, alpha=float(1)/9, min_periods=0, ignore_na=False).mean()).shift(1)
    part2 = (part1.rolling(window=12, min_periods=12).mean() - part1.rolling(window=26, min_periods=26).mean()).ewm(adjust=False, alpha=float(1)/10, min_periods=0, ignore_na=False).mean()
    return part2

def alpha170(data, dependencies=['close','volume','high', 'amount'], max_window=20):
    # ((RANK(1/CLOSE)*VOLUME)/MEAN(VOLUME,20))*(HIGH*RANK(HIGH-CLOSE)/(SUM(HIGH,5)/5))-RANK(VWAP-DELAY(VWAP,5))
    # vwap = data['amount'] / (data['volume']*100)
    vwap = data['vwap']
    part1 = (1.0/data['close']).rank(axis=0,pct=True) * data['volume'] / (data['volume'].rolling(window=20,min_periods=20).mean())
    part2 = ((data['high']-data['close']).rank(axis=0,pct=True) * data['high']) / (data['high'].rolling(window=5,min_periods=5).sum()/5.0)
    part3 = (vwap.diff(5)).rank(axis=0,pct=True)
    return (part1*part2-part3)
    
def alpha171(data, dependencies=['low', 'close', 'Open', 'high'], max_window=1):
    # (-1*(LOW-CLOSE)*(OPEN^5))/((CLOSE-HIGH)*(CLOSE^5))
    part1 = (data['low']-data['close']) * (data['Open'] ** 5) * (-1)
    part2 = (data['close']-data['high']) * (data['close'] ** 5)
    return round(part1/part2,8)

def alpha172(data, dependencies=['ADX'], max_window=20):
    # 就是DMI-ADX
    # HD  HIGH-DELAY(HIGH,1)
    # LD  DELAY(LOW,1)-LOW
    # TR  MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))

    # MEAN(ABS(
    #     SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)
    #     -SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))
    # /(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)
    #     +SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))
    # *100,6)
    hd=data['high'].diff(1)
    ld=-data['low'].diff(1)
    tr=np.maximum(np.maximum(data['high']-data['low'],(data['high']-data['close'].shift(1)).abs()),(data['low']-data['close'].shift(1)).abs())
    part1=(((ld>0)&(ld>hd))*ld).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()-(((hd>0)&(hd>ld))*hd).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()
    part2=(((ld>0)&(ld>hd))*ld).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()+(((hd>0)&(hd>ld))*hd).rolling(window=14, min_periods=14).sum()*100/tr.rolling(window=14, min_periods=14).sum()
    alpha = (part1/part2).abs().rolling(6).mean()*100
    return alpha

def alpha173(data, dependencies=['close'], max_window=39):
    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    den = data['close'].ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    part1 = 3 * den
    part2 = 2 * (den.ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean())
    part3 = ((np.log(data['close']).ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()) .ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()) .ewm(adjust=False, alpha=float(2)/13, min_periods=0, ignore_na=False).mean()
    return part1-part2+part3

def alpha174(data, dependencies=['close'], max_window=41):
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    part1 = data['close'].rolling(window=20,min_periods=20).std()
    part1[data['close'].diff(1)<=0] = 0.0
    part1[:19] = np.nan
    return part1.ewm(adjust=False, alpha=float(1)/20, min_periods=0, ignore_na=False).mean()

def alpha175(data, dependencies=['low','high','close'], max_window=7):
    # MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    alpha = np.maximum(data['high']-data['low'], abs(data['close'].shift(1)-data['high']))
    alpha = np.maximum(alpha, abs(data['close'].shift(1)-data['low']))
    return alpha.rolling(window=6,min_periods=6).mean()

def alpha176(data, dependencies=['close','high','low','volume'], max_window=18):
    # CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))),RANK(VOLUME),6)
    part1 = ((data['close'] - data['low'].rolling(window=12,min_periods=12).min()) / (data['high'].rolling(window=12,min_periods=12).max()-data['low'].rolling(window=12,min_periods=12).min())).rank(axis=0, pct=True)
    part2 = data['volume'].rank(axis=0, pct=True)
    return part1.rolling(window=6,min_periods=6).corr(part2)

def alpha177(data, dependencies=['high'], max_window=20):
    # ((20-HIGHDAY(HIGH,20))/20)*100
    return (20 - data['high'].rolling(window=20, min_periods=20).apply(lambda x: 19-x.argmax(axis=0))) * 5.0

def alpha178(data, dependencies=['close', 'volume'], max_window=2):
    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    return (data['close'].pct_change(periods=1) * data['volume'])

def alpha179(data, dependencies=['low','amount','volume'], max_window=62):
    # RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
    part1 = ((data['amount']/data['volume']).rolling(window=4,min_periods=4).corr(data['volume'])).rank(axis=0,pct=True)
    part2 = (((data['volume'].rolling(window=50,min_periods=50).mean()).rank(axis=0,pct=True)).rolling(window=12,min_periods=12).corr(data['low'].rank(axis=0,pct=True))).rank(axis=0,pct=True)
    return part1 * part2

def alpha180(data, dependencies=['volume', 'close'], max_window=68):
    # (MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))
    condition = data['volume'].rolling(window=20, min_periods=20).mean() < data['volume']
    alpha = abs(data['close'].diff(7)).rolling(window=60, min_periods=60).apply(lambda x: stats.rankdata(x)[-1]/60.0) * np.sign(data['close'].diff(7)) * (-1)
    alpha[~condition] = -1 * data['volume'][~condition]
    return round(alpha, 8)


def alpha181(data, dependencies=['close', 'bm_index_close'], max_window=40):
    # SUM(RET-MEAN(RET,20)-(BANCHMARK_INDEX_CLOSE-MEAN(BANCHMARK_INDEX_CLOSE,20))^2,20)/SUM((BANCHMARK_INDEX_CLOSE-MEAN(BANCHMARK_INDEX_CLOSE,20))^3)
    # 优化：数值取对数，否则ret 跟 index 可能不在一个量级上，导致全部结果趋同。
    bm = np.log(data['bm_index_close'])
    bm_mean = bm - bm.rolling(window=20, min_periods=20).mean()
    # print(bm_mean)
    # bm_mean = pd.DataFrame(data=np.repeat(bm_mean.values.reshape(len(bm_mean.values),1), len(data['close'].columns), axis=1), index=data['close'].index, columns=data['close'].columns)
    ret = np.log(data['close']).pct_change(periods=1)
    part1 = (ret-ret.rolling(window=20,min_periods=20).mean()-bm_mean**2).rolling(window=20,min_periods=20).sum()
    part2 = (bm_mean ** 3).rolling(window=20,min_periods=20).sum()
    return part1 / part2

def alpha182(data, dependencies=['close','Open', 'bm_index_open', 'bm_index_close'], max_window=20):
    # COUNT((CLOSE>OPEN & BANCHMARK_INDEX_CLOSE>BANCHMARK_INDEX_OPEN) OR (CLOSE<OPEN &BANCHMARK_INDEX_CLOSE<BANCHMARK_INDEX_OPEN),20)/20
    bm = data['bm_index_close'] > data['bm_index_open']
    bm_2 = data['bm_index_close'] < data['bm_index_open']
    # bm = pd.DataFrame(data=np.repeat(bm.values.reshape(len(bm.values),1), len(data['close'].columns), axis=1), index=data['close'].index, columns=data['close'].columns)
    condition_1 = np.logical_and(data['close']>data['Open'], bm)
    condition_2 = np.logical_and(data['close']<data['Open'], bm_2)
    alpha = np.logical_or(condition_1, condition_2).rolling(window=20, min_periods=20).mean()
    return alpha
    
def alpha183(data, dependencies=['close'], max_window=72):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    part1 = ((data['close']-data['close'].rolling(window=24,min_periods=24).mean()).rolling(window=24,min_periods=24).sum()).rolling(window=24,min_periods=24).max()
    part2 = ((data['close']-data['close'].rolling(window=24,min_periods=24).mean()).rolling(window=24,min_periods=24).sum()).rolling(window=24,min_periods=24).min()
    part3 = data['close'].rolling(window=24,min_periods=24).std()
    return part1-part2/part3

def alpha184(data, dependencies=['close','Open'], max_window=201):
    # RANK(CORR(DELAY(OPEN-CLOSE,1),CLOSE,200))+RANK(OPEN-CLOSE)
    part1 = (((data['Open']-data['close']).shift(1)).rolling(window=200,min_periods=200).corr(data['close'])).rank(axis=0,pct=True)
    part2 = (data['Open']-data['close']).rank(axis=0,pct=True)
    return part1+part2

def alpha185(data, dependencies=['close', 'Open'], max_window=1):
    # RANK(-1*(1-OPEN/CLOSE)^2)
    return ((1.0-data['Open']/data['close']) ** 2) * (-1)

def alpha186(data, dependencies=['low','high','close'], max_window=20):
    # 就是ADXR
#     (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) - SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14)) 
#     /
#     (SUM((LD>0  &  LD>HD)?LD:0,14)*100/SUM(TR,14) + SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
#     +
#     DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) - SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14)) 
#            / (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14) + SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
#          ,6))/2
    dm_plus = data['close'].diff(1).fillna(0)
    dm_subtract = data['low'].diff(1).fillna(0)
    condition_plus = (dm_plus<dm_subtract) | (dm_plus<0)
    condition_sub = (dm_subtract < dm_plus) | (dm_subtract<0)
    dm_plus[condition_plus] = 0
    dm_subtract[condition_sub] = 0
    
    close_delay = data['close'].shift(1)
    tr_a = data['high'] - data['low']
    tr_b = data['high'] - close_delay
    tr_c = data['low'] - close_delay
    tr = pd.concat([tr_a,tr_b,tr_c],axis=1).max(axis=1)
    
    tr_sum = tr.rolling(14).sum()
    
    PDI = dm_plus.rolling(14).sum() * 100 / tr_sum
    MDI = dm_subtract.rolling(14).sum() * 100 / tr_sum
    DX = np.abs(PDI-MDI)/(PDI+MDI) * 100
    ADX = MEAN(DX,6)
    ADXR =(ADX+ADX.shift(1))/2
    
    return ADXR



def alpha187(data, dependencies=['Open', 'high'], max_window=21):
    # SUM(OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN,OPEN-DELAY(OPEN,1)),20)
    part1 = np.maximum(data['high']-data['Open'], data['Open'].diff(1))
    part1[data['Open'].diff(1)<=0] = 0.0
    return part1.rolling(window=20, min_periods=20).sum()

def alpha188(data, dependencies=['low', 'high'], max_window=11):
    # ((HIGH-LOW\u2013SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    sma = (data['high']-data['low']).ewm(adjust=False, alpha=float(2)/11, min_periods=0, ignore_na=False).mean()
    return ((data['high']-data['low']-sma)/sma) * 100

def alpha189(data, dependencies=['close'], max_window=12):
    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    return abs(data['close']-data['close'].rolling(window=6,min_periods=6).mean()).rolling(window=6,min_periods=6).mean()

def alpha190(data, dependencies=['close'], max_window=40):
    # LOG((COUNT(RET>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)
    # *SUMIF((RET-(CLOSE/DELAY(CLOSE,19))^(1/20)-1)^2,20,RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1)
    # /(COUNT(RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1,20)
    # *SUMIF((RET-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,RET>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))
    ret = data['close'].pct_change(periods=1)
    ret_19 = (data['close']/data['close'].shift(19))**0.05-1.0
    part1 = (ret>ret_19).rolling(window=20, min_periods=20).sum()-1.0
    part2 = (np.minimum(ret-ret_19, 0.0) ** 2).rolling(window=20,min_periods=20).sum()
    part3 = (ret<ret_19).rolling(window=20, min_periods=20).sum()
    part4 = (np.maximum(ret-ret_19, 0.0) ** 2).rolling(window=20,min_periods=20).sum()
    return np.log(part1*part2/part3/part4)

def alpha191(data, dependencies=['volume', 'low', 'close', 'high'], max_window=25):
    # CORR(MEAN(VOLUME,20),LOW,5)+(HIGH+LOW)/2-CLOSE
    part1 = (data['volume'].rolling(window=20,min_periods=20).mean()).rolling(window=5,min_periods=5).corr(data['low'])
    return (part1 + data['high']*0.5+data['low']*0.5-data['close'])
