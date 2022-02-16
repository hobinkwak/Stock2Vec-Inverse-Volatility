## Inverse Volatility Allocation by Sto2vec Clustering

### 

### Data
- ETF (2008.01 ~ 2022.02)
  - VTI : Vanguard Total Stock Market Index Fund ETF
  - VEA : Vanguard Developed Markets Index Fund ETF
  - VWO : Vanguard Emerging Markets Stock Index Fund ETF
  - IAU : iShares COMEX Gold Trust ETF
  - DBC : Invesco DB Commodity Index Tracking Fund
  - XLB : Materials Select Sector SPDR Fund
  - XLE : Energy Select Sector SPDR Fund
  - XLF : Financial Select Sector SPDR Fund
  - XLI : Industrial Select Sector SPDR Fund
  - XLK : Technology Select Sector SPDR Fund
  - XLP : Consumer Staples Select Sector SPDR Fund
  - XLU : Utilities Select Sector SPDR Fund
  - XLV : Health Care Select Sector SPDR Fund
  - XLY : 미 경기소비재 ETF

### Implemenation
```
python main.py
```

## Requirements
```
finance-datareader==0.9.31
gensim==4.1.2
matplotlib
numpy==1.19.3
pandas==1.3.4
scikit-learn==1.0.2
scipy==1.8.0
tqdm==4.62.3
```
