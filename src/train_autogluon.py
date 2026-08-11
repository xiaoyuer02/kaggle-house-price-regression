from autogluon.tabular import TabularPredictor, TabularDataset
import numpy as np

# load data
train_data = TabularDataset('data/train.csv')

# preprocessing
col, label = 'Id', 'SalePrice'

train_data[label] = np.log1p(train_data[label])

# train   
predictor = TabularPredictor(label = label).fit(train_data.drop(columns = [col]))

# predict
import pandas as pd
test_data = TabularDataset('data/test.csv')

test_data[label] = np.log1p(test_data[label])

# predict
preds = predictor.predict(test_data.drop(columns = [col]))

# submit
submission = pd.DataFrame({
    'Id': test_data[col], 
    'SalePrice': np.expm1(preds)})
submission.to_csv('submission_autogluon.csv', index = False)
