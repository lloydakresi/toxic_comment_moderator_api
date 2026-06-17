import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
from datasets import Dataset
from eda import training_set, class_cols

y = training_set.select(class_cols).to_numpy()
X_dummy = np.zeros((training_set.height, 1))

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.1, random_state=42
)

train_idx, val_idx = next(msss.split(X_dummy, y))
train_set = training_set[train_idx]
val_set = training_set[val_idx]
print(train_set.height)

train_ds = Dataset.from_polars(train_set)
val_ds = Dataset.from_polars(val_set)
