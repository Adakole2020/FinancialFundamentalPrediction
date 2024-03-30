import math
from tsai.basics import prepare_forecasting_data

import warnings
import torch
from torch.utils.data import DataLoader, Dataset

import spacetimeformer as stf
from spacetimeformer.data import DataModule
from sklearn.preprocessing import StandardScaler
from typing import List
import pandas as pd
import numpy as np

class FundamentalsCSVSeries:
    """
    Class for handling financial fundamentals data from a CSV file.
    """

    def __init__(
        self,
        val_split: float = 0.15,
        test_split: float = 0.15,
        context_length: int = 18,
        prediction_length: int = 8,
        time_features: List[str] = ["year", "month"],
        normalize: bool = True,
    ):
        """
        Initialize the FundamentalsCSVSeries object.

        Parameters:
        - val_split: float, optional (default=0.15)
            The split ratio for the validation set.
        - test_split: float, optional (default=0.15)
            The split ratio for the test set.
        - context_length: int, optional (default=18)
            The number of previous timesteps given to the model in order to make predictions.
        - prediction_length: int, optional (default=8)
            The number of timesteps to predict.
        - time_features: List[str], optional (default=["year", "month"])
            The time features to use for modeling.
        - normalize: bool, optional (default=True)
            Whether to normalize the data.

        """
        #CHANGED_FILE
        raw_df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSdLYZtOd12U14IGzypjzIX1q69OMuhW_AsTaOgdZc7UgqJSvuBIC8V85kZeQHnqiCxaB7Ezsru_ri7/pub?gid=198200420&single=true&output=csv")
        
        # Preprocessing steps
        raw_df = raw_df[raw_df['quarter'].notnull()].reset_index(drop = True)
        raw_df['eps_surprise'] = np.where(raw_df['eps_normalized_consensus_mean'] == 0, raw_df['eps_normalized_actual'], (raw_df["eps_normalized_actual"] - raw_df["eps_normalized_consensus_mean"])/raw_df["eps_normalized_consensus_mean"])
        raw_df.drop(columns = ["eps_normalized_actual"], inplace = True)
        raw_df = raw_df.sort_values(['symbol', 'quarter']).reset_index(drop=True)
        target_cols = ["eps_surprise"]
        group_cols = ["symbol"]
        
        self.time_col_name = 'quarter'
        assert self.time_col_name in raw_df.columns

        time_df = pd.to_datetime(raw_df[self.time_col_name])
        df = stf.data.timefeatures.time_features(
            time_df,
            raw_df,
            time_col_name=self.time_col_name,
            use_features=time_features,
        )
        self.time_cols = df.columns.difference(raw_df.columns)
        self.target_cols = target_cols
        self.context_cols = df.columns[~(df.columns.isin([*self.time_cols, *group_cols, self.time_col_name, *self.target_cols]))].tolist()
        self.categorical_context_cols = ["Sector", "Industry Group", "Industry", "Sub-Industry"]
        for col in self.categorical_context_cols:
            df[col] = df[col].astype('category')
        
        # Select columns with 'float64' dtype
        float64_cols = list(df.select_dtypes(include='float64'))

        # Convert selected columns to 'float32' dtype
        df[float64_cols] = df[float64_cols].astype('float32')
            
        # Convert categorical columns to numerical codes
        df[self.categorical_context_cols] = df[self.categorical_context_cols].apply(lambda x: x.cat.codes)
        
        self.normalize = normalize
        self._scaler = StandardScaler()
        if normalize:
            self._scaler = self._scaler.fit(
                df[[x for x in self.context_cols if x not in self.categorical_context_cols]].values
            )
            
        df = self.apply_scaling_df(df)
        
        # Fill missing values with a large negative number
        df[float64_cols] = df[float64_cols].fillna(value=-1e2)
        
        grouped_df= df.groupby(group_cols)
        ctxt_x_train, trgt_x_train = np.empty((0, context_length, len(self.time_cols))), np.empty((0, prediction_length, len(self.time_cols)))
        ctxt_y_train, trgt_y_train = np.empty((0, context_length, len(self.context_cols)+len(self.target_cols))), np.empty((0, prediction_length, len(self.target_cols)))
        ctxt_x_val, trgt_x_val = np.empty((0, context_length, len(self.time_cols))), np.empty((0, prediction_length, len(self.time_cols)))
        ctxt_y_val, trgt_y_val = np.empty((0, context_length, len(self.context_cols)+len(self.target_cols))), np.empty((0, prediction_length, len(self.target_cols)))
        ctxt_x_test, trgt_x_test = np.empty((0, context_length, len(self.time_cols))), np.empty((0, prediction_length, len(self.time_cols)))
        ctxt_y_test, trgt_y_test = np.empty((0, context_length, len(self.context_cols)+len(self.target_cols))), np.empty((0, prediction_length, len(self.target_cols)))
        
        test_group_identifiers = []

        for group in grouped_df.groups.keys():
            mini_df = self._arrange_cols(grouped_df.get_group(group).reset_index(drop=True))
            if len(mini_df.index) < prediction_length + context_length:
                continue

            ctxt_y, trgt_y = prepare_forecasting_data(mini_df, fcst_history=context_length, fcst_horizon=prediction_length, x_vars=self.target_cols+[x for x in self.context_cols if x not in self.categorical_context_cols]+self.categorical_context_cols, y_vars=self.target_cols)
            ctxt_x, trgt_x = prepare_forecasting_data(mini_df, fcst_history=context_length, fcst_horizon=prediction_length, x_vars=self.time_cols, y_vars=self.time_cols)


            ctxt_y, trgt_y = np.einsum('ijk -> ikj', ctxt_y), np.einsum('ijk -> ikj', trgt_y)
            ctxt_x, trgt_x = np.einsum('ijk -> ikj', ctxt_x), np.einsum('ijk -> ikj', trgt_x)
            test_start = math.ceil(test_split*ctxt_x.shape[0]) if test_split < 1 else int(test_split)
            valid_start = math.ceil(val_split*ctxt_x.shape[0]) + test_start if val_split < 1 else int(val_split) + test_start
            

            if test_start != 0:
                ctxt_y_test = np.concatenate((ctxt_y_test, ctxt_y[-test_start:]), axis=0)
                trgt_y_test = np.concatenate((trgt_y_test, trgt_y[-test_start:]), axis=0)

                ctxt_x_test = np.concatenate((ctxt_x_test, ctxt_x[-test_start:]), axis=0)
                trgt_x_test = np.concatenate((trgt_x_test, trgt_x[-test_start:]), axis=0)
                
                test_group_identifiers.extend([group]*test_start)
                
            if valid_start != 0:
                ctxt_y_val = np.concatenate((ctxt_y_val, ctxt_y[-valid_start:-test_start]), axis=0) if test_start != 0 else np.concatenate((ctxt_y_val, ctxt_y[-valid_start:]), axis=0)
                trgt_y_val = np.concatenate((trgt_y_val, trgt_y[-valid_start:-test_start]), axis=0) if test_start != 0 else np.concatenate((trgt_y_val, trgt_y[-valid_start:]), axis=0)

                ctxt_x_val = np.concatenate((ctxt_x_val, ctxt_x[-valid_start:-test_start]), axis=0) if test_start != 0 else np.concatenate((ctxt_x_val, ctxt_x[-valid_start:]), axis=0)
                trgt_x_val = np.concatenate((trgt_x_val, trgt_x[-valid_start:-test_start]), axis=0) if test_start != 0 else np.concatenate((trgt_x_val, trgt_x[-valid_start:]), axis=0)

                ctxt_y_train = np.concatenate((ctxt_y_train, ctxt_y[:-valid_start]), axis=0)
                trgt_y_train = np.concatenate((trgt_y_train, trgt_y[:-valid_start]), axis=0)

                ctxt_x_train = np.concatenate((ctxt_x_train, ctxt_x[:-valid_start]), axis=0)
                trgt_x_train = np.concatenate((trgt_x_train, trgt_x[:-valid_start]), axis=0)
                
            else:
                ctxt_y_train = np.concatenate((ctxt_y_train, ctxt_y), axis=0)
                trgt_y_train = np.concatenate((trgt_y_train, trgt_y), axis=0)

                ctxt_x_train = np.concatenate((ctxt_x_train, ctxt_x), axis=0)
                trgt_x_train = np.concatenate((trgt_x_train, trgt_x), axis=0)
                
                
        self._train_data = (ctxt_x_train, ctxt_y_train, trgt_x_train, trgt_y_train) # (4, n_samples, n_timesteps, n_features)
        self._val_data = (ctxt_x_val, ctxt_y_val, trgt_x_val, trgt_y_val) # (4, n_samples, n_timesteps, n_features)
        self._test_data = (ctxt_x_test, ctxt_y_test, trgt_x_test, trgt_y_test) # (4, n_samples, n_timesteps, n_features)
        self.test_group_identifiers = np.array(test_group_identifiers)
        
        
    
    @property
    def train_data(self):
        """
        Get the training data.

        Returns:
        - train_data: tuple
            A tuple containing the training data arrays.
        """
        return self._train_data

    @property
    def val_data(self):
        """
        Get the validation data.

        Returns:
        - val_data: tuple
            A tuple containing the validation data arrays.
        """
        return self._val_data

    @property
    def test_data(self):
        """
        Get the test data.

        Returns:
        - test_data: tuple
            A tuple containing the test data arrays.
        """
        return self._test_data

    def length(self, split):
        """
        Get the length of a specific split.

        Parameters:
        - split: str
            The split name ("train", "val", or "test").

        Returns:
        - length: int
            The length of the specified split.
        """
        return {
            "train": self._train_data[0].shape[0],
            "val": self._val_data[0].shape[0],
            "test": self._test_data[0].shape[0],
        }[split]
        
    def _arrange_cols(self, df):
        """
        Rearrange the columns of the DataFrame.

        Parameters:
        - df: pandas.DataFrame
            The DataFrame to rearrange.

        Returns:
        - df: pandas.DataFrame
            The rearranged DataFrame.
        """
        df = df[self.time_cols.tolist() + self.target_cols + [x for x in self.context_cols if x not in self.categorical_context_cols] + self.categorical_context_cols]
        return df
    
    def apply_scaling_df(self, df):
        """
        Apply scaling to the DataFrame.

        Parameters:
        - df: pandas.DataFrame
            The DataFrame to apply scaling to.

        Returns:
        - scaled: pandas.DataFrame
            The scaled DataFrame.
        """
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = [x for x in self.context_cols if x not in self.categorical_context_cols]
        dtype = df[cols].values.dtype
        scaled[cols] = self._scaler.transform(scaled[cols]).astype(dtype)
        return scaled

    def apply_scaling(self, array):
        """
        Apply scaling to the array.

        Parameters:
        - array: numpy.ndarray
            The array to apply scaling to.

        Returns:
        - scaled: numpy.ndarray
            The scaled array.
        """
        if not self.normalize:
            return array
        dim = array.shape[-1]
        return (array - self._scaler.mean_[:dim]) / self._scaler.scale_[:dim]


    def reverse_scaling_df(self, df):
        """
        Reverse the scaling of the DataFrame.

        Parameters:
        - df: pandas.DataFrame
            The DataFrame to reverse scaling.

        Returns:
        - scaled: pandas.DataFrame
            The DataFrame with scaling reversed.
        """
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = self.context_cols
        dtype = df[cols].values.dtype
        scaled[cols] = self._scaler.inverse_transform(scaled[cols]).astype(dtype)
        return scaled

    def reverse_scaling(self, array):
        """
        Reverse the scaling of the array.

        Parameters:
        - array: numpy.ndarray
            The array to reverse scaling.

        Returns:
        - scaled: numpy.ndarray
            The array with scaling reversed.
        """
        if not self.normalize:
            return array
        dim = array.shape[-1]
        return (array * self._scaler.scale_[:dim]) + self._scaler.mean_[:dim]
    
    @classmethod
    def add_cli(self, parser):
        """
        Add command-line arguments for the FundamentalsCSVSeries class.

        Parameters:
        - parser: argparse.ArgumentParser
            The argument parser object.
        """
        parser.add_argument(
            "--context_points",
            type=int,
            default=18,
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default=8,
            help="number of future timesteps to predict",
        )
        
    
class FundamentalsDset(Dataset):
    """
    Dataset class for handling financial fundamentals data.

    Args:
        csv_time_series (FundamentalsCSVSeries): The CSV time series data.
        split (str, optional): The split of the dataset (train, val, or test). Defaults to "train".
    """

    def __init__(
        self,
        csv_time_series: FundamentalsCSVSeries,
        split: str = "train",
    ):
        """
        Initializes the FundamentalsDset class.

        Args:
            csv_time_series (FundamentalsCSVSeries): The CSV time series data.
            split (str, optional): The split of the dataset (train, val, or test). Defaults to "train".
        """
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = csv_time_series
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.series.length(self.split)

    def _torch(self, *dfs):
        """
        Converts the input data to torch tensors.

        Args:
            *dfs: Variable number of input data arrays.

        Returns:
            tuple: Tuple of torch tensors.
        """
        return tuple(torch.from_numpy(x).float() for x in dfs)

    def __getitem__(self, i):
        """
        Returns the i-th item from the dataset.

        Args:
            i (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple of torch tensors representing the i-th item.
        """
        if self.split == "train":
            return self._torch(self.series.train_data[0][i], self.series.train_data[1][i], self.series.train_data[2][i], self.series.train_data[3][i])
        elif self.split == "val":
            return self._torch(self.series.val_data[0][i], self.series.val_data[1][i], self.series.val_data[2][i], self.series.val_data[3][i])
        else:
            return self._torch(self.series.test_data[0][i], self.series.test_data[1][i], self.series.test_data[2][i], self.series.test_data[3][i])
        
        
class FundamentalsDataModule(DataModule):
    """
    Data module for handling financial fundamentals data.
    """

    def __init__(
        self,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int,
        collate_fn=None,
        overfit: bool = False,
    ):
        """
        Initializes the FundamentalsDataModule.

        Args:
            dataset_kwargs (dict): Keyword arguments for the dataset.
            batch_size (int): The batch size for the data loader.
            workers (int): The number of workers for data loading.
            collate_fn (callable, optional): Function to use for custom collation of samples. Defaults to None.
            overfit (bool, optional): Whether to overfit the data. Defaults to False.
        """
        super().__init__(FundamentalsDset, dataset_kwargs, batch_size, workers, collate_fn, overfit)
        self.batch_size = batch_size
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.series = FundamentalsCSVSeries(**dataset_kwargs)
        self.datasetCls = FundamentalsDset
        self.workers = workers
        self.collate_fn = collate_fn
        if overfit:
            warnings.warn("Overriding val and test dataloaders to use train set!")
        self.overfit = overfit
        
    def _make_dloader(self, split, shuffle=False):
        """
        Creates a data loader for the specified split.

        Args:
            split (str): The split to create the data loader for.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            DataLoader: The data loader for the specified split.
        """
        if self.overfit:
            split = "train"
            shuffle = True
        return DataLoader(
            self.datasetCls(self.series, split=split),
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )