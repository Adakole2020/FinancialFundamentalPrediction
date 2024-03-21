import math
from tsai.basics import prepare_forecasting_data

import warnings
import torch
from torch.utils.data import DataLoader, Dataset

import spacetimeformer as stf
from spacetimeformer.data import DataModule

class FundamentalsCSVSeries:
    def __init__(
        self,
        val_split: float = 0.15,  # int or float indicating the split for the validation set
        test_split: float = 0.15, # int or float indicating the split for the test set
        context_length: int = 18, # Use the context of 18 quarters to predict the next 8 quarters
        prediction_length: int = 8,
        time_features: List[str] = [
            "year",
            "month"
        ],
        normalize: bool = True,):
        
        raw_df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSdLYZtOd12U14IGzypjzIX1q69OMuhW_AsTaOgdZc7UgqJSvuBIC8V85kZeQHnqiCxaB7Ezsru_ri7/pubhtml?gid=198200420&single=true")
        raw_df = raw_df[raw_df['quarter'].notnull()].reset_index(drop = True)
        raw_df['eps_surprise'] = np.where(raw_df['eps_normalized_consensus_mean'] == 0, raw_df['eps_normalized_actual'], (raw_df["eps_normalized_actual"] - raw_df["eps_normalized_consensus_mean"])/raw_df["eps_normalized_consensus_mean"])
        raw_df.drop(columns = ["eps_normalized_actual"], inplace = True)
        raw_df = raw_df.sort_values(['symbol', 'quarter']).reset_index(drop=True)
        
        self.time_col_name = time_col_name
        assert self.time_col_name in raw_df.columns
        
        if not target_cols:
            target_cols = raw_df.columns.tolist()
            target_cols.remove(time_col_name)
            
        if ignore_cols:
            if ignore_cols == "all":
                ignore_cols = raw_df.columns.difference(target_cols).tolist()
                ignore_cols.remove(self.time_col_name)
            raw_df.drop(columns=ignore_cols, inplace=True)

        time_df = pd.to_datetime(raw_df[self.time_col_name], format="%Y-%m")
        df = stf.data.timefeatures.time_features(
            time_df,
            raw_df,
            time_col_name=self.time_col_name,
            use_features=time_features,
        )
        self.time_cols = df.columns.difference(raw_df.columns)
        self.target_cols = ["eps_surprise"]
        group_cols = ["symbol"]
        self.context_cols = df.columns[~(df.columns.isin([*self.time_cols, *self.group_cols]))]
        self.categorical_context_cols = ["Sector", "Industry Group", "Industry", "Sub-Industry"]
        for col in self.categorical_context_cols:
            df[col] = df[col].astype('category')
        
        # Select columns with 'float64' dtype
        float64_cols = list(df.select_dtypes(include='float64'))

        # The same code again calling the columns
        df[float64_cols] = df[float64_cols].astype('float32')
        df[float64_cols] = df[float64_cols].fillna(value=np.NAN)
            
        df[self.categorical_context_cols] = df[self.categorical_context_cols].apply(lambda x: x.cat.codes)
        
        self.normalize = normalize
        if normalize:
            self._scaler = self._scaler.fit(
                df[[x for x in self.context_cols if x not in self.categorical_context_cols]].values
            )
        df = self.apply_scaling_df(df)

        grouped_df= df.groupby(group_cols)
        ctxt_x_train, trgt_x_train = np.empty((0, len(self.time_cols), context_length)), np.empty((0, len(self.time_cols), prediction_length))
        ctxt_y_train, trgt_y_train = np.empty((0, len(self.context_cols), context_length)), np.empty((0, len(self.target_cols), prediction_length))
        ctxt_x_val, trgt_x_val = np.empty((0, len(self.time_cols), context_length)), np.empty((0, len(self.time_cols), prediction_length))
        ctxt_y_val, trgt_y_val = np.empty((0, len(self.context_cols), context_length)), np.empty((0, len(self.target_cols), prediction_length))
        ctxt_x_test, trgt_x_test = np.empty((0, len(self.time_cols), context_length)), np.empty((0, len(self.time_cols), prediction_length))
        ctxt_y_test, trgt_y_test = np.empty((0, len(self.context_cols), context_length)), np.empty((0, len(self.target_cols), prediction_length))

        for group in grouped_df.groups.keys():
            mini_df = self._arrange_cols(grouped_df.get_group(group).reset_index(drop=True))
            if len(mini_df.index) < prediction_length + context_length:
                continue

            ctxt_y, trgt_y = prepare_forecasting_data(mini_df, fcst_history=context_length, fcst_horizon=prediction_length, x_vars=self.context_cols, y_vars=self.target_cols)
            ctxt_x, trgt_x = prepare_forecasting_data(mini_df, fcst_history=context_length, fcst_horizon=prediction_length, x_vars=self.time_cols, y_vars=self.time_cols)

            test_start = math.ceil(test_split*ctxt_x.shape[0]) if test_split < 1 else test_split
            valid_start = math.ceil(val_split*ctxt_x.shape[0]) + test_start if val_split < 1 else val_split + test_start

            if test_start != 0:
                ctxt_y_test = np.concatenate((ctxt_y_test, ctxt_y[-test_start:]), axis=0)
                trgt_y_test = np.concatenate((trgt_y_test, trgt_y[-test_start:]), axis=0)

                ctxt_x_test = np.concatenate((ctxt_x_test, ctxt_x[-test_start:]), axis=0)
                trgt_x_test = np.concatenate((trgt_x_test, trgt_x[-test_start:]), axis=0)
            
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
                
        self._train_data = np.array([ctxt_x_train, trgt_x_train, ctxt_y_train, trgt_y_train]) # (4, n_samples, n_features, n_timesteps)
        self._val_data = np.array([ctxt_x_val, trgt_x_val, ctxt_y_val, trgt_y_val]) # (4, n_samples, n_features, n_timesteps)
        self._test_data = np.array([ctxt_x_test, trgt_x_test, ctxt_y_test, trgt_y_test]) # (4, n_samples, n_features, n_timesteps)
        
        
    
    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def length(self, split):
        return {
            "train": self._train_data.shape[1],
            "val": self._val_data.shape[1],
            "test": self._test_data.shape[1],
        }[split]
        
    def _arrange_cols(self, df):
        df = df[self.time_cols.tolist() + self.target_cols + [x for x in self.context_cols if x not in self.categorical_context_cols] + self.categorical_context_cols]
        return df
    
    def apply_scaling_df(self, df):
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = [x for x in self.context_cols if x not in self.categorical_context_cols]
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values - self._scaler.mean_.astype(dtype)
        ) / self._scaler.scale_.astype(dtype)
        return scaled

    def apply_scaling(self, array):
        if not self.normalize:
            return array
        dim = array.shape[-1]
        return (array - self._scaler.mean_[:dim]) / self._scaler.scale_[:dim]


    def reverse_scaling_df(self, df):
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = self.context_cols
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values * self._scaler.scale_.astype(dtype)
        ) + self._scaler.mean_.astype(dtype)
        return scaled

    def reverse_scaling(self, array):
        if not self.normalize:
            return array
        # self._scaler is fit for exo_cols
        # if the array dim is less than this length we start
        # slicing from the target cols
        dim = array.shape[-1]
        return (array * self._scaler.scale_[:dim]) + self._scaler.mean_[:dim]
        
    
class FundamentalsDset(Dataset):
    def __init__(
        self,
        csv_time_series: FundamentalsCSVSeries,
        split: str = "train",
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = csv_time_series
        
    def __len__(self):
        return len(self.series.length(self.split))

    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    def __getitem__(self, i):
        if self.split == "train":
            return self._torch(self.series.train_data[:, i])
        elif self.split == "val":
            return self._torch(self.series.val_data[:, i])
        else:
            return self._torch(self.series.test_data[:, i])
        
    @classmethod
    def add_cli(self, parser):
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
        parser.add_argument(
            "--time_resolution",
            type=int,
            default=1,
        )
        
        
class FundamentalsDataModule(DataModule):
    def __init__(
        self,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int,
        collate_fn=None,
        overfit: bool = False,
    ):
        super().__init__()
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