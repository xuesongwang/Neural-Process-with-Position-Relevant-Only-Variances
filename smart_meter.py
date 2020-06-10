from pathlib import Path
import pandas as pd
import numpy as np
import torch


class SmartMeterDataSet(torch.utils.data.Dataset):
    def __init__(self, df, num_context=40, num_extra_target=10, label_names=['energy(kWh/hh)']):
        self.df = df
        self.num_context = num_context
        self.num_extra_target = num_extra_target
        self.label_names = label_names

    def get_rows(self, i):
        rows = self.df.iloc[i: i + (self.num_context + self.num_extra_target)].copy()
        rows['tstp'] = (rows['tstp'] - rows['tstp'].iloc[0]).dt.total_seconds() / 86400.0
        rows = rows.sort_values('tstp')

        # make sure tstp, which is our x axis, is the first value
        columns = ['tstp'] + list(set(rows.columns) - set(['tstp'])) + ['future']
        rows['future'] = 0.
        rows = rows[columns]

        # This will be the last row, and will change it upon sample to let the model know some points are in the future

        x = rows[['tstp']].copy()
        y = rows[self.label_names].copy()
        return x, y

    def __getitem__(self, i):
        x, y = self.get_rows(i)
        return x.values, y.values

    def __len__(self):
        return len(self.df) - (self.num_context + self.num_extra_target)

def get_smartmeter_df(indir=Path('../NP/data/smart-meters-in-london'), use_logy=False):
    csv_files = '../NP/data/smart-meters-in-london/halfhourly_dataset/block_0.csv'
    df = pd.read_csv(csv_files, parse_dates=[1], na_values=['Null'])
    #     print(df.info())

    df = df.groupby('tstp').mean()
    df['tstp'] = df.index
    df.index.name = ''

    # Load weather data
    df_weather = pd.read_csv(indir / 'weather_hourly_darksky.csv', parse_dates=[3])

    use_cols = ['visibility', 'windBearing', 'temperature', 'time', 'dewPoint',
                'pressure', 'apparentTemperature', 'windSpeed',
                'humidity']
    df_weather = df_weather[use_cols].set_index('time')

    # Resample to match energy data
    df_weather = df_weather.resample('30T').ffill()

    # Normalise
    weather_norms = dict(mean={'visibility': 11.2,
                               'windBearing': 195.7,
                               'temperature': 10.5,
                               'dewPoint': 6.5,
                               'pressure': 1014.1,
                               'apparentTemperature': 9.2,
                               'windSpeed': 3.9,
                               'humidity': 0.8},
                         std={'visibility': 3.1,
                              'windBearing': 90.6,
                              'temperature': 5.8,
                              'dewPoint': 5.0,
                              'pressure': 11.4,
                              'apparentTemperature': 6.9,
                              'windSpeed': 2.0,
                              'humidity': 0.1})

    for col in df_weather.columns:
        df_weather[col] -= weather_norms['mean'][col]
        df_weather[col] /= weather_norms['std'][col]

    df = pd.concat([df, df_weather], 1).dropna()

    # Also find bank holidays
    df_hols = pd.read_csv(indir / 'uk_bank_holidays.csv', parse_dates=[0])
    holidays = set(df_hols['Bank holidays'].dt.round('D'))

    df['holiday'] = df.tstp.apply(lambda dt: dt.floor('D') in holidays).astype(int)

    # Add time features
    time = df.tstp
    df["month"] = time.dt.month / 12.0
    df['day'] = time.dt.day / 310.0
    df['week'] = time.dt.week / 52.0
    df['hour'] = time.dt.hour / 24.0
    df['minute'] = time.dt.minute / 24.0
    df['dayofweek'] = time.dt.dayofweek / 7.0

    # Drop nan and 0's
    df = df[df['energy(kWh/hh)'] != 0]
    df = df.dropna()

    if use_logy:
        df['energy(kWh/hh)'] = np.log(df['energy(kWh/hh)'] + 1e-4)
    df = df.sort_values('tstp')

    # split data
    n_split = -int(len(df) * 0.1)
    df_train = df[:3 * n_split]
    df_val = df[3 * n_split:n_split]
    df_test = df[n_split:]
    return df_train, df_val, df_test

def collate_fns(max_num_context, max_num_extra_target):
    def collate_fn(batch):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(3, max_num_context)
        num_extra_target = np.random.randint(3, max_num_extra_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Last feature will show how far in time a point is from out last context
        assert (np.diff(x[:, :, 0], 1) >= 0).all(), 'first features should be ordered e.g. seconds'
        # assert (x[:, max_num_context, -1] == 0.).all(), 'last features should be empty'
        # time = x[:, :, 0]
        # t0 = x[:, max_num_context, 0][:, None]
        # x[:, :, -1] = time - t0  # Feature to let the model know this is past data

        inds = np.random.choice(range(x.shape[1]), size=(num_context + num_extra_target), replace=False)
        x_context = x[:, inds][:, :num_context]
        y_context = y[:, inds][:, :num_context]

        x_target = x[:, inds][:, num_context:]
        y_target = y[:, inds][:, num_context:]

        return x_context, y_context, x_target, y_target

    return collate_fn

class SmartMeterDataLoader:
    def __init__(self, max_context=50, max_target = 50, batch_size = 16):
        super().__init__()
        self._dfs = None
        hparams = {}
        hparams["num_context"] = max_context
        hparams["num_extra_target"] = max_target
        hparams["batch_size"] = batch_size
        hparams["num_workers"] = 4
        self.hparams = hparams

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_val, df_test = get_smartmeter_df()
            # self._dfs = dict(df_train=df_train[:600], df_test=df_test[:600])
            self._dfs = dict(df_train=df_train, df_val = df_val, df_test=df_test)
        return self._dfs

    def train_dataloader(self):
        df_train = self._get_cache_dfs()['df_train']
        data_train = SmartMeterDataSet(
            df_train, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            collate_fn=collate_fns(self.hparams["num_context"], self.hparams["num_extra_target"]),
            num_workers=self.hparams["num_workers"])

    def val_dataloader(self):
        df_val = self._get_cache_dfs()['df_val']
        data_val = SmartMeterDataSet(
            df_val, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_val,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            collate_fn=collate_fns(self.hparams["num_context"], self.hparams["num_extra_target"]),
        )

    def test_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
        data_test = SmartMeterDataSet(
            df_test, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"]),
        )


if __name__ == '__main__':
    dataloader = SmartMeterDataLoader()
    dataloader_train = dataloader.train_dataloader()
    # dataloader_val = dataloader.val_dataloader()
    # dataloader_test = dataloader.test_dataloader()

    print(len(dataloader_train))
    for batch in dataloader_train:
        x_context, y_context, x_target, y_target = batch