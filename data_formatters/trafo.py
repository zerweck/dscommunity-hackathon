"""Custom formatting functions for Tine-Trafo dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import datetime
import pandas as pd
import sklearn.preprocessing

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class TrafoFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the sandy tradeopt dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('timestampUtc', DataTypes.DATE, InputTypes.TIME),
      ('value', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('t_2mc', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('aswdifd_s_0', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('aswdir_s_0', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('gr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('tp_0', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_60m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_90m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_120m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_150m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_180m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_210m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('wspd_240m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('tlp.ep1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('tlp.ez2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('h0dynamic', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('typeDayInt', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('hourMinuteInt', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),

      ('bridgeDay', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('id', DataTypes.REAL_VALUED, InputTypes.ID)
  ]
  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=datetime.datetime(2020, 6, 1), test_boundary=datetime.datetime(2020, 9, 1)):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """
    print('Formatting train-valid-test splits.')
    fixed_params = self.get_fixed_params()
    lookback = fixed_params['num_encoder_steps']

    df['timestampUtc'] = pd.to_datetime(df['timestampUtc'])
    index = df['timestampUtc']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary - datetime.timedelta(lookback / 96)) &
                   (index < test_boundary)]
    test = df.loc[index >= test_boundary - datetime.timedelta(lookback / 96)]
    print("Obtained dataset with nrow train: {0}, valid: {1}, test: {2}".format(len(train), len(valid), len(test)))
    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(targets)
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Transform real inputs per entity
    df_list = []
    for identifier, sliced in df.groupby(id_col):

      # Filter out any trajectories that are too short
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
          sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 4 * 24 * 4,
        'num_encoder_steps': 3 * 24 * 4,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 160,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 1,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1
