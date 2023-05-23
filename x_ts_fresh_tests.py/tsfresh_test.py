# from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
#     load_robot_execution_failures
# download_robot_execution_failures()
# timeseries, y = load_robot_execution_failures()
# timeseries = timeseries.iloc[:, :3]
# timeseries[timeseries['id'] == 2].plot(subplots=True,
#                                        sharex=True,
#                                        figsize=(10, 10))
# plt.show()
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
settings = ComprehensiveFCParameters()

# # filtering data

df_shift, _ = make_forecasting_frame(data.close,
                                     kind="price",
                                     max_timeshift=20,
                                     rolling_direction=1)

X = extract_features(df_shift,
                     column_id="id",
                     column_sort="time",
                     column_value="value",
                     impute_function=impute,
                     show_warnings=False,
                     default_fc_parameters=settings)

import pandas as pd

for i in range(100):
    print(i)
    if i == 0:
        df = pd.DataFrame(data=X[i])
        df['id'] = i + 1
        df['time'] = df.index + 0
    else:
        df2 = pd.DataFrame(data=X[i])
        df2['id'] = i + 1
        df2['time'] = df2.index + 0
        df = pd.concat([df, df2])

df = df[['id', 'time', 0]]
extracted_features = extract_features(df, column_id="id", column_sort="time")

impute(extracted_features)
features_filtered = select_features(extracted_features, y)

X = features_filtered.values
