import os
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wavelet_demo(feature,data_path):
    """
    Take one feature of one stock and do decomposition
    """
    # Compute
    feature_df = pd.read_csv(data_path,parse_dates=["date"], index_col = "date", usecols=["date",feature,"comp_name"])
    selected_comp_name = feature_df[feature_df[feature].notna()].groupby("comp_name").size().idxmax()
    select_feature =  feature_df[feature_df["comp_name"] == selected_comp_name][feature]
    approx_coeffs = np.array([pywt.dwt(select_feature, "haar", mode="symmetric")[0]]).T
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(select_feature, label=f"Original")
    ax.plot(select_feature.index[::2],approx_coeffs, label='DWT Approximation Coefficients', color='r')
    ax.set_title(f"Signal for {selected_comp_name}")
    ax.legend()
    fig.tight_layout()
    return fig