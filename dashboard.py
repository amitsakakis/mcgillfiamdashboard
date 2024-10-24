import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import helpers as hp
from display_ticker_tape import display_ticker_tape

################
## Page setup ##
################
st.set_page_config(
    page_title="FIAM Vol Skew - Wavelet Transform Model", page_icon="🏄", layout="wide"
)

col = st.columns((2, 3, 2))

############################
## Global text variables ##
############################

INTRO_WAVELET_1 = "We approach the problem of predicting stock returns with methods from signal processing. \
    With the many features available in the given data set, we aim first to remove as much noise from them as possible. \
    Specifically, we define noise as random fluctuations in each feature, which then by definition do not carry any useful information. As the nature of the data is discrete, we rely on the discrete wavelet transform (DWT) to decompose each feature in its more fundamental frequencies. We specifically choose the Haar wavelet, which is a simple up and down step function."

INTRO_WAVELET_2 = "From the transform, we obtain approximation coefficients \
    which contain the low-frequency information of the signal, or in other words its general trend (we discard detail coefficients, which represent finer frequencies). \
    Specifically, this is analogous to performing a low-pass filter on any signal. We apply the same on the stock_return variable, which \
    we want to predict out-of-sample."

INTRO_XGB = (
    "With the data de-noised, we train an XGBoost model on these approximation coefficients using an \
    expanding 1 year window. We predict stock return coefficients 1 year out of sample, for each month in that year, and then perform an \
    inverse discrete wavelet transform (IDWT) using only the approximation coefficients. This effectively reconstructs the predicted return time-series. We also perform a similar procedure using a simpler Decision Tree Regressor model and notice \
    that the XGBoost far outperforms in terms of the MSE metric."
)

# Update the file paths to the smaller data sets for demo purposes
RAW_DATA_PATH = "./oracle_costco_data.csv"
PREDICTED_RETURNS_PATH = "./predicted_returns_wavelet_final2.csv"

################################
## Interactive widget helpers ##
################################


def denoise_box_select_demo():
    map_dic = {
        "Market Beta": "beta_60m",
        "Profit Margin": "ebit_sale",
        "Earnings per Share": "eps_actual",
        "Hiring Rate": "emp_gr1",
        "Dividend Yield": "div12m_me",
        "Standardized Earnings Surprise": "niq_su",
        "Operating Cash Flow to Assets": "ocf_at",
    }
    feature = st.selectbox(
        "Select a feature to investigate and de-noise across time with a DWT",
        map_dic.keys(),
        placeholder="Select a feature...",
    )
    if feature:
        fig = hp.wavelet_demo(map_dic.get(feature), RAW_DATA_PATH)
        st.pyplot(fig)
        st.caption(
            "By default, the company which has the most data points for that feature is chosen. Note also that the \
            feature is scaled by this transformation, but it is removed during data normalization (before the model training)."
        )
    else:
        return


def tabular_predicted_df():
    model_dic = {
        "XGBoost": "XGB",
    }
    model = st.selectbox(
        "Select a model to display sample return predictions",
        model_dic.keys(),
        placeholder="Select a model...",
    )
    if model:
        df = pd.read_csv(PREDICTED_RETURNS_PATH, index_col="date")
        df.rename(
            columns={
                model_dic.get(model): "Predicted Returns",
                "stock_exret": "Real Returns",
                "permno": "Stock ID",
                f"MSE_{model_dic.get(model)}": "MSE",
            },
            inplace=True,
        )

        new_df = df[["Stock ID", "Predicted Returns", "Real Returns", "MSE"]]
        st.subheader("Sample Predicted Returns")
        st.table(new_df.head(10))

        # Return the top 10 selected stock IDs
        return new_df["Stock ID"].head(10).tolist()
    else:
        return []


def stock_selection_demo():
    df = pd.read_csv(PREDICTED_RETURNS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    stocks = df["comp_name"].unique()

    selected_stock = st.selectbox("Select a stock for portfolio analysis:", stocks)

    if selected_stock:
        selected_data = df[df["comp_name"] == selected_stock]

        # Calculate R² value
        y_true = selected_data["stock_exret"]
        y_pred = selected_data["XGB"]
        r2 = 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - y_true.mean()) ** 2))

        # Calculate Hit Ratio
        hit_ratio = (y_true * y_pred > 0).mean() * 100  # Percentage of correct predictions

        # Plotting the graph
        fig, ax = plt.subplots()
        ax.plot(selected_data["date"], y_pred, label="Predicted", color="blue")
        ax.plot(selected_data["date"], y_true, label="Real", color="orange")

        # Set plot labels and title
        ax.set_ylabel("Returns (%)")
        ax.set_title(f"Predicted vs Real Returns for {selected_stock}")

        # Customize the x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        plt.xticks(rotation=0)

        # Add legend with R² and Hit Ratio
        legend_text = f"R²: {r2:.2f} | Hit Ratio: {hit_ratio:.1f}%"
        ax.legend(title=legend_text)

        st.pyplot(fig)



def display_performance_metrics():
    df = pd.read_csv(PREDICTED_RETURNS_PATH)
    model_dic = {"XGBoost": "XGB"}
    model = st.selectbox("Select a model for performance metrics", model_dic.keys())

    if model:
        y_true = df["stock_exret"]
        y_pred = df[model_dic[model]]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")


################
##### MAIN #####
################


def main():
    # Layout columns and content
    col = st.columns((2, 3, 2))

    # Place the ticker tape inside the middle column for alignment
    with col[1]:
        display_ticker_tape()  # Display ticker tape at the top

        st.title("Vol Skew Team Model")
        st.header("Discrete Wavelet Decomposition")
        st.write(INTRO_WAVELET_1)
        denoise_box_select_demo()
        st.write(INTRO_WAVELET_2)

        st.header("XGBoost Predictor Model")
        st.write(INTRO_XGB)

        selected_stocks = tabular_predicted_df()
        st.caption("The MSE metric applies to the single predicted period that includes the returns above.")

        st.subheader("Stock Selection")
        stock_selection_demo()

        st.header("Performance Metrics")
        display_performance_metrics()

if __name__ == "__main__":
    main()

