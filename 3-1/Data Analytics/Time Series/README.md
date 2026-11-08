
# STL (Seasonal-Trend Decomposition using LOESS)

**STL (Seasonal-Trend Decomposition using LOESS)** is a versatile time series analysis technique used to break down a sequence of data into three fundamental underlying components:

\[
Y_t = T_t + S_t + R_t
\]

- **\(Y_t\) (Observed Data):** The original value at time \(t\).
- **\(T_t\) (Trend-Cycle):** The long-term direction, smooth growth, or decline in the series.
- **\(S_t\) (Seasonal Component):** Repeating, cyclical patterns that occur at fixed periodic intervals (e.g., monthly peaks, seasonal highs).
- **\(R_t\) (Remainder / Residual):** The random noise or unexplained variation after removing trend and seasonality.

---

## 1. What is LOESS?

The core engine behind STL is **LOESS (Locally Estimated Scatterplot Smoothing)**.

Instead of fitting a single global regression line (like a straight line) through an entire dataset, LOESS performs non-parametric regression by fitting multiple weighted local lines or polynomials across small windows (subsets) of data.

- Points closer to the target estimation point receive higher weight, while points farther away receive lower weight.
- This allows STL to flexibly capture non-linear trends and evolving seasonal patterns over time without assuming a rigid mathematical shape.

---

## 2. The Two-Loop Architecture of STL

STL extracts components through an iterative process consisting of an **Inner Loop** and an **Outer Loop**.

```text
  ┌─────────────────────────────────────────────────────────┐
  │                       OUTER LOOP                        │
  │  Calculates robustness weights to reduce outlier influence│
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                       INNER LOOP                        │
  │  Step 1: Detrending         (Y_t - T_t)                │
  │  Step 2: Cycle-Subseries    Group by season             │
  │  Step 3: Seasonality Filter LOESS smoothing + Low-pass  │
  │  Step 4: Deseasonalizing   (Y_t - S_t)                │
  │  Step 5: Trend Extraction   LOESS smoothing for T_t     │
  └─────────────────────────────────────────────────────────┘
```

### The Inner Loop

The inner loop updates the seasonal component \(S_t\) and trend component \(T_t\) through six primary steps:

1. **Detrending:** The current estimate of the trend (\(T_t\)) is subtracted from the original data (\(Y_t - T_t\)).
2. **Cycle-Subseries Separation:** The detrended series is split into subseries corresponding to each seasonal point (for example, placing all Januaries together, all Februaries together, etc.).
3. **LOESS Smoothing of Subseries:** LOESS is applied independently to each seasonal subseries to capture changes in seasonality over time.
4. **Low-Pass Filtering:** A series of moving averages and a LOESS smoother extract any remaining low-frequency trend leakage from the smoothed seasonal values.
5. **Deseasonalizing:** The final seasonal component (\(S_t\)) is subtracted from the original data (\(Y_t - S_t\)).
6. **Trend Extraction:** LOESS smoothing is applied to the deseasonalized data to calculate the updated trend component (\(T_t\)).

### The Outer Loop

The outer loop computes **robustness weights** based on the magnitude of the remainder term (\(R_t = Y_t - T_t - S_t\)).

- If an observation has an unusually large remainder (an anomaly or outlier), it is assigned a small weight in the next inner loop pass.
- This makes STL resistant to outliers like severe weather events or temporary structural shocks.

---

## 3. Practical Example: Supermarket Ice Cream Sales

Consider tracking monthly ice cream sales across a calendar year:

- **Original Sales (\(Y_t\)):** ₹7 Lakhs in April, increasing to ₹12 Lakhs in June.
- **Trend (\(T_t\)):** Overall long-term expansion due to business growth over multiple years (e.g., growing from ₹5 Lakhs → ₹8 Lakhs → ₹10 Lakhs).
- **Seasonality (\(S_t\)):** Recurring surges every summer (May–July) due to warm weather.
- **Remainder (\(R_t\)):** A sudden, unseasonal drop caused by unexpected heavy rain during a specific week.

**Component Decomposition Matrix**:

| Month | Actual Sales (\(Y_t\)) | Trend (\(T_t\)) | Seasonality (\(S_t\)) | Remainder (\(R_t\)) |
| --- | --- | --- | --- | --- |
| **April** | ₹7.0 Lakhs | ₹2.8 Lakhs | ₹2.2 Lakhs | ₹2.0 Lakhs |
| **May** | ₹12.2 Lakhs | ₹2.85 Lakhs | ₹3.3 Lakhs | ₹6.05 Lakhs |
| **June** | ₹14.0 Lakhs | ₹2.9 Lakhs | ₹3.4 Lakhs | ₹7.7 Lakhs |

---

## 4. Key Advantages of STL

- **Handles Any Frequency:** Unlike classical decomposition methods restricted to monthly or quarterly data, STL can handle any arbitrary seasonal period (daily, weekly, hourly).
- **Dynamic Seasonality:** The seasonal component is allowed to change over time rather than remaining strictly fixed.
- **Outlier Robustness:** The outer loop ensures that isolated spikes or missing values do not distort the underlying trend or seasonal estimates.
- **Preprocessing Tool:** STL is often used as a preprocessing step prior to fitting forecasting algorithms like ARIMA or Exponential Smoothing.
# Extracting Features from Time Series

Extracting features—such as **height**, **average**, and **energy**—from a generated time series model or its predictions transforms raw numerical streams into structured, descriptive metrics. These statistical and spectral representations allow machine learning models and business systems to analyze patterns, classify behavior, and make accurate predictions.

---

## 1. Feature Categories for Time Series Analysis

When analyzing time series models or prediction residuals, features are generally grouped into three main categories:

- **Statistical Features (Time Domain):** Capture central tendencies, dispersion, and structural shape.
- **Morphological / Shape Features:** Capture peaks, valleys, troughs, and spatial properties (like height).
- **Spectral / Energy Features (Frequency Domain):** Capture signal strength, periodicity, and variance distribution across frequencies.

---

## 2. Deep Dive into Specific Features

### A. Height (Peak Amplitude / Range)

- **Definition:** Height measures the maximum vertical distance between a peak and a trough, or the absolute maximum magnitude of a signal/prediction curve over a rolling window.
- **Mathematical Representation:**

\[
\text{Height} = \max(Y_t) - \min(Y_t)
\]

Alternatively, peak height can be measured relative to a baseline:

\[
H_p = Y_{\text{peak}} - Y_{\text{baseline}}
\]

- **Predictive Value:** In business forecasting (such as demand spikes or website traffic), tracking height identifies the severity of surges. In anomaly detection or sensor data, an abnormal increase in height indicates sudden structural shifts or faults.

### B. Average (Mean / Central Tendency)

- **Definition:** The mathematical expectation or average of the signal values over a defined window size.
- **Mathematical Representation:**

\[
\mu = \frac{1}{N} \sum_{t=1}^{N} Y_t
\]

- **Predictive Value:** Averages establish the baseline level around which time series fluctuate. By tracking rolling averages, models can filter out high-frequency noise and focus on intermediate shifts in trend.

### C. Energy (Signal Energy / Power)

- **Definition:** Derived from signal processing, energy measures the sum of squared values of the time series or prediction residuals over a window. It quantifies the overall dynamic "power" or fluctuation intensity of the sequence.
- **Mathematical Representation:**

\[
\text{Energy} = \sum_{t=1}^{N} \vert Y_t \vert^2
\]

Often, **Root Mean Square (RMS)** or normalized energy is used:

\[
\text{RMS} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} Y_t^2}
\]

- **Predictive Value:** High energy signifies high volatility or aggressive fluctuations, while low energy indicates a stable, flat series. This is critical for predicting regime changes, structural instability, or financial volatility.

---

## 3. The Feature Extraction & Analysis Pipeline

To utilize these features for predictive modeling, follow a structured workflow:

```text
  ┌─────────────────────────────────────────────────────────┐
  │                 1. Raw Time Series Data                 │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┤
  │                 2. Rolling Window Extraction            │
  │     (Segmenting data into fixed temporal chunks)        │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 3. Feature Computation                  │
  │   - Height (Max - Min amplitude)                        │
  │   - Average (Rolling mean)                              │
  │   - Energy (Sum of squares / RMS)                       │
  │   - Additional: Variance, Skewness, Entropy             │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 4. Predictive Modeling                  │
  │    (Feeding feature matrix into ML/DL classifiers)      │
  └─────────────────────────────────────────────────────────┘
```

1. **Windowing (Segmentation):** Break the continuous time series or model outputs into overlapping or non-overlapping temporal windows (e.g., sliding windows of 24 hours).
2. **Feature Matrix Construction:** For each window, compute the vector of features:

\[
\mathbf{X}_w = [\text{Height}, \text{Average}, \text{Energy}, \text{Variance}, \dots]
\]

3. **Target Labeling:** Align these feature vectors with target variables (e.g., predicting whether a system will fail in the next window, or forecasting future demand volume).
4. **Model Training:** Feed the feature matrix into machine learning algorithms (such as Random Forest, XGBoost, or Support Vector Machines) or deep learning layers to execute final predictions.

---

## 4. Why Extract Features Instead of Using Raw Data?

- **Dimensionality Reduction:** Compressing thousands of raw timestamps into concise statistical markers (like average and energy) speeds up model training.
- **Noise Invariance:** Features like rolling averages and energy metrics filter out random, irrelevant fluctuations while retaining structural properties.
- **Interpretability:** Unlike black-box neural network weights, features like "height" and "energy" have clear physical and business meanings, making model decisions easier to audit and explain.
# Measures of Forecast Accuracy

Measures of forecast accuracy are quantitative metrics used to evaluate how well a forecasting model performs by comparing its predicted values against the actual observed values.

At the core of these metrics is the **forecast error (\(e_t\))**, which is defined as the difference between the actual value (\(A_t\)) and the forecasted value (\(F_t\)) at time \(t\):

\[
e_t = A_t - F_t
\]

Building directly on your study notes, here is a deep and structured breakdown of the primary forecast accuracy metrics.

---

## 1. Mean Forecast Error (MFE) / Bias

MFE measures the average forecast error over a dataset, indicating whether a model tends to over-forecast or under-forecast (uncovering systematic bias).

\[
\text{MFE} = \frac{1}{N} \sum_{t=1}^{N} (A_t - F_t)
\]

**Interpretation:**

- **\(\text{MFE} > 0\)**: The model generally **under-forecasts** (predictions are lower than actuals on average).
- **\(\text{MFE} < 0\)**: The model generally **over-forecasts** (predictions are higher than actuals on average).
- **\(\text{MFE} \approx 0\)**: The model is unbiased, meaning positive and negative errors cancel each other out.

**Limitation:** Because positive and negative errors offset one another, a model with massive errors can still have an MFE close to zero.

---

## 2. Mean Absolute Error (MAE)

MAE calculates the average magnitude of absolute errors, treating positive and negative deviations equally by converting them to absolute values.

\[
\text{MAE} = \frac{1}{N} \sum_{t=1}^{N} \vert A_t - F_t \vert
\]

**Interpretation:** Easy to understand because it is in the exact same units as the original data. A **lower MAE** indicates a more accurate model.

**Limitation:** MAE penalizes all errors equally regardless of their size, meaning it doesn't place extra weight on dangerously large outlier errors.

---

## 3. Mean Squared Error (MSE)

MSE squares each individual error before summing them up, which heavily penalizes larger errors.

\[
\text{MSE} = \frac{1}{N} \sum_{t=1}^{N} (A_t - F_t)^2
\]

**Interpretation:** A **lower MSE** represents a better-performing model.

**Limitation:** Because errors are squared, the resulting value is no longer in the original data units (e.g., if data is in dollars, MSE is in squared dollars), making intuitive interpretation difficult.

---

## 4. Root Mean Squared Error (RMSE)

RMSE is simply the square root of the MSE, bringing the error metric back into the original scale of the data.

\[
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (A_t - F_t)^2}
\]

**Interpretation:** Measures typical prediction error. Like MAE, a **lower RMSE** indicates better forecasting accuracy. Because of the squaring step prior to the square root, RMSE heavily penalizes large errors, making it extremely useful when large misses are particularly undesirable.

---

## 5. Percentage of Mean Absolute Error (PMAE / MAPE)

Often referred to as Mean Absolute Percentage Error (MAPE), this expresses forecast error as a percentage of the actual values.

\[
\text{PMAE} = \frac{100\%}{N} \sum_{t=1}^{N} \left\vert \frac{A_t - F_t}{A_t} \right\vert
\]

**Interpretation:** Extremely intuitive and easy to interpret as a percentage. A **lower PMAE** implies a better forecast. It allows comparison across datasets with completely different scales.

**Limitation:** It can become undefined, infinite, or heavily skewed when actual values (\(A_t\)) are equal to zero or very close to zero.
# Measures of Forecast Accuracy

Measures of forecast accuracy are quantitative metrics used to evaluate how closely a forecasting model's predicted values (\(F_t\)) match the actual observed values (\(A_t\)). They help determine model reliability, compare different forecasting techniques, and optimize model parameters.

---

## 1. Core Foundation: Forecast Error (\(e_t\))

The fundamental building block for all error-based metrics is the individual forecast error:

\[
e_t = A_t - F_t
\]

- **\(A_t\)**: Actual value at time \(t\)
- **\(F_t\)**: Forecasted value at time \(t\)
- **\(e_t\)**: Forecast error

---

## 2. Mean Forecast Error (MFE) / Bias

MFE measures the average of all errors, tracking whether a model tends to over-forecast or under-forecast.

**Formula:**

\[
\text{MFE} = \frac{1}{n} \sum_{t=1}^{n} (A_t - F_t)
\]

**Interpretation:**

- \(\text{MFE} > 0\): The model generally **under-forecasts** (actual values are typically higher than predicted).
- \(\text{MFE} < 0\): The model generally **over-forecasts** (predictions are typically higher than actuals).
- \(\text{MFE} \approx 0\): Indicates **unbiased** forecasting or zero bias, meaning positive and negative errors cancel each other out.

---

## 3. Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and forecast values. Because it uses absolute values, positive and negative errors do not cancel each other out.

**Formula:**

\[
\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \vert A_t - F_t \vert
\]

**Interpretation:**

- **Easy to understand**: Expressed in the exact same units as the original data.
- **Lower MAE** indicates a better, more accurate forecast.
- Less sensitive to extreme outliers compared to squared error metrics because it doesn't heavily penalize large individual errors.

---

## 4. Mean Squared Error (MSE)

MSE squares each individual error before summing and averaging them, placing a much higher penalty on large errors.

**Formula:**

\[
\text{MSE} = \frac{1}{n} \sum_{t=1}^{n} (A_t - F_t)^2
\]

**Interpretation:**

- **Lower MSE** indicates a better model.
- Because errors are squared, large forecasting mistakes disproportionately affect the final score, making MSE ideal when large errors are particularly dangerous or costly.
- **Limitation**: The resulting unit is squared relative to the original data, making intuitive interpretation harder.

---

## 5. Root Mean Squared Error (RMSE)

RMSE is simply the square root of the MSE. It brings the error metric back into the original scale/units of the data while retaining the heavy penalty on large errors.

**Formula:**

\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

**Interpretation:**

- Measures typical prediction error.
- Has the **same units as the original data**, making it much easier to interpret than MSE.
- **Lower RMSE** indicates better forecasting accuracy.

---

## 6. Percentage of Mean Absolute Error (PMAE) / MAPE

PMAE (often referred to as Mean Absolute Percentage Error, or MAPE) expresses the forecast error as a percentage of the actual values, allowing for easy comparison across datasets with completely different scales.

**Formula:**

\[
\text{PMAE} = \frac{100\%}{n} \sum_{t=1}^{n} \left\vert \frac{A_t - F_t}{A_t} \right\vert
\]

**Interpretation:**

- **Easy to interpret as a percentage** (e.g., "the model is off by 8.5% on average").
- **Lower PMAE** indicates a better forecast.
- **Limitation**: Can become undefined, unstable, or misleadingly infinite when actual values (\(A_t\)) are equal to zero or very close to zero.
```


