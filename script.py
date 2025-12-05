import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


st.title("Serverless Cost Analysis Dashboard")

# Parsing the QUOTED CSV format (kept as-is)
with open("Serverless_Data.csv", "r") as f:
    lines = f.readlines()

# Remove outer quotes from EVERY line and split
header_line = lines[0].strip()[1:-1]  # Remove first and last quotes
header = [col.strip() for col in header_line.split(",")]

data_rows = []
for line in lines[1:]:
    if line.strip():
        row_line = line.strip()[1:-1]  # Remove outer quotes from data rows
        row = [field.strip() for field in row_line.split(",")]
        data_rows.append(row)

# Create DataFrame
serverless_functions_dataframe = pd.DataFrame(data_rows, columns=header)
serverless_functions_dataframe.columns = (
    serverless_functions_dataframe.columns.str.strip()
)

st.write(f"Loaded {len(serverless_functions_dataframe)} functions")
st.write("Columns:", list(serverless_functions_dataframe.columns))
st.dataframe(serverless_functions_dataframe.head())

# Convert numeric columns
numeric_columns_list = [
    "InvocationsPerMonth",
    "AvgDurationMs",
    "MemoryMB",
    "ColdStartRate",
    "ProvisionedConcurrency",
    "GBSeconds",
    "DataTransferGB",
    "CostUSD",
]
for column_name in numeric_columns_list:
    serverless_functions_dataframe[column_name] = pd.to_numeric(
        serverless_functions_dataframe[column_name], errors="coerce"
    )

st.write(
    f"Total cost: ${serverless_functions_dataframe['CostUSD'].sum():.2f}"
)

# Exercise 1: Top cost contributors
st.header("Exercise 1: Top Cost Contributors")

cost_sorted_dataframe = serverless_functions_dataframe.sort_values(
    "CostUSD", ascending=False
)
total_monthly_cost = serverless_functions_dataframe["CostUSD"].sum()
cost_sorted_dataframe["CumulativeCost"] = cost_sorted_dataframe["CostUSD"].cumsum()
cost_sorted_dataframe["CumulativePercentage"] = (
    cost_sorted_dataframe["CumulativeCost"] / total_monthly_cost
) * 100

top_eighty_percentage_count = (
    cost_sorted_dataframe["CumulativePercentage"] <= 80
).sum()
st.write(
    f"{top_eighty_percentage_count} functions = 80% of ${total_monthly_cost:.2f}"
)

# Bar chart for top cost contributors
fig1 = px.bar(
    cost_sorted_dataframe.head(20),
    x="FunctionName",
    y="CostUSD",
    color="Environment",
    title="Top 20 Functions by Cost",
)
st.plotly_chart(fig1)

# Scatter plot cost vs invocations (aligns with assignment text)
fig1_scatter = px.scatter(
    serverless_functions_dataframe,
    x="InvocationsPerMonth",
    y="CostUSD",
    color="Environment",
    hover_name="FunctionName",
    title="Cost vs Invocation Frequency",
)
st.plotly_chart(fig1_scatter)

# Exercise 2: Memory right-sizing
st.header("Exercise 2: Memory Right-Sizing Candidates")

# More realistic criteria: low efficiency (duration/memory ratio)
serverless_functions_dataframe["DurationPerMB"] = (
    serverless_functions_dataframe["AvgDurationMs"]
    / serverless_functions_dataframe["MemoryMB"]
)

memory_rightsizing_candidates = serverless_functions_dataframe[
    (serverless_functions_dataframe["DurationPerMB"] < 0.3)
    & (serverless_functions_dataframe["MemoryMB"] > 1024)
    & (serverless_functions_dataframe["Environment"] == "production")
].sort_values("CostUSD", ascending=False)

st.write(f"Found {len(memory_rightsizing_candidates)} over-provisioned candidates")
st.dataframe(
    memory_rightsizing_candidates[
        ["FunctionName", "AvgDurationMs", "MemoryMB", "DurationPerMB", "CostUSD"]
    ]
)

# Exercise 3: Provisioned concurrency
st.header("Exercise 3: Provisioned Concurrency")

provisioned_concurrency_functions = serverless_functions_dataframe[
    serverless_functions_dataframe["ProvisionedConcurrency"] > 0
]

st.write(f"{len(provisioned_concurrency_functions)} functions with PC")
st.dataframe(
    provisioned_concurrency_functions[
        ["FunctionName", "ColdStartRate", "ProvisionedConcurrency", "CostUSD"]
    ]
)

# Scatter plot to compare PC vs cold starts and cost
if not provisioned_concurrency_functions.empty:
    fig3_scatter = px.scatter(
        provisioned_concurrency_functions,
        x="ProvisionedConcurrency",
        y="ColdStartRate",
        size="CostUSD",
        color="Environment",
        hover_name="FunctionName",
        title="Provisioned Concurrency vs Cold Start Rate",
    )
    st.plotly_chart(fig3_scatter)

# Exercise 4: Low-value workloads
st.header("Exercise 4: Low-Value Workloads")

total_invocations = serverless_functions_dataframe["InvocationsPerMonth"].sum()
serverless_functions_dataframe["InvocationPercentage"] = (
    serverless_functions_dataframe["InvocationsPerMonth"] / total_invocations
) * 100

low_value_workloads = serverless_functions_dataframe[
    (serverless_functions_dataframe["InvocationPercentage"] < 1)
    & (serverless_functions_dataframe["CostUSD"] > 10)
]

st.write(f"{len(low_value_workloads)} low-value functions")
st.dataframe(
    low_value_workloads[
        ["FunctionName", "InvocationsPerMonth", "InvocationPercentage", "CostUSD"]
    ]
)

# Exercise 5: Cost forecasting
st.header("Exercise 5: Cost Model")

serverless_functions_dataframe["CalculatedGBSeconds"] = (
    serverless_functions_dataframe["InvocationsPerMonth"]
    * (serverless_functions_dataframe["AvgDurationMs"] / 1000)
    * (serverless_functions_dataframe["MemoryMB"] / 1024)
)

X = serverless_functions_dataframe[
    ["CalculatedGBSeconds", "DataTransferGB"]
].fillna(0)
y = serverless_functions_dataframe["CostUSD"].fillna(0)

model = LinearRegression().fit(X, y)

st.write(
    f"Compute coef: {model.coef_[0]:.8f}, "
    f"DataTransfer coef: {model.coef_[1]:.4f}"
)

# Optional: show predicted vs actual cost for quick sanity check
serverless_functions_dataframe["PredictedCost"] = model.predict(X)
st.dataframe(
    serverless_functions_dataframe[
        ["FunctionName", "CostUSD", "PredictedCost", "CalculatedGBSeconds", "DataTransferGB"]
    ].head(20)
)

# Exercise 6: Containerization
st.header("Exercise 6: Container Candidates")

container_candidates = serverless_functions_dataframe[
    (serverless_functions_dataframe["AvgDurationMs"] > 3000)
    & (serverless_functions_dataframe["MemoryMB"] > 2048)
    & (serverless_functions_dataframe["InvocationsPerMonth"] < 1000)
]

st.write(f"{len(container_candidates)} candidates")
st.dataframe(
    container_candidates[
        ["FunctionName", "AvgDurationMs", "MemoryMB", "InvocationsPerMonth", "CostUSD"]
    ]
)
