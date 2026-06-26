from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime


app = Flask(__name__)

# Load and clean dataset once
df = pd.read_csv("synthetic_1000_patients_3_months.csv")

# Strip spaces from column names and string data
df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

# Convert datatypes
df['RecordMonth'] = pd.to_datetime(df['RecordMonth'], errors='coerce')
numeric_cols = ["Glucose", "Weight", "SystolicBP", "DiastolicBP", "Cholesterol"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing critical values
df.dropna(subset=["PatientID", "RecordMonth", "Glucose"], inplace=True)

# Optional: filter out unrealistic values (domain knowledge based)
df = df[(df['Glucose'] > 30) & (df['Glucose'] < 500)]  # Realistic glucose range
df = df[(df['Weight'] > 20) & (df['Weight'] < 300)]
df = df[(df['SystolicBP'] > 50) & (df['SystolicBP'] < 250)]
df = df[(df['Cholesterol'] > 50) & (df['Cholesterol'] < 400)]
# Create line chart (Univariate)
def create_line_chart(patient_data, next_month, predicted_glucose):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(patient_data["RecordMonth"], patient_data["Glucose"], marker='o', label="Actual Glucose Level")
    ax.plot([patient_data["RecordMonth"].iloc[-1], next_month],
            [patient_data["Glucose"].iloc[-1], predicted_glucose],
            'r--o', label="Predicted Month", linewidth=2)
    ax.axvline(x=next_month, color='red', linestyle='dotted')
    ax.scatter(next_month, predicted_glucose, color='red', zorder=5)

    ax.set_title("Glucose Level Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Glucose Level")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

# Create bar chart (Bivariate)
def create_bar_chart(values, labels, title):
    percentages = [v / sum(values) * 100 if sum(values) > 0 else 0 for v in values]
    bar_colors = ['#1976d2', '#43a047', '#ffa000', '#d32f2f'][:len(values)]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_title(title)
    ax.set_ylabel("Glucose Value")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

# Create multivariate chart
def create_multivariate_chart(patient_data):
    features = ["Weight", "SystolicBP", "Cholesterol"]
    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))

    for i, feature in enumerate(features):
        axes[i].scatter(patient_data[feature], patient_data["Glucose"], color='#007acc', alpha=0.7)
        axes[i].set_title(f"Glucose vs {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Glucose")

        X = patient_data[feature].values.reshape(-1, 1)
        y = patient_data["Glucose"].values
        model = LinearRegression().fit(X, y)
        x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_preds = model.predict(x_vals)
        axes[i].plot(x_vals, y_preds, color='red')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

@app.route("/", methods=["GET", "POST"])
def index():
    chart = None
    bar_chart = None
    multivariate_chart = None
    patient_id = ""
    patient_info = {}
    predicted_glucose = None

    if request.method == "POST":
        patient_id = request.form["patient_id"].strip().upper()
        patient_data = df[df["PatientID"].str.strip() == patient_id].copy()

        if not patient_data.empty:
            patient_data["RecordMonth"] = pd.to_datetime(patient_data["RecordMonth"].str.strip(), errors='coerce')
            patient_data["Glucose"] = pd.to_numeric(patient_data["Glucose"], errors='coerce')
            patient_data.dropna(subset=["RecordMonth", "Glucose"], inplace=True)
            patient_data = patient_data.sort_values("RecordMonth")

            X = np.arange(len(patient_data)).reshape(-1, 1)
            y = patient_data["Glucose"].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)

            next_index = len(X)
            next_month = patient_data["RecordMonth"].max() + pd.DateOffset(months=1)
            predicted_glucose = round(model.predict([[next_index]])[0][0], 2)

            sample_row = patient_data.iloc[0]
            patient_info = {
                "PatientID": sample_row.get("PatientID", "N/A"),
                "Weight": sample_row.get("Weight", "N/A"),
                "SystolicBP": sample_row.get("SystolicBP", "N/A"),
                "DiastolicBP": sample_row.get("DiastolicBP", "N/A"),
                "Cholesterol": sample_row.get("Cholesterol", "N/A")
            }

            chart = create_line_chart(patient_data, next_month, predicted_glucose)

            last3 = list(patient_data["Glucose"].tail(3))
            labels = [d.strftime("%b %Y") for d in patient_data["RecordMonth"].tail(3)] + ["Predicted"]
            values = last3 + [predicted_glucose]
            bar_chart = create_bar_chart(values, labels, "Recent & Predicted Glucose Values")

            # 🔴 Add multivariate chart
            multivariate_chart = create_multivariate_chart(patient_data)

    return render_template(
        "index.html",
        chart=chart,
        bar_chart=bar_chart,
        multivariate_chart=multivariate_chart,
        patient_id=patient_id,
        patient_info=patient_info,
        predicted_glucose=predicted_glucose
    )

# Load dataset once
df = pd.read_csv("synthetic_1000_patients_3_months.csv")
df.columns = df.columns.str.strip()

# Create line chart (Univariate) for manual-entry
def create_manual_line_chart(m1, m2, m3, predicted):
    months = pd.date_range(end=datetime.today(), periods=3, freq='M')
    next_month = months[-1] + pd.DateOffset(months=1)
    values = [m1, m2, m3]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, values, marker='o', label="Actual Glucose Level")
    ax.plot([months[-1], next_month], [values[-1], predicted], 'r--o', label="Predicted Month", linewidth=2)
    ax.axvline(x=next_month, color='red', linestyle='dotted')
    ax.scatter(next_month, predicted, color='red', zorder=5)

    ax.set_title("Glucose Level Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Glucose Level")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

# Create bar chart (Bivariate)
def create_manual_bar_chart(values, labels, title):
    percentages = [v / sum(values) * 100 if sum(values) > 0 else 0 for v in values]
    bar_colors = ['#1976d2', '#43a047', '#ffa000', '#d32f2f'][:len(values)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_title(title)
    ax.set_ylabel("Glucose Value")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)

    for bar, val, pct in zip(bars, values, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

# Create multivariate chart (similar to index.html)
def create_manual_multivariate_chart(weight, systolic, cholesterol, glucose_values):
    features = ["Weight", "SystolicBP", "Cholesterol"]
    values = [weight, systolic, cholesterol]

    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))
    for i, (feature, value) in enumerate(zip(features, values)):
        X = np.array([value - 1, value, value + 1]).reshape(-1, 1)
        y = np.array(glucose_values)

        axes[i].scatter(X.flatten(), y[:3], color="#007acc", alpha=0.7)
        axes[i].set_title(f"Glucose vs {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Glucose")

        model = LinearRegression().fit(X, y[:3])
        x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_preds = model.predict(x_vals)
        axes[i].plot(x_vals, y_preds, color='red')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return chart

@app.route('/manual-entry', methods=['GET', 'POST'])
def manual_entry():
    if request.method == "POST":
        try:
            age = max(1, int(request.form["age"]))
            gender = int(request.form["gender"])  # Assume: 0 = Male, 1 = Female

            weight = float(request.form["weight"])
            weight = max(30, min(weight, 300))

            systolic = int(request.form["systolic_bp"])
            systolic = max(70, min(systolic, 250))

















































            

            diastolic = int(request.form["diastolic_bp"])
            diastolic = max(40, min(diastolic, 150))

            cholesterol = int(request.form["cholesterol"])
            cholesterol = max(50, min(cholesterol, 400))

            m1 = max(40, min(int(request.form["month1"]), 500))
            m2 = max(40, min(int(request.form["month2"]), 500))
            m3 = max(40, min(int(request.form["month3"]), 500))

        except (ValueError, KeyError):
            return render_template("manual-entry.html", prediction="Invalid input. Please enter valid numeric values.")

        # Predict next month's glucose
        months = np.array([[1], [2], [3]])
        glucose = np.array([m1, m2, m3])
        model = LinearRegression().fit(months, glucose)
        predicted = int(model.predict(np.array([[4]]))[0])

        # Generate charts
        line_graph = create_manual_line_chart(m1, m2, m3, predicted)
        bar_graph = create_manual_bar_chart([m1, m2, m3, predicted],
                                            ["Month 1", "Month 2", "Month 3", "Predicted"],
                                            "Monthly Glucose Comparison")
        multivariate_graph = create_manual_multivariate_chart(weight, systolic, cholesterol, [m1, m2, m3])

        return render_template("manual-entry.html",
                               prediction=predicted,
                               line_graph=line_graph,
                               bar_graph=bar_graph,
                               multivariate_graph=multivariate_graph)

    return render_template("manual-entry.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
