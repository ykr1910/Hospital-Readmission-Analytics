import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import plotly.express as px

st.set_page_config(
    page_title="Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide"
)



st.markdown("""
<style>

.kpi-card {
    background-color: #e8f0f8;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}

.kpi-card::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(
        90deg,
        #2563eb 0%,
        #60a5fa 50%,
        #93c5fd 100%
    );
}

.kpi-title {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #111827;
}
.section-divider {
    height: 4px;
    background-color: #2a2f3a !important;
    border: none !important;
    border-radius: 3px;
    margin: 40px 0;
}


</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Database settings
DB_HOST = "127.0.0.1"
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# PostgreSQL connection
DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)



# Dashboard title
st.title("🏥 Hospital Readmission Analytics")

st.write(
    "Analysis of 30-day hospital readmissions using patient demographics, "
    "admission characteristics, diagnoses, and encounter history."
)

st.sidebar.title("Dashboard Filters")

st.sidebar.write("Use the filters to explore the dataset.")

readmission_filter = st.sidebar.selectbox(
    "Readmission status",
    ["All", "No readmission", "Within 30 days", "After 30 days"]
)

if st.sidebar.button("Refresh Dashboard"):
    st.rerun()



# Get overall readmission KPI
query = """
SELECT
    COUNT(*) AS total_encounters,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmitted_within_30_days,
    ROUND(
        100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart;
"""

kpi = pd.read_sql(query, engine)

# Display KPI
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">30-Day Readmission Rate</div>
            <div class="kpi-value">{kpi.loc[0, 'readmission_rate']}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Encounters</div>
            <div class="kpi-value">{kpi.loc[0, 'total_encounters']:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">30-Day Readmissions</div>
            <div class="kpi-value">{kpi.loc[0, 'readmitted_within_30_days']:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">Total Unique Patients</div>
            <div class="kpi-value">71,518</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Patient profile
st.header("Patient Profile")
st.write(
    "Patient demographics and encounter characteristics "
    "that may be associated with hospital readmission."
)
# AGE query
st.subheader("30-Day Readmission Rate by Age Group")

age_query = """
SELECT
    age_group,
    COUNT(*) AS total_encounters,
    ROUND(
        100.0 * SUM(
            CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart

GROUP BY age_group
ORDER BY age_group;
"""

age_data = pd.read_sql(age_query, engine)

age_order = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100"
]

age_data["age_group"] = pd.Categorical(
    age_data["age_group"],
    categories=age_order,
    ordered=True
)

age_data = age_data.sort_values("age_group")

age_data["bar_color"] = [
    "#0369a1" if i % 2 == 0 else "#7dd3fc"
    for i in range(len(age_data))
]

fig_age = px.bar(
    age_data,
    x="age_group",
    y="readmission_rate",
    title="30-Day Readmission Rate by Age Group",
    color="bar_color",
    color_discrete_map={
        "#0369a1": "#0369a1",
        "#7dd3fc": "#7dd3fc"
    },
    hover_data={
        "bar_color": False,
        "readmission_rate": ":.2f"
    },
    labels={
        "age_group": "Age Group",
        "readmission_rate": "Readmission Rate (%)"
    }
)

fig_age.update_layout(
    height=450,
    font=dict(size=16),
    showlegend=False,
    xaxis={
        "categoryorder": "array",
        "categoryarray": [
            "0-10",
            "10-20",
            "20-30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100"
        ]
    }
)
fig_age.update_traces(
    texttemplate="%{y:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_age,
    use_container_width=True
)

# GENDER
st.subheader("30-Day Readmission Rate by Gender")

gender_query = """
SELECT
    gender,
    ROUND(
        100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart
WHERE gender <> 'Unknown'
GROUP BY gender
ORDER BY gender;
"""

gender_data = pd.read_sql(gender_query, engine)

fig_gender = px.pie(
    gender_data,
    names="gender",
    values="readmission_rate",
    hole=0.5,
    title="30-Day Readmission Rate by Gender",
    color_discrete_sequence=["#2563eb", "#60a5fa"]
)

fig_gender.update_layout(
    height=500,
    font=dict(size=16)
)

st.plotly_chart(
    fig_gender,
    use_container_width=True
)
st.markdown(
    '<div class="section-divider"></div>',
    unsafe_allow_html=True
)


# Length of Stay chart
st.header("Readmission Analysis")
st.subheader("30-Day Readmission Rate by Length of Stay")

stay_query = """
SELECT
    time_in_hospital,
    ROUND(
        100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart
GROUP BY time_in_hospital
ORDER BY time_in_hospital;
"""

stay_data = pd.read_sql(stay_query, engine)

fig_stay = px.line(
    stay_data,
    x="time_in_hospital",
    y="readmission_rate",
    markers=True,
    text="readmission_rate",
    title="30-Day Readmission Rate by Length of Stay",
    labels={
        "time_in_hospital": "Days in Hospital",
        "readmission_rate": "Readmission Rate (%)"
    }
)

fig_stay.update_traces(
    line=dict(color="#0891b2", width=4),
    marker=dict(size=9)
)

fig_stay.update_layout(
    height=450,
    font=dict(size=16)
)
fig_stay.update_traces(
    line=dict(color="#0891b2", width=4),
    marker=dict(size=9),
    texttemplate="%{text:.2f}%",
    textposition="top center",
    cliponaxis=False
)

st.plotly_chart(
    fig_stay,
    use_container_width=True
)


# Prior Visits chart
st.subheader("30-Day Readmission Rate by Prior Visits")

prior_query = """
SELECT
    CASE
        WHEN prior_visits = 0 THEN '0'
        WHEN prior_visits BETWEEN 1 AND 2 THEN '1-2'
        WHEN prior_visits BETWEEN 3 AND 5 THEN '3-5'
        WHEN prior_visits BETWEEN 6 AND 9 THEN '6-9'
        ELSE '10+'
    END AS prior_visit_group,

    COUNT(*) AS total_encounters,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN readmitted = '<30' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS readmission_rate

FROM healthcare.readmission_mart

GROUP BY
    CASE
        WHEN prior_visits = 0 THEN '0'
        WHEN prior_visits BETWEEN 1 AND 2 THEN '1-2'
        WHEN prior_visits BETWEEN 3 AND 5 THEN '3-5'
        WHEN prior_visits BETWEEN 6 AND 9 THEN '6-9'
        ELSE '10+'
    END

ORDER BY MIN(prior_visits);
"""

prior_data = pd.read_sql(prior_query, engine)

# Correct order for prior visit groups
prior_order = ["0", "1-2", "3-5", "6-9", "10+"]

prior_data["prior_visit_group"] = pd.Categorical(
    prior_data["prior_visit_group"].astype(str),
    categories=prior_order,
    ordered=True
)

prior_data = prior_data.sort_values("prior_visit_group")

# Alternating colors
prior_data["bar_color"] = [
    "#4338ca" if i % 2 == 0 else "#818cf8"
    for i in range(len(prior_data))
]

# Create bar chart
fig_prior = px.bar(
    prior_data,
    x="prior_visit_group",
    y="readmission_rate",
    title="30-Day Readmission Rate by Prior Visits",

    color="bar_color",

    color_discrete_map={
        "#4338ca": "#4338ca",
        "#818cf8": "#818cf8"
    },

    hover_data={
        "bar_color": False,
        "readmission_rate": ":.2f"
    },

    labels={
        "prior_visit_group": "Prior Visit Group",
        "readmission_rate": "Readmission Rate (%)"
    }
)

# Force x-axis to be categorical
fig_prior.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=prior_order
)

# Chart formatting
fig_prior.update_layout(
    height=450,
    font=dict(size=16),
    showlegend=False
)
fig_prior.update_traces(
    texttemplate="%{y:.2f}%",
    textposition="outside",
    cliponaxis=False
)

# Display chart
st.plotly_chart(
    fig_prior,
    use_container_width=True
)


# DIAGNOSIS count
st.subheader("30-Day Readmission Rate by Diagnosis Count")

diagnosis_query = """
SELECT
    diagnosis_count,
    COUNT(*) AS total_encounters,
    ROUND(
        100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart
WHERE diagnosis_count > 0
GROUP BY diagnosis_count
ORDER BY diagnosis_count;
"""

diagnosis_data = pd.read_sql(diagnosis_query, engine)

fig_diagnosis = px.line(
    diagnosis_data,
    x="diagnosis_count",
    y="readmission_rate",
    markers=True,
    text="readmission_rate",
    title="30-Day Readmission Rate by Diagnosis Count",
    color_discrete_sequence=["#7c3aed"],
    labels={
        "diagnosis_count": "Number of Diagnoses",
        "readmission_rate": "Readmission Rate (%)"
    }
)

fig_diagnosis.update_layout(
    height=450,
    font=dict(size=16)
)
fig_diagnosis.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="top center",
    cliponaxis=False
)

st.plotly_chart(
    fig_diagnosis,
    use_container_width=True
)



# Specialty
st.subheader("30-Day Readmission Rate by Medical Specialty")

specialty_query = """
SELECT
    medical_specialty,
    COUNT(*) AS total_encounters,
    ROUND(
        100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.readmission_mart
WHERE medical_specialty <> 'Unknown'
GROUP BY medical_specialty
HAVING COUNT(*) >= 500
ORDER BY readmission_rate DESC;
"""

specialty_data = pd.read_sql(specialty_query, engine)

specialty_data["bar_color"] = [
    "#0f3d5e" if i % 2 == 0 else "#4f9ac4"
    for i in range(len(specialty_data))
]

fig_specialty = px.bar(
    specialty_data,
    x="readmission_rate",
    y="medical_specialty",
    orientation="h",
    title="30-Day Readmission Rate by Medical Specialty",
    color="bar_color",
    color_discrete_map={
        "#0f3d5e": "#0f3d5e",
        "#4f9ac4": "#4f9ac4"
    },
    hover_data={
        "bar_color": False,
        "readmission_rate": ":.2f"
    },
    labels={
        "readmission_rate": "Readmission Rate (%)",
        "medical_specialty": "Medical Specialty"
    }
)

fig_specialty.update_layout(
    height=650,
    font=dict(size=16),
    yaxis={"categoryorder": "total ascending"},
    showlegend=False,
    xaxis=dict(
        range=[0, specialty_data["readmission_rate"].max() + 2]
    )
)
fig_specialty.update_traces(
    texttemplate="%{x:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_specialty,
    use_container_width=True
)


# readmission status
st.subheader("Readmission Status")

status_query = """
SELECT
    readmitted,
    COUNT(*) AS total_encounters
FROM healthcare.readmission_mart
GROUP BY readmitted
ORDER BY
    CASE
        WHEN readmitted = 'NO' THEN 1
        WHEN readmitted = '>30' THEN 2
        WHEN readmitted = '<30' THEN 3
    END;
"""

status_data = pd.read_sql(status_query, engine)

status_order = ["NO", ">30", "<30"]

status_data["readmitted"] = pd.Categorical(
    status_data["readmitted"],
    categories=status_order,
    ordered=True
)

status_data = status_data.sort_values("readmitted")

if readmission_filter == "All":
    fig_status = px.pie(
        status_data,
        names="readmitted",
        values="total_encounters",
        hole=0.5,
        title="Readmission Status",
        color="readmitted",
        color_discrete_map={
    "NO": "#c4b5fd",
    ">30": "#8b5cf6",
    "<30": "#5b21b6"
}

    )

    fig_status.update_layout(
        height=500,
        font=dict(size=16)
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True
    )

else:
    status_map = {
        "No readmission": "NO",
        "Within 30 days": "<30",
        "After 30 days": ">30"
    }

    selected_status = status_map[readmission_filter]

    filtered_status = status_data[
        status_data["readmitted"] == selected_status
    ]

    fig_selected_status = px.bar(
        filtered_status,
        x="readmitted",
        y="total_encounters",
        title=f"Selected Status: {readmission_filter}",
        labels={
            "readmitted": "Readmission Status",
            "total_encounters": "Number of Encounters"
        }
    )

    fig_selected_status.update_traces(
        width=0.35,
        texttemplate="%{y:,}",
        textposition="outside"
    )

    fig_selected_status.update_layout(
        height=400,
        font=dict(size=16),
        showlegend=False
    )

    st.plotly_chart(
        fig_selected_status,
        use_container_width=True
    )
st.markdown(
    '<div class="section-divider"></div>',
    unsafe_allow_html=True
)



# Diagnosis
st.header("Diagnosis Analysis")

st.write(
    "Primary diagnosis categories with higher 30-day readmission rates."
)

diagnosis_rate_query = """
SELECT
    diagnosis_1,
    COUNT(*) AS total_encounters,
    ROUND(
        100.0 * SUM(
            CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS readmission_rate
FROM healthcare.diagnoses d
JOIN healthcare.admissions a
    ON d.encounter_id = a.encounter_id
GROUP BY diagnosis_1
HAVING COUNT(*) >= 500
ORDER BY readmission_rate DESC
LIMIT 10;
"""

diagnosis_rate_data = pd.read_sql(
    diagnosis_rate_query,
    engine
)

# Treat diagnosis codes as categories, not numbers
diagnosis_rate_data["diagnosis_1"] = (
    diagnosis_rate_data["diagnosis_1"]
    .astype(str)
)

# Alternating colors
diagnosis_rate_data["bar_color"] = [
    "#1d4ed8" if i % 2 == 0 else "#60a5fa"
    for i in range(len(diagnosis_rate_data))
]

# Create horizontal bar chart
fig_primary_diagnosis = px.bar(
    diagnosis_rate_data,
    x="readmission_rate",
    y="diagnosis_1",
    orientation="h",
    title="Top Primary Diagnoses by 30-Day Readmission Rate",

    color="bar_color",

    color_discrete_map={
        "#1d4ed8": "#1d4ed8",
        "#60a5fa": "#60a5fa"
    },

    hover_data={
        "bar_color": False,
        "readmission_rate": ":.2f"
    },

    labels={
        "readmission_rate": "Readmission Rate (%)",
        "diagnosis_1": "Primary Diagnosis"
    }
)

# Force diagnosis codes to be categorical
fig_primary_diagnosis.update_yaxes(
    type="category",
    categoryorder="total ascending"
)

# Chart formatting
fig_primary_diagnosis.update_layout(
    height=550,
    font=dict(size=16),
    showlegend=False,
    xaxis=dict(
        range=[0, diagnosis_rate_data["readmission_rate"].max() + 2]
    )
)
fig_primary_diagnosis.update_traces(
    texttemplate="%{x:.2f}%",
    textposition="outside"
)

# Display chart
st.plotly_chart(
    fig_primary_diagnosis,
    use_container_width=True
)
st.markdown(
    '<div class="section-divider"></div>',
    unsafe_allow_html=True
)


# Summary
st.header("Dataset Summary")

summary_query = """
SELECT
    COUNT(*) AS total_encounters,
    COUNT(DISTINCT patient_key) AS unique_patients,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) AS readmitted_within_30_days,
    SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) AS readmitted_after_30_days,
    SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END) AS not_readmitted
FROM healthcare.readmission_mart;
"""

summary_data = pd.read_sql(summary_query, engine)

st.dataframe(summary_data)


st.subheader("Export Data")

export_query = """
SELECT *
FROM healthcare.readmission_mart;
"""

# EXPORT
export_data = pd.read_sql(export_query, engine)

csv_data = export_data.to_csv(index=False)

st.download_button(
    label="Download Readmission Data",
    data=csv_data,
    file_name="readmission_analysis_data.csv",
    mime="text/csv"
)


# DATA QUALITY
st.subheader("Data Quality")

quality_query = """
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT encounter_id) AS unique_encounters,
    COUNT(DISTINCT patient_key) AS unique_patients
FROM healthcare.readmission_mart;
"""

quality_data = pd.read_sql(quality_query, engine)

qcol1, qcol2, qcol3 = st.columns(3)

with qcol1:
    st.metric(
        "Rows",
        f"{quality_data.loc[0, 'total_rows']:,}"
    )

with qcol2:
    st.metric(
        "Unique Encounters",
        f"{quality_data.loc[0, 'unique_encounters']:,}"
    )

with qcol3:
    st.metric(
        "Unique Patients",
        f"{quality_data.loc[0, 'unique_patients']:,}"
    )

# DASHBOARD ABOUT
with st.expander("About this dashboard"):
    st.write(
        """
        This dashboard analyzes hospital readmissions using the diabetic
        hospital encounter dataset.

        A 30-day readmission is defined as an encounter where the
        readmitted value is '<30'.

        The dashboard uses PostgreSQL analytical tables and retrieves
        aggregated results through SQL queries.

        Key engineered features include prior visits, diagnosis count,
        medication count, patient key, and age group.
        """
    )


st.caption(
    "Data source: Diabetic Hospital Readmission Dataset | "
    "Processed through the project ETL pipeline and stored in PostgreSQL."
)


# --------------------------------------------------
# 30-DAY READMISSION PREDICTION (Docker API)
# --------------------------------------------------
import requests

st.markdown(
    '<div class="section-divider"></div>',
    unsafe_allow_html=True
)

st.header("30-Day Readmission Prediction")
st.write(
    "Enter encounter characteristics to obtain a prediction "
    "from the containerized XGBoost model."
)

API_URL = os.getenv("PREDICTION_API_URL", "http://127.0.0.1:8000")

with st.form("readmission_prediction_form"):
    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        pred_age = st.selectbox(
            "Age Group",
            [
                "0-10", "10-20", "20-30", "30-40", "40-50",
                "50-60", "60-70", "70-80", "80-90", "90-100"
            ],
            index=7
        )
        pred_gender = st.selectbox(
            "Gender", ["Male", "Female", "Unknown"]
        )
        pred_admission_type = st.number_input(
            "Admission Type ID", min_value=1, max_value=8, value=1
        )
        pred_los = st.number_input(
            "Days in Hospital", min_value=1, max_value=14, value=5
        )

    with pred_col2:
        pred_prior_visits = st.number_input(
            "Prior Visits", min_value=0, value=3
        )
        pred_diagnoses = st.number_input(
            "Diagnosis Count", min_value=1, max_value=16, value=8
        )
        pred_medications = st.number_input(
            "Medication Count", min_value=0, value=12
        )
        pred_specialty = st.selectbox(
            "Medical Specialty",
            [
                "InternalMedicine",
                "Family/GeneralPractice",
                "Emergency/Trauma",
                "Cardiology",
                "Surgery-General",
                "Nephrology",
                "Unknown",
                "Other"
            ]
        )

    predict_clicked = st.form_submit_button("Predict Readmission")

if predict_clicked:
    payload = {
        "age_group": pred_age,
        "gender": pred_gender,
        "admission_type_id": int(pred_admission_type),
        "time_in_hospital": int(pred_los),
        "prior_visits": int(pred_prior_visits),
        "diagnosis_count": int(pred_diagnoses),
        "medication_count": int(pred_medications),
        "medical_specialty": pred_specialty
    }

    try:
        with st.spinner("Requesting prediction from Docker API..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

        probability = result["probability"]
        threshold = result["threshold"]
        predicted_class = result["predicted_readmission_within_30_days"]

        st.subheader("Prediction Result")
        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Model Probability",
            f"{probability:.2%}"
        )
        result_col2.metric(
            "Decision Threshold",
            f"{threshold:.2%}"
        )
        result_col3.metric(
            "Predicted Class",
            str(predicted_class)
        )

        if predicted_class == 1:
            st.warning(
                "The model classified this encounter as positive "
                "for readmission within 30 days."
            )
        else:
            st.info(
                "The model classified this encounter as negative "
                "for readmission within 30 days."
            )

        st.caption(
            f"Model: {result['model']}. "
            "Academic prototype only; not medical advice or a "
            "validated clinical assessment."
        )

    except requests.RequestException as exc:
        st.error(
            "Could not reach the prediction API. Confirm that the "
            "Docker container is running and the API URL is correct."
        )
        st.caption(f"Technical details: {exc}")




# ---------------- API Monitoring ----------------
import requests

st.divider()
st.header("API Monitoring")

try:
    response = requests.get(
        "http://127.0.0.1:8000/metrics",
        timeout=5
    )
    response.raise_for_status()
    metrics = response.json()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total API Requests",
        metrics["total_requests"]
    )
    col2.metric(
        "Successful Predictions",
        metrics["successful_predictions"]
    )
    col3.metric(
        "Failed Requests",
        metrics["failed_requests"]
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Error Rate",
        f'{metrics["error_rate_percent"]:.2f}%'
    )
    col5.metric(
        "Average API Latency",
        f'{metrics["average_latency_ms"]:.2f} ms'
    )

    st.caption(
        "Latest prediction (UTC): "
        + str(metrics["latest_prediction_timestamp_utc"])
    )

except requests.RequestException as exc:
    st.warning(
        "Monitoring metrics are unavailable. "
        "Check that the prediction API container is running."
    )





# ---------------- Data Drift Monitoring ----------------
st.divider()
if st.button("Refresh Monitoring"):
    st.rerun()
st.header("Data Drift Monitoring")

try:
    response = requests.get(
        "http://127.0.0.1:8000/drift",
        timeout=10
    )
    response.raise_for_status()
    drift_data = response.json()

    status = drift_data["status"]

    if status == "insufficient_data":
        st.warning(
            "Insufficient production data for a reliable drift review."
        )
    elif status == "review":
        st.warning(
            "Potential feature drift detected. Investigate the affected features."
        )
    else:
        st.success("No feature drift flags detected by the configured thresholds.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Production Samples",
        drift_data["current_samples"]
    )
    col2.metric(
        "Minimum Samples",
        drift_data["minimum_samples"]
    )
    col3.metric(
        "Reference Samples",
        drift_data.get("reference_samples", 0)
    )

    feature_results = drift_data.get("features", {})

    feature_results = drift_data.get("features", {})

    if feature_results:
        drift_table = pd.DataFrame.from_dict(
            feature_results,
            orient="index"
        ).rename_axis("Feature").reset_index()

        st.subheader("Feature-Level Drift Results")
        st.dataframe(drift_table, use_container_width=True)

    elif drift_data["status"] == "insufficient_data":
        st.info(
        f"Feature-level drift scores will appear after at least "
        f'{drift_data["minimum_samples"]} successful predictions. '
        f'Currently collected: {drift_data["current_samples"]}.'
    )



    st.caption(
        "PSI measures numerical distribution shifts; total variation "
        "distance measures categorical distribution shifts. "
        "Drift is not a measure of model accuracy."
    )

except requests.RequestException:
    st.warning(
        "Drift monitoring is unavailable. Check that the API container is running."
    )
except (KeyError, ValueError) as exc:
    st.error(f"Could not interpret the drift response: {exc}")





# ---------------- Deployed Model Information ----------------
st.divider()
st.header("Deployed Model Information")

try:
    response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=5
    )
    response.raise_for_status()
    model_info = response.json()

    st.write(f"**Model:** {model_info['model']}")
    st.write(f"**Version:** {model_info['model_version']}")
    st.write(f"**Deployment Stage:** {model_info['deployment_stage']}")
    st.write(
        f"**Decision Threshold:** "
        f"{model_info['threshold']:.4f}"
    )

    evaluation = model_info["evaluation_metrics"]

    st.subheader("Test Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{evaluation['roc_auc']:.4f}")
    col2.metric("F1 Score", f"{evaluation['f1']:.4f}")
    col3.metric("Recall", f"{evaluation['recall']:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("Precision", f"{evaluation['precision']:.4f}")
    col5.metric("Brier Score", f"{evaluation['brier_score']:.4f}")

except requests.RequestException:
    st.warning(
        "Model information is unavailable. "
        "Check that the prediction API is running."
    )
except (KeyError, ValueError, TypeError):
    st.error("The API returned unexpected model metadata.")
