

\# Model Monitoring and Retraining Policy



\## 1. Purpose



This policy defines how the Hospital Readmission Analytics

prototype monitors input drift, prediction service health,

model performance, and conditions for retraining.



This is an academic decision-support prototype. It is not

clinically validated and must not independently determine

patient care.



\## 2. Current Model



\- Model: XGBoost classifier

\- Registry name: HospitalReadmissionXGBoost

\- Registered version: 1

\- Decision threshold: 0.1336142122745514

\- Test ROC-AUC: 0.6311

\- Test recall: 0.4211

\- Test precision: 0.1661

\- Test F1: 0.2383

\- Test Brier score: 0.0914



These metrics are baseline results from the held-out test set.

They are not guarantees of future production performance.



\## 3. Monitoring Categories



\### 3.1 Service Health



Monitor:

\- API health and availability

\- Prediction request count

\- Prediction latency

\- Failed prediction requests

\- Input validation failures



Action:

\- Investigate repeated failures or sustained latency increases.

\- Check application logs, container health, and dependencies.

\- Preserve the current working deployment while investigating.



\### 3.2 Input Data Quality



Monitor:

\- Missing required fields

\- Invalid or unexpected feature values

\- Unexpected categorical values

\- Changes in feature distributions



Action:

\- Investigate schema changes and data pipeline problems.

\- Do not retrain solely to compensate for invalid input data.



\### 3.3 Feature Drift



The API currently calculates numerical Population Stability

Index (PSI) and categorical total variation distance.



Current implementation:

\- Minimum successful monitoring samples: 30

\- Review threshold: 0.2

\- Reference features: models/reference\_features.csv



Interpretation:

\- Below 0.1: lower observed drift under this heuristic.

\- 0.1 to below 0.2: investigate changes.

\- 0.2 or above: trigger a documented review.



These are provisional project thresholds, not universal

statistical guarantees. A sample of 30 is only a minimum

operational gate and may be inadequate for reliable drift

assessment, especially for rare categories.



Action:

\- Verify sample size and data quality.

\- Compare current and reference distributions.

\- Investigate changes in patient mix, source systems, and

&#x20; feature engineering.

\- Do not automatically retrain based on drift alone.



\### 3.4 Model Performance



When reliable outcome labels become available, calculate:

\- Recall

\- Precision

\- F1 score

\- ROC-AUC

\- Brier score

\- Calibration curve

\- Performance by age group and gender



Compare results with the documented test-set baseline.



A proposed review trigger is:

\- Recall falls by at least 5 percentage points, or

\- ROC-AUC falls by at least 0.03, or

\- Brier score increases by at least 0.01.



These are provisional engineering thresholds. Confirm sample

sizes, uncertainty, data quality, and label delay before

concluding that performance has deteriorated.



The API currently monitors inputs and operational predictions.

It does not establish real-world model performance unless

reliable outcome labels are subsequently linked to predictions.



\## 4. Retraining Criteria



Consider retraining when one or more of the following occurs:

\- Sustained, material feature drift is confirmed.

\- Reliable labelled data shows performance deterioration.

\- The source data or feature definitions change.

\- A sufficiently representative new dataset becomes available.

\- A documented evaluation identifies a meaningful limitation

&#x20; that a candidate model could address.



Before retraining:

1\. Investigate data quality and pipeline issues.

2\. Confirm the availability and reliability of outcome labels.

3\. Preserve the existing model and evaluation artifacts.

4\. Use a reproducible patient-group-aware split.

5\. Compare Logistic Regression, Random Forest, and XGBoost.

6\. Track parameters and metrics with MLflow.

7\. Evaluate recall, precision, F1, ROC-AUC, Brier score,

&#x20;  calibration, and subgroup performance.



Retraining is not automatically justified by a calendar date

or a drift alert alone.



\## 5. Candidate Model Approval



A candidate model must:

\- Pass data and pipeline validation.

\- Be evaluated on data not used for fitting or threshold

&#x20; selection.

\- Be compared with the currently approved model.

\- Have its metrics, limitations, and subgroup results recorded.

\- Be registered as a new model version in MLflow.

\- Pass API and deployment smoke tests.



Promotion requires documented review and approval.



Do not overwrite the existing production model merely because

a new model has been trained.



\## 6. Deployment and Rollback



Before deployment:

\- Preserve the current working container and model version.

\- Test the candidate in an isolated environment.

\- Verify API health and prediction response structure.

\- Verify logging and monitoring behavior.



After deployment:

\- Check health, latency, errors, and prediction behavior.

\- Investigate unexpected changes.

\- Roll back to the previously approved version if serious

&#x20; technical regressions occur.



Keep rollback artifacts until the new deployment has been

verified.



\## 7. Monitoring Review Record



For each review, record:

\- Review date

\- Model version

\- Monitoring window and sample count

\- Data quality findings

\- Drift results

\- Available labelled performance metrics

\- Subgroup findings

\- Decision and justification

\- Follow-up actions and responsible reviewer



Possible decisions:

\- Continue monitoring

\- Investigate data quality

\- Collect more labelled outcomes

\- Retrain and evaluate a candidate

\- Approve a new version

\- Roll back a deployment



\## 8. Limitations



The reference dataset is historical and does not represent

proof of current hospital conditions. Drift thresholds and

performance triggers in this document are provisional

engineering rules for an academic prototype.



No model version should be described as clinically safe or

validated without appropriate independent clinical and

prospective validation.
