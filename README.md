# Hiver SDE Intern — AI Customer Support Agent

## Overview

This project builds a lightweight AI customer support agent using the Customer Support on Twitter dataset. The system focuses on Apple Support (@AppleSupport) and performs three tasks:

1. Classifies incoming customer messages into support intents.
2. Retrieves historically similar customer-support interactions to ground a draft response.
3. Decides whether the message should be automatically handled or escalated to a human.

The implementation uses classical NLP methods so that the complete pipeline remains lightweight, interpretable, and reproducible on a normal laptop.

---

## Dataset

The project uses the Kaggle **Customer Support on Twitter** dataset.

The original dataset contains approximately 3 million tweets from multiple customer-support brands.

For this project, @AppleSupport was selected because it provides a large number of customer messages and historical support replies.

The raw Kaggle dataset is intentionally **not included in this repository** because it is approximately 516 MB.

Download the dataset from Kaggle and place the extracted 	wcs.csv file at:

    data/twcs/twcs.csv

The expected columns are:

    tweet_id
    author_id
    inbound
    created_at
    text
    response_tweet_id
    in_response_to_tweet_id

---

## Environment Setup

Python 3.10+ is recommended.

Create and activate a virtual environment:

    python -m venv .venv

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

---

## Pipeline

The overall pipeline is:

    Raw Twitter Dataset
            |
            v
    AppleSupport Extraction
            |
            v
    Intent Labeling
            |
            v
    Intent Classification
            |
            v
    Historical Reply Retrieval
            |
            v
    Reply Drafting
            |
            v
    Auto-handle / Escalate Decision

---

## 1. Extract Apple Support Data

After downloading 	wcs.csv, run:

    python src/explore_apple.py

The Apple Support subset can then be prepared using the project scripts.

The resulting working dataset is:

    data/apple_support_raw.csv

---

## 2. Generate Intent Labels

Run:

    python src/label_data.py

The project uses the following 11 intent categories:

- ios_update
- device_performance
- keyboard_input
- connectivity
- battery_power
- apps_services
- music_media
- account_activation
- payments_purchases
- hardware_accessories
- other

The taxonomy was derived from recurring themes in the Apple Support customer messages.

---

## 3. Train and Evaluate Intent Classifiers

Two lightweight baselines were evaluated:

### Baseline 1 — TF-IDF + Logistic Regression

Accuracy on the weakly labelled dataset:

    94.83%

Macro F1:

    91.41%

Weighted F1:

    94.88%

Run:

    python src/train_baseline.py

### Baseline 2 — TF-IDF + Linear SVM

Accuracy on the weakly labelled dataset:

    98.28%

Macro F1:

    96.50%

Weighted F1:

    98.28%

Run:

    python src/train_baseline_svm.py

These numbers are diagnostic results on weak labels and should not be interpreted as human-grounded performance.

---

## 4. Golden Evaluation Set

A 200-example stratified evaluation set was created and manually reviewed.

The final golden set is:

    data/golden_set_final.xlsx

The examples cover all 11 intent categories.

The golden examples are removed from the classifier training data before evaluation to prevent direct train/test leakage.

Run:

    python src/evaluate_classifier.py

### Leakage-free result

Accuracy:

    87.50%

Macro F1:

    86.59%

Weighted F1:

    87.35%

This is the main intent-classification evaluation result.

---

## 5. Historical Reply Retrieval

Historical customer-to-AppleSupport reply pairs are reconstructed using the tweet reply relationships in the dataset.

The retrieval system uses TF-IDF vectors and cosine similarity over historical customer messages.

Run:

    python src/build_reply_pairs.py

The evaluation corpus excludes golden evaluation examples.

Run:

    python src/evaluate_retriever.py

### Retrieval results

Recall@1:

    38.50%

Recall@3:

    60.50%

Recall@5:

    69.50%

Mean Reciprocal Rank:

    0.5018

These metrics measure whether the retrieved historical interaction has the same intent as the manually reviewed query. They are an intent-consistency retrieval diagnostic, not a direct measurement of final response quality.

---

## 6. Support Agent

The main agent is implemented in:

    src/support_agent.py

The agent combines:

- TF-IDF + Linear SVM intent classification
- historical reply retrieval
- issue-specific intent correction
- similarity-based confidence
- escalation rules

The hybrid correction layer is used for cases where closely related words such as "update" and "battery" can cause a classifier ambiguity.

---

## 7. Escalation Policy

The system escalates messages when:

- the customer explicitly asks for a human
- historical retrieval confidence is low
- the message belongs to payments_purchases
- an account issue has limited historical support
- the message falls into the other category

Evaluation on the 200-example golden set:

    AUTO_HANDLE: 99 / 200 (49.5%)
    ESCALATE:    101 / 200 (50.5%)

The objective is conservative automation: uncertain or unsupported cases are routed to a human instead of producing an unreliable automated response.

---

## 8. Reply Quality Evaluation

Historical replies were evaluated using a structured human-labelled quality rubric covering:

- Relevance
- Helpfulness
- Groundedness
- Overall quality

Average scores across 200 examples:

    Relevance:    3.21 / 5
    Helpfulness:  2.61 / 5
    Groundedness: 3.38 / 5
    Overall:      2.90 / 5

The evaluation artifacts are available in:

    data/reply_quality_eval.csv

The judge rubric is available in:

    src/judge_prompt.txt

The reply-quality evaluation was reviewed using human-labelled judgements across relevance, helpfulness, groundedness, and overall quality.

---

## 9. Failure Analysis

The major observed failure patterns include:

1. Generic or insufficient responses
2. Wrong historical example causing irrelevance
3. Irrelevant responses
4. Low retrieval confidence

Detailed examples are available in:

    data/failure_analysis.csv

The failures show that high intent-classification accuracy does not automatically produce high-quality customer replies. Retrieval quality and response usefulness remain important bottlenecks.

---

## What Is Misleading About the Headline Number?

The 87.50% intent accuracy should not be interpreted as production-level performance.

Important limitations include:

- The golden set contains only 200 examples.
- The dataset is highly imbalanced across real-world issues.
- other represents a substantial portion of the evaluation set.
- The classifier is evaluated on a single selected brand.
- The response-quality evaluation is substantially weaker than the intent-classification result.
- Retrieval Recall@1 is only 38.50%.
- The reply-quality evaluation was based on human-labelled judgements.

Therefore, the headline number demonstrates that the intent-classification component works reasonably well on the reviewed sample, but it does not establish that the complete support agent is production-ready.

---

## Reproducibility

The repository contains the source code, evaluation set, evaluation outputs, and documentation needed to understand the system.

The large raw Kaggle dataset and generated intermediate datasets are excluded from GitHub using .gitignore.

A fresh run follows this general sequence:

    pip install -r requirements.txt
    python src/explore_apple.py
    python src/label_data.py
    python src/train_baseline.py
    python src/train_baseline_svm.py
    python src/evaluate_classifier.py
    python src/build_reply_pairs.py
    python src/evaluate_retriever.py
    python src/evaluate_escalation.py
    python src/failure_analysis.py

---

## AI Assistance

AI coding assistants were used for implementation scaffolding, debugging, analysis support, and documentation.

The final implementation, intent taxonomy, evaluation design, leakage controls, failure analysis, and reported results were reviewed and modified as part of the project development process.

---

## Project Structure

    .
    +-- src/
    ¦   +-- support_agent.py
    ¦   +-- label_data.py
    ¦   +-- train_baseline.py
    ¦   +-- train_baseline_svm.py
    ¦   +-- evaluate_classifier.py
    ¦   +-- build_reply_pairs.py
    ¦   +-- reply_retriever.py
    ¦   +-- evaluate_retriever.py
    ¦   +-- escalation.py
    ¦   +-- evaluate_escalation.py
    ¦   +-- failure_analysis.py
    ¦
    +-- data/
    ¦   +-- golden_set_final.xlsx
    ¦   +-- golden_predictions.csv
    ¦   +-- retrieval_predictions.csv
    ¦   +-- reply_quality_eval.csv
    ¦   +-- escalation_eval.csv
    ¦   +-- failure_analysis.csv
    ¦
    +-- README.md
    +-- REPORT.md
    +-- DECISION_LOG.md
    +-- requirements.txt
    +-- .gitignore

