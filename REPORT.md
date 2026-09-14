# Hiver SDE Intern Take-Home Report

## 1. Problem Framing

The goal of this project is to build an AI customer-support agent from the Customer Support on Twitter dataset.

The system focuses on @AppleSupport and addresses three tasks:

1. Identify the intent of an incoming customer message.
2. Retrieve historically similar support interactions to ground a response.
3. Decide whether the message can be auto-handled or should be escalated.

The system is deliberately lightweight and uses classical NLP methods so that the complete pipeline can be reproduced on a standard laptop.

---

## 2. Data Selection and Preparation

The Customer Support on Twitter dataset contains approximately 3 million tweets covering multiple brands.

Apple Support was selected because it provides a large number of customer messages and historical support interactions.

The selected Apple Support subset contains:

- 97,896 inbound customer messages
- 106,860 outbound messages
- 78,406 customer-to-AppleSupport reply pairs

The tweet relationships were reconstructed using in_response_to_tweet_id and esponse_tweet_id.

The final intent taxonomy contains 11 categories:

| Intent | Description |
|---|---|
| ios_update | iOS/software update issues |
| device_performance | Slow, freezing or performance problems |
| keyboard_input | Keyboard or typing issues |
| connectivity | Wi-Fi, Bluetooth, network or connection issues |
| battery_power | Battery drain, charging or power issues |
| apps_services | App, iCloud or Apple service issues |
| music_media | Music, media or playback issues |
| account_activation | Login, activation or account access |
| payments_purchases | Purchases, billing or payment issues |
| hardware_accessories | Device hardware and accessories |
| other | Messages outside the supported categories |

The taxonomy was derived from recurring topics in the Apple Support messages rather than imposed from an external intent schema.

---

## 3. Baseline Models

Two text-classification baselines were implemented using TF-IDF features.

### Baseline A: TF-IDF + Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 94.83% |
| Macro F1 | 91.41% |
| Weighted F1 | 94.88% |

### Baseline B: TF-IDF + Linear SVM

| Metric | Score |
|---|---:|
| Accuracy | 98.28% |
| Macro F1 | 96.50% |
| Weighted F1 | 98.28% |

The Linear SVM was selected as the stronger baseline.

However, both results are measured against weak labels generated from heuristic rules. Therefore, they are treated as development diagnostics rather than the final performance claim.

---

## 4. Golden Evaluation

A stratified set of 200 customer messages was created across the 11 intent categories.

The final labels were manually reviewed.

To prevent train/test leakage, all 200 golden examples were removed from the classifier training data before evaluation.

Training examples:

- Original: 97,896
- Golden examples removed: 200
- Final training set: 97,696

### Leakage-free result

| Metric | Score |
|---|---:|
| Accuracy | **87.50%** |
| Macro F1 | **86.59%** |
| Weighted F1 | **87.35%** |

The 87.50% accuracy is the primary intent-classification result.

The difference between the weak-label result and the golden result demonstrates why evaluation against reviewed examples is more informative than evaluating only against automatically generated labels.

---

## 5. Historical Reply Retrieval

The support agent retrieves similar historical customer messages using TF-IDF and cosine similarity.

Golden examples were excluded from the retrieval corpus to reduce evaluation leakage.

The evaluation contains 200 customer queries.

| Metric | Score |
|---|---:|
| Recall@1 | 38.50% |
| Recall@3 | 60.50% |
| Recall@5 | 69.50% |
| MRR | 0.5018 |

These metrics measure whether a retrieved historical example has the same intent as the query.

This is an intent-consistency retrieval diagnostic, not a direct measure of whether the generated response is helpful.

---

## 6. Reply Quality

Retrieved historical replies were evaluated using an LLM-as-judge rubric.

The judge scored:

- Relevance
- Helpfulness
- Groundedness
- Overall quality

Results over 200 examples:

| Dimension | Average |
|---|---:|
| Relevance | 3.21 / 5 |
| Helpfulness | 2.61 / 5 |
| Groundedness | 3.38 / 5 |
| Overall | 2.90 / 5 |

The results reveal an important distinction: successful intent classification does not necessarily produce a useful customer-support response.

Many historical replies are short acknowledgements or generic troubleshooting messages, and retrieval can occasionally select an example from the wrong issue.

### Human–LLM Judge Calibration

Independent human calibration of the reply-quality judge was not completed in this run.

Model-generated provisional comparison scores were intentionally not reported as genuine human ratings. This avoids presenting synthetic agreement as human evidence.

This is a limitation of the current evaluation and is a priority for the next iteration.

---

## 7. Escalation Policy

The system uses conservative escalation rules.

A message is escalated when:

- the customer explicitly requests a human;
- historical retrieval confidence is low;
- the intent is payments_purchases;
- an account issue has limited historical support;
- the message falls into other.

Evaluation on the 200 golden examples:

| Decision | Count | Percentage |
|---|---:|---:|
| AUTO_HANDLE | 99 | 49.5% |
| ESCALATE | 101 | 50.5% |

The policy intentionally prefers escalation when confidence or historical support is insufficient.

---

## 8. Failure Analysis

The main observed response-quality failures were:

### 1. Generic or insufficient responses

The retrieved response may acknowledge the problem without giving useful issue-specific guidance.

### 2. Wrong historical example causing irrelevance

The retriever can select a historically similar message whose underlying issue is different from the current customer problem.

### 3. Irrelevant responses

Very short or ambiguous customer messages can result in poorly matched historical responses.

### 4. Low retrieval confidence

Some customer messages have no sufficiently similar historical example.

The detailed examples are stored in data/failure_analysis.csv.

These failures indicate that retrieval quality is currently a larger bottleneck for response quality than the basic text classifier.

---

## 9. What Is Misleading About My Headline Number?

The headline intent accuracy of **87.50%** is useful but should not be interpreted as production-level performance.

There are several limitations:

- The evaluation contains only 200 examples.
- It represents one selected brand.
- The evaluation distribution is stratified and therefore may not match real production traffic.
- other is a relatively large category.
- The classifier evaluates intent, not the quality of the final support response.
- Retrieval Recall@1 is only 38.50%.
- Overall LLM-judged reply quality is 2.90/5.
- Independent human calibration of the LLM judge was not completed.

Therefore, the 87.50% number should be interpreted as evidence that the intent component performs reasonably well on the reviewed evaluation sample, rather than evidence that the complete AI support agent is production-ready.

---

## 10. Two Important Engineering Decisions

### Leakage Prevention

The golden evaluation examples were explicitly removed from both:

- classifier training data
- historical retrieval corpus

This prevents the system from receiving an exact evaluation example during evaluation.

### Conservative Escalation

The system does not attempt to automate every message.

Low-confidence, unsupported, payment-related, account-related, and explicitly human-requested cases are routed to a human.

This prioritizes safe handling over maximizing the percentage of automatically handled messages.

---

## 11. Next Week

The next iteration would focus on improving response quality rather than only classifier accuracy.

### Priority 1 — Better Retrieval

Use stronger semantic embeddings or a hybrid lexical + semantic retriever to improve historical-example matching.

### Priority 2 — Better Response Generation

Instead of directly returning a historical reply, use retrieved examples as evidence for a structured response-generation step.

### Priority 3 — Human Calibration

Have humans independently score a sample of replies using the same relevance, helpfulness, groundedness and overall rubric, then measure agreement with the LLM judge.

### Priority 4 — Harder Evaluation

Expand the golden set and include more difficult, ambiguous and cross-intent examples.

### Priority 5 — Monitoring

Track intent confidence, retrieval similarity, escalation rate and response-quality feedback after deployment.

---

## 12. Conclusion

The project demonstrates an end-to-end AI support workflow:

**Intent Classification → Historical Retrieval → Response Grounding → Escalation**

The strongest result is an **87.50% accuracy / 86.59% macro F1** on a 200-example manually reviewed, leakage-controlled golden set.

At the same time, the evaluation shows that classification accuracy alone is insufficient. Historical retrieval and response quality remain the main areas for improvement.

The system therefore provides a practical baseline and a clear path toward a more reliable customer-support agent.

