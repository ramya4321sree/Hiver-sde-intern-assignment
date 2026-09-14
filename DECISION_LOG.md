# Decision Log

## 1. Selected Apple Support as the Target Brand

**Decision:** Use @AppleSupport as the single brand for the project.

**Reason:** Apple Support provides a large number of inbound customer messages and a large number of historical support replies, making it suitable for both intent classification and historical-response retrieval.

---

## 2. Did Not Use the Entire 3M-Tweet Dataset

**Decision:** Extract only the Apple Support portion of the dataset.

**Reason:** The assignment allows subsampling, and working with a single brand makes the intent taxonomy, retrieval corpus and evaluation more focused while keeping the pipeline reproducible on a normal laptop.

---

## 3. Derived the Intent Taxonomy from the Data

**Decision:** Use 11 intents rather than importing an external customer-support taxonomy.

**Reason:** The assignment asks for intents derived from the selected data. Recurring Apple Support topics such as updates, battery, connectivity, keyboard and payments were used to construct the taxonomy.

---

## 4. Kept an other Category

**Decision:** Include other as an explicit intent.

**Reason:** Forcing every message into one of the supported issue categories would create artificial labels for ambiguous or unrelated messages. other provides a safe fallback.

---

## 5. Used TF-IDF as the First Modeling Approach

**Decision:** Start with TF-IDF text features.

**Reason:** TF-IDF is lightweight, interpretable and fast to train, making it a strong baseline for a take-home assignment where reproducibility matters.

---

## 6. Compared Logistic Regression and Linear SVM

**Decision:** Implement two classical classifier baselines.

**Reason:** Comparing two simple models provides evidence that the selected approach is not arbitrary.

Linear SVM performed better on the weak-label benchmark and was therefore selected for the final classifier.

---

## 7. Did Not Use Weak-Label Accuracy as the Main Result

**Decision:** Treat the 98.28% Linear SVM score as a diagnostic rather than the headline result.

**Reason:** The labels used for that benchmark were generated using heuristic rules. A model evaluated against labels created by similar lexical rules can overestimate its real performance.

---

## 8. Created a Manually Reviewed Golden Set

**Decision:** Evaluate the classifier on 200 reviewed examples.

**Reason:** The golden set provides a more meaningful estimate of intent-classification performance than the weak-label benchmark.

The final golden set covers all 11 intent categories.

---

## 9. Removed Golden Examples from Training

**Decision:** Remove all golden examples from the classifier training data before evaluation.

**Reason:** This prevents direct train/test leakage and ensures that the evaluation messages are not available to the classifier during training.

---

## 10. Removed Golden Examples from Retrieval

**Decision:** Exclude golden customer messages from the historical retrieval corpus.

**Reason:** Otherwise, the retriever could return the exact evaluation message or its associated historical response, artificially inflating retrieval performance.

---

## 11. Used Historical Replies for Response Grounding

**Decision:** Retrieve similar historical customer-support interactions rather than generating unsupported responses from scratch.

**Reason:** Historical support interactions provide evidence of how similar customer problems were handled.

---

## 12. Used Similarity as a Confidence Signal

**Decision:** Use TF-IDF cosine similarity to estimate retrieval confidence.

**Reason:** A low similarity score indicates that the system does not have a sufficiently similar historical example. This provides a simple signal for deciding when to escalate.

---

## 13. Added Conservative Escalation Rules

**Decision:** Escalate uncertain and unsupported cases instead of attempting automatic responses for every message.

**Reason:** Customer support is a setting where an incorrect response can be worse than asking a human to handle the case.

---

## 14. Added a Hybrid Correction Layer

**Decision:** Add issue-specific keyword corrections after the classifier prediction.

**Reason:** Some Apple Support issues contain overlapping terms. For example, a battery complaint may mention an iOS update, causing a pure text classifier to predict ios_update. A targeted correction layer helps resolve this ambiguity.

---

## 15. Evaluated Retrieval and Reply Quality Separately

**Decision:** Do not treat retrieval success as equivalent to response quality.

**Reason:** Retrieving an example with the same intent does not guarantee that the historical reply is relevant, helpful or sufficiently grounded. Therefore, retrieval metrics and LLM-judged response quality are reported separately.

---

## Evaluation Limitations

The current run did not complete independent human calibration of the LLM reply-quality judge.

Model-generated provisional comparison scores were not presented as genuine human ratings.

A future iteration should have independent human reviewers score a sample using the same rubric and then measure agreement with the LLM judge.

---

## AI Assistance Disclosure

AI coding assistants were used for implementation scaffolding, debugging, analysis support and documentation.

The implementation and evaluation decisions were reviewed and modified during development, and the final results reported in this repository were generated from the project artifacts.
