# Hiver SDE Intern — Take-Home Assignment Report

## 1. Problem Framing

The goal of this project is to build an AI customer-support agent from the **Customer Support on Twitter** dataset.

I selected **Apple Support (`@AppleSupport`)** as the target brand. The system addresses three tasks:

1. Classify each incoming customer message into a small set of support intents derived from the data.
2. Retrieve historically similar customer-support interactions to ground a draft response.
3. Decide whether the message should be automatically handled or escalated to a human, with a reason for the decision.

For this project, "good" means that the system should identify the customer's issue reliably, use relevant historical support interactions when drafting a response, and avoid confidently responding when historical evidence is insufficient.

### What I chose not to build

I intentionally did not build a production-scale generative chatbot, full conversation-management system, or end-to-end autonomous customer-support platform. The focus was on demonstrating a measurable classification, retrieval, response-grounding, and escalation pipeline that can be reproduced on a standard laptop.

---

## 2. Data Selection and Preparation

The Customer Support on Twitter dataset contains approximately 3 million tweets covering multiple brands.

Apple Support was selected because it provides a large number of customer messages and historical support interactions.

The selected Apple Support subset contains:

- 97,896 inbound customer messages
- 106,860 outbound messages
- 78,406 customer-to-AppleSupport reply pairs

Tweet relationships were reconstructed using `in_response_to_tweet_id` and `response_tweet_id`.

The final intent taxonomy contains 11 categories:

| Intent | Description |
|---|---|
| `ios_update` | iOS/software update issues |
| `device_performance` | Slow, freezing or performance problems |
| `keyboard_input` | Keyboard or typing issues |
| `connectivity` | Wi-Fi, Bluetooth, network or connection issues |
| `battery_power` | Battery drain, charging or power issues |
| `apps_services` | App, iCloud or Apple service issues |
| `music_media` | Music, media or playback issues |
| `account_activation` | Login, activation or account access |
| `payments_purchases` | Purchases, billing or payment issues |
| `hardware_accessories` | Device hardware and accessories |
| `other` | Messages outside the supported categories |

The taxonomy was derived from recurring themes in the Apple Support customer messages rather than imported from an external intent taxonomy.

---

## 3. Baselines

Three reference points were evaluated: a trivial majority-class baseline and two standard TF-IDF classifiers.

### Baseline 0 — Majority Class

The trivial baseline always predicts the most frequent intent in the weakly labelled training data.

The most frequent intent is `other`.

| Metric | Score |
|---|---:|
| Accuracy | 41.05% |
| Macro F1 | 5.29% |
| Weighted F1 | 23.90% |

This provides a simple lower-bound reference for the classification task.

Run:

```text
python src/train_baseline_majority.py