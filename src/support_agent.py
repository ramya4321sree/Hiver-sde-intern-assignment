import sys
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from escalation import decide_escalation


TRAIN_FILE = "data/apple_labeled.csv"
REPLY_FILE = "data/apple_reply_pairs.csv"


class SupportAgent:

    def __init__(self):

        print("Loading support data...")

        # ==========================================
        # 1. LOAD INTENT TRAINING DATA
        # ==========================================

        self.train_df = pd.read_csv(TRAIN_FILE)

        self.train_df["clean_text"] = (
            self.train_df["clean_text"]
            .fillna("")
            .astype(str)
        )

        self.train_df["intent"] = (
            self.train_df["intent"]
            .fillna("other")
            .astype(str)
        )

        # ==========================================
        # 2. INTENT CLASSIFIER
        # ==========================================

        self.intent_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True
        )

        X_train = self.intent_vectorizer.fit_transform(
            self.train_df["clean_text"]
        )

        self.intent_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        self.intent_model.fit(
            X_train,
            self.train_df["intent"]
        )

        # ==========================================
        # 3. LOAD HISTORICAL CUSTOMER → REPLY PAIRS
        # ==========================================

        self.reply_df = pd.read_csv(REPLY_FILE)

        self.reply_df["customer_text"] = (
            self.reply_df["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.reply_df["apple_support_reply"] = (
            self.reply_df["apple_support_reply"]
            .fillna("")
            .astype(str)
        )

        # ==========================================
        # 4. RETRIEVAL MODEL
        # ==========================================

        self.reply_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.reply_matrix = self.reply_vectorizer.fit_transform(
            self.reply_df["customer_text"]
        )

        print(
            "Intent training examples:",
            len(self.train_df)
        )

        print(
            "Historical reply pairs:",
            len(self.reply_df)
        )

        print("Support agent ready!")

    # ==================================================
    # CLEAN MESSAGE
    # ==================================================

    def clean_message(self, message):

        text = message.lower()

        text = re.sub(
            r"@\w+",
            "",
            text
        )

        text = re.sub(
            r"https?://\S+",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ==================================================
    # CLASSIFY INTENT
    # ==================================================

    def classify_intent(self, message):

        cleaned = self.clean_message(message)

        vector = self.intent_vectorizer.transform(
            [cleaned]
        )

        probabilities = (
            self.intent_model.predict_proba(vector)[0]
        )

        best_index = probabilities.argmax()

        classifier_intent = (
            self.intent_model.classes_[best_index]
        )

        classifier_confidence = float(
            probabilities[best_index]
        )

        return (
            classifier_intent,
            classifier_confidence
        )

    # ==================================================
    # RETRIEVE HISTORICAL CASES
    # ==================================================

    def retrieve_replies(
        self,
        message,
        top_k=5
    ):

        cleaned = self.clean_message(message)

        vector = self.reply_vectorizer.transform(
            [cleaned]
        )

        scores = cosine_similarity(
            vector,
            self.reply_matrix
        ).flatten()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            results.append(
                {
                    "customer_text":
                        self.reply_df.iloc[index]["customer_text"],

                    "reply":
                        self.reply_df.iloc[index]["apple_support_reply"],

                    "similarity":
                        float(scores[index])
                }
            )

        return results

    # ==================================================
    # INFER INTENT FROM HISTORICAL EXAMPLES
    # ==================================================

    def infer_intent_from_history(
        self,
        message,
        historical_results
    ):

        message_lower = self.clean_message(message)

        # Keywords associated with each intent.
        # Used only as supporting evidence from the
        # retrieved historical customer messages.

        intent_keywords = {

            "battery_power": [
                "battery",
                "charging",
                "charge",
                "drain",
                "dies",
                "power"
            ],

            "connectivity": [
                "wifi",
                "wi-fi",
                "bluetooth",
                "network",
                "connection",
                "connected",
                "disconnect"
            ],

            "ios_update": [
                "ios update",
                "software update",
                "upgrade",
                "downgrade"
            ],

            "keyboard_input": [
                "keyboard",
                "typing",
                "type",
                "letters"
            ],

            "music_media": [
                "music",
                "itunes",
                "airplay",
                "audio",
                "video"
            ],

            "account_activation": [
                "apple id",
                "password",
                "activation",
                "activate",
                "login",
                "sign in",
                "locked"
            ],

            "payments_purchases": [
                "apple pay",
                "payment",
                "billing",
                "purchase",
                "charged",
                "refund"
            ],

            "hardware_accessories": [
                "airpods",
                "headphones",
                "earphones",
                "cable",
                "charger",
                "button",
                "screen",
                "display",
                "accessory"
            ],

            "apps_services": [
                "app",
                "icloud",
                "photos",
                "safari",
                "mail",
                "email",
                "facetime",
                "app store"
            ],

            "device_performance": [
                "slow",
                "freeze",
                "freezing",
                "crash",
                "crashing",
                "lag",
                "bug",
                "glitch",
                "stuck",
                "error",
                "not working"
            ]
        }

        scores = {
            intent: 0.0
            for intent in intent_keywords
        }

        # Score the original customer message.
        for intent, keywords in intent_keywords.items():

            for keyword in keywords:

                if keyword in message_lower:

                    scores[intent] += 2.0

        # Add evidence from retrieved historical messages.
        for result in historical_results:

            historical_text = (
                result["customer_text"].lower()
            )

            similarity = result["similarity"]

            for intent, keywords in intent_keywords.items():

                for keyword in keywords:

                    if keyword in historical_text:

                        scores[intent] += similarity

        if not scores:
            return None

        best_intent = max(
            scores,
            key=scores.get
        )

        # Only use this evidence when there is
        # meaningful support.
        if scores[best_intent] < 2.0:

            return None

        return best_intent

    # ==================================================
    # FINAL INTENT
    # ==================================================

    def get_final_intent(
        self,
        classifier_intent,
        classifier_confidence,
        historical_results,
        message
    ):

        history_intent = (
            self.infer_intent_from_history(
                message,
                historical_results
            )
        )

        # If historical evidence strongly points to
        # another intent, prefer it.
        if history_intent is not None:

            # Give historical evidence priority when
            # the classifier's prediction conflicts
            # with obvious issue-specific keywords.

            if history_intent != classifier_intent:

                cleaned = self.clean_message(message)

                strong_issue_terms = {
                    "battery_power": [
                        "battery",
                        "battery drain",
                        "battery draining"
                    ],

                    "connectivity": [
                        "wifi",
                        "bluetooth",
                        "connection"
                    ],

                    "keyboard_input": [
                        "keyboard",
                        "typing"
                    ],

                    "payments_purchases": [
                        "payment",
                        "charged",
                        "refund",
                        "purchase"
                    ]
                }

                for intent, terms in strong_issue_terms.items():

                    if intent == history_intent:

                        for term in terms:

                            if term in cleaned:

                                return intent

        return classifier_intent

    # ==================================================
    # CREATE DRAFT
    # ==================================================

    def create_draft(
        self,
        historical_reply
    ):

        reply = historical_reply.strip()

        # Remove old Twitter handles
        reply = re.sub(
            r"@\w+",
            "",
            reply
        )

        # Remove old URLs
        reply = re.sub(
            r"https?://\S+",
            "",
            reply
        )

        # Remove extra spaces
        reply = re.sub(
            r"\s+",
            " ",
            reply
        )

        return reply.strip()

    # ==================================================
    # COMPLETE AGENT
    # ==================================================

    def handle(self, message):

        # Classifier prediction
        classifier_intent, classifier_confidence = (
            self.classify_intent(message)
        )

        # Historical retrieval
        historical_results = (
            self.retrieve_replies(
                message,
                top_k=5
            )
        )

        # Final intent
        final_intent = self.get_final_intent(
            classifier_intent,
            classifier_confidence,
            historical_results,
            message
        )

        # Best historical example
        best_match = historical_results[0]

        # Escalation
        decision = decide_escalation(
            intent=final_intent,
            retrieval_similarity=best_match["similarity"],
            customer_text=message
        )

        # Draft response
        draft = self.create_draft(
            best_match["reply"]
        )

        return {
            "customer_message": message,

            "classifier_intent":
                classifier_intent,

            "classifier_confidence":
                classifier_confidence,

            "final_intent":
                final_intent,

            "historical_customer_message":
                best_match["customer_text"],

            "historical_reply":
                best_match["reply"],

            "similarity":
                best_match["similarity"],

            "draft_reply":
                draft,

            "decision":
                decision["decision"],

            "reason":
                decision["reason"]
        }


# ======================================================
# RUN AGENT
# ======================================================

if __name__ == "__main__":

    agent = SupportAgent()

    print()
    print("=" * 70)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 70)

    message = input(
        "\nCustomer message: "
    ).strip()

    if not message:

        print(
            "Please enter a customer message."
        )

        sys.exit()

    result = agent.handle(
        message
    )

    print()
    print("=" * 70)
    print("AGENT RESULT")
    print("=" * 70)

    print()
    print("Customer:")
    print(result["customer_message"])

    print()
    print("Classifier intent:")
    print(result["classifier_intent"])

    print()
    print(
        "Classifier confidence:",
        f"{result['classifier_confidence']:.4f}"
    )

    print()
    print("FINAL INTENT:")
    print(result["final_intent"])

    print()
    print("Historical customer example:")
    print(result["historical_customer_message"])

    print()
    print("Historical AppleSupport reply:")
    print(result["historical_reply"])

    print()
    print(
        "Retrieval similarity:",
        f"{result['similarity']:.4f}"
    )

    print()
    print("Draft reply:")
    print(result["draft_reply"])

    print()
    print("Decision:")
    print(result["decision"])

    print()
    print("Reason:")
    print(result["reason"])

    print()
    print("=" * 70)