import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = "data/apple_reply_pairs.csv"


class ReplyRetriever:

    def __init__(self, data_file=DATA_FILE):

        print("Loading historical reply pairs...")

        self.df = pd.read_csv(data_file)

        self.df["customer_text"] = (
            self.df["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.df["apple_support_reply"] = (
            self.df["apple_support_reply"]
            .fillna("")
            .astype(str)
        )

        # TF-IDF representation of historical customer messages
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

        print("Historical examples:", len(self.df))
        print("TF-IDF matrix:", self.matrix.shape)
        print("Retriever ready!")


    def retrieve(self, query, top_k=3):

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get highest scoring examples
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({
                "customer_text": self.df.iloc[idx]["customer_text"],
                "reply": self.df.iloc[idx]["apple_support_reply"],
                "similarity": float(scores[idx])
            })

        return results


if __name__ == "__main__":

    retriever = ReplyRetriever()

    print()
    print("=" * 60)
    print("TESTING REPLY RETRIEVER")
    print("=" * 60)

    query = input(
        "\nEnter a customer message: "
    ).strip()

    results = retriever.retrieve(
        query,
        top_k=3
    )

    print()

    for i, result in enumerate(results, start=1):

        print("-" * 60)
        print(f"RESULT {i}")
        print("-" * 60)

        print("Customer example:")
        print(result["customer_text"])

        print()
        print("Historical AppleSupport reply:")
        print(result["reply"])

        print()
        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

    print()
    print("=" * 60)