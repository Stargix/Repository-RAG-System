import pandas as pd
import json
from retrieve import retrieve_repository

def recall_at_10(relevant, retrieved):
    top_10 = retrieved[:10]
    relevant_set = set(relevant)
    retrieved_set = set(top_10)

    # Calculate hits and total relevant items
    hits = len(relevant_set & retrieved_set)
    total_relevant = len(relevant_set)
    
    return hits / total_relevant


if __name__ == "__main__":
     
    with open("escrcpy-commits-generated.json", "r") as f:
            test_data = json.load(f)

    df_test = pd.DataFrame(test_data)
    questions = df_test["question"].tolist()

    sources = []
    for query in questions:
        metadata = retrieve_repository(query, n_results=10)
        source = [result["source"] for result in metadata]
        sources.append(source)

        sources = [[source.replace("\\", "/") for source in source_list] for source_list in sources]

    df_test["predicted_sources"] = sources

    df_test["recall@10"] = df_test.apply(
        lambda row: recall_at_10(row["files"], row["predicted_sources"]),
        axis=1
    )

    average_recall = df_test["recall@10"].mean()
    print(f"Average Recall@10: {average_recall:.2f}")

    df_test.to_csv("evaluation_results.csv", index=False)