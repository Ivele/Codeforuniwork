import json
import csv
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def init_models():
    return {
        "semantic_model": SentenceTransformer('all-MiniLM-L6-v2'),
        "zero_shot": pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    }


def validate_item(item, models, config):
    anomalies = []

    if not item.get("current_segment"):
        anomalies.append(("missing_value", "Пустой вопрос"))
    if not item.get("context"):
        anomalies.append(("missing_value", "Пустой контекст"))
    if not item.get("target"):
        anomalies.append(("missing_value", "Пустой ответ"))

    if len(item['context'].split()) < 5:
        anomalies.append(("short_context", "Короткий контекст"))

    if len(item['target'].split()) > 20:
        anomalies.append(("long_target", "Длинный ответ"))

    try:
        emb1 = models["semantic_model"].encode(item["context"])
        emb2 = models["semantic_model"].encode(item["target"])
        sim = cosine_similarity([emb1], [emb2])[0][0]

        if sim < config["semantic_threshold"]:
            anomalies.append(("low_semantic_similarity", f"Сходство: {sim:.2f}"))

        result = models["zero_shot"](
            item["context"],
            candidate_labels=[item["target"], "противоречие", "нерелевантно"]
        )

        if result['labels'][0] in ["противоречие", "нерелевантно"]:
            anomalies.append(("conceptual_inconsistency", f"Метка: {result['labels'][0]}"))

        contradiction = models["zero_shot"](
            item["context"],
            candidate_labels=["подтверждение", "противоречие", "нейтрально"],
            hypothesis_template="Ответ: {}"
        )
        if contradiction['labels'][0] == "противоречие" and contradiction['scores'][0] > config["contradiction_threshold"]:
            anomalies.append(("contradictory_context", f"Противоречие: {contradiction['scores'][0]:.2f}"))

    except Exception as e:
        anomalies.append(("error", str(e)))

    return anomalies


def save_to_csv(valid, invalid):
    with open("valid_records.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["current_segment", "context", "target"])
        writer.writeheader()
        writer.writerows(valid)

    with open("invalid_records.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["current_segment", "context", "target", "anomalies"])
        writer.writeheader()
        for item in invalid:
            row = item.copy()
            row["anomalies"] = "; ".join(f"{a[0]}: {a[1]}" for a in item["anomalies"])
            writer.writerow(row)


def visualize_summary(total, valid_count, anomaly_counter):
    data = [{"type": "valid", "count": valid_count}] + [
        {"type": key, "count": value} for key, value in anomaly_counter.items()
    ]

    fig1 = px.treemap(data, path=["type"], values="count", title="Treemap: Распределение записей")
    fig1.show()

    fig2 = px.bar(data[1:], x="type", y="count", title="Bar Chart: Кол-во аномалий", text="count")
    fig2.update_layout(xaxis_tickangle=-45)
    fig2.show()


def main(filepath):
    config = {
        "semantic_threshold": 0.5,
        "contradiction_threshold": 0.7,
    }

    models = init_models()
    data = load_data(filepath)
    anomaly_counter = defaultdict(int)

    valid = []
    invalid = []

    for item in data:
        anomalies = validate_item(item, models, config)
        if not anomalies:
            valid.append(item)
        else:
            invalid.append({**item, "anomalies": anomalies})
            for a_type, _ in anomalies:
                anomaly_counter[a_type] += 1

    total = len(data)
    valid_count = len(valid)

    print(f"\nОбработано: {total}")
    print(f"Валидные: {valid_count}")
    print(f"Невалидные: {len(invalid)}")
    print("\nТоп аномалий:")
    for k, v in anomaly_counter.items():
        print(f"  - {k}: {v}")

    save_to_csv(valid, invalid)
    visualize_summary(total, valid_count, anomaly_counter)


if __name__ == "__main__":
    main("infa.json")
