from collections import defaultdict
import random


def balance_dataset(dataset, random_state=42):
    """
    Her yazardan eşit sayıda chunk alır.
    Fazla chunk'ları rastgele örnekleyerek dengeler.
    """
    random.seed(random_state)

    by_author = defaultdict(list)
    for chunk, author in dataset:
        by_author[author].append((chunk, author))

    min_count = min(len(items) for items in by_author.values())
    print(f"Yazar başına chunk sayısı: {min_count}")

    balanced = []
    for author, items in by_author.items():
        balanced.extend(random.sample(items, min_count))

    random.shuffle(balanced)
    return balanced