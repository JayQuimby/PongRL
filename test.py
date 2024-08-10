import random

# Example list of tuples, where index 2 is the score
items = [
    ('item1', 'value1', 10),
    ('item2', 'value2', 30),
    ('item3', 'value3', 20),
    ('item4', 'value4', 40)
]

# Extract the weights (scores) from the list of tuples
weights = [item[2] for item in items]

# Sample one item based on the weights
selected_item = random.sample(items, weights=weights, k=2)

print(selected_item)