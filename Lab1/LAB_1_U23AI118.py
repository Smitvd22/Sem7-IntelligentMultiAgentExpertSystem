# Semantic Network with Inheritance

triples = [
    ("bird", "can", "fly"),
    ("bird", "has", "wings"),
    ("animal", "has", "skin"),
    ("bird", "is-a", "animal"),
    ("penguin", "is-a", "bird"),
    ("penguin", "cannot", "fly"),
    ("shark", "is-a", "fish"),
    ("fish", "is-a", "animal"),
    ("fish", "has", "gills"),
]


def get_parent(node):
    for s, r, o in triples:
        if s == node and r == "is-a":
            return o
    return None


def inheritance_chain(node):
    chain = []
    while node:
        chain.append(node)
        node = get_parent(node)
    return chain


def query(node, relation):
    current = node

    while current:
        # Check exception first
        for s, r, o in triples:
            if s == current and r == "cannot" and relation == "can":
                return f"{node} cannot {o}"

        # Check normal property
        for s, r, o in triples:
            if s == current and r == relation:
                return f"{node} {relation} {o}"

        current = get_parent(current)

    return "Unknown"


# Demonstration

print("Knowledge Base:")
for t in triples:
    print(t)

print("\nInheritance Chains:")
for node in ["penguin", "bird", "shark"]:
    print(f"{node}: {' -> '.join(inheritance_chain(node))}")

print("\nQueries:")
queries = [
    ("penguin", "can"),
    ("bird", "can"),
    ("penguin", "has"),
    ("shark", "has"),
    ("shark", "can"),
]

for node, rel in queries:
    print(f"{node} {rel}? -> {query(node, rel)}")

triples.extend([
    ("ostrich", "is-a", "bird"),
    ("ostrich", "cannot", "fly"),
    ("ostrich", "can", "run fast"),
])

print(query("ostrich", "can"))      # ostrich cannot fly
print(query("ostrich", "has"))      # ostrich has wings