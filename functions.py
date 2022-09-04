import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f"Writing to {name}.pkl")
    return None


def load_obj(name):
    name_ext = name[-4:]
    if name_ext != ".pkl":
        name = name + ".pkl"
    print(f"Loading {name}")
    with open(name, 'rb') as f:
        return pickle.load(f)
    return None


