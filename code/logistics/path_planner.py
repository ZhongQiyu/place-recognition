import networkx


class PathPlanner(networkx.Graph):
    """
    A Python class that plans the path to take
    during a visit to the main campus, for data
    collection.
    """
    # see if the syntax is correct
    trail = networkx.Graph()
