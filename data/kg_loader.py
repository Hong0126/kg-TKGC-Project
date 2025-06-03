"""
Simple loader: reads <s,r,o,t_start,t_end> CSV/TSV and returns a NetworkX MultiDiGraph.
Edge attrs:  rel, time=(start,end)
"""
import csv, networkx as nx
from pathlib import Path

def load_kg(file_path: Path) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    with open(file_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for s, r, o, ts, te in reader:
            g.add_edge(s, o, rel=r, time=(int(ts), int(te)))
    return g