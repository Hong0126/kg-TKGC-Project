def build_prompt(
    subj, rel, obj, interval, subgraph, style: str = "nl"
) -> str:
    triples = []
    for u, v, _, d in subgraph.edges(data=True, keys=True):
        ts, te = d["time"]
        if style == "triple":
            triples.append(f"({u}, {d['rel']}, {v}, [{ts}, {te}])")
        else:
            triples.append(f"From {ts} to {te}, {u} {d['rel']} {v}.")
    context = "\n".join(triples)
    q = (f"Given the facts above, between which years was it true that "
         f"{subj} {rel} {obj}?")
    return context + "\n" + q + "\nAnswer:"