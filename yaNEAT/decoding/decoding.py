import tensorflow as tf
from ..NEAT import Genome, Innovation
from collections import deque, defaultdict, OrderedDict
from typing import Dict, Set, List


def decode_NEAT(genome: Genome, innovations: List[Innovation]):
    """Decode a NEAT genome into its corresponding ANN phenotype."""
    def get_execution_order(end: Set[int], incoming: Dict[int, Set[int]]):
        q = deque(end)
        visited = defaultdict(bool)
        for v in end:
            visited[v] = True
        traversal = []
        while len(q) > 0:
            v = q.pop()
            traversal.append(v)
            for u in incoming[v]:
                if not visited[u]:
                    visited[u] = True
                    # TODO not sure this is quite right
                    q.appendleft(u)
                traversal.append(u)
        traversal = reversed(traversal)
        traversal = OrderedDict((x, None) for x in traversal)
        return list(traversal)

    input_nodes = set(genome.nodes)
    output_nodes = set(genome.nodes)

    recurrent_nodes = set()
    for source, destination in genome.connections:
        if destination in input_nodes:
            input_nodes.remove(destination)
        if source in output_nodes and source != destination:
            # TODO is this correct? There seems to be a flaw here...
            # an output node is any node that does not have a non-recurrent outgoing edge
            output_nodes.remove(source)
        if source == destination:
            # recurrent connections
            recurrent_nodes.add(source)

    incoming = defaultdict(lambda: [set(), None]) # pair of set of node-weight pairs and activation
    for innov, gene in genome.genes.items():
        innov = innovations[innov]
        incoming[innov.destination][0].add((innov.source, gene.weight))
        if incoming[innov.destination][1] is None:
            incoming[innov.destination][1] = gene.activation
        else:
            assert incoming[innov.destination][1] == gene.activation, "all activations going to same node must be same"

    # this is ugly
    execution_order = get_execution_order(output_nodes,
                                          defaultdict(set,
                                                      ((k, set(node for node, _ in v)) for (k, (v, _)) in incoming.items())))

    # build a mapping of destination nodes to weight vector-source node pairs
    weight_vectors = {}
    for destination, (pairs, _) in incoming.items():
        if len(pairs) > 0:
            wv = tf.constant([w for _, w in pairs])
            sources = set(s for s, _ in pairs)
            weight_vectors[destination] = (wv, sources)

    @tf.function
    def network(input, state, node_input_map, node_state_map):
        computed = {}

        for n in execution_order:
            if n in input_nodes:
                computed[n] = input[node_input_map[n]]
            elif n in recurrent_nodes:
                computed[n] = state[node_state_map[n]]
            else:
                # MUST be a hidden or output node
                assert n in weight_vectors
                # collect inputs to this neuron; I don't like having to do it this way
                x = tf.stack([computed[m] for m in weight_vectors[n][1]])
                w = weight_vectors[n][0]
                computed[n] = tf.tensordot(w, x, 1)

        return tf.stack([computed[n] for n in output_nodes])

    return network