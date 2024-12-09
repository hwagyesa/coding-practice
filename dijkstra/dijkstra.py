#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
import heapq
from math import isclose
from pathlib import Path
from random import randint
from typing import Dict, List, Literal, Optional, Tuple, cast


def shortest_path_dijkstra(
    graph: Dict[int, Dict[int, float]], start: int, end: int
) -> Tuple[float, List[int] | None]:
    # Start at the start node.
    parents = {}
    costs = {}
    cost_heap: List[tuple[float, int]] = []
    nodes = graph.keys()
    inf = float("inf")

    # initialize
    for node in nodes:
        parents[node] = None
        costs[node] = inf
    # populate start info
    heapq.heappush(cost_heap, (0.0, start))
    # main loop
    while cost_heap:
        cost, node = heapq.heappop(cost_heap)
        if cost <= costs[node]:  # prevents revisiting/re-processing
            # Process this node's neighbors
            costs[node] = cost
            for neighbor, weight in graph[node].items():
                if costs[node] + weight < costs[neighbor]:
                    # New short path; update
                    costs[neighbor] = costs[node] + weight
                    heapq.heappush(cost_heap, (costs[neighbor], neighbor))
                    parents[neighbor] = node
    # reconstruct the path
    dist = costs[end]
    if costs[end] == inf:
        return inf, None

    path = []
    node = end
    while node != start:
        path.append([parents[node], node])
        node = parents[node]
    path.reverse()

    assert path[0][0] == start
    return dist, path


def test_toy_graph():
    node_list = [0, 1, 2, 3, 4, 5]
    edge_list = [[1, 2, 3], [4, 5], [1], [2, 4, 5], [5], []]
    graph = {}
    for node, edges in zip(node_list, edge_list):
        graph[node] = {}
        for edge in edges:
            graph[node][edge] = edge
    dist, path = shortest_path_dijkstra(graph, 0, 5)
    assert isclose(dist, 6), f"Expected 6, got {dist}"
    assert path == [[0, 1], [1, 5]], f"Expected [[0, 1], [1, 5]], got {path}"
