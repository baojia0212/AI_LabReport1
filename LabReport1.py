import time
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

GRAPH = {'A':['D','B'], 'B':['C','E', 'G'], 'D':['C'], 'C':['A'], 'E':['H'], 'H':['G','F'], 'G':['F'], 'F':[]}


def normalize(graph):
    result = {str(k).strip().upper(): sorted(set(str(v).strip().upper() for v in vs)) for k, vs in graph.items()}
    for vs in list(result.values()):
        for v in vs:
            result.setdefault(v, [])
    return dict(sorted(result.items()))


def search(graph, start, goal=None, method='BFS'):
    graph = normalize(graph)
    start = start.upper().strip()
    goal = goal.upper().strip() if goal else None
    if start not in graph or (goal and goal not in graph):
        raise ValueError('Start and goal must exist in the graph.')
    if method not in ('BFS', 'DFS'):
        raise ValueError('Method must be BFS or DFS.')

    frontier = deque([start]) if method == 'BFS' else [start]
    discovered = {start}
    parent = {start: None}
    depth = {start: 0}
    order, trace = [], []

    while frontier:
        node = frontier.popleft() if method == 'BFS' else frontier.pop()
        order.append(node)

        if node != goal:
            neighbors = graph[node] if method == 'BFS' else list(reversed(graph[node]))
            for neighbor in neighbors:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    frontier.append(neighbor)

        trace.append({
            'step': len(order),
            'current': node,
            'visited': order.copy(),
            'frontier': list(frontier),
            'depth': depth[node],
            'parent': parent[node],
        })

        if node == goal:
            break

    path = []
    if goal is not None and goal in order:
        cursor = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()

    return {
        'method': method,
        'order': order,
        'path': path,
        'found': bool(path) if goal else None,
        'trace': trace,
        'depths': depth,
        'parents': parent,
        'explored': len(order),
        'max_frontier': max([1] + [len(t['frontier']) for t in trace]),
    }


def edges_from_text(text):
    """One directed edge per line: A B or A,B; isolated nodes: A."""
    graph = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip().replace(',', ' ')
        if not line:
            continue
        tokens = line.upper().split()
        if len(tokens) not in (1, 2):
            raise ValueError(f'Invalid line: {line}')
        graph.setdefault(tokens[0], [])
        if len(tokens) == 2:
            graph[tokens[0]].append(tokens[1])
            graph.setdefault(tokens[1], [])
    if not graph:
        raise ValueError('Graph cannot be empty.')
    if len(graph) > 100:
        raise ValueError('Maximum 100 nodes.')
    return normalize(graph)


DEFAULT_EDGES = '\n'.join(
    f'{u} {v}' if vs else u
    for u, ns in GRAPH.items()
    for vs in (ns or [None])
    for v in ([vs] if vs else [None])
).replace(' None', '')


st.set_page_config(page_title='BSD3513 | BFS & DFS Lab#1', page_icon='', layout='wide')
st.title('BFS & DFS | Interactive Uninformed Search Algorithms')
st.caption('BSD3513 Artificial Intelligence Lab Report 1 · Huang Bao Jia · Breadth First Search  & Depth First Search · Alphabetical tie-breaking · UMPSA')

with st.expander('Learning guide', expanded=False):
    st.markdown('**BFS** explores level by level using a FIFO queue; **DFS** explores deeply using a LIFO stack. For adjacency-list implementations both require **O(V + E)** time. BFS finds minimum-hop paths in unweighted graphs; DFS does not guarantee this.')

with st.sidebar:
    st.header('Graph editor')
    edge_text = st.text_area('Directed edges (one per line: A B)', DEFAULT_EDGES, height=220)
    st.caption('A line containing only a node represents an isolated node. Edges are directed.')

try:
    graph = edges_from_text(edge_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

nodes = list(graph)
a, b, c = st.columns(3)
start = a.selectbox('Starting node', nodes)
goal = b.selectbox('Goal node (optional)', ['No goal'] + nodes)
method = c.selectbox('Algorithm', ['Both', 'BFS', 'DFS'])
goal = None if goal == 'No goal' else goal

chosen_methods = ['BFS', 'DFS'] if method == 'Both' else [method]
results = {m: search(graph, start, goal, m) for m in chosen_methods}


def draw_graph(result, step):
    g = nx.DiGraph()
    for node, neighbors in graph.items():
        g.add_node(node)
        for neighbor in neighbors:
            g.add_edge(node, neighbor)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = nx.spring_layout(g, seed=12)
    seen = result['trace'][step - 1]['visited']
    colors = ['#10b981' if n == result['trace'][step - 1]['current'] else '#93c5fd' if n in seen else '#e5e7eb' for n in g.nodes]
    nx.draw_networkx(g, pos, ax=ax, with_labels=True, node_color=colors, node_size=950, font_weight='bold', arrows=True, arrowsize=19, edge_color='#64748b')
    ax.axis('off')
    return fig

for m, r in results.items():
    st.subheader(f'{m} search')
    st.write('**Expansion order:** ' + ' → '.join(r['order']))
    if goal:
        st.write('**Goal path:** ' + (' → '.join(r['path']) if r['found'] else 'Not reachable'))
    x, y, z = st.columns(3)
    x.metric('Expanded nodes', r['explored'])
    y.metric('Maximum recorded frontier', r['max_frontier'])
    z.metric('Goal path edges', len(r['path']) - 1 if r['found'] else '—')
    
    # Controls for manual step inspection and Auto-run
    ctrl_1, ctrl_2, ctrl_3 = st.columns([3, 1.2, 0.8])
    with ctrl_1:
        if len(r['trace']) > 1:
            step = st.slider(f'{m}: inspect expansion step', 1, len(r['trace']), len(r['trace']), key=f'slider_{m}')
        else:
            step = 1
    with ctrl_2:
        speed = st.slider('Auto-run speed (s)', 0.2, 2.0, 0.8, 0.1, key=f'speed_{m}')
    with ctrl_3:
        st.write('')  # vertical alignment spacing
        play = st.button('Auto-run', key=f'play_{m}', type='primary')

    stage = st.empty()

    def render_step(s):
        with stage.container():
            left, right = st.columns([1.2, 1])
            with left:
                fig = draw_graph(r, s)
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.write('**Step trace**')
                st.dataframe(pd.DataFrame(r['trace']), hide_index=True, use_container_width=True)
                st.info('Current frontier: ' + str(r['trace'][s - 1]['frontier']))

    if play:
        for s in range(1, len(r['trace']) + 1):
            render_step(s)
            time.sleep(speed)
    else:
        render_step(step)

st.divider()

# Automatic level & summary 
st.subheader('Levels and Summary')
full_results = {m: search(graph, start, None, m) for m in chosen_methods}

cols = st.columns(len(chosen_methods))
for col, m in zip(cols, chosen_methods):
    r_full = full_results[m]
    lv = {}
    for n in r_full['order']:
        d = r_full['depths'][n]
        lv.setdefault(d, []).append(n)
    with col:
        st.markdown(f"**{m}: nodes per level** (full order: {' → '.join(r_full['order'])})")
        st.dataframe(
            pd.DataFrame([{'Level': str(d), 'Nodes': ', '.join(sorted(v))} for d, v in sorted(lv.items())]),
            hide_index=True,
            use_container_width=True
        )

summary_df = pd.DataFrame([{
    'Algorithm': m,
    'Data structure': 'Queue (FIFO)' if m == 'BFS' else 'Stack (LIFO)',
    'Visit order (full run)': ' → '.join(full_results[m]['order']),
    'Max level': max(full_results[m]['depths'].values()) if full_results[m]['depths'] else 0,
    'Max frontier': results[m]['max_frontier'],
    'Path to goal': (' → '.join(results[m]['path']) if results[m]['found'] else 'Not reachable') if goal else 'No goal',
    'Path edges': str(len(results[m]['path']) - 1) if (goal and results[m]['found']) else '—'
} for m in chosen_methods])

st.markdown('**Algorithm Summary**')
st.dataframe(summary_df, hide_index=True, use_container_width=True)