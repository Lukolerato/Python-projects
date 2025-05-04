# autor: Dominika Szaradowska
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import heapq # kopiec
matplotlib.use('Qt5Agg')
plt.ion()


points = []
G = None

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        ax.plot(x, y, 'bo')
        fig.canvas.draw()


def on_key(event):
    if event.key == 'enter':
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)
        build_graph()


def build_graph():
    global G
    G = nx.Graph()
    n = len(points)

    for i, (x, y) in enumerate(points):
        G.add_node(i, pos=(x, y))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            G.add_edge(i, j, weight=dist)

    draw_graph()
    plt.title("Klikaj na krawędzie które chcesz USUNĄĆ, ENTER kończy wybieranie krawędzi")

    fig.canvas.mpl_connect('button_press_event', on_edge_click)
    fig.canvas.mpl_connect('key_press_event', on_start_selection)


def draw_graph(highlight_edges=None):
    plt.clf()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}

    if highlight_edges:
        edge_colors = ['red' if e in highlight_edges or (e[1], e[0]) in highlight_edges else 'gray' for e in G.edges()]
    else:
        edge_colors = 'gray'

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    fig.canvas.draw()


def on_edge_click(event):
    if event.xdata is None or event.ydata is None:
        return
    click_point = np.array([event.xdata, event.ydata])
    pos = nx.get_node_attributes(G, 'pos')

    min_dist = float('inf')
    closest_edge = None

    for u, v in G.edges():
        point_u = np.array(pos[u])
        point_v = np.array(pos[v])
        midpoint = (point_u + point_v) / 2
        dist = np.linalg.norm(click_point - midpoint)
        if dist < min_dist:
            min_dist = dist
            closest_edge = (u, v)

    if closest_edge and min_dist < 0.3:  # próg bliskości kliknięcia
        print(f"Usuwam krawędź: {closest_edge}")
        G.remove_edge(*closest_edge)
        draw_graph()


def on_start_selection(event):
    if event.key == 'enter':
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        plt.title("Kliknij w wierzchołek startowy")
        fig.canvas.mpl_connect('button_press_event', lambda e: select_start_node(e, G))


def select_start_node(event, G):
    if event.xdata is None or event.ydata is None:
        return
    click_point = np.array([event.xdata, event.ydata])
    pos = nx.get_node_attributes(G, 'pos')

    min_dist = float('inf')
    selected_node = None

    for i, (x, y) in pos.items():
        dist = np.linalg.norm(click_point - np.array([x, y]))
        if dist < min_dist:
            min_dist = dist
            selected_node = i

    if selected_node is not None:
        run_dijkstra(G, selected_node)


def dijkstra(G, start_node):
    # każdemu wierzchołkowi przypisujemy nieskończoność (poza startowym)
    dist = {node: float('inf') for node in G.nodes()}
    dist[start_node] = 0  # odległość startowego wierzchołka do samego siebie = 0

    # zapisujemy, z którego wierzchołka przyszliśmy do danego (do odtworzenia ścieżki)
    previous = {node: None for node in G.nodes()}

    # tworzymy kolejkę priorytetową (kopiec), zaczynamy od wierzchołka startowego
    heap = [(0, start_node)]  # elementy to (odległość, wierzchołek)

    # dopóki kolejka nie jest pusta – przetwarzamy kolejne wierzchołki
    while heap:
        current_dist, u = heapq.heappop(heap)  # wyciągamy wierzchołek o najmniejszej odległości

        # jeśli aktualna odległość jest większa niż już zapisana, pomijamy (znaleźliśmy lepszą wcześniej)
        if current_dist > dist[u]:
            continue

        # dla każdego sąsiada u:
        for v in G.neighbors(u):
            weight = G[u][v]['weight']  # pobieramy wagę krawędzi u-v
            alt = dist[u] + weight      # obliczamy potencjalną nową odległość (przez u)

            # jeśli ta ścieżka jest krótsza niż znana dotąd – aktualizujemy
            if alt < dist[v]:
                dist[v] = alt           # zapisujemy nową krótszą odległość
                previous[v] = u         # zapamiętujemy, że do v doszliśmy z u
                heapq.heappush(heap, (alt, v))  # wrzucamy sąsiada do kolejki z nową odległością

    # po zakończeniu budujemy ścieżki do wszystkich wierzchołków
    paths = {}
    for node in G.nodes():
        if dist[node] < float('inf'):  # jeśli węzeł jest osiągalny
            path = []
            current = node
            # cofamy się od końcowego wierzchołka do startowego
            while current is not None:
                path.insert(0, current)  # dodajemy węzeł na początek listy ścieżki
                current = previous[current]  # idziemy do poprzednika
            paths[node] = path  # zapisujemy pełną ścieżkę
        else:
            paths[node] = None  # jeśli węzeł jest nieosiągalny → brak ścieżki

    # zwracamy słownik odległości i ścieżek
    return dist, paths


def run_dijkstra(G, start_node):
    lengths, paths = dijkstra(G, start_node)

    print(f"\nNajkrótsze odległości od wierzchołka {start_node}:")
    for target in sorted(G.nodes()):
        if target in lengths:
            print(f"Do {target}: odległość = {lengths[target]:.2f}, ścieżka = {paths[target]}")
        else:
            print(f"Do {target}: brak ścieżki")

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = []

    for u, v in G.edges():
        edge_in_path = any((u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]) for path in paths.values())
        edge_colors.append('red' if edge_in_path else 'gray')

    plt.clf()
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, edge_color=edge_colors, width=2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Najkrótsze ścieżki od {start_node}")
    plt.show()


# główne okno
fig, ax = plt.subplots()
plt.title("Klikaj punkty, ENTER kończy wybieranie punktów")
cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show(block=True)
