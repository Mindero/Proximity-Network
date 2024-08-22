#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <istream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <bitset>
#include <array>
#include <queue>
#include <iomanip>
#include <numeric>
#include <string>
#include <fstream>
#include <iterator>
#include <functional>
#include <stack>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;

//mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
mt19937_64 rng(0);

clock_t startTime;
double getCurrentTime() {
    return (double)(clock() - startTime) / CLOCKS_PER_SEC;
}
// 1e18 = 1000000000000000000
// 1e5 = 100000
// 1e9 = 1000000000

const int mod = 998244353;
const int INF = static_cast<int>(2e9) + 5;
const ll INF_64 = static_cast<ll>(1e18) + 5;
const ld eps = 1e-7;
const ld PI = acos(-1.0);

const char* filename_edges = "edges_coauthorship.csv";
const char* filename_nodes = "nodes_coauthorship.csv";
const char* filenameout = "clique_stats.txt";

struct edge {
    int v, u;
};

vector<vector<int>> g;
vector<int> used;
vector<edge> ed;
map<string, int> id;
map<int, string> id_country;
map<int, string> rev_id;
int nodes_size = 34304;
int edges_size = 48889;

int create_id(const string& author) {
    if (id.count(author)) return id[author];
    int sz = id.size();
    id[author] = sz;
    rev_id[sz] = author;
    return sz;
}
void add_edge(int v, int u) {
    g[v].push_back(ed.size());
    g[u].push_back(ed.size());
    ed.push_back({ v, u });
}
vector<string> get_name_coauthorship(string& info) {
    int lf = 0;
    int sz = info.size();
    string name_author, country;
    while (lf < sz && info[lf] != ',') {
        name_author += info[lf];
        ++lf;
    }
    ++lf;
    int rg = sz - 1;
    while (rg >= 0 && info[rg] != ',') --rg;
    --rg;
    while (rg >= 0 && info[rg] != ',') {
        country += info[rg];
        --rg;
    }
    reverse(country.begin(), country.end());
    return { name_author, country };
}

void read_graph() {
    freopen(filename_nodes, "r", stdin);
    string info;
    string header;
    getline(cin, header);
    g.assign(nodes_size, {});
    for (int i = 0; i < nodes_size; ++i) {
        getline(cin, info);
        string name_author, country;
        /*
        int j = 0;
        int sz = info.size();
        while (j < sz && info[j] != '\"') {
            name_author += info[j];
            ++j;
        }
        assert(j != sz);
        // Считываем страну автора
        ++j;
        while (j < sz && info[j] != ',') {
            country += info[j];
            ++j;
        }
        */
        auto result = get_name_coauthorship(info);
        name_author = result[0], country = result[1];
        int id = create_id(name_author);
        id_country[id] = country;
    }
    freopen(filename_edges, "r", stdin);
    getline(cin, header);
    for (int i = 0; i < edges_size; ++i) {
        getline(cin, info);
        // Считываем первого автора
        int sz = info.size();
        string first_author;
        int j = 0;
        while (j < sz && info[j] != ',') {
            first_author += info[j];
            ++j;
        }
        assert(j != sz);
        // Считываем второго автора
        ++j;
        string second_author;
        while (j < sz && info[j] != ',') {
            second_author += info[j];
            ++j;
        }
        assert(j != sz);

        assert(id.count(first_author) && id.count(second_author));
        int first_id = id[first_author];
        int second_id = id[second_author];
        add_edge(first_id, second_id);
    }
    cout << id.size() << ' ' << ed.size() << '\n';
    cout << "Read finished\n";
    assert(nodes_size == id.size() && edges_size == ed.size());
}
// Выделяет компоненты
void bfs_components(int v, int comp) {
    used[v] = comp;
    queue<int> q;
    q.push(v);
    while (!q.empty()) {
        v = q.front();
        q.pop();
        for (int e : g[v]) {
            int u = ed[e].u ^ ed[e].v ^ v;
            if (used[u] == -1) {
                used[u] = comp;
                q.push(u);
            }
        }
    }
}
void take_giant_component() {
    used.assign(nodes_size, -1);
    vector<int> comp_size;
    for (int i = 0; i < nodes_size; ++i) {
        if (used[i] == -1) {
            int sz = comp_size.size();
            bfs_components(i, sz);
            comp_size.push_back(0);
        }
    }
    for (int i = 0; i < nodes_size; ++i) {
        comp_size[used[i]]++;
    }
    int largest_component = max_element(comp_size.begin(), comp_size.end()) - comp_size.begin();
    vector<vector<int>> h(nodes_size);
    for (int i = 0; i < h.size(); ++i) {
        if (used[i] != largest_component) {
            nodes_size--;
            continue;
        }
        for (int e : g[i]) {
            int u = ed[e].u ^ ed[e].v ^ i;
            if (used[u] != largest_component) continue;
            h[i].push_back(e);
        }
    }
    g = h;
    edges_size = 0;
    for (int i = 0; i < h.size(); ++i) {
        edges_size += h[i].size();
    }
    edges_size /= 2;
    cout << "Giant component stats: " << nodes_size << ' ' << edges_size << '\n';
}

// Функция считает максимальный диаметр и средний кратчайший путь
void calc_unweighted() {
    int min_diametr = 0;
    ll sum_distance = 0;
    for (int start = 0; start < g.size(); ++start) {
        if (start % 1000 == 0) {
            cout << start << '\n';
        }
        queue<int> q;
        q.push(start);
        vector<int> d(g.size(), INF);
        vector<bool> inq(g.size(), 0);
        d[start] = 0;
        inq[start] = 1;
        while (!q.empty()) {
            int v = q.front();
            min_diametr = max(min_diametr, d[v]);
            sum_distance += d[v];
            q.pop();
            inq[v] = 0;
            for (int e : g[v]) {
                int u = ed[e].u ^ ed[e].v ^ v;
                if (d[u] > d[v] + 1) {
                    d[u] = d[v] + 1;
                    if (!inq[u])
                    {
                        q.push(u);
                        inq[u] = 1;
                    }
                }
            }
        }
    }
    cout << "Diametr: " << min_diametr << "\n";
    cout << "Aver. short path: " << sum_distance << '\n';
}

void max_clique_graph(int times) {
    vector<int> d(nodes_size, 0);

    for (int i = 0; i < g.size(); ++i) {
        //if (g[i].size() == 0) continue;
        d[i] = nodes_size - int(g[i].size()) - 1;
    }
    // {Count of neighbours from clique ,degree of vertex, node_id}
    set<vector<int>> nodes_init;
    for (int i = 0; i < g.size(); ++i) {
        //if (g[i].size() == 0) continue;
        nodes_init.insert({ 0, -d[i], i });
    }
    uniform_int_distribution<int> uni(0, nodes_size - 1);
    set<pair<int, int>> used_cliques;
    vector<int> deleted_vertex(nodes_size, 0);
    for (int i = 0; i < times; ++i) {
        set<vector<int>> nodes = nodes_init;
        vector<int> max_clique;
        used.assign(nodes_size, 0);
        vector<int> cnt(nodes_size, 0);
        int start_vertex = uni(rng);
        used[start_vertex] = 1;
        max_clique.push_back(start_vertex);
        nodes.erase({ cnt[start_vertex], -d[start_vertex], start_vertex });
        for (int e : g[start_vertex]) {
            int u = ed[e].u ^ ed[e].v ^ start_vertex;
            nodes.erase({ cnt[u], d[u], u });
            d[u]--;
            cnt[u]++;
            if (!used[u] && !deleted_vertex[u]) nodes.insert({ cnt[u], -d[u], u });
        }
        while (!nodes.empty()) {
            int v = nodes.rbegin()->back();
            if (deleted_vertex[v]) break;
            used[v] = 1;
            max_clique.push_back(v);
            set<vector<int>> nodes1;
            for (int e : g[v]) {
                int u = ed[e].u ^ ed[e].v ^ v;
                d[u]--;
                cnt[u]++;
                if (!used[u] && !deleted_vertex[u]) nodes1.insert({ cnt[u], -d[u], u });
            }
            nodes = nodes1;
        }
        int nodes_count = max_clique.size();
        int edges_count = 0;
        set<int> save_nodes(max_clique.begin(), max_clique.end());
        for (int v = 0; v < g.size(); ++v) {
            if (!save_nodes.count(v)) continue;
            for (int e : g[v]) {
                int u = ed[e].u ^ ed[e].v ^ v;
                if (u >= v || !save_nodes.count(u)) continue;
                ++edges_count;
            }
        }
        ld percent = ld(edges_count) / (nodes_count * 1ll * (nodes_count - 1) / 2);
        // ФИЛЬТРАЦИЯ
        if (nodes_count <= 20 || percent - 0.5 < 0 || used_cliques.count({ nodes_count, edges_count })) {
            --i;
            continue;
        }
        //used_cliques.insert({ nodes_count, edges_count });
        string name = ("clique " + to_string(i) + " nodes.txt");
        const char* filenameout_nodes_clique = name.c_str();
        freopen(filenameout_nodes_clique, "w", stdout);
        cout << "id,country\n";
        for (int v : max_clique) {
            deleted_vertex[v] = 1;
            cout << rev_id[v] << "," << id_country[v] << '\n';
        }
        name = ("clique " + to_string(i) + " edges.txt");
        const char* filenameout_edges_clique = name.c_str();
        freopen(filenameout_edges_clique, "w", stdout);
        cout << "source,target\n";
        for (int v = 0; v < g.size(); ++v) {
            if (!save_nodes.count(v)) continue;
            for (int e : g[v]) {
                int u = ed[e].u ^ ed[e].v ^ v;
                if (u >= v || !save_nodes.count(u)) continue;
                cout << rev_id[v] << ',' << rev_id[u] << '\n';
            }
        }
        freopen(filenameout, "a", stdout);
        cout << i << " FILE, SIZE OF CLIQUE = " << nodes_count << "; NUMBER OF EDGES = " << edges_count << "; % of all edges = " << percent << '\n';
    }
}

int32_t main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(NULL);
    cout << setprecision(5) << fixed;
    freopen(filenameout, "w", stdout);
    read_graph();
    //take_giant_component();
    max_clique_graph(5);
    //calc_unweighted();
}