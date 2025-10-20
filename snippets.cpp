#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
#define int long long
typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> pbds;
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());
#define fr(i, a, b) for (int i = (a); i < (int)(b); ++i)
#define frr(i, a, b) for (int i = (a); i > (int)(b); --i)
#define ina(x, n)               \
    for (int i = 0; i < n; ++i) \
        cin >> x[i];
#define in(n)    \
    long long n; \
    cin >> n;
#define double long double
#define pb push_back
#define vi vector<int>
#define mii map<int, int>
#define vvi vector<vector<int>>
#define vpi vector<pair<int, int>>
#define mdi map<double, int>
#define mci map<char, int>
#define mide map<int, deque<int>>
#define pi pair<int, int>
#define si set<int>
#define oyes cout << "YES" << endl;
#define ono cout << "NO" << endl;
#define oyess cout << "Yes" << endl;
#define onoo cout << "No" << endl;
#define ve1 cout << "-1" << endl;
#define ff first
#define ss second
#define all(x) (x).begin(), (x).end()
#define rev(x) reverse((x).begin(), (x).end())
using int64 = long long;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

void solve()
{
    in(n);
}

signed main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    in(t);
    while (t--)
        solve();
    return 0;
}

int modexp(int base, int exponent)
{
    int result = 1;
    base = base % mod;
    while (exponent > 0)
    {
        if (exponent % 2 == 1)
        {
            result = (result * base) % mod;
        }
        exponent = exponent >> 1;
        base = (base * base) % mod;
    }
    return result;
}

int pow1(int a, int b)
{
    int res = 1;
    while (b > 0)
    {
        if (b & 1)
        {
            res = res * a;
        }
        a = a * a;
        b >>= 1;
    }
    return res;
}

string dectobin(int n)
{
    if (n == 0)
        return "0";
    string binary;
    while (n > 0)
    {
        binary.push_back((n % 2) + '0');
        n /= 2;
    }
    reverse(binary.begin(), binary.end());
    return binary;
}

int bintodec(const string &binary)
{
    int decimal = 0, base = 1;
    for (int i = binary.length() - 1; i >= 0; i--)
    {
        decimal += (binary[i] - '0') * base;
        base *= 2;
    }
    return decimal;
}

vector<int> sieve(int n)
{
    vector<bool> isPrime(n + 1, true);
    vector<int> primes;
    isPrime[0] = isPrime[1] = false;
    for (int p = 2; p * p <= n; p++)
    {
        if (isPrime[p])
        {
            for (int multiple = p * p; multiple <= n; multiple += p)
            {
                isPrime[multiple] = false;
            }
        }
    }
    for (int i = 2; i <= n; i++)
    {
        if (isPrime[i])
        {
            primes.push_back(i);
        }
    }
    return primes;
}

int modinv(int a, int m)
{
    int res = 1, b = m - 2;
    while (b)
    {
        if (b & 1)
            res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

const int N = 2e5 + 5, mod = 1e9 + 7;
int64_t fact[N];
int64_t pw(int64_t a, int64_t b)
{
    int64_t r = 1;
    while (b > 0)
    {
        if (b & 1)
            r = (r * a) % mod;
        b /= 2;
        a = (a * a) % mod;
    }
    return r;
}
int64_t C(int64_t n, int64_t k)
{
    // fact[0] = 1;
    // for (int64_t i = 1; i < N; ++i)
    //     fact[i] = (fact[i - 1] * i) % mod;
    // Add above part before taking input t
    if (n < k)
        return 0LL;
    return (fact[n] * pw((fact[n - k] * fact[k]) % mod, mod - 2)) % mod;
}

int logi(int x, int b)
{
    int ans = 0;
    int y = b;
    while (y <= x)
    {
        if (y >= 2e18 / b)
            y = 2e18;
        else
            y *= b;

        ans++;
    }

    return ans;
}

int mex(vi &nums)
{
    unordered_set<int> numSet(nums.begin(), nums.end());
    int mex = 0;
    while (numSet.count(mex))
    {
        ++mex;
    }
    return mex;
}

const int N = 2e6 + 5;
vector<int> par(N);
vector<int> sizes(N);

void make(int n)
{
    for (int i = 1; i <= n; i++)
    {
        par[i] = i;
        sizes[i] = 1;
    }
}

int find(int v)
{
    if (par[v] == v)
        return v;
    return par[v] = find(par[v]);
}

void Union(int a, int b)
{
    a = find(a);
    b = find(b);
    if (a != b)
    {
        if (sizes[a] < sizes[b])
            swap(a, b);
        par[b] = a;
        sizes[a] += sizes[b];
    }
}

void bfs(int start, vvi &graph, vi &lvl)
{
    int n = graph.size();
    vi vis(n + 1);
    queue<int> q;
    q.push(start);
    vis[start] = true;
    lvl[start] = 1;
    while (!q.empty())
    {
        int node = q.front();
        q.pop();
        for (int neighbor : graph[node])
        {
            if (!vis[neighbor])
            {
                vis[neighbor] = true;
                lvl[neighbor] = lvl[node] + 1;
                q.push(neighbor);
            }
        }
    }
}

int nchoosek(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    k = min(k, n - k);
    if (k == 0)
        return 1;
    int res = 1;
    for (int i = 1; i <= k; i++)
    {
        int num = n - k + i;
        int den = i;
        int g = __gcd(num, den);
        num /= g;
        den /= g;
        g = __gcd(res, den);
        res /= g;
        den /= g;
        int tmp = (int)res * num;
        res = (int)(tmp / den);
    }
    return res;
}

void ncn(int n)
{
    vvi C(n + 1, vi(n + 1));
    fr(i, 0, n + 1)
    {
        fr(j, 0, i + 1)
        {
            if (j == 0 || j == i)
            {
                C[i][j] = 1;
            }
            else
            {
                C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
            }
        }
    }
}

pair<int, int> dfs(int node, int parent, vvi &tree) // diameter of tree
{
    pair<int, int> farthest = {0, node};
    for (int neighbor : tree[node])
    {
        if (neighbor == parent)
            continue;
        pair<int, int> res = dfs(neighbor, node, tree);
        res.first++;
        if (res.first > farthest.first)
            farthest = res;
    }
    return farthest;
}

int dia(int n, vvi &tree)
{
    pair<int, int> firstDfs = dfs(0, -1, tree);
    pair<int, int> secondDfs = dfs(firstDfs.second, -1, tree);
    return secondDfs.first;
}

bool dfs(int node, int par, vvi &gr, vi &col)
{
    for (int child : gr[node])
    {
        if (child != par)
        {
            if (col[child] == -1)
            {
                col[child] = !col[node];
                if (!dfs(child, node, gr, col))
                    return false;
            }
            else if (col[child] == col[node])
            {
                return false;
            }
        }
    }
    return true;
}

map<int, int> primefac(int n)
{
    map<int, int> fac;
    while (n % 2 == 0)
    {
        fac[2]++;
        n /= 2;
    }
    for (int i = 3; i * i <= n; i += 2)
    {
        while (n % i == 0)
        {
            fac[i]++;
            n /= i;
        }
    }
    if (n > 1)
    {
        fac[n]++;
    }
    return fac;
}
struct func
{
    // using int64 = long long;
    // mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    int64 mulmod(int64 a, int64 b, int64 mod)
    {
        __int128 res = (__int128)a * b % mod;
        return (int64)res;
    }

    int64 power(int64 a, int64 b, int64 mod)
    {
        int64 res = 1;
        while (b)
        {
            if (b & 1)
                res = mulmod(res, a, mod);
            a = mulmod(a, a, mod);
            b >>= 1;
        }
        return res;
    }

    bool isPrime(int64 n)
    {
        if (n < 2)
            return false;
        for (int64 p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
        {
            if (n % p == 0)
                return n == p;
        }
        int64 d = n - 1, s = 0;
        while ((d & 1) == 0)
            d >>= 1, s++;
        for (int64 a : {2, 325, 9375, 28178, 450775, 9780504, 1795265022})
        {
            if (a % n == 0)
                continue;
            int64 x = power(a, d, n);
            if (x == 1 || x == n - 1)
                continue;
            bool comp = true;
            for (int i = 1; i < s; i++)
            {
                x = mulmod(x, x, n);
                if (x == n - 1)
                {
                    comp = false;
                    break;
                }
            }
            if (comp)
                return false;
        }
        return true;
    }

    int64 f(int64 x, int64 c, int64 mod)
    {
        return (mulmod(x, x, mod) + c) % mod;
    }

    int64 pollard(int64 n)
    {
        if (n % 2 == 0)
            return 2;
        int64 x = uniform_int_distribution<int64>(2, n - 2)(rng);
        int64 y = x;
        int64 c = uniform_int_distribution<int64>(1, n - 1)(rng);
        int64 d = 1;
        while (d == 1)
        {
            x = f(x, c, n);
            y = f(f(y, c, n), c, n);
            d = __gcd(abs(x - y), n);
            if (d == n)
                return pollard(n);
        }
        return d;
    }

    vector<int64> factorize(int64 n)
    {
        if (n == 1)
            return {};
        if (isPrime(n))
            return {n};
        int64 d = pollard(n);
        auto left = factorize(d);
        auto right = factorize(n / d);
        left.insert(left.end(), right.begin(), right.end());
        return left;
    }
};

vector<int> dijkstra(vector<vector<pair<int, int>>> &adj, int n, int src)
{
    vector<int> dist(n);
    int ans = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    for (int i = 0; i < n; i++)
        dist[i] = LLONG_MAX;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty())
    {
        int dis = pq.top().first;
        int node = pq.top().second;
        pq.pop();
        if (dis != dist[node])
        {
            continue;
        }
        for (auto it : adj[node])
        {
            int edgeweight = it.second;
            int adjnode = it.first;
            if (dis + edgeweight < dist[adjnode])
            {
                dist[adjnode] = dis + edgeweight;
                pq.push({dist[adjnode], adjnode});
            }
        }
    }
    return dist;
}

void dfs(int node, int par, vvi &tr, vi &ss, vi &par2) // dfs for subtree size
{
    if (node == 1)
        par2[node] = 0;
    else
        par2[node] = par;
    ss[node] = 1;
    for (int i : tr[node])
    {
        if (i != par)
        {
            dfs(i, node, tr, ss, par2);
            ss[node] += ss[i];
        }
    }
}

int longest_common_subsequence(const string &x, const string &y)
{
    int n = x.size(), m = y.size();
    int dp[n + 1][m + 1] = {};
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (x[i - 1] == y[j - 1])
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
            else
                dp[i][j] = max({dp[i][j], dp[i - 1][j], dp[i][j - 1]});
        }
    }
    return dp[n][m];
}

pair<int, vi> lis(const string &s) // longest increasing subsequence
{
    if (s.empty())
        return {0, {}};
    vi lis;
    vi ind;
    vi prev(s.size(), -1);
    for (int i = 0; i < s.size(); ++i)
    {
        auto it = lower_bound(all(lis), s[i]);
        int pos = it - lis.begin();
        if (it == lis.end())
        {
            lis.push_back(s[i]);
            ind.push_back(i);
        }
        else
        {
            *it = s[i];
            ind[pos] = i;
        }
        if (pos > 0)
        {
            prev[i] = ind[pos - 1];
        }
    }
    vi lisind;
    for (int i = ind.back(); i != -1; i = prev[i])
    {
        lisind.pb(i);
    }
    reverse(lisind.begin(), lisind.end());
    return {lis.size(), lisind};
}

pair<int, vi> lnds(const string &s) // longest non decressing subsequence
{
    if (s.empty())
        return {0, {}};
    vi lnds;
    vi ind;
    vi prev(s.size(), -1);
    fr(i, 0, s.size())
    {
        auto it = upper_bound(lnds.begin(), lnds.end(), s[i]);
        int pos = it - lnds.begin();
        if (pos < lnds.size())
        {
            lnds[pos] = s[i];
            ind[pos] = i;
        }
        else
        {
            lnds.push_back(s[i]);
            ind.push_back(i);
        }
        if (pos > 0)
        {
            prev[i] = ind[pos - 1];
        }
    }
    vi lndsInd;
    for (int i = ind.back(); i != -1; i = prev[i])
    {
        lndsInd.push_back(i);
    }
    reverse(lndsInd.begin(), lndsInd.end());
    return {lnds.size(), lndsInd};
}

signed main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    return 0;
}

// {
//     const int MAXN = 100005;
//     const int LOG = 17; // log2(MAXN) ~ 17 for N up to 100000

//     struct Edge
//     {
//         int to, weight;
//     };

//     vector<Edge> tree[MAXN];
//     int up[MAXN][LOG];      // Binary Lifting table for ancestors
//     int maxEdge[MAXN][LOG]; // Max edge weight in the path to the ancestor
//     int depth[MAXN];        // Depth of each node

//     void dfs(int node, int parent, int w)
//     {
//         up[node][0] = parent;
//         maxEdge[node][0] = w;
//         depth[node] = depth[parent] + 1;

//         for (int i = 1; i < LOG; i++)
//         {
//             up[node][i] = up[up[node][i - 1]][i - 1];
//             maxEdge[node][i] = max(maxEdge[node][i - 1], maxEdge[up[node][i - 1]][i - 1]);
//         }

// for (auto &edge : tree[node])
// {
//     int next = edge.to, weight = edge.weight;
//     if (next != parent)
//     {
//         dfs(next, node, weight);
//     }
// }
//     }

//     // Finds LCA and returns max edge weight on the path from u to v
//     int getMaxEdge(int u, int v)
//     {
//         if (depth[u] < depth[v])
//             swap(u, v);
//         int maxW = 0;

//         // Bring u and v to the same depth
//         for (int i = LOG - 1; i >= 0; i--)
//         {
//             if (depth[u] - (1 << i) >= depth[v])
//             {
//                 maxW = max(maxW, maxEdge[u][i]);
//                 u = up[u][i];
//             }
//         }

//         if (u == v)
//             return maxW;
//         for (int i = LOG - 1; i >= 0; i--)
//         {
//             if (up[u][i] != up[v][i])
//             {
//                 maxW = max({maxW, maxEdge[u][i], maxEdge[v][i]});
//                 u = up[u][i];
//                 v = up[v][i];
//             }
//         }

//         return max({maxW, maxEdge[u][0], maxEdge[v][0]});
//     }
// dfs(1, 0, 0); -> preprocessing in solve function

// }