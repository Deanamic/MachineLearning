#include <bits/stdc++.h>
using namespace std;

struct cosa {
  int a, b;
  string s;
};

int main() {
  freopen("submission-knn.csv", "r", stdin);
  freopen("submission-knn-send.csv", "w", stdout);
  string rip;
  cin >> rip;
  cout << rip << endl;
  vector<cosa> v;
  while(cin >> rip) {
    int x = 0;
    while(rip[x] != ',') ++x;
    pair<int,int> p = {rip[0] - '0', stoi(rip.substr(1,x-1))};
    v.push_back({p.first, p.second, rip.substr(x, rip.size()-x)});
  }
  sort(v.begin(), v.end(), [](auto a, auto b) {
                             return (a.a != b.a ? a.a < b.a : a.b < b.b);
                           });
  for(auto i : v) cout << i.a << '_' << i.b << i.s << endl;
}
