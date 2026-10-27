// Necessary composition + carbon-carbon connectivity screen. Hash collisions
// only cause extra exact InChI computations; hashes NEVER decide exclusions.
// Standard InChI H/charge/tautomer normalization and salt/metal disconnection
// preserve the induced carbon-carbon adjacency graph (bond order is ignored).
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

static uint64_t mix(uint64_t h,uint64_t x) {
    h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
    return h;
}
extern "C" uint64_t screening_hash(const char* input) {
    try {
        std::string s(input);
        std::map<std::string,int> composition;
        std::vector<std::vector<int>> graph;
        std::vector<int> atoms,branches;
        std::map<int,int> rings;
        int current=-1;
        auto edge=[&](int a,int b) {
            if(a<0||b<0) return;
            int ca=atoms[a],cb=atoms[b];
            if(ca>=0&&cb>=0){graph[ca].push_back(cb);graph[cb].push_back(ca);}
        };
        for(size_t i=0;i<s.size();) {
            char ch=s[i];
            if(ch=='(') {if(current<0)return 0;branches.push_back(current);++i;continue;}
            if(ch==')') {if(branches.empty())return 0;current=branches.back();branches.pop_back();++i;continue;}
            if(ch=='.') {current=-1;++i;continue;}
            if(ch=='-'||ch=='='||ch=='#'||ch==':'||ch=='/'||ch=='\\') {++i;continue;}
            if(std::isdigit(ch)||ch=='%') {
                if(current<0)return 0;
                int label;
                if(ch=='%') {
                    if(i+2>=s.size()||!std::isdigit(s[i+1])||!std::isdigit(s[i+2]))return 0;
                    label=(s[i+1]-'0')*10+s[i+2]-'0';i+=3;
                }else{label=ch-'0';++i;}
                auto it=rings.find(label);
                if(it==rings.end())rings[label]=current;
                else {edge(current,it->second);rings.erase(it);}
                continue;
            }
            std::string element;
            if(ch=='[') {
                size_t end=s.find(']',i+1);
                if(end==std::string::npos)return 0;
                ++i;while(i<end&&std::isdigit(s[i]))++i;
                if(i>=end||s[i]=='*')return 0;
                char first=s[i++];
                if(!std::isalpha(first))return 0;
                element+=std::toupper(first);
                if(std::isupper(first)&&i<end&&std::islower(s[i]))element+=s[i++];
                else if((first=='s'&&i<end&&s[i]=='e')||(first=='a'&&i<end&&s[i]=='s'))element+=s[i++];
                i=end+1;
            }else {
                std::string legal="BCNOPSFIbcnops";
                if(legal.find(ch)==std::string::npos)return 0;
                element+=std::toupper(ch);++i;
                if((ch=='C'&&i<s.size()&&s[i]=='l')||(ch=='B'&&i<s.size()&&s[i]=='r'))element+=s[i++];
            }
            if(element!="H")composition[element]++;
            int carbon=-1;
            if(element=="C"){carbon=graph.size();graph.emplace_back();}
            atoms.push_back(carbon);
            int next=atoms.size()-1;
            edge(current,next);current=next;
        }
        if(!branches.empty()||!rings.empty()||atoms.empty())return 0;
        std::vector<uint64_t> labels;
        for(auto&neighbors:graph)labels.push_back(mix(0x84222325cbf29ce4ULL,neighbors.size()));
        for(int round=0;round<3;++round){
            std::vector<uint64_t> next(labels.size());
            for(size_t a=0;a<labels.size();++a){
                std::vector<uint64_t> ns;for(int b:graph[a])ns.push_back(labels[b]);
                std::sort(ns.begin(),ns.end());
                uint64_t h=mix(0x3141592653589793ULL,labels[a]);
                for(auto value:ns)h=mix(h,value);
                next[a]=h;
            }
            labels.swap(next);
        }
        std::sort(labels.begin(),labels.end());
        uint64_t h=0xcbf29ce484222325ULL;
        for(auto& entry:composition){for(char c:entry.first)h=mix(h,c);h=mix(h,entry.second);h=mix(h,0xff);}
        h=mix(h,graph.size());for(auto value:labels)h=mix(h,value);
        return h?h:1;
    }catch(...){return 0;}
}
