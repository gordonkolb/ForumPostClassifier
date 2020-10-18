#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cstring>
#include "csvstream.h"
#include <map>
#include <set>

using namespace std;

set<string> unique_words(const string &str) {
    // Fancy modern C++ and STL way to do it
    istringstream source{str};
    return {istream_iterator<string>{source},
        istream_iterator<string>{}};
}

class Classifier{
public:
    
    Classifier() : posts(0), label(), lwpair(), words() {}
    
    void read_trainer(string file_in, bool debug){
        
        map<string, string> row;
        csvstream csvin(file_in);
        if(debug){
            cout << "training data:" << endl;
        }
        while (csvin >> row){
            posts++;
            if (debug){
                cout << "  label = " << row["tag"] << ", content = " <<
                row["content"] << endl;
            }
            ++label[row["tag"]];
            set<string> copy_words = unique_words(row["content"]);
            for(const auto &milah:copy_words){
                ++words[milah];
                ++lwpair[pair<string, string>(row["tag"], milah)];
            }
        }
        int x = posts;
        cout << "trained on " << x << " examples"  << endl;
        
        if(debug){
            cout << "vocabulary size = " << words.size() << endl;
        }
        cout << endl;
    }
    
    double log_prior(string l){
        return log(label[l] / posts);
    }
    
    double log_likelihood(string l, string w){
        if (lwpair[pair<string, string>(l, w)] != 0){
            return log(lwpair[pair<string, string>(l,w)] / label[l]);
        }
        else if(words[w] != 0){
            return log(words[w] / posts);
        }
        else{
            return log(1 / posts);
        }
    }
    
    void read_test(string file_name){
        map<string, string> row;
        map<string, double> test_words;
        set<string> copy_words;
        string correct_label;
        int correct_count = 0;
        int total_count = 0;
        csvstream csvin(file_name);
        cout << "test data:" << endl;
        while(csvin >> row){
            ++total_count;
            copy_words = unique_words(row["content"]);
            correct_label = row["tag"];
            pair<string, double> prediction =
                max_label_score(labels_and_scores(copy_words));
            cout << "  correct = " << correct_label << ", predicted = "
            << prediction.first << ", log-probability score = "
            << prediction.second << endl;
            cout << "  content = " << row["content"] << endl;
            cout << endl;
            
            if (prediction.first == correct_label){
                correct_count++;
            }
        }
        cout << "performance: " << correct_count << " / " << total_count
        << " posts predicted correctly" << endl;
        
    }
    
    pair<string, double> max_label_score(map<string, double> lbscore){
        pair<string, double> prediction;
        prediction.first = lbscore.begin()->first;
        prediction.second = lbscore.begin()->second;
        
        for (const auto & lbls : lbscore){
            if (lbls.second > prediction.second){
                prediction.first = lbls.first;
                prediction.second = lbls.second;
            }
        }
        return prediction;
    }
    
    map<string, double> labels_and_scores(set<string> test_words){
        map<string, double> lbscore;
        
        for(const auto &lbls : label){
            lbscore[lbls.first] = label_probability(lbls.first, test_words);
        }
        return lbscore;
    }
    
    double label_probability(string l, set<string> test_words){
        double total = log_prior(l);
        for(const auto &wrd: test_words){
            total += log_likelihood(l, wrd);
        }
        return total;
    }
    
    
    void print_helper(){
        cout << "classifier parameters:" << endl;
        for(auto const &p : lwpair){
            cout << "  " << p.first.first << ":" << p.first.second << ", "
            << "count = " << p.second
            << ", log-likelihood = "
            << log_likelihood(p.first.first, p.first.second) << endl;
        }
    }
    
    void output_data(){
        cout << "classes:" << endl;
        for(auto const &lbls : label){
            string keeper = lbls.first;
            cout << "  " << keeper << ", " << label[keeper] << " examples, " <<
            "log-prior = " << log_prior(keeper) << endl;
        }
        
    }
    
    
private:
    double posts;
    map<string, double> label;
    map<pair<string, string>, double> lwpair;
    map<string, double> words;
};


int main(int argc, char * argv[]){
    if(argc != 4 && argc != 3){
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 1;
    }
    else if(argc == 4 && strcmp(argv[3], "--debug")){
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 2;
    }
    
    cout.precision(3);
    bool debug;
    if(argc == 4){
        debug = true;
    }
    else{
        debug = false;
    }
    
    ifstream fin;
    fin.open(argv[1]);
    if(!fin.is_open()){
        std::cout << "Error opening " << argv[1] << std::endl;
        return 3;
    }
    fin.open(argv[2]);
    if(!fin.is_open()){
        std::cout << "Error opening " << argv[2] << std::endl;
        return 3;
    }
    
    Classifier c;
    c.read_trainer(argv[1], debug);
    if(debug){
        c.output_data();
        c.print_helper();
        cout << endl;
    }
    
    c.read_test(argv[2]);
    
    
    return 0;
}
