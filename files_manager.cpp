#include "main_header.h"

int box_file_writer (const vector<box>& boxes, const string& path) {
    try {
        //cout << "Writing  array contents to file..." << endl;
        //open file for writing
        ofstream fw(path, std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //store array contents to text file
            for (const auto box: boxes) {
                fw << "ID: " << box.ID << "; [" << box.p0x << ", " <<  box.p0y << ", " << box.width << ", " <<  box.height << "]\n";
            }
            fw.close();
        }
        else cout << "Problem with opening file";
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    //cout << "Done!" << endl;
    return 0;
}

int box_file_reader (vector<box>& boxes, const string& path) {
    try {
        //cout << "Reading array contents from file..." << endl;
        string line;
        //open file for reading
        ifstream fw(path, ifstream::in);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            while (getline(fw, line)) {
                string str;
                vector<string> vec;
                stringstream ss(line);
                // Use while loop to check the getline() function condition.
                while (getline(ss, str, ' ')) {
                    vec.push_back(str);
                }
                boxes.emplace_back(
                        stoi(vec[1].substr(0, vec[1].size()-1)),
                        stoi(vec[2].substr(1, vec[2].size()-2)),
                        stoi(vec[3].substr(0, vec[3].size()-1)),
                        stoi(vec[4].substr(0, vec[4].size()-1)),
                        stoi(vec[5].substr(0, vec[5].size()-1)),
                        -1,
                        Mat::zeros(stoi(vec[5].substr(0, vec[5].size()-1)),
                                   stoi(vec[4].substr(0, vec[4].size()-1)),
                                   CV_8UC3)
                        );
            }
            fw.close();
        }
        else cout << "Problem with opening file";
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    //cout << "Done!" << endl;
    return 0;
}
