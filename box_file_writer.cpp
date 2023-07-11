#include "main_header.h"

int box_file_writer (const vector<box>& boxes, const string& path) {
    try {
        cout << "\nWriting  array contents to file...";
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
    cout << "\nDone!" << endl;
    return 0;
}
