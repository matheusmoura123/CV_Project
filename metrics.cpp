#include "main_header.h"

double boxes_IoU (const box& box1, const box& box2) {
    vector<int> c{0, 0, 0};
    for (int y = min(box1.p0y, box2.p0y); y <= max(box1.p0y+box1.height, box2.p0y+box2.height); ++y) {
        for (int x = min(box1.p0x, box2.p0x); x <= max(box1.p0x+box1.width, box2.p0x+box2.width); ++x) {
            if ((y >= box1.p0y & y <= box1.p0y+box1.height) & (x >= box1.p0x & x <= box1.p0x+box1.width)) {
                if ((y >= box2.p0y & y <= box2.p0y+box2.height) & (x >= box2.p0x & x <= box2.p0x+box2.width)) {
                    c[0]++;
                } else c[1]++;
            } else if ((y >= box2.p0y & y <= box2.p0y+box2.height) & (x >= box2.p0x & x <= box2.p0x+box2.width)) {
                c[2]++;
            }
        }
    }
    double IoU = c[0]/(c[0]+c[1]+c[2]);
    cout << IoU << endl;
    return IoU;
}