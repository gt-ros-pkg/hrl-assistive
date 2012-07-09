#ifndef HSL_RGB_CONVERSIONS_H
#define HSL_RGB_CONVERSIONS_H
#include<cmath>
using namespace std;

void extractRGB(float rgb, uint8_t& r, uint8_t& g, uint8_t& b);
void extractHSL(float rgb, double& h, double& s, double& l);
void writeRGB(uint8_t r, uint8_t g, uint8_t b, float& rgb);
void writeHSL(double h, double s, double l, float& rgb);
void HSLToRGB(double h, double s, double l, uint8_t& r, uint8_t& g, uint8_t& b);
void RGBToHSL(uint8_t r, uint8_t g, uint8_t b, double& h, double& s, double& l);

// H in [0, 360], S in [0, 100], L in [0, 100]
// R,G,B in [0, 255] 

void extractRGB(float rgb, uint8_t& r, uint8_t& g, uint8_t& b) {
    r = ((uint8_t*) &rgb)[2];
    g = ((uint8_t*) &rgb)[1];
    b = ((uint8_t*) &rgb)[0];
}

void extractHSL(float rgb, double& h, double& s, double& l) { 
    uint8_t r, g, b;
    extractRGB(rgb, r, g, b);
    RGBToHSL(r, g, b, h, s, l);
}

void writeRGB(uint8_t r, uint8_t g, uint8_t b, float& rgb) {
    ((uint8_t*) &rgb)[3] = 0xff;
    ((uint8_t*) &rgb)[2] = r;
    ((uint8_t*) &rgb)[1] = g;
    ((uint8_t*) &rgb)[0] = b;
}

void writeHSL(double h, double s, double l, float& rgb) {
    uint8_t r, g, b;
    HSLToRGB(h, s, l, r, g, b);
    writeRGB(r, g, b, rgb);
}

void RGBToHSL(uint8_t r, uint8_t g, uint8_t b, double& h, double& s, double& l) {
    double rd = r / 255.0, gd = g / 255.0, bd = b / 255.0;
    double min_color = min(rd, min(gd, bd));
    double max_color = max(rd, max(gd, bd));
    l = (min_color + max_color) / 2.0;
    if(min_color == max_color) {
        s = 0.0; h = 0.0;
        l *= 100.0;
        return;
    }
    if(l < 0.5) 
        s = (max_color - min_color) / (max_color + min_color);
    else
        s = (max_color - min_color) / (2.0 - max_color - min_color);
    if(rd == max_color)
        h = (gd - bd) / (max_color - min_color);
    else if(gd == max_color)
        h = 2.0 + (bd - rd) / (max_color - min_color);
    else 
        h = 4.0 + (rd - bd) / (max_color - min_color);
    h *= 60.0;
    if(h < 0)
        h += 360.0;
    s *= 100.0;
    l *= 100.0;
}

void HSLToRGB(double h, double s, double l, uint8_t& r, uint8_t& g, uint8_t& b) {
    h /= 360.0;
    s /= 100.0;
    l /= 100.0;
    double rd, gd, bd;
    if(s == 0) {
        rd = l; gd = l; bd = l;
    } else {
        double temp2;
        if(l < 0.5)
            temp2 = l * (1.0 + s);
        else
            temp2 = l + s - l*s;
        double temp1 = 2.0 * l - temp2;
        double rtemp3 = h + 1.0 / 3.0;
        if(rtemp3 < 0) rtemp3 += 1.0;
        if(rtemp3 > 1) rtemp3 -= 1.0;
        double gtemp3 = h;
        if(gtemp3 < 0) gtemp3 += 1.0;
        if(gtemp3 > 1) gtemp3 -= 1.0;
        double btemp3 = h - 1.0 / 3.0;
        if(btemp3 < 0) btemp3 += 1.0;
        if(btemp3 > 1) btemp3 -= 1.0;
        if(6.0 * rtemp3 < 1.0) rd = temp1 + (temp2 - temp1) * 6.0 * rtemp3;
        else if(2.0 * rtemp3 < 1.0) rd = temp2;
        else if(3.0 * rtemp3 < 2.0) rd = temp1 + (temp2 - temp1) * (2.0/3.0 - rtemp3) * 6.0;
        else rd = temp1;
        if(6.0 * gtemp3 < 1.0) gd = temp1 + (temp2 - temp1) * 6.0 * gtemp3;
        else if(2.0 * gtemp3 < 1.0) gd = temp2;
        else if(3.0 * gtemp3 < 2.0) gd = temp1 + (temp2 - temp1) * (2.0/3.0 - gtemp3) * 6.0;
        else gd = temp1;
        if(6.0 * btemp3 < 1.0) bd = temp1 + (temp2 - temp1) * 6.0 * btemp3;
        else if(2.0 * btemp3 < 1.0) bd = temp2;
        else if(3.0 * btemp3 < 2.0) bd = temp1 + (temp2 - temp1) * (2.0/3.0 - btemp3) * 6.0;
        else bd = temp1;
    }
    r = rd * 255.0;
    g = gd * 255.0;
    b = bd * 255.0;
}

#endif // HSL_RGB_CONVERSIONS_H
