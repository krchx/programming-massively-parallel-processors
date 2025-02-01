#ifndef PNGHAND_HPP
#define PNGHAND_HPP

#include <string>

class PngHand
{
public:
    PngHand(const std::string &filename);
    int Width() const;
    int Height() const;
    unsigned char *Data() const;
    void Save(const std::string &filename, unsigned char *data) const;
    void SaveGrey(const std::string &filename, unsigned char *grey) const;

private:
    int width;
    int height;
    unsigned char *data;
    void ReadFile(const std::string &filename);
};

#endif // PNGHAND_HPP