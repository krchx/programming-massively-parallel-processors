#include <libpng16/png.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include "PNGHand.hpp"

PngHand::PngHand(const std::string &filename)
{
    ReadFile(filename);
}

int PngHand::Width() const
{
    return width;
}

int PngHand::Height() const
{
    return height;
}

unsigned char *PngHand::Data() const
{
    return data;
}

void PngHand::SaveGrey(const std::string &filename, unsigned char *grey) const
{
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        throw std::runtime_error("Could not create PNG write structure");
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        throw std::runtime_error("Could not create PNG info structure");
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_bytep row_pointers[height];
    for (int i = 0; i < height; i++)
    {
        row_pointers[i] = (png_bytep)(grey + i * width);
    }
    png_set_rows(png, info, row_pointers);
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void PngHand::Save(const std::string &filename, unsigned char *data) const
{
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        throw std::runtime_error("Could not create PNG write structure");
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        throw std::runtime_error("Could not create PNG info structure");
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_bytep row_pointers[height];
    for (int i = 0; i < height; i++)
    {
        row_pointers[i] = (png_bytep)(data + i * width * 3);
    }
    png_set_rows(png, info, row_pointers);
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void PngHand::ReadFile(const std::string &filename)
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        fclose(fp);
        throw std::runtime_error("File is not a PNG image: " + filename);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        throw std::runtime_error("Could not create PNG read structure");
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        throw std::runtime_error("Could not create PNG info structure");
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);
    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (width == 0 || height == 0)
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw std::runtime_error("Invalid image dimensions");
    }

    if (bit_depth != 8)
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw std::runtime_error("Image must be of 8-bit depth");
    }
    if (color_type != PNG_COLOR_TYPE_RGB)
    {
        switch (color_type)
        {
        case PNG_COLOR_TYPE_GRAY:
            png_set_gray_to_rgb(png);
            break;

        case PNG_COLOR_TYPE_PALETTE:
            png_set_palette_to_rgb(png);
            break;

        case PNG_COLOR_TYPE_GRAY_ALPHA:
            png_set_gray_to_rgb(png);
            png_set_strip_alpha(png);
            break;

        case PNG_COLOR_TYPE_RGB_ALPHA:
            png_set_strip_alpha(png);
            break;

        default:
            png_destroy_read_struct(&png, &info, NULL);
            fclose(fp);
            throw std::runtime_error("Unsupported color type");
        }
    }

    png_bytep *row_pointers = new png_bytep[height];
    data = new unsigned char[width * height * 3];
    for (int i = 0; i < height; i++)
    {
        row_pointers[i] = (png_bytep)(data + i * width * 3);
    }
    png_read_image(png, row_pointers);
    delete[] row_pointers;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
}