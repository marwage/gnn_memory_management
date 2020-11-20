// Copyright 2020 Marcel Wagenländer

#include <string>

void alzheimer(std::string dataset, int chunk_size);


int main() {
    alzheimer("flickr", 1 << 14);
}
