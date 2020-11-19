// Copyright 2020 Marcel Wagenländer

#include <string>

void alzheimer_chunked(std::string dataset, int chunk_size);


int main() {
    alzheimer_chunked("flickr", 1 << 14);
}
