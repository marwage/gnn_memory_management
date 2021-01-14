#include <iostream>


int main() {
    std::string mode = "mean";
    std::string input = "mean";

    if (mode.compare(input) == 0) {
        std::cout << "equal" << std::endl;
    }
}
