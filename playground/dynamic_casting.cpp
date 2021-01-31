// Copyright 2021 Marcel Wagenländer

#include <vector>
#include <iostream>

class Base {
protected:
    long value;

public:
    Base();
    virtual void foo();
};

class Derivative : public Base {
protected:
    long another_value;

public:
    Derivative();
    void foo() override;
};

Base::Base() {
    value = 0;
}

void Base::foo() {
}

Derivative::Derivative() {
    value = 1;
    another_value = 2;
}

void Derivative::foo() {
}

int main() {
    std::vector<Base> vec;
    vec.push_back(Base());
    vec.push_back(Derivative());

    for (Base &cla : vec) {
    Derivative *der_ptr = dynamic_cast<Derivative *>(&cla);
    std::cout << "Pointer: " << der_ptr << std::endl;
    }
}

