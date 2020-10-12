// Copyright 2020 Marcel Wagenländer

void check_divmv();
void check_log_softmax_forward();
void check_loss();

int main() {
    check_divmv();
    check_log_softmax_forward();
    check_loss();
}
