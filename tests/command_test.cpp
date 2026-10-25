#include "tinynpu/command.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    const float a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const float b[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    float c[4] = {};

    tinynpu::Event event;
    tinynpu::CommandQueue queue;

    tinynpu::GemmCommand command;
    command.a = a;
    command.b = b;
    command.c = c;
    command.m = 2;
    command.n = 2;
    command.k = 3;
    command.completion_event = &event;

    assert(queue.submit(command));
    assert(queue.status() == tinynpu::CommandStatus::Submitted);
    assert(!event.is_complete());

    assert(queue.wait());
    assert(queue.status() == tinynpu::CommandStatus::Completed);
    assert(event.is_complete());

    const float expected[] = {
        58.0f,  64.0f,
        139.0f, 154.0f
    };

    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(c[i] - expected[i]) < 1e-5f);
    }

    std::cout << "command_test passed\n";
    return 0;
}