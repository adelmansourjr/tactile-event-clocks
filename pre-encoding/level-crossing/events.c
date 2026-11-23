#include <stddef.h>

size_t send_delta(const double *values, size_t rows, size_t columns,
                  double step, double *times, int *labels, size_t capacity) {
    double latest[256];
    if (columns > 256 || rows < 2 || step <= 0.0) return capacity + 1;
    for (size_t j = 0; j < columns; ++j) latest[j] = values[j];
    size_t n = 0;
    for (size_t i = 1; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            const double right = values[i * columns + j];
            const double left = values[(i - 1) * columns + j];
            const double delta = right - left;
            if (delta == 0.0) continue;
            double next = latest[j];
            const int sign = delta > 0.0 ? 1 : -1;
            while (sign * (right - next) >= step) {
                next += sign * step;
                if (n >= capacity) return capacity + 1;
                times[n] = (double)(i - 1) + (next - left) / delta;
                labels[n] = (int)j + (sign > 0 ? (int)columns : 0);
                ++n;
            }
            latest[j] = next;
        }
    }
    return n;
}
