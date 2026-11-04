#include "MatrixOps.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace MatrixOps {

Matrix transpose(const Matrix& A) {
    if (A.empty()) return {};
    const std::size_t m = A.size();
    const std::size_t n = A[0].size();
    Matrix T(n, Vector(m, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty())
        throw std::invalid_argument("multiply: matriz vacia");
    const std::size_t m = A.size();
    const std::size_t n = A[0].size();
    const std::size_t p = B[0].size();
    if (B.size() != n)
        throw std::invalid_argument("multiply: dimensiones incompatibles");

    Matrix C(m, Vector(p, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const double aik = A[i][k];
            for (std::size_t j = 0; j < p; ++j)
                C[i][j] += aik * B[k][j];
        }
    return C;
}

Vector multiply(const Matrix& A, const Vector& x) {
    if (A.empty()) return {};
    if (A[0].size() != x.size())
        throw std::invalid_argument("multiply(A,x): dimensiones incompatibles");
    Vector y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("dot: tamanos distintos");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

Vector solve(Matrix A, Vector b) {
    const std::size_t n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n)
        throw std::invalid_argument("solve: el sistema no es cuadrado");

    for (std::size_t col = 0; col < n; ++col) {
        // Pivoteo parcial: se busca el mayor valor absoluto de la columna.
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col])) pivot = r;

        if (std::fabs(A[pivot][col]) < 1e-12)
            throw std::runtime_error("solve: matriz singular o mal condicionada");

        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);

        // Normalizacion de la fila pivote.
        const double p = A[col][col];
        for (std::size_t j = col; j < n; ++j) A[col][j] /= p;
        b[col] /= p;

        // Eliminacion del resto de filas.
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = A[r][col];
            if (factor == 0.0) continue;
            for (std::size_t j = col; j < n; ++j) A[r][j] -= factor * A[col][j];
            b[r] -= factor * b[col];
        }
    }
    return b;
}

Matrix inverse(Matrix A) {
    const std::size_t n = A.size();
    if (n == 0 || A[0].size() != n)
        throw std::invalid_argument("inverse: la matriz no es cuadrada");

    Matrix I(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) I[i][i] = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col])) pivot = r;
        if (std::fabs(A[pivot][col]) < 1e-12)
            throw std::runtime_error("inverse: matriz singular");

        std::swap(A[col], A[pivot]);
        std::swap(I[col], I[pivot]);

        const double p = A[col][col];
        for (std::size_t j = 0; j < n; ++j) { A[col][j] /= p; I[col][j] /= p; }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            if (f == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                A[r][j] -= f * A[col][j];
                I[r][j] -= f * I[col][j];
            }
        }
    }
    return I;
}

Vector column(const Matrix& A, std::size_t j) {
    Vector c;
    c.reserve(A.size());
    for (const auto& row : A) c.push_back(row.at(j));
    return c;
}

void print(const Matrix& A, const std::string& label, std::size_t maxRows) {
    std::cout << label << " (" << A.size() << " x "
              << (A.empty() ? 0 : A[0].size()) << ")\n";
    const std::size_t lim = std::min(maxRows, A.size());
    for (std::size_t i = 0; i < lim; ++i) {
        std::cout << "  [ ";
        for (double v : A[i]) std::cout << std::fixed << std::setprecision(4) << v << " ";
        std::cout << "]\n";
    }
    if (lim < A.size()) std::cout << "  ... (" << A.size() - lim << " filas mas)\n";
}

void print(const Vector& v, const std::string& label, std::size_t maxItems) {
    std::cout << label << " (n=" << v.size() << "): [ ";
    const std::size_t lim = std::min(maxItems, v.size());
    for (std::size_t i = 0; i < lim; ++i)
        std::cout << std::fixed << std::setprecision(4) << v[i] << " ";
    if (lim < v.size()) std::cout << "...";
    std::cout << "]\n";
}

double mean(const Vector& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double variance(const Vector& v) {
    if (v.empty()) return 0.0;
    const double mu = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - mu) * (x - mu);
    return s / static_cast<double>(v.size());
}

double stddev(const Vector& v) { return std::sqrt(variance(v)); }

double minimum(const Vector& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x < m) m = x;
    return m;
}

double maximum(const Vector& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x > m) m = x;
    return m;
}

} // namespace MatrixOps
