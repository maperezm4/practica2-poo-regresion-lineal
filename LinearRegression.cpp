#include "LinearRegression.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
std::string Metrics::toString() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << "MSE = "  << mse
       << " | RMSE = " << rmse
       << " | MAE = "  << mae
       << " | R2 = "   << r2;
    return os.str();
}

// ---------------------------------------------------------------------------
// Constructores / destructor
// ---------------------------------------------------------------------------
LinearRegression::LinearRegression(double learningRate,
                                   std::size_t epochs,
                                   Solver solver,
                                   double tolerance)
    : weights_(),
      bias_(0.0),
      learningRate_(learningRate),
      epochs_(epochs),
      tolerance_(tolerance),
      solver_(solver),
      scalingType_(ScalingType::NONE),
      scaleCenter_(),
      scaleSpread_(),
      trained_(false),
      iterationsRun_(0),
      finalCost_(0.0) {}

LinearRegression::LinearRegression(const LinearRegression& other)
    : weights_(other.weights_),
      bias_(other.bias_),
      learningRate_(other.learningRate_),
      epochs_(other.epochs_),
      tolerance_(other.tolerance_),
      solver_(other.solver_),
      scalingType_(other.scalingType_),
      scaleCenter_(other.scaleCenter_),
      scaleSpread_(other.scaleSpread_),
      trained_(other.trained_),
      iterationsRun_(other.iterationsRun_),
      finalCost_(other.finalCost_) {}

LinearRegression& LinearRegression::operator=(const LinearRegression& other) {
    if (this != &other) {
        weights_       = other.weights_;
        bias_          = other.bias_;
        learningRate_  = other.learningRate_;
        epochs_        = other.epochs_;
        tolerance_     = other.tolerance_;
        solver_        = other.solver_;
        scalingType_   = other.scalingType_;
        scaleCenter_   = other.scaleCenter_;
        scaleSpread_   = other.scaleSpread_;
        trained_       = other.trained_;
        iterationsRun_ = other.iterationsRun_;
        finalCost_     = other.finalCost_;
    }
    return *this;
}

LinearRegression::~LinearRegression() = default;

// ---------------------------------------------------------------------------
// Validacion
// ---------------------------------------------------------------------------
void LinearRegression::validate(const Matrix& X, const Vector& y) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("fit/score: la matriz X esta vacia");
    if (X.size() != y.size())
        throw std::invalid_argument("fit/score: X y y tienen distinto numero de muestras");
    const std::size_t n = X[0].size();
    for (const auto& row : X)
        if (row.size() != n)
            throw std::invalid_argument("fit/score: filas de X con distinto numero de columnas");
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------
void LinearRegression::fit(const Matrix& X, const Vector& y) {
    validate(X, y);
    weights_.assign(X[0].size(), 0.0);
    bias_          = 0.0;
    iterationsRun_ = 0;

    if (solver_ == Solver::NORMAL_EQUATION)
        fitNormalEquation(X, y);
    else
        fitGradientDescent(X, y);

    trained_   = true;
    finalCost_ = computeCost(X, y);
}

void LinearRegression::fitGradientDescent(const Matrix& X, const Vector& y) {
    const std::size_t m = X.size();
    const std::size_t n = X[0].size();
    double previousCost = std::numeric_limits<double>::max();

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        Vector gradW(n, 0.0);
        double gradB = 0.0;

        // Gradiente del MSE: dJ/dw_j = (2/m) * sum((y_hat - y) * x_j)
        for (std::size_t i = 0; i < m; ++i) {
            double yHat = bias_;
            for (std::size_t j = 0; j < n; ++j) yHat += weights_[j] * X[i][j];
            const double error = yHat - y[i];
            for (std::size_t j = 0; j < n; ++j) gradW[j] += error * X[i][j];
            gradB += error;
        }

        const double factor = 2.0 / static_cast<double>(m);
        for (std::size_t j = 0; j < n; ++j)
            weights_[j] -= learningRate_ * factor * gradW[j];
        bias_ -= learningRate_ * factor * gradB;

        ++iterationsRun_;

        const double cost = computeCost(X, y);
        if (!std::isfinite(cost))
            throw std::runtime_error(
                "fit: el costo divergio. Reduzca learningRate o escale los datos.");
        if (std::fabs(previousCost - cost) < tolerance_) break;
        previousCost = cost;
    }
}

void LinearRegression::fitNormalEquation(const Matrix& X, const Vector& y) {
    const std::size_t m = X.size();
    const std::size_t n = X[0].size();

    // Se agrega una columna de unos para absorber el bias dentro de los pesos.
    Matrix Xb(m, Vector(n + 1, 1.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) Xb[i][j + 1] = X[i][j];

    const Matrix Xt  = MatrixOps::transpose(Xb);
    const Matrix XtX = MatrixOps::multiply(Xt, Xb);
    const Vector Xty = MatrixOps::multiply(Xt, y);

    const Vector theta = MatrixOps::solve(XtX, Xty);

    bias_ = theta[0];
    for (std::size_t j = 0; j < n; ++j) weights_[j] = theta[j + 1];
    iterationsRun_ = 1;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------
double LinearRegression::predict(const Vector& x) const {
    if (!trained_)
        throw std::runtime_error("predict: el modelo no ha sido entrenado");
    if (x.size() != weights_.size())
        throw std::invalid_argument("predict: numero de caracteristicas incorrecto");

    double yHat = bias_;
    for (std::size_t j = 0; j < x.size(); ++j) yHat += weights_[j] * x[j];
    return yHat;
}

Vector LinearRegression::predict(const Matrix& X) const {
    if (!trained_)
        throw std::runtime_error("predict: el modelo no ha sido entrenado");
    Vector yHat;
    yHat.reserve(X.size());
    for (const auto& row : X) yHat.push_back(predict(row));
    return yHat;
}

// ---------------------------------------------------------------------------
// score
// ---------------------------------------------------------------------------
Metrics LinearRegression::score(const Matrix& X, const Vector& y) const {
    validate(X, y);
    const Vector yHat = predict(X);
    const std::size_t m = y.size();

    double sse = 0.0, sae = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double e = yHat[i] - y[i];
        sse += e * e;
        sae += std::fabs(e);
    }

    const double yMean = MatrixOps::mean(y);
    double sst = 0.0;
    for (double yi : y) sst += (yi - yMean) * (yi - yMean);

    Metrics mt;
    mt.mse  = sse / static_cast<double>(m);
    mt.rmse = std::sqrt(mt.mse);
    mt.mae  = sae / static_cast<double>(m);
    mt.r2   = (sst > 0.0) ? 1.0 - sse / sst : 0.0;
    return mt;
}

double LinearRegression::computeCost(const Matrix& X, const Vector& y) const {
    const std::size_t m = X.size();
    double sse = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        double yHat = bias_;
        for (std::size_t j = 0; j < X[i].size(); ++j) yHat += weights_[j] * X[i][j];
        const double e = yHat - y[i];
        sse += e * e;
    }
    return sse / static_cast<double>(m);
}

// ---------------------------------------------------------------------------
// data_scaling
// ---------------------------------------------------------------------------
Matrix LinearRegression::data_scaling(const Matrix& X, ScalingType type, bool fitParameters) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("data_scaling: matriz vacia");

    const std::size_t m = X.size();
    const std::size_t n = X[0].size();

    if (type == ScalingType::NONE) {
        scalingType_ = type;
        return X;
    }

    if (fitParameters) {
        scalingType_ = type;
        scaleCenter_.assign(n, 0.0);
        scaleSpread_.assign(n, 1.0);

        for (std::size_t j = 0; j < n; ++j) {
            const Vector col = MatrixOps::column(X, j);
            if (type == ScalingType::STANDARDIZATION) {
                scaleCenter_[j] = MatrixOps::mean(col);
                const double sd = MatrixOps::stddev(col);
                scaleSpread_[j] = (sd < 1e-12) ? 1.0 : sd;  // evita division por cero
            } else { // MIN_MAX
                const double mn = MatrixOps::minimum(col);
                const double mx = MatrixOps::maximum(col);
                scaleCenter_[j] = mn;
                scaleSpread_[j] = (mx - mn < 1e-12) ? 1.0 : (mx - mn);
            }
        }
    } else {
        if (scaleCenter_.size() != n)
            throw std::runtime_error(
                "data_scaling: no hay parametros de escala previos para este dataset");
    }

    Matrix Z(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Z[i][j] = (X[i][j] - scaleCenter_[j]) / scaleSpread_[j];

    return Z;
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------
const Vector& LinearRegression::getWeights() const { return weights_; }
double LinearRegression::getBias() const { return bias_; }
bool LinearRegression::isTrained() const { return trained_; }
std::size_t LinearRegression::getIterationsRun() const { return iterationsRun_; }
double LinearRegression::getFinalCost() const { return finalCost_; }

std::string LinearRegression::toString() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << "y_hat = ";
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        if (j > 0) os << " + ";
        os << weights_[j] << "*x" << (j + 1);
    }
    os << (bias_ >= 0 ? " + " : " - ") << std::fabs(bias_);
    return os.str();
}

void LinearRegression::summary() const {
    std::cout << "--- LinearRegression ---\n"
              << "  entrenado      : " << (trained_ ? "si" : "no") << "\n"
              << "  n_caracteristicas: " << weights_.size() << "\n"
              << "  solver         : "
              << (solver_ == Solver::NORMAL_EQUATION ? "ecuaciones normales"
                                                     : "descenso de gradiente")
              << "\n"
              << "  iteraciones    : " << iterationsRun_ << "\n"
              << "  costo final    : " << std::fixed << std::setprecision(8)
              << finalCost_ << "\n"
              << "  modelo         : " << toString() << "\n";
}
