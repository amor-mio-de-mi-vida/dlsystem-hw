#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // 注意数组越界

    int iterations = (m + batch - 1) / batch;
    float* x = (float*)malloc(sizeof(float) * batch * n);
    char* yy = (char*)malloc(sizeof(float) * batch);

    float* Z = (float*)malloc(sizeof(float) * batch * k);

    float* Z_ = (float*)malloc(sizeof(float) * batch);

    float* grad = (float*)malloc(sizeof(float) * k * n);

    for (int i = 0; i < iterations; i++) {

        // copy X (batch * input_dimension) (batch * n)
        memcpy(x, X + i * batch, sizeof(float) * batch * n);
        // copy y (batch)
        memcpy(yy, y + i * batch, sizeof(char) * batch);

        memset(Z, 0, sizeof(float) * batch * k);
        memset(Z_, 0, sizeof(float) * batch);
        memset(grad, 0, sizeof(float) * k * n);

        // matmul x @ theta
        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < k; j++) {
                for (size_t l = 0; l < n; k++) {
                    Z[i * batch + j] += x[i * batch + l] * theta[l * n + j];
                }
            }
        }

        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < k; j++) {
                Z[i * batch + j] = exp(Z[i * batch + j]);
                Z_[i] += Z[i * batch + j];
            }
        }

        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < k; j++) {
                Z[i * batch + j] /= Z_[i];
            }
        }

        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < k; j++) {
                if (j == int(yy[i] + '0')) {
                    Z[i * batch + j] -= 1;
                }
            }
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                for (size_t l = 0; l < batch; l++) {
                    grad[i * n + j] += x[l * batch + i] * Z[l * batch + j];
                }
            }
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                theta[i * n + j] -= lr * grad[i * n + j] / batch;
            }
        }
    }

    free(x);
    free(yy);
    free(Z);
    free(Z_);
    free(grad);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
