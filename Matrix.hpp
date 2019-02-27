#ifndef CPP_EX3_MATRIX_HPP
#define CPP_EX3_MATRIX_HPP

#define MATRIX_SIZES_NOT_MATCH  "matrices size not match"
#define NOT_SQR_MTRX_MSG "cant transpose non square matrix"

#include <iostream>
#include <algorithm>
#include <utility>

#include "Complex.h"
#include <vector>
#include <stdexcept>
#include <eigen3/Eigen/Dense>
#include <stack>
#include <chrono>
#include <thread>
#include <future>
#include <iterator>

using namespace std;

/** generic matrix class */
template<class T>
class Matrix
{

private:
    static bool parallel;
    unsigned int m{0};
    unsigned int n{0};
    vector<T> _cells;

public:

    using iterator = typename vector<T>::iterator;
    using const_iterator = typename vector<T>::const_iterator;
    typedef Matrix<T> self_type;

    /** non-const begin iterator */
    iterator begin()
    {
        return _cells.begin();
    }

    /** non-const end iterator */
    iterator end()
    {
        return _cells.end();
    }

/** const begin iterator */
    const_iterator cbegin()
    {
        return _cells.cbegin();
    }

/** const end iterator */
    const_iterator cend()
    {
        return _cells.cend();
    }

    /** default ctor */
    Matrix<T>() : m(1), n(1), _cells(1, 0)
    {};

    /** row col ctor */
    Matrix<T>(unsigned int
              rows, unsigned int
              cols) : m(rows),
                      n(cols),
                      _cells(rows * cols, 0)
    {}

    /** copy ctor */
    Matrix<T>(
            const self_type &a) : m(a.m), n(a.n), _cells(a._cells.begin(), a._cells.end())
    {}

    /** row col vector ctor */
    Matrix<T>(unsigned int
              rows, unsigned int
              cols,
              const vector<T> &cells) : Matrix<T>(rows, cols)
    {
        if (cells.size() != rows * cols)
        {
            throw invalid_argument("given vector size is not match");
        }
        unsigned int k, i, j;
        for (k = 0, i = 0; (i < rows); i++)
        {
            for (j = 0; (j < cols); k++, j++)
            {
                _cells[i * n + j] = cells[k];
            }
        }
    }

    /** move ctor */
    Matrix(self_type && m) noexcept : m(0), n(0), _cells(vector<T>())
    {
        *this = move(m);
    }

    /** destructor */
    ~Matrix()
    {
        _cells.clear();
    };

    /** transpose */
    self_type trans()
    {
        if (!isSquareMatrix())
        {
            throw invalid_argument(NOT_SQR_MTRX_MSG);
        }
        self_type newMat(n, m);
        unsigned int i, j;
        for (i = 0; (i < m); i++)
        {
            for (j = 0; (j < n); j++)
            {
                newMat._cells[j * n + i] = _cells[i * n + j];
            }
        }
        return newMat;
    };

    /** checks if this is square matrix */
    bool isSquareMatrix() const
    { return m == n; };

/** rows getter */
    unsigned int rows() const
    { return m; };

    /** cols getter */
    unsigned int cols() const
    { return n; };

    /** check sizes */
    static void checkSizes(unsigned int m1, unsigned int m2, unsigned int n1, unsigned int n2)
    {
        checkSizes(m1, m2);
        checkSizes(n1, n2);
    }

    /** check sizes */
    static void checkSizes(unsigned int m1, unsigned int m2)
    {
        if (m1 != m2)
        {
            throw range_error(MATRIX_SIZES_NOT_MATCH);
        }
    }

    /** perform a given linear transformation */
    static void linearTransformation(const self_type &a, const self_type &b, self_type &result, T(op)(T, T))
    {
        checkSizes(a.m, b.m, a.n, b.n);
        for (unsigned int i = 0; i < a.m; i++)
        {
            for (unsigned int j = 0; j < a.n; j++)
            {
                result._cells[i * a.n + j] = op(a._cells[i * a.n + j], b._cells[i * a.n + j]);
            }
        }
    }

    /** addition in multithreading */
    static void addP(const self_type &a, const self_type &b, self_type &result)
    {
        auto computeRow = [&result, &a, &b](int row)
        {
            for (unsigned int col = 0; col < a.cols(); col++)
            {
                result._cells[row * a.cols() + col] =
                        a._cells[row * a.cols() + col] + b._cells[row * a.cols() + col];
            }
        };

        vector<future<void>> tasks;
        tasks.reserve(result.rows() * result.cols());

        for (unsigned int row = 0; row < result.rows(); row++)
        {
            auto task = async(launch::async, computeRow, row);
            tasks.push_back(move(task));
        }

        for (auto &task: tasks)
        {
            task.get();
        }
    }

    /** addition operator overloading */
    friend self_type operator+(const self_type &a, const self_type &b)
    {
        self_type result(a.m, a.n);
        if (Matrix<T>::parallel)
        {
            addP(a, b, result);
        }
        else
        {
            auto add = [](T b, T c)
            {
                return b + c;
            };
            linearTransformation(a, b, result, add);
        }
        return result;
    };


    /** substruction operator overloading */
    friend self_type operator-(const self_type &a, const self_type &b)
    {
        checkSizes(a.m, b.m, a.n, b.n);
        self_type result(a.m, a.n);

        auto sub = [](T b, T c)
        {
            return b - c;
        };

        linearTransformation(a, b, result, sub);
        return result;
    };

    /** multiplication operator overloading */
    friend self_type operator*(const self_type &a, const self_type &b)
    {
        checkSizes(a.n, b.m);
        Matrix<T> result(a.rows(), b.cols());

        if (Matrix<T>::parallel)
        {
            multiP(a, b, result);
        }
        else
        {
            multi(a, b, result);
        }
        return result;
    };

    /** multiplication in multithreading*/
    static void multiP(const self_type &a, const self_type &b, self_type &result)
    {
        auto computeRow = [&result, &a, &b](unsigned int row)
        {
            T value;
            for (unsigned int col = 0; col < b.n; col++)
            {
                value = T();
                for (unsigned int k = 0; k < a.n; k++)
                {
                    value += a._cells[row * a.cols() + k] * b._cells[k * b.cols() + col];
                }
                result._cells[row * result.cols() + col] = value;
            }
        };

        vector<future<void>> tasks;
        tasks.reserve(result.rows());

        for (unsigned int row = 0; row < result.rows(); row++)
        {
            auto task = async(launch::async, computeRow, row);
            tasks.push_back(move(task));
        }
        for (auto &task: tasks)
        {
            task.get();
        }
    }

    /** multiplication single threading */
    static void multi(const self_type &a, const self_type &b, self_type &result)
    {
        T value;
        for (unsigned int row = 0; row < a.rows(); row++)
        {
            for (unsigned int col = 0; col < b.n; col++)
            {
                value = T();
                for (unsigned int k = 0; k < a.n; k++)
                {
                    value += a._cells[row * a.cols() + k] * b._cells[k * b.cols() + col];
                }
                result._cells[row * result.cols() + col] = value;
            }
        }
    }

    /** assignment overloading */
    Matrix& operator=(const self_type &a)
    {
        if (this == &a)
        {
            return *this;
        }
        m = a.m;
        n = a.n;
        _cells = vector<T>(n * m, 0);
        for (unsigned int i = 0; i < m; i++)
        {
            for (unsigned int j = 0; (j < n); j++)
            {
                _cells[i * n + j] = a._cells[i * a.n + j];
            }
        }
        return *this;
    };

    /** comparer overloading */
    bool operator==(const self_type &a)
    {
        if (m != a.m || n != a.n)
        { return false; }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((*this)(i, j) != a(i, j))
                {
                    return false;
                };
            }
        }
        return true;
    };

    /** comparer overloading */
    bool operator!=(const self_type &a)
    {
        return !(*this == a);
    };

    /** getter for row/col */
    const T operator()(unsigned int rows, unsigned int cols) const
    {
        return _cells.at(rows * n + cols);
    };

    /** getter for row/col */
    T &operator()(unsigned int rows, unsigned int cols)
    {
        return _cells.at(rows * n + cols);
    };

    /** out streaming overloading */
    friend ostream &operator<<(ostream &out, const self_type &a)
    {
        unsigned int i, j;

        for (i = 0; (i < a.m); i++)
        {
            for (j = 0; (j < a.n); j++)
            {
                out << a(i, j) << '\t';
            }
            out << endl;
        }
        return out;
    }

    /** in streaming overloading */
    friend istream &operator>>(istream &in, self_type &a)
    {
        unsigned int i, j;
        for (i = 0; (i < a.m); i++)
        {
            for (j = 0; (j < a.n); j++)
            {
                in >> a(i, j);
            }
        }
        return in;
    }

    /** multithreading switch */
    static void setParallel(bool b)
    {
        string mode;
        Matrix<T>::parallel = b;
        if (b)
        {
            mode = "Parallel";
        }
        else
        {
            mode = "non-Parallel";
        }
        cout << "Generic Matrix mode changed to " << mode << " mode." << endl;
    }

};

/** specialization for transposing complex matrix */
template<>
Matrix<Complex> Matrix<Complex>::trans()
{
    if (!isSquareMatrix())
    {
        throw invalid_argument(NOT_SQR_MTRX_MSG);
    }
    Matrix<Complex> newMat(n, m);
    unsigned int i, j;
    for (i = 0; (i < m); i++)
    {
        for (j = 0; (j < n); j++)
        {
            newMat._cells[j * m + i] = _cells[i * m + j].conj();
        }
    }
    return newMat;


};

template<class T>
bool Matrix<T>::parallel = false;

#endif //CPP_EX3_MATRIX_HPP
