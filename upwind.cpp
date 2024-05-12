#include <fstream>

int main()
{
    const unsigned int n = 150, m = 150;
    const double h = 1.0 / (n - 1), t = 1.0 / (m - 1);
    double* T1 = new double[n], * T2 = new double[n];
    std::ofstream output("out.txt");
    for (int i = 0; i < n; ++i)
        if (i > 0 && i < 11)
            T1[i] = 1;
        else
            T1[i] = 0;

    for (int i = 0; i < n; ++i)
        output << T1[i] << std::endl;

    for (int j = 0; j < m - 1; ++j)
    {
        for (int i = 1; i < n; ++i)
            T2[i] = t / h * (T1[i - 1] - T1[i]) + T1[i];

        for (int i = 0; i < n; ++i)
            T1[i] = T2[i];

        for (int i = 0; i < n; ++i)
            output << T1[i] << std::endl;
    }
    output.close();
    delete[] T1, T2;
    return 0;
}