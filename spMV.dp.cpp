#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <dpct/sparse_utils.hpp>

#include <dpct/blas_utils.hpp>

#include <dpct/lib_common_utils.hpp>

int main() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::queue *handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // Initialize cuSPARSE library
    handle = &q_ct1;

    // Create stream and assign to handle
    dpct::queue_ptr stream;
    stream = dev_ct1.create_queue();

    handle = stream;

    // Define a simple CSR matrix
    // | 1 0 0 |
    // | 0 2 0 |
    // | 4 0 3 |
    int h_csrRowPtr[] = {0, 1, 2, 4};
    int h_csrColInd[] = {0, 1, 0, 2};
    float h_csrVal[] = {1.0f, 2.0f, 4.0f, 3.0f};
    float h_x[] = {1.0f, 2.0f, 3.0f};
    float h_y[3] = {0};

    // Device arrays
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y;

    // Allocate device memory
    d_csrRowPtr = (int *)sycl::malloc_device(sizeof(h_csrRowPtr), q_ct1);
    d_csrColInd = (int *)sycl::malloc_device(sizeof(h_csrColInd), q_ct1);
    d_csrVal = (float *)sycl::malloc_device(sizeof(h_csrVal), q_ct1);
    d_x = (float *)sycl::malloc_device(sizeof(h_x), q_ct1);
    d_y = (float *)sycl::malloc_device(sizeof(h_y), q_ct1);

    // Copy data to device
    q_ct1.memcpy(d_csrRowPtr, h_csrRowPtr, sizeof(h_csrRowPtr)).wait();
    q_ct1.memcpy(d_csrColInd, h_csrColInd, sizeof(h_csrColInd)).wait();
    q_ct1.memcpy(d_csrVal, h_csrVal, sizeof(h_csrVal)).wait();
    q_ct1.memcpy(d_x, h_x, sizeof(h_x)).wait();

    // Create matrix and vector descriptors
    /*
    DPCT1007:0: Migration of cusparseCreateCsr is not supported.
    */
    cusparseCreateCsr(&matA, 3, 3, 4, d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      oneapi::mkl::index_base::zero,
                      dpct::library_data_t::real_float);
    /*
    DPCT1007:1: Migration of cusparseCreateDnVec is not supported.
    */
    cusparseCreateDnVec(&vecX, 3, d_x, dpct::library_data_t::real_float);
    /*
    DPCT1007:2: Migration of cusparseCreateDnVec is not supported.
    */
    cusparseCreateDnVec(&vecY, 3, d_y, dpct::library_data_t::real_float);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Determine buffer size required for the cusparseSpMV operation
    size_t bufferSize = 0;
    /*
    DPCT1007:3: Migration of cusparseSpMV_bufferSize is not supported.
    */
    cusparseSpMV_bufferSize(handle, oneapi::mkl::transpose::nontrans, &alpha,
                            matA, vecX, &beta, vecY,
                            dpct::library_data_t::real_float,
                            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);

    // Allocate the buffer
    void* dBuffer = NULL;
    dBuffer = (void *)sycl::malloc_device(bufferSize, q_ct1);

    // Perform matrix-vector multiplication: y = A*x
    /*
    DPCT1007:4: Migration of cusparseSpMV is not supported.
    */

    // SYCL Interop
    cudaStream_t nativeStream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
    cudaStreamCreate(&handle);
    cusparseSetStream(handle, stream);

    cusparseSpMV(handle, oneapi::mkl::transpose::nontrans, &alpha, matA, vecX,
                 &beta, vecY, dpct::library_data_t::real_float,
                 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // Copy result back to the host
    q_ct1.memcpy(h_y, d_y, sizeof(h_y)).wait();

    // Print result
    for (int i = 0; i < 3; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Cleanup
    sycl::free(d_csrRowPtr, q_ct1);
    sycl::free(d_csrColInd, q_ct1);
    sycl::free(d_csrVal, q_ct1);
    sycl::free(d_x, q_ct1);
    sycl::free(d_y, q_ct1);
    sycl::free(dBuffer, q_ct1);
    /*
    DPCT1007:5: Migration of cusparseDestroySpMat is not supported.
    */
    cusparseDestroySpMat(matA);
    /*
    DPCT1007:6: Migration of cusparseDestroyDnVec is not supported.
    */
    cusparseDestroyDnVec(vecX);
    /*
    DPCT1007:7: Migration of cusparseDestroyDnVec is not supported.
    */
    cusparseDestroyDnVec(vecY);
    handle = nullptr;

    return 0;
}
