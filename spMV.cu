#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>

int main() {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // Initialize cuSPARSE library
    cusparseCreate(&handle);

    // Create stream and assign to handle
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cusparseSetStream(handle, stream);

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
    cudaMalloc(&d_csrRowPtr, sizeof(h_csrRowPtr));
    cudaMalloc(&d_csrColInd, sizeof(h_csrColInd));
    cudaMalloc(&d_csrVal, sizeof(h_csrVal));
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));

    // Copy data to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, sizeof(h_csrRowPtr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, sizeof(h_csrColInd), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, sizeof(h_csrVal), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    // Create matrix and vector descriptors
    cusparseCreateCsr(&matA, 3, 3, 4, d_csrRowPtr, d_csrColInd, d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 3, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, d_y, CUDA_R_32F);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Determine buffer size required for the cusparseSpMV operation
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);

    // Allocate the buffer
    void* dBuffer = NULL;
    cudaMalloc(&dBuffer, bufferSize);

    // Perform matrix-vector multiplication: y = A*x
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 3; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    return 0;
}
