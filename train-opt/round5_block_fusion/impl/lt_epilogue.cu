#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

static void check(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", static_cast<int>(status));
}
class LinearPlan {
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t a, b, out;
  cublasLtMatmulAlgo_t algo;
  int64_t m, n, k;
  bool tanh;
public:
  LinearPlan(int64_t rows, int64_t outputs, int64_t inputs, bool bf16, bool use_tanh)
      : m(rows), n(outputs), k(inputs), tanh(use_tanh) {
    cudaDataType_t dtype = bf16 ? CUDA_R_16BF : CUDA_R_32F;
    auto compute = bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
    check(cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F));
    cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    cublasLtEpilogue_t epi=tanh ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_BIAS;
    check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi)));
    if(tanh) check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,&n,sizeof(n)));
    check(cublasLtMatrixLayoutCreate(&a,dtype,k,n,k));
    check(cublasLtMatrixLayoutCreate(&b,dtype,k,m,k));
    check(cublasLtMatrixLayoutCreate(&out,dtype,n,m,n));
    cublasLtMatmulPreference_t pref;
    check(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace=32*1024*1024;
    check(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t result;
    int count=0;
    auto status=cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(),desc,a,b,out,out,pref,1,&result,&count);
    cublasLtMatmulPreferenceDestroy(pref);
    check(status);
    TORCH_CHECK(count>0,"No cuBLASLt epilogue algorithm for this shape/dtype");
    algo=result.algo;
  }
  ~LinearPlan(){cublasLtMatmulDescDestroy(desc);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(out);}
  void run(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor y,torch::Tensor aux,torch::Tensor workspace){
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && bias.is_contiguous(),"contiguous Lt inputs required");
    TORCH_CHECK(x.size(0)==m && x.size(1)==k && w.size(0)==n && w.size(1)==k,"shape differs from cached Lt plan");
    void* bias_ptr=bias.data_ptr();
    check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_ptr,sizeof(bias_ptr)));
    if(tanh){void* aux_ptr=aux.data_ptr();check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,&aux_ptr,sizeof(aux_ptr)));}
    float alpha=1.f,beta=0.f;
    check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),desc,&alpha,w.data_ptr(),a,x.data_ptr(),b,
                        &beta,y.data_ptr(),out,y.data_ptr(),out,&algo,workspace.data_ptr(),workspace.numel(),at::cuda::getCurrentCUDAStream()));
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){pybind11::class_<LinearPlan>(m,"LinearPlan")
  .def(pybind11::init<int64_t,int64_t,int64_t,bool,bool>()).def("run",&LinearPlan::run);}
