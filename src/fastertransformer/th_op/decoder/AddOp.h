#include "src/fastertransformer/kernels/add_bias_transpose_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace th = torch;

namespace torch_ext {

th::Tensor
add(th::Tensor input, th::Tensor bias);

}  // namespace torch_ext