#include "src/fastertransformer/th_op/decoder/AddOp.h"

namespace th = torch;
namespace torch_ext {
th::Tensor
add(th::Tensor input, th::Tensor bias)
{
CHECK_TH_CUDA(input);
CHECK_CONTIGUOUS(input);
CHECK_TH_CUDA(bias);
CHECK_CONTIGUOUS(bias);
auto output = th::empty_like(input);

    
fastertransformer::invokeAdd(torch_ext::get_ptr<float>(input),
                              torch_ext::get_ptr<float>(bias),
                              torch_ext::get_ptr<float>(output));

return output;
}

}  // namespace torch_ext

static auto add = th::RegisterOperators("fastertransformer::add", &torch_ext::add);