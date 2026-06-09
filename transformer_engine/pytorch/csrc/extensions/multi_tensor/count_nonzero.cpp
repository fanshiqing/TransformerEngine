/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

at::Tensor multi_tensor_count_nonzero_cuda(int chunk_size, at::Tensor noop_flag,
                                           std::vector<std::vector<at::Tensor>> tensor_lists) {
  auto int64_options = tensor_lists[0][0].options().dtype(at::kLong);

  // Fixed-size per-block scratch (depth_to_max_blocks[0] == 320) plus the scalar return.
  auto output = at::zeros({320}, int64_options);
  auto ret = at::empty({1}, int64_options);

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto ret_cu = makeTransformerEngineTensor(ret);

  nvte_multi_tensor_count_nonzero_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(),
                                       num_lists, num_tensors, output_cu.data(), ret_cu.data(),
                                       at::cuda::getCurrentCUDAStream());

  return ret;
}

}  // namespace transformer_engine::pytorch
