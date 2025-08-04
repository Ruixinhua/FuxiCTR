# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
对比学习(Contrastive Learning)模型模块

提供各种基础模型的对比学习版本实现，包括：
- PNNCL: PNN with Contrastive Learning
- DCNv3CL: DCNv3 with Contrastive Learning  
- DCNv2CL: DCNv2 with Contrastive Learning
- MaskNetCL: MaskNet with Contrastive Learning

所有CL模型共享统一的对比学习基础组件。
"""

from .src.base import ContrastiveLearningBase
from .src.PNNCL import PNNCL
from .src.DCNv3CL import DCNv3CL
from .src.DCNv2CL import DCNv2CL
from .src.MaskNetCL import MaskNetCL

__all__ = [
    'ContrastiveLearningBase',
    'PNNCL',
    'DCNv3CL', 
    'DCNv2CL',
    'MaskNetCL'
] 