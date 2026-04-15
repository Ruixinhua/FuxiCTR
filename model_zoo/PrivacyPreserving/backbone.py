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
Backbone factory for PrivacyPreserving models.

This module re-exports from the unified backbone in fuxictr.pytorch.backbone
to ensure all model series (KD, PrivacyPreserving, DTDN, DTCN) share the
same backbone implementations.
"""

from fuxictr.pytorch.backbone import (
    PNNBackbone,
    FinalNetBackbone,
    DCNv3Backbone,
    BACKBONE_REGISTRY,
    build_backbone,
)
