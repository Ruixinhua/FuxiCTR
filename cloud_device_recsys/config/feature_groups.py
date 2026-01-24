# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Feature Group Management Module

This module manages feature groups for cloud-device collaborative recommendation:
- FG1: Non-personalized features (item + context) - available everywhere
- FG2: Cloud-personalized features (user behavioral, low privacy) - available everywhere
- FG3: Device-only features (strong privacy) - device only
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
import logging


class FeatureGroup(Enum):
    """Feature group enumeration"""
    FG1 = "non_personalized"      # Item + Context features
    FG2 = "cloud_personalized"    # User behavioral (low privacy)
    FG3 = "device_only"           # Strong privacy features
    
    @classmethod
    def from_string(cls, s: str) -> 'FeatureGroup':
        """Convert string to FeatureGroup enum"""
        mapping = {
            'FG1': cls.FG1, 'fg1': cls.FG1, 'non_personalized': cls.FG1,
            'FG2': cls.FG2, 'fg2': cls.FG2, 'cloud_personalized': cls.FG2,
            'FG3': cls.FG3, 'fg3': cls.FG3, 'device_only': cls.FG3,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown feature group: {s}")


@dataclass
class FeatureGroupConfig:
    """Configuration for feature group management"""
    # Default feature group assignments based on feature name patterns
    fg1_patterns: List[str] = field(default_factory=lambda: [
        'cate_id', 'brand', 'price', 'campaign_id', 'customer', 'adgroup_id',
        'pid', 'btag', 'cand_item_', 'site_', 'app_', 'banner_', 'C1', 'C14',
        'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour', 'weekday', 'weekend'
    ])
    fg2_patterns: List[str] = field(default_factory=lambda: [
        '_his', '_seq', 'exposure_', 'click_', 'ipv_', 'exp_item_'
    ])
    fg3_patterns: List[str] = field(default_factory=lambda: [
        'user_id', 'userid', 'age', 'gender', 'age_level', 'gender_code',
        'cms_segid', 'cms_group_id', 'pvalue_level', 'shopping_level',
        'occupation', 'new_user_class_level', 'device_id', 'device_ip'
    ])


class FeatureGroupManager:
    """
    Manages feature group assignments and filtering for cloud-device pipeline.
    
    Usage:
        manager = FeatureGroupManager()
        
        # Auto-assign based on feature names
        manager.auto_assign_groups(feature_map)
        
        # Get features for cloud modules (FG1 + FG2)
        cloud_features = manager.get_cloud_features()
        
        # Get features for device modules (FG1 + FG2 + FG3)
        device_features = manager.get_device_features()
    """
    
    def __init__(self, config: Optional[FeatureGroupConfig] = None):
        self.config = config or FeatureGroupConfig()
        self.feature_assignments: Dict[str, FeatureGroup] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def assign_feature(self, feature_name: str, group: FeatureGroup) -> None:
        """Manually assign a feature to a group"""
        self.feature_assignments[feature_name] = group
        self.logger.debug(f"Assigned '{feature_name}' to {group.value}")
    
    def auto_assign_groups(self, feature_map: Any) -> Dict[str, FeatureGroup]:
        """
        Automatically assign features to groups based on name patterns.
        
        Args:
            feature_map: FuxiCTR FeatureMap object with features dict
            
        Returns:
            Dictionary mapping feature names to their assigned groups
        """
        features = feature_map.features if hasattr(feature_map, 'features') else feature_map
        
        for feature_name in features.keys():
            if feature_name in self.feature_assignments:
                continue  # Skip if already assigned
            
            # Check explicit feature_group in feature spec
            spec = features[feature_name]
            if isinstance(spec, dict) and 'feature_group' in spec:
                group_str = spec['feature_group']
                self.feature_assignments[feature_name] = FeatureGroup.from_string(group_str)
                continue
            
            # Auto-assign based on patterns
            assigned = False
            
            # Check FG3 patterns first (most restrictive)
            for pattern in self.config.fg3_patterns:
                if pattern.lower() in feature_name.lower():
                    self.feature_assignments[feature_name] = FeatureGroup.FG3
                    assigned = True
                    break
            
            if not assigned:
                # Check FG2 patterns (behavioral/sequence features)
                for pattern in self.config.fg2_patterns:
                    if pattern.lower() in feature_name.lower():
                        self.feature_assignments[feature_name] = FeatureGroup.FG2
                        assigned = True
                        break
            
            if not assigned:
                # Default to FG1 (non-personalized)
                self.feature_assignments[feature_name] = FeatureGroup.FG1
        
        self.logger.info(f"Auto-assigned {len(self.feature_assignments)} features to groups")
        self._log_assignment_summary()
        
        return self.feature_assignments
    
    def _log_assignment_summary(self) -> None:
        """Log summary of feature group assignments"""
        fg1_count = sum(1 for g in self.feature_assignments.values() if g == FeatureGroup.FG1)
        fg2_count = sum(1 for g in self.feature_assignments.values() if g == FeatureGroup.FG2)
        fg3_count = sum(1 for g in self.feature_assignments.values() if g == FeatureGroup.FG3)
        
        self.logger.info(f"Feature group summary: FG1={fg1_count}, FG2={fg2_count}, FG3={fg3_count}")
    
    def get_features_by_group(self, group: FeatureGroup) -> Set[str]:
        """Get all feature names in a specific group"""
        return {name for name, g in self.feature_assignments.items() if g == group}
    
    def get_cloud_features(self) -> Set[str]:
        """Get features available in cloud (FG1 + FG2)"""
        return self.get_features_by_group(FeatureGroup.FG1) | \
               self.get_features_by_group(FeatureGroup.FG2)
    
    def get_device_features(self) -> Set[str]:
        """Get features available on device (FG1 + FG2 + FG3)"""
        return self.get_features_by_group(FeatureGroup.FG1) | \
               self.get_features_by_group(FeatureGroup.FG2) | \
               self.get_features_by_group(FeatureGroup.FG3)
    
    def filter_feature_map(self, feature_map: Any, allowed_groups: List[FeatureGroup]) -> Dict:
        """
        Filter feature map to only include features from allowed groups.
        
        Args:
            feature_map: FuxiCTR FeatureMap object or features dict
            allowed_groups: List of feature groups to include
            
        Returns:
            Filtered features dictionary
        """
        features = feature_map.features if hasattr(feature_map, 'features') else feature_map
        allowed_features = set()
        
        for group in allowed_groups:
            allowed_features.update(self.get_features_by_group(group))
        
        filtered = {name: spec for name, spec in features.items() 
                   if name in allowed_features}
        
        self.logger.info(f"Filtered features: {len(features)} -> {len(filtered)}")
        return filtered
    
    def validate_cloud_module(self, used_features: Set[str]) -> bool:
        """
        Validate that a cloud module doesn't use FG3 features.
        
        Args:
            used_features: Set of feature names used by the module
            
        Returns:
            True if valid (no FG3 features used), False otherwise
        """
        fg3_features = self.get_features_by_group(FeatureGroup.FG3)
        leaked_features = used_features & fg3_features
        
        if leaked_features:
            self.logger.error(f"Privacy violation! Cloud module uses FG3 features: {leaked_features}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, str]:
        """Export assignments to dictionary"""
        return {name: group.value for name, group in self.feature_assignments.items()}
    
    def from_dict(self, assignments: Dict[str, str]) -> None:
        """Import assignments from dictionary"""
        for name, group_str in assignments.items():
            self.feature_assignments[name] = FeatureGroup.from_string(group_str)
    
    def save_to_csv(self, filepath: str) -> None:
        """Save feature assignments to CSV file"""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature_name', 'feature_group'])
            for name, group in sorted(self.feature_assignments.items()):
                writer.writerow([name, group.value])
        self.logger.info(f"Saved feature assignments to {filepath}")
    
    def load_from_csv(self, filepath: str) -> None:
        """Load feature assignments from CSV file"""
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.feature_assignments[row['feature_name']] = \
                    FeatureGroup.from_string(row['feature_group'])
        self.logger.info(f"Loaded {len(self.feature_assignments)} assignments from {filepath}")
