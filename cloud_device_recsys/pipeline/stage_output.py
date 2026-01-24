# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Stage Output Data Structures

This module defines the data structures for passing data between pipeline stages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
import pickle
import json
import os
import csv
from datetime import datetime


@dataclass
class CandidateItem:
    """A single candidate item with its features and scores"""
    item_id: Any
    features: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSet:
    """
    A set of candidate items for a single request/user.
    
    This is the primary data structure passed between pipeline stages.
    """
    request_id: str  # Unique identifier for this request
    user_id: Optional[Any] = None
    user_features: Dict[str, Any] = field(default_factory=dict)
    candidates: List[CandidateItem] = field(default_factory=list)
    context_features: Dict[str, Any] = field(default_factory=dict)
    
    # Stage metadata
    source_stage: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def add_candidate(self, item_id: Any, features: Dict = None, 
                     score: float = 0.0, embedding: np.ndarray = None) -> None:
        """Add a candidate to the set"""
        self.candidates.append(CandidateItem(
            item_id=item_id,
            features=features or {},
            score=score,
            embedding=embedding
        ))
    
    def get_top_k(self, k: int) -> List[CandidateItem]:
        """Get top-k candidates by score"""
        sorted_candidates = sorted(self.candidates, key=lambda x: x.score, reverse=True)
        return sorted_candidates[:k]
    
    def get_item_ids(self) -> List[Any]:
        """Get list of all item IDs"""
        return [c.item_id for c in self.candidates]
    
    def get_scores(self) -> np.ndarray:
        """Get array of all scores"""
        return np.array([c.score for c in self.candidates])
    
    def update_scores(self, new_scores: np.ndarray) -> None:
        """Update scores for all candidates"""
        assert len(new_scores) == len(self.candidates)
        for i, score in enumerate(new_scores):
            self.candidates[i].score = float(score)
    
    def filter_by_score(self, min_score: float) -> 'CandidateSet':
        """Return new CandidateSet with candidates above min_score"""
        filtered = CandidateSet(
            request_id=self.request_id,
            user_id=self.user_id,
            user_features=self.user_features.copy(),
            context_features=self.context_features.copy(),
            source_stage=self.source_stage
        )
        filtered.candidates = [c for c in self.candidates if c.score >= min_score]
        return filtered
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'request_id': self.request_id,
            'user_id': self.user_id,
            'user_features': self.user_features,
            'context_features': self.context_features,
            'source_stage': self.source_stage,
            'timestamp': self.timestamp,
            'candidates': [
                {
                    'item_id': c.item_id,
                    'features': c.features,
                    'score': c.score,
                    'metadata': c.metadata
                }
                for c in self.candidates
            ]
        }


@dataclass
class StageOutput:
    """
    Container for stage output including results and metrics.
    
    This is what each pipeline stage returns after processing.
    """
    stage_name: str
    candidate_sets: List[CandidateSet] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing info
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
    
    def mark_complete(self) -> None:
        """Mark the stage as complete and compute duration"""
        self.end_time = datetime.now().isoformat()
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.duration_seconds = (end - start).total_seconds()
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the output"""
        self.metrics[name] = value
    
    def get_total_candidates(self) -> int:
        """Get total number of candidates across all sets"""
        return sum(len(cs.candidates) for cs in self.candidate_sets)
    
    def save(self, filepath: str) -> None:
        """Save stage output to file (pickle format)"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'StageOutput':
        """Load stage output from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Save stage output to CSV files.
        
        Creates:
        - {stage_name}_candidates.csv: All candidates with scores
        - {stage_name}_metrics.csv: Stage metrics
        
        Returns:
            Dictionary mapping output type to filepath
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # Save candidates
        candidates_path = os.path.join(output_dir, f"{self.stage_name}_candidates.csv")
        with open(candidates_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['request_id', 'user_id', 'item_id', 'score', 'rank'])
            
            for cs in self.candidate_sets:
                sorted_candidates = sorted(cs.candidates, key=lambda x: x.score, reverse=True)
                for rank, candidate in enumerate(sorted_candidates, 1):
                    writer.writerow([
                        cs.request_id,
                        cs.user_id,
                        candidate.item_id,
                        f"{candidate.score:.6f}",
                        rank
                    ])
        saved_files['candidates'] = candidates_path
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{self.stage_name}_metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            writer.writerow(['stage_name', self.stage_name])
            writer.writerow(['duration_seconds', f"{self.duration_seconds:.2f}"])
            writer.writerow(['total_requests', len(self.candidate_sets)])
            writer.writerow(['total_candidates', self.get_total_candidates()])
            for name, value in sorted(self.metrics.items()):
                writer.writerow([name, f"{value:.6f}" if isinstance(value, float) else value])
        saved_files['metrics'] = metrics_path
        
        return saved_files
    
    def to_summary_dict(self) -> Dict:
        """Get summary dictionary for logging/display"""
        return {
            'stage_name': self.stage_name,
            'num_requests': len(self.candidate_sets),
            'total_candidates': self.get_total_candidates(),
            'avg_candidates_per_request': self.get_total_candidates() / max(1, len(self.candidate_sets)),
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics
        }
