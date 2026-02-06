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
    label: Optional[int] = None # Added for ground truth in candidate lists
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
                     score: float = 0.0, label: int = None, 
                     embedding: np.ndarray = None) -> None:
        """Add a candidate to the set"""
        self.candidates.append(CandidateItem(
            item_id=item_id,
            features=features or {},
            score=score,
            label=label,
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

    def save_parquet(self, output_dir: str) -> None:
        """
        Save stage output to Parquet format (directory).
        
        Creates:
        - {output_dir}/candidates.parquet: Candidate items
        - {output_dir}/requests.parquet: Request metadata (user features, etc)
        - {output_dir}/metadata.json: Stage metrics and other metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Flatten Candidates Optimized
        # Use list of tuples instead of list of dicts for faster creation
        cand_tuples = []
        for i, cs in enumerate(self.candidate_sets):
            rid = cs.request_id
            # Micro-optimization: avoid dot access in loop
            for rank, c in enumerate(cs.candidates):
                cand_tuples.append((
                    rid,
                    c.item_id,
                    c.score,
                    c.label,
                    rank,
                    i
                ))

        if cand_tuples:
            df_cand = pd.DataFrame(cand_tuples, columns=['request_id', 'item_id', 'score', 'label', 'rank', 'list_index'])
            df_cand.to_parquet(os.path.join(output_dir, 'candidates.parquet'), index=False)
        else:
            # Save empty schema
            pd.DataFrame(columns=['request_id', 'item_id', 'score', 'label', 'rank', 'list_index']).to_parquet(
                os.path.join(output_dir, 'candidates.parquet'), index=False
            )
            
        # 2. Save Requests
        # Complex objects like user_features dicts need serialization
        req_data = []
        for i, cs in enumerate(self.candidate_sets):
            req_data.append({
                'request_id': cs.request_id,
                'user_id': str(cs.user_id), # Ensure consistent type
                'user_features': json.dumps(cs.user_features, default=str), # Serialize dict
                'context_features': json.dumps(cs.context_features, default=str),
                'source_stage': cs.source_stage,
                'timestamp': cs.timestamp,
                'list_index': i
            })
            
        if req_data:
            df_req = pd.DataFrame(req_data)
            df_req.to_parquet(os.path.join(output_dir, 'requests.parquet'), index=False)
        else:
            pd.DataFrame(columns=['request_id', 'user_id', 'user_features', 'context_features', 'source_stage', 'timestamp', 'list_index']).to_parquet(
                os.path.join(output_dir, 'requests.parquet'), index=False
            )

        # 3. Save Metadata
        metadata = {
            'stage_name': self.stage_name,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load_parquet(cls, output_dir: str) -> 'StageOutput':
        """Load stage output from Parquet directory"""
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Directory {output_dir} not found")
            
        # 1. Load Metadata
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        stage_output = cls(stage_name=meta['stage_name'])
        stage_output.metrics = meta.get('metrics', {})
        stage_output.metadata = meta.get('metadata', {})
        stage_output.start_time = meta.get('start_time')
        stage_output.end_time = meta.get('end_time')
        stage_output.duration_seconds = meta.get('duration_seconds', 0.0)
        
        # 2. Load Requests
        req_path = os.path.join(output_dir, 'requests.parquet')
        if not os.path.exists(req_path):
            return stage_output
            
        df_req = pd.read_parquet(req_path)
        if df_req.empty:
            return stage_output
            
        # Sort by list_index to preserve order
        df_req = df_req.sort_values('list_index')
        
        # Pre-create CandidateSets map to maintain order
        # Assuming list_index corresponds to position in candidate_sets
        candidate_sets_map = {} # list_index -> CandidateSet
        
        # Vectorized JSON decoding is tricky, loop is acceptable for Requests (usually much fewer than candidates)
        for _, row in df_req.iterrows():
            cs = CandidateSet(
                request_id=row['request_id'],
                user_id=row['user_id'],
                user_features=json.loads(row['user_features']),
                context_features=json.loads(row['context_features']),
                source_stage=row['source_stage'],
                timestamp=row['timestamp']
            )
            candidate_sets_map[row['list_index']] = cs
            stage_output.candidate_sets.append(cs)
            
        # 3. Load Candidates Optimized
        cand_path = os.path.join(output_dir, 'candidates.parquet')
        if os.path.exists(cand_path):
            df_cand = pd.read_parquet(cand_path)
            
            if not df_cand.empty:
                # Optimized grouping using Numpy splitting
                # 1. Sort by list_index (primary) and rank (secondary)
                # Note: Parquet might already be sorted, but ensure it.
                df_cand = df_cand.sort_values(['list_index', 'rank'])
                
                # 2. Convert columns to numpy arrays for fast access
                list_indices = df_cand['list_index'].values
                item_ids = df_cand['item_id'].values
                scores = df_cand['score'].values
                labels = df_cand['label'].values
                
                # 3. Find split indices
                # unique_indices correspond to the Values
                # split_indices are where the value changes
                unique_indices, split_indices = np.unique(list_indices, return_index=True)
                # np.unique returns sorted unique values. split_indices needs to be sorted for np.split?
                # Actually, if list_indices is sorted, we can use return_index to get start of each group.
                # However, unique_indices might not be in the order they appear if we didn't sort?
                # We sorted df_cand, so list_indices is monotonic.
                
                # split_indices[0] is always 0. np.split expects indices where to cut.
                # slice(split_indices[i], split_indices[i+1])
                
                # Handle edge case: split_indices doesn't give the end.
                split_indices = list(split_indices[1:]) + [len(list_indices)]
                
                # But unique_indices corresponds to the Group Key (list_index)
                
                # Alternative: Use itertools.groupby or just simple slicing if we know counts
                # Using np.unique(..., return_counts=True) is safer/easier
                unique_keys, counts = np.unique(list_indices, return_counts=True)
                # unique_keys are sorted.
                
                # We need to map unique_keys (list_index) to the data chunks
                # Since we sorted by list_index, the chunks are contiguous.
                
                # Split arrays
                # current end index
                cum_counts = np.cumsum(counts)
                # np.split requires splitting points (indices). 
                # e.g. [10, 20] means split at 10, then at 20.
                # cum_counts[:-1] gives the split points.
                
                split_item_ids = np.split(item_ids, cum_counts[:-1])
                split_scores = np.split(scores, cum_counts[:-1])
                split_labels = np.split(labels, cum_counts[:-1])
                
                # Iterate and assign
                # unique_keys matches the order of splits because we sorted by list_index
                for k, i_ids, i_scores, i_labels in zip(unique_keys, split_item_ids, split_scores, split_labels):
                    if k in candidate_sets_map:
                        cs = candidate_sets_map[k]
                        
                        # Fastest way to create list of objects: List Comprehension with zip
                        # Casting numpy types to python types if necessary (CandidateItem expects float/int usually)
                        # but keeping as numpy scalars is often fine or faster.
                        # Explicit float() cast is safer for json serialization later.
                        
                        # Handle Null labels (None)
                        # If label column has NaN/None, numpy array might be float.
                        # Check dtype or handle per item.
                        
                        # Doing a bulk cast to object or list of dicts -> CandidateItem
                        # Straight list comp:
                        cs.candidates = [
                            CandidateItem(
                                item_id=iid,
                                score=float(s),
                                label=int(l) if not np.isnan(l) else None
                            ) 
                            for iid, s, l in zip(i_ids, i_scores, i_labels)
                        ]
                            
        return stage_output
    
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
