"""
Module version tracking for self-evolving DSPy agents.

Tracks module versions with scores, timestamps, and supports rollback.
"""

import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy


@dataclass
class ModuleVersion:
    """Represents a single version of a DSPy module."""
    
    version: int
    score: float
    timestamp: str
    metadata: Dict[str, Any]
    module_state: Optional[bytes] = None  # Pickled module
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding module_state for JSON)."""
        data = asdict(self)
        data.pop('module_state', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleVersion':
        """Create from dictionary."""
        return cls(**data)


class VersionTracker:
    """
    Manages version history for DSPy modules.
    
    Provides versioning, rollback, and best version selection.
    """
    
    def __init__(self, module_name: str, storage_dir: str = "./data/module_versions"):
        """
        Initialize version tracker.
        
        Args:
            module_name: Name of the module to track
            storage_dir: Base directory for version storage
        """
        self.module_name = module_name
        self.storage_dir = Path(storage_dir) / module_name
        self.versions: List[ModuleVersion] = []
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing versions
        self._load_versions()
    
    def _load_versions(self):
        """Load version history from disk."""
        metadata_file = self.storage_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.versions = [ModuleVersion.from_dict(v) for v in data]
            except Exception as e:
                print(f"Warning: Could not load version metadata: {e}")
    
    def _save_metadata(self):
        """Save version metadata to disk."""
        metadata_file = self.storage_dir / "metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(
                    [v.to_dict() for v in self.versions],
                    f,
                    indent=2
                )
        except Exception as e:
            print(f"Warning: Could not save version metadata: {e}")
    
    def add_version(
        self,
        module: dspy.Module,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new module version.
        
        Args:
            module: DSPy module to save
            score: Performance score for this version
            metadata: Optional metadata dict
            
        Returns:
            Version number
        """
        version_num = len(self.versions)
        timestamp = datetime.utcnow().isoformat()
        
        # Pickle the module
        try:
            module_state = pickle.dumps(module)
        except Exception as e:
            print(f"Warning: Could not pickle module: {e}")
            module_state = None
        
        # Create version entry
        version = ModuleVersion(
            version=version_num,
            score=score,
            timestamp=timestamp,
            metadata=metadata or {},
            module_state=module_state
        )
        
        self.versions.append(version)
        
        # Save module to disk
        if module_state:
            version_file = self.storage_dir / f"v{version_num}.pkl"
            try:
                with open(version_file, 'wb') as f:
                    f.write(module_state)
            except Exception as e:
                print(f"Warning: Could not save module file: {e}")
        
        # Update metadata
        self._save_metadata()
        
        return version_num
    
    def get_version(self, version: int) -> Optional[dspy.Module]:
        """
        Load a specific module version.
        
        Args:
            version: Version number to load
            
        Returns:
            DSPy module or None if not found
        """
        if version < 0 or version >= len(self.versions):
            return None
        
        version_file = self.storage_dir / f"v{version}.pkl"
        
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'rb') as f:
                module_state = f.read()
                return pickle.loads(module_state)
        except Exception as e:
            print(f"Warning: Could not load module version {version}: {e}")
            return None
    
    def get_current(self) -> Optional[dspy.Module]:
        """
        Get the current (latest) module version.
        
        Returns:
            Latest DSPy module or None
        """
        if not self.versions:
            return None
        
        return self.get_version(self.versions[-1].version)
    
    def get_best(self) -> Optional[dspy.Module]:
        """
        Get the best performing module version.
        
        Returns:
            Best DSPy module or None
        """
        if not self.versions:
            return None
        
        best_version = max(self.versions, key=lambda v: v.score)
        return self.get_version(best_version.version)
    
    def get_best_version_number(self) -> Optional[int]:
        """
        Get the version number of the best performing module.
        
        Returns:
            Version number or None
        """
        if not self.versions:
            return None
        
        best_version = max(self.versions, key=lambda v: v.score)
        return best_version.version
    
    def rollback_to(self, version: int) -> bool:
        """
        Rollback to a specific version.
        
        This truncates the version history after the specified version.
        
        Args:
            version: Version number to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        if version < 0 or version >= len(self.versions):
            return False
        
        # Truncate version list
        self.versions = self.versions[:version + 1]
        
        # Update metadata
        self._save_metadata()
        
        return True
    
    def get_version_info(self, version: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific version.
        
        Args:
            version: Version number
            
        Returns:
            Version info dict or None
        """
        if version < 0 or version >= len(self.versions):
            return None
        
        return self.versions[version].to_dict()
    
    def get_all_versions_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all versions.
        
        Returns:
            List of version info dicts
        """
        return [v.to_dict() for v in self.versions]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of version history."""
        if not self.versions:
            return {"count": 0}
        
        scores = [v.score for v in self.versions]
        return {
            "count": len(self.versions),
            "best_version": self.get_best_version_number(),
            "best_score": max(scores),
            "latest_version": self.versions[-1].version,
            "latest_score": self.versions[-1].score,
            "avg_score": sum(scores) / len(scores),
        }
