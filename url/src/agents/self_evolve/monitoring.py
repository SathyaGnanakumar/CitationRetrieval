"""
Monitoring and logging for self-evolving DSPy agents.

Provides structured logging and CSV export for analysis.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvolutionLogger:
    """
    Structured logging for optimization events.
    
    Logs to console and file with CSV export for analysis.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize evolution logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"evolution_{timestamp}.log"
        self.csv_file = self.log_dir / f"evolution_{timestamp}.csv"
        
        # Setup file logger
        self.logger = logging.getLogger("evolution")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # CSV data
        self.events: List[Dict[str, Any]] = []
        
        self.logger.info("Evolution logger initialized")
    
    def log_evaluation(
        self,
        query_count: int,
        avg_score: float,
        min_score: float,
        max_score: float,
        module_version: Optional[int] = None
    ):
        """
        Log an evaluation event.
        
        Args:
            query_count: Number of queries processed
            avg_score: Average score
            min_score: Minimum score
            max_score: Maximum score
            module_version: Current module version
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "evaluation",
            "query_count": query_count,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "module_version": module_version
        }
        self.events.append(event)
        
        self.logger.info(
            f"Evaluation: {query_count} queries, "
            f"avg={avg_score:.4f}, min={min_score:.4f}, max={max_score:.4f}"
        )
    
    def log_optimization_start(
        self,
        module_name: str,
        trainset_size: int,
        valset_size: int,
        current_score: float
    ):
        """
        Log optimization start.
        
        Args:
            module_name: Name of module being optimized
            trainset_size: Training set size
            valset_size: Validation set size
            current_score: Current score before optimization
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "optimization_start",
            "module_name": module_name,
            "trainset_size": trainset_size,
            "valset_size": valset_size,
            "current_score": current_score
        }
        self.events.append(event)
        
        self.logger.info(
            f"Optimization START: {module_name}, "
            f"train={trainset_size}, val={valset_size}, "
            f"current_score={current_score:.4f}"
        )
    
    def log_optimization_complete(
        self,
        module_name: str,
        new_score: float,
        old_score: float,
        improvement: float,
        duration_seconds: float
    ):
        """
        Log optimization completion.
        
        Args:
            module_name: Name of module optimized
            new_score: Score after optimization
            old_score: Score before optimization
            improvement: Improvement amount
            duration_seconds: Optimization duration
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "optimization_complete",
            "module_name": module_name,
            "new_score": new_score,
            "old_score": old_score,
            "improvement": improvement,
            "duration_seconds": duration_seconds
        }
        self.events.append(event)
        
        self.logger.info(
            f"Optimization COMPLETE: {module_name}, "
            f"new_score={new_score:.4f}, old_score={old_score:.4f}, "
            f"improvement={improvement:+.4f} ({improvement/old_score*100:+.1f}%), "
            f"duration={duration_seconds:.1f}s"
        )
    
    def log_deployment(
        self,
        module_name: str,
        version: int,
        score: float,
        reason: str
    ):
        """
        Log module deployment.
        
        Args:
            module_name: Name of module deployed
            version: Version number
            score: Module score
            reason: Deployment reason
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "deployment",
            "module_name": module_name,
            "version": version,
            "score": score,
            "reason": reason
        }
        self.events.append(event)
        
        self.logger.info(
            f"Deployment: {module_name} v{version}, "
            f"score={score:.4f}, reason={reason}"
        )
    
    def log_rollback(
        self,
        module_name: str,
        from_version: int,
        to_version: int,
        reason: str
    ):
        """
        Log module rollback.
        
        Args:
            module_name: Name of module rolled back
            from_version: Version rolled back from
            to_version: Version rolled back to
            reason: Rollback reason
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "rollback",
            "module_name": module_name,
            "from_version": from_version,
            "to_version": to_version,
            "reason": reason
        }
        self.events.append(event)
        
        self.logger.warning(
            f"Rollback: {module_name} v{from_version}→v{to_version}, "
            f"reason={reason}"
        )
    
    def export_csv(self):
        """Export events to CSV file."""
        if not self.events:
            return
        
        # Get all unique keys
        keys = set()
        for event in self.events:
            keys.update(event.keys())
        keys = sorted(keys)
        
        # Write CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.events)
        
        self.logger.info(f"Exported {len(self.events)} events to {self.csv_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dict with summary statistics
        """
        if not self.events:
            return {}
        
        # Count event types
        event_counts = {}
        for event in self.events:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Get optimization statistics
        optimizations = [e for e in self.events if e.get("event_type") == "optimization_complete"]
        if optimizations:
            improvements = [e["improvement"] for e in optimizations if "improvement" in e]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        else:
            avg_improvement = 0
        
        # Get deployment statistics
        deployments = [e for e in self.events if e.get("event_type") == "deployment"]
        
        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "optimizations": len(optimizations),
            "avg_improvement": avg_improvement,
            "deployments": len(deployments),
            "log_file": str(self.log_file),
            "csv_file": str(self.csv_file)
        }
