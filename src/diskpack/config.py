"""
Configuration and type definitions for circle packing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

# Type aliases
Polygon = np.ndarray
Point = np.ndarray
GridKey = Tuple[int, int]
Circle = Tuple[float, float, float]  # (x, y, radius)


class PackingMode(Enum):
    """Available packing strategies."""
    RANDOM = "random"
    HEX_GRID = "hex_grid"
    FRONT = "front"
    HYBRID = "hybrid"


@dataclass
class PackingConfig:
    """
    Configuration parameters for the circle packing algorithm.
    
    Quick Start
    -----------
    # Best density for simple convex shapes (squares, rectangles, circles):
    config = PackingConfig()  # Uses random sampling (default)
    
    # Best density for complex/concave shapes (stars, L-shapes, letters):
    config = PackingConfig(use_hybrid_packing=True)
    
    # Fixed radius circles - fastest:
    config = PackingConfig(fixed_radius=5.0)
    
    # Fixed radius circles - best density on complex shapes:
    config = PackingConfig(fixed_radius=5.0, use_hybrid_packing=True)
    
    
    Algorithm Selection Guide
    -------------------------
    | Shape Type          | Best Algorithm              | Config                          |
    |---------------------|-----------------------------|---------------------------------|
    | Simple convex       | Random sampling (default)   | PackingConfig()                 |
    | Complex/concave     | Hybrid                      | use_hybrid_packing=True         |
    | Fixed radius, speed | Hex grid (default for fixed)| fixed_radius=X                  |
    | Fixed radius, dense | Hybrid                      | fixed_radius=X, use_hybrid_packing=True |
    | Artistic/organic    | Random sampling             | use_hex_grid=False              |
    
    
    Tuning Hybrid Mode
    ------------------
    Hybrid mode works in phases:
      Phase 1: Place large circles (>= hybrid_large_threshold * max_radius)
      Phase 2: Place medium circles (>= hybrid_medium_threshold * max_radius)
      Phase 3: Fill remaining space with random sampling
    
    For complex shapes with tight corners (stars, letters):
      - Use higher thresholds: hybrid_large_threshold=0.5, hybrid_medium_threshold=0.25
      - This prioritizes filling corners with appropriately-sized circles
    
    For simpler shapes where you want hybrid's speed:
      - Use lower thresholds: hybrid_large_threshold=0.3, hybrid_medium_threshold=0.1
      - This lets the front-based algorithm place more circles before random cleanup
    
    
    Parameters
    ----------
    padding : float, default=1.5
        Minimum gap between circles and between circles and polygon edges.
        
    min_radius : float, default=1.0
        Smallest circle that will be placed. Circles below this size are skipped.
        
    fixed_radius : float, optional
        If set, all circles will have exactly this radius. Enables hex grid mode
        by default (override with use_hex_grid=False for organic placement).
        
    use_hex_grid : bool, default=True
        Use hexagonal grid for fixed radius packing. Fastest option but creates
        a regular pattern. Set to False for organic/random placement.
        
    use_front_packing : bool, default=False
        Use front-based algorithm. Good for filling corners but may produce
        many small circles. Usually better to use use_hybrid_packing instead.
        
    use_hybrid_packing : bool, default=False
        Use state-of-the-art multi-phase algorithm. Best for complex shapes.
        Combines front-based (for corners) with random sampling (for gaps).
        
    max_failed_attempts : int, default=200
        Stop after this many consecutive failed placement attempts.
        Higher values may find more circles but take longer.
        
    sample_batch_size : int, default=50
        Number of random points sampled per iteration. Higher values find
        better placements per iteration but use more memory.
        
    hybrid_large_threshold : float, default=0.5
        Phase 1 minimum radius as fraction of max possible radius.
        Only circles >= this fraction are placed in Phase 1.
        
    hybrid_medium_threshold : float, default=0.25
        Phase 2 minimum radius as fraction of max possible radius.
        Only circles >= this fraction are placed in Phase 2.
        
    verbose : bool, default=False
        Print progress information during packing.
    """
    # Basic parameters
    padding: float = 1.5
    min_radius: float = 1.0
    fixed_radius: Optional[float] = None
    
    # Algorithm selection
    use_hex_grid: bool = True
    use_front_packing: bool = False
    use_hybrid_packing: bool = False
    
    # Performance tuning
    max_failed_attempts: int = 200
    sample_batch_size: int = 50
    grid_resolution_divisor: float = 25
    mega_circle_threshold: float = 0.5
    ray_cast_epsilon: float = 1e-10
    
    # Hybrid mode parameters
    hybrid_large_threshold: float = 0.5
    hybrid_medium_threshold: float = 0.25
    
    # Output
    verbose: bool = False


@dataclass
class PackingProgress:
    """Tracks the current state of the packing algorithm."""
    circles_placed: int = 0
    failed_attempts: int = 0
    max_failed_attempts: int = 200
    phase: str = ""

    @property
    def progress_ratio(self) -> float:
        """How close to stopping (0.0 = just started, 1.0 = done)."""
        return self.failed_attempts / self.max_failed_attempts if self.max_failed_attempts > 0 else 0

    def __str__(self) -> str:
        phase_str = f"[{self.phase}] " if self.phase else ""
        return f"{phase_str}Placed: {self.circles_placed} | Failed: {self.failed_attempts}/{self.max_failed_attempts} ({self.progress_ratio:.0%})"
