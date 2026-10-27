"""Taxonomy DAG Index & Fast Integer Wu-Palmer Semantic Similarity.

Builds a high-performance in-memory integer DAG for HPO and MONDO ontologies:
- Extracts is_a edges from OBO files, storing terms and parents as compact 32-bit ints.
- Maps alt_ids directly to primary canonical integer IDs.
- Calculates longest-path depth from root (depth[root] = 1).
- Computes transitive ancestors, pre-filtering generic terms (depth <= GENERIC_DEPTH).
- Computes Deepest Common Ancestors (DCA) via fast integer set intersection.
- Caches results using 64-bit packed integer keys ((min_id << 32) | max_id).
- Accepts int (e.g. 8807, 2761) or CURIE strings ("MONDO:0008807", "HP:0002761").
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

GENERIC_DEPTH_DEFAULT = 2


class TaxonomyIndex:
    """High-performance integer-based DAG and LCA index for an ontology."""

    def __init__(
        self,
        obo_path: Union[str, Path],
        prefix: str,
        generic_depth: int = GENERIC_DEPTH_DEFAULT,
    ) -> None:
        self.obo_path = Path(obo_path)
        self.prefix = prefix.upper().rstrip(":") + ":"
        self.generic_depth = generic_depth
        self.release_key: str = ""

        # Integer-keyed data structures
        self.parents: Dict[int, List[int]] = {}
        self.alt_id_map: Dict[int, int] = {}
        self.depth: Dict[int, int] = {}
        self.ancestors: Dict[int, frozenset[int]] = {}
        # 64-bit packed integer key: (min_id << 32) | max_id -> score
        self._pair_cache: Dict[int, float] = {}

        self._build_index()

    def _to_int_code(self, val: Union[int, str, None]) -> Optional[int]:
        """Convert input term identifier (int code, CURIE string, or numeric string) to int."""
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip()
        if not s:
            return None
        if ":" in s:
            s = s.split(":")[-1].strip()
        if s.isdigit():
            return int(s)
        digits = "".join(filter(str.isdigit, s))
        return int(digits) if digits else None

    def _parse_term_id(self, raw_str: str) -> Optional[int]:
        """Parse raw OBO id/alt_id/is_a string with prefix to integer ID."""
        cleaned = raw_str.split("!")[0].strip().upper()
        if cleaned.startswith(self.prefix):
            num_part = cleaned[len(self.prefix):].strip()
            if num_part.isdigit():
                return int(num_part)
        return None

    def _build_index(self) -> None:
        """Parse OBO file, construct integer DAG, compute depths and ancestors."""
        if not self.obo_path.exists():
            raise FileNotFoundError(f"Ontology OBO file not found: {self.obo_path}")

        stat = self.obo_path.stat()
        self.release_key = f"{self.obo_path.name}:{stat.st_size}:{int(stat.st_mtime)}"

        raw_parents: Dict[int, List[int]] = {}
        alt_map: Dict[int, int] = {}

        current_id: Optional[int] = None
        is_term = False
        is_obsolete = False
        term_parents: List[int] = []

        with open(self.obo_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    if current_id is not None and not is_obsolete:
                        raw_parents[current_id] = term_parents
                    current_id = None
                    is_term = True
                    is_obsolete = False
                    term_parents = []
                elif line.startswith("[") and line != "[Term]":
                    if current_id is not None and not is_obsolete:
                        raw_parents[current_id] = term_parents
                    current_id = None
                    is_term = False
                elif is_term:
                    if line.startswith("id: "):
                        current_id = self._parse_term_id(line[4:])
                    elif line.startswith("alt_id: "):
                        alt_id = self._parse_term_id(line[8:])
                        if alt_id is not None and current_id is not None:
                            alt_map[alt_id] = current_id
                    elif line.startswith("is_obsolete: true"):
                        is_obsolete = True
                    elif line.startswith("is_a: "):
                        parent_id = self._parse_term_id(line[6:])
                        if parent_id is not None:
                            term_parents.append(parent_id)

        if current_id is not None and not is_obsolete:
            raw_parents[current_id] = term_parents

        self.alt_id_map = alt_map

        # Canonicalize parent links
        resolved_parents: Dict[int, List[int]] = {}
        for tid, plist in raw_parents.items():
            res_p = []
            for p in plist:
                canon_p = self.alt_id_map.get(p, p)
                if canon_p in raw_parents:
                    res_p.append(canon_p)
            resolved_parents[tid] = res_p

        self.parents = resolved_parents

        # 1. Compute longest-path depth from root (root = 1)
        depth: Dict[int, int] = {}
        visiting: Set[int] = set()

        def compute_depth(node: int) -> int:
            if node in depth:
                return depth[node]
            if node in visiting:
                return 1  # Cycle break fallback
            visiting.add(node)
            par = self.parents.get(node, [])
            if not par:
                d = 1
            else:
                d = 1 + max(compute_depth(p) for p in par)
            visiting.remove(node)
            depth[node] = d
            return d

        for node in self.parents:
            compute_depth(node)
        self.depth = depth

        # 2. Compute transitive ancestors anc[t] = {t} | union(anc[p])
        raw_anc: Dict[int, Set[int]] = {}

        def compute_anc(node: int) -> Set[int]:
            if node in raw_anc:
                return raw_anc[node]
            anc = {node}
            for p in self.parents.get(node, []):
                anc.update(compute_anc(p))
            raw_anc[node] = anc
            return anc

        for node in self.parents:
            compute_anc(node)

        # 3. Pre-filter generic ancestors (depth <= generic_depth) once at build time
        # This eliminates filtering loops during query time!
        self.ancestors = {
            node: frozenset(x for x in raw_anc[node] if self.depth.get(x, 0) > self.generic_depth)
            for node in self.parents
        }

        logger.info(
            "Integer TaxonomyIndex built for %s: %d terms, %d alt_ids, max depth %d (release=%s)",
            self.prefix,
            len(self.parents),
            len(self.alt_id_map),
            max(self.depth.values()) if self.depth else 0,
            self.release_key,
        )

    def wu_palmer(self, term_a: Union[str, int, None], term_b: Union[str, int, None]) -> float:
        """Compute Wu-Palmer similarity between two ontology terms using fast integer operations.

        LCA search:
          common = anc[a] & anc[b] (generic ancestors depth <= GENERIC_DEPTH already excluded)
          dca = max(common) by depth
          wuPalmer(a, b) = 2 * depth[dca] / (depth[a] + depth[b])
          a == b -> 1.0; no dca -> 0.0
        """
        code_a = self._to_int_code(term_a)
        code_b = self._to_int_code(term_b)

        if code_a is None or code_b is None:
            return 0.0

        canon_a = self.alt_id_map.get(code_a, code_a)
        canon_b = self.alt_id_map.get(code_b, code_b)

        if canon_a not in self.depth or canon_b not in self.depth:
            return 0.0

        if canon_a == canon_b:
            return 1.0

        # Fast 64-bit packed integer key for unordered pair
        pair_key = (canon_a << 32) | canon_b if canon_a < canon_b else (canon_b << 32) | canon_a
        cached = self._pair_cache.get(pair_key)
        if cached is not None:
            return cached

        anc_a = self.ancestors.get(canon_a)
        anc_b = self.ancestors.get(canon_b)

        if not anc_a or not anc_b:
            self._pair_cache[pair_key] = 0.0
            return 0.0

        # Fast C-level frozenset intersection
        common = anc_a & anc_b
        if not common:
            self._pair_cache[pair_key] = 0.0
            return 0.0

        # Deepest Common Ancestor
        dca = max(common, key=self.depth.__getitem__)
        dca_depth = self.depth[dca]
        depth_a = self.depth[canon_a]
        depth_b = self.depth[canon_b]

        denom = depth_a + depth_b
        score = (2.0 * dca_depth) / float(denom) if denom > 0 else 0.0
        score = min(max(score, 0.0), 1.0)

        self._pair_cache[pair_key] = score
        return score


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_HPO_TAXONOMY: Optional[TaxonomyIndex] = None
_MONDO_TAXONOMY: Optional[TaxonomyIndex] = None


def get_hpo_taxonomy(obo_path: Union[str, Path]) -> TaxonomyIndex:
    """Get or initialize singleton integer TaxonomyIndex for HPO."""
    global _HPO_TAXONOMY
    if _HPO_TAXONOMY is None:
        _HPO_TAXONOMY = TaxonomyIndex(obo_path, prefix="HP")
    return _HPO_TAXONOMY


def get_mondo_taxonomy(obo_path: Union[str, Path]) -> TaxonomyIndex:
    """Get or initialize singleton integer TaxonomyIndex for MONDO."""
    global _MONDO_TAXONOMY
    if _MONDO_TAXONOMY is None:
        _MONDO_TAXONOMY = TaxonomyIndex(obo_path, prefix="MONDO")
    return _MONDO_TAXONOMY
