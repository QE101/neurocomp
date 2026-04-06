"""Core type definitions for the neuromorphic graph architecture."""

from enum import IntEnum


class NodeType(IntEnum):
    """Four biologically-motivated node types."""
    EXCITATORY = 0  # ~80%, two-compartment, carry representational content
    PV = 1          # ~7%, fast perisomatic inhibition, gap-junction coupled
    SST = 2         # ~7%, dendritic inhibition, gates apical input
    VIP = 3         # ~6%, inhibits SST (disinhibition / surprise circuit)


class EdgeType(IntEnum):
    """Seven edge types with distinct dynamics and routing."""
    # Chemical (5 subtypes)
    DRIVING = 0              # EXC→EXC, targets basal compartment (feedforward)
    MODULATORY = 1           # EXC→EXC, targets apical compartment (feedback)
    INHIB_PERISOMATIC = 2    # PV→EXC, scales output gain
    INHIB_DENDRITIC = 3      # SST→EXC/VIP, gates apical input
    DISINHIBITION = 4        # VIP→SST, suppresses SST output (attention/surprise)
    # Non-chemical (2 types)
    ELECTRICAL = 5           # PV↔PV, bidirectional gap junctions, non-plastic
    RETROGRADE = 6           # EXC→incoming edges, post-to-pre suppression


class NodeRole(IntEnum):
    """Functional role within the predictive coding hierarchy."""
    NONE = 0            # Not assigned (inhibitory nodes, or pre-hierarchy)
    ERROR = 1           # Computes prediction error: output ∝ |basal - apical|
    REPRESENTATION = 2  # Holds model state, generates predictions downward


class HierarchyLevel(IntEnum):
    """Which level of the hierarchy a node belongs to."""
    UNASSIGNED = 0
    LEVEL_1 = 1   # Lowest — fast, sensory
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5   # Highest — slowest, most abstract


class Compartment(IntEnum):
    """Target compartments for message routing."""
    BASAL = 0        # Bottom-up evidence (driving input)
    APICAL = 1       # Top-down predictions/context (modulatory input)
    SOMA = 2         # Output / soma (perisomatic inhibition target)


# Edge type → target compartment mapping
EDGE_TARGET_COMPARTMENT = {
    EdgeType.DRIVING: Compartment.BASAL,
    EdgeType.MODULATORY: Compartment.APICAL,
    EdgeType.INHIB_PERISOMATIC: Compartment.SOMA,
    EdgeType.INHIB_DENDRITIC: Compartment.APICAL,
    EdgeType.DISINHIBITION: Compartment.SOMA,  # suppresses SST output directly
    EdgeType.ELECTRICAL: Compartment.SOMA,      # direct coupling
    EdgeType.RETROGRADE: None,                   # targets edges, not compartments
}

# Valid source→target type constraints per edge type
EDGE_TYPE_CONSTRAINTS = {
    EdgeType.DRIVING: {
        "source_types": {NodeType.EXCITATORY},
        "target_types": {NodeType.EXCITATORY},
    },
    EdgeType.MODULATORY: {
        "source_types": {NodeType.EXCITATORY},
        "target_types": {NodeType.EXCITATORY},
    },
    EdgeType.INHIB_PERISOMATIC: {
        "source_types": {NodeType.PV},
        "target_types": {NodeType.EXCITATORY},
    },
    EdgeType.INHIB_DENDRITIC: {
        "source_types": {NodeType.SST},
        "target_types": {NodeType.EXCITATORY, NodeType.VIP},
    },
    EdgeType.DISINHIBITION: {
        "source_types": {NodeType.VIP},
        "target_types": {NodeType.SST},
    },
    EdgeType.ELECTRICAL: {
        "source_types": {NodeType.PV},
        "target_types": {NodeType.PV},
    },
    EdgeType.RETROGRADE: {
        "source_types": {NodeType.EXCITATORY},
        "target_types": {NodeType.EXCITATORY},
    },
}

# Default node type ratios
DEFAULT_NODE_RATIOS = {
    NodeType.EXCITATORY: 0.80,
    NodeType.PV: 0.07,
    NodeType.SST: 0.07,
    NodeType.VIP: 0.06,
}
